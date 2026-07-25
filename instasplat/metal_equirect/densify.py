"""MCMC-style densification / pruning (gsplat 3DGUT uses MCMC)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from instasplat.metal_equirect.gaussians import GaussianModel


@dataclass
class DensifyState:
    """Running abs-grad accumulator for means."""

    xyz_grad_norm: torch.Tensor | None = None
    denom: torch.Tensor | None = None

    def reset(self, n: int, device: torch.device) -> None:
        self.xyz_grad_norm = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)

    def ensure(self, n: int, device: torch.device) -> None:
        if (
            self.xyz_grad_norm is None
            or self.denom is None
            or self.xyz_grad_norm.numel() != n
        ):
            self.reset(n, device)

    @torch.no_grad()
    def accumulate(self, model: GaussianModel) -> None:
        if model.means.grad is None:
            return
        self.ensure(model.n, model.means.device)
        g = model.means.grad.detach().norm(dim=-1)
        assert self.xyz_grad_norm is not None and self.denom is not None
        self.xyz_grad_norm += g
        self.denom += 1.0

    def mean_grad(self) -> torch.Tensor:
        assert self.xyz_grad_norm is not None and self.denom is not None
        return self.xyz_grad_norm / self.denom.clamp(min=1.0)


def _cat_param(model: GaussianModel, name: str, extra: torch.Tensor) -> None:
    param = getattr(model, name)
    setattr(model, name, nn.Parameter(torch.cat([param.data, extra], dim=0)))


@torch.no_grad()
def _split_gaussians(model: GaussianModel, idx: torch.Tensor, scale_div: float = 1.6) -> None:
    if idx.numel() == 0:
        return
    device = model.means.device
    means = model.means.data[idx].clone()
    scales = model.get_scales()[idx]
    quats = model.get_quats()[idx].clone()
    f_dc = model.f_dc.data[idx].clone()
    opac = model.opacities.data[idx].clone() - 0.5
    log_scales = model.scales.data[idx].clone() - torch.log(
        torch.tensor(scale_div, device=device, dtype=means.dtype)
    )
    f_rest = None
    if model.sh_degree >= 1 and isinstance(model.f_rest, nn.Parameter) and model.f_rest.numel():
        f_rest = model.f_rest.data[idx].clone()

    eps = torch.randn_like(means) * (scales / scale_div)
    child_means = torch.cat([means + eps, means - eps], dim=0)
    child_opac = torch.cat([opac, opac], dim=0)
    child_scales = torch.cat([log_scales, log_scales], dim=0)
    child_quats = torch.cat([quats, quats], dim=0)
    child_fdc = torch.cat([f_dc, f_dc], dim=0)
    child_frest = torch.cat([f_rest, f_rest], dim=0) if f_rest is not None else None

    keep = torch.ones(model.n, dtype=torch.bool, device=device)
    keep[idx] = False
    model.prune_mask(keep)

    _cat_param(model, "means", child_means)
    _cat_param(model, "opacities", child_opac)
    _cat_param(model, "scales", child_scales)
    _cat_param(model, "quats", child_quats)
    _cat_param(model, "f_dc", child_fdc)
    if child_frest is not None:
        _cat_param(model, "f_rest", child_frest)


@torch.no_grad()
def densify_and_prune(
    model: GaussianModel,
    state: DensifyState,
    *,
    grad_threshold: float = 0.0002,
    min_opacity: float = 0.005,
    max_scale: float = 0.5,
    max_gaussians: int = 80_000,
    clone_scale_frac: float = 0.02,
) -> dict[str, int]:
    """
    Clone small high-grad Gaussians, split large ones, prune transparent/huge.

    Inspired by 3DGS densify + gsplat MCMC (simplified for Metal equirect).
    """
    stats = {"cloned": 0, "split": 0, "pruned": 0, "relocated": 0}
    if model.n == 0:
        return stats
    state.ensure(model.n, model.means.device)
    grads = state.mean_grad()
    opac = model.get_opacity()
    scales = model.get_scales()
    scale_max = scales.max(dim=-1).values

    keep = (opac > min_opacity) & (scale_max < max_scale * 10)
    n_before = model.n
    if int(keep.sum()) < n_before and int(keep.sum()) > 16:
        model.prune_mask(keep)
        grads = grads[keep]
        opac = model.get_opacity()
        scales = model.get_scales()
        scale_max = scales.max(dim=-1).values
        stats["pruned"] = n_before - model.n

    if model.n >= max_gaussians:
        state.reset(model.n, model.means.device)
        return stats

    high_grad = grads >= grad_threshold
    clone_mask = high_grad & (scale_max <= clone_scale_frac)
    split_mask = high_grad & (scale_max > clone_scale_frac)

    clone_idx = torch.nonzero(clone_mask, as_tuple=False).view(-1)
    split_idx = torch.nonzero(split_mask, as_tuple=False).view(-1)

    room = max_gaussians - model.n
    if clone_idx.numel() > 0 and room > 0:
        clone_idx = clone_idx[:room]
        model.densify_clone(clone_idx, scale_div=1.0)
        stats["cloned"] = int(clone_idx.numel())
        room = max_gaussians - model.n

    if split_idx.numel() > 0 and room > 0:
        # Recompute indices after clone (masks invalid) — only split from original set
        # that still exist: use top high-grad large scales again
        opac = model.get_opacity()
        scales = model.get_scales()
        scale_max = scales.max(dim=-1).values
        # Approximate: split largest high-opacity gaussians
        score = opac * scale_max
        k = min(int(split_idx.numel()), max(0, room // 2), max(1, model.n // 40))
        if k > 0:
            new_split = torch.topk(score, k).indices
            _split_gaussians(model, new_split)
            stats["split"] = int(new_split.numel())

    # MCMC-lite relocate near-transparent Gaussians
    opac = model.get_opacity()
    scales = model.get_scales()
    soft = (opac < 0.05) & (opac > min_opacity)
    soft_idx = torch.nonzero(soft, as_tuple=False).view(-1)
    if soft_idx.numel() > 0:
        noise = torch.randn_like(model.means.data[soft_idx]) * scales[soft_idx].mean(
            dim=-1, keepdim=True
        ).clamp(min=1e-4)
        model.means.data[soft_idx] += noise
        stats["relocated"] = int(soft_idx.numel())

    state.reset(model.n, model.means.device)
    return stats
