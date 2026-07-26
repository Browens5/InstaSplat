"""gsplat-style densification: 2D absgrad, Adam-preserving prune/clone/split."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from instasplat.metal_equirect.gaussians import GaussianModel
from instasplat.metal_equirect.optim_utils import (
    cat_with_optimizer,
    prune_with_optimizer,
    replace_opacity_with_optimizer,
)

# Prefer pure PyTorch selection above this size (Rust path copies full arrays to CPU).
_RUST_SELECT_MAX_N = 100_000


@dataclass
class DensifyState:
    """Running screen-space absgrad accumulator (AbsGS / gsplat DefaultStrategy)."""

    absgrad_2d: torch.Tensor | None = None
    denom: torch.Tensor | None = None
    # Fallback when 2D grads unavailable
    xyz_grad_norm: torch.Tensor | None = None

    def reset(self, n: int, device: torch.device) -> None:
        self.absgrad_2d = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)
        self.xyz_grad_norm = torch.zeros(n, device=device)

    def ensure(self, n: int, device: torch.device) -> None:
        if (
            self.absgrad_2d is None
            or self.denom is None
            or self.absgrad_2d.numel() != n
            or self.absgrad_2d.device != device
        ):
            self.reset(n, device)

    @torch.no_grad()
    def accumulate(
        self,
        model: GaussianModel,
        *,
        mean_2d: torch.Tensor | None = None,
        selected: torch.Tensor | None = None,
    ) -> None:
        """
        Prefer ``|grad mean_2d|`` (screen-space). Fall back to 3D ``means.grad``.

        ``selected`` maps rows of ``mean_2d`` back into the full model when the
        rasterizer capped the Gaussian set.
        """
        self.ensure(model.n, model.means.device)
        assert self.absgrad_2d is not None and self.denom is not None
        assert self.xyz_grad_norm is not None

        used_2d = False
        if mean_2d is not None and mean_2d.grad is not None:
            g = mean_2d.grad.detach().norm(dim=-1)
            if selected is None:
                if g.numel() == model.n:
                    self.absgrad_2d += g
                    self.denom += 1.0
                    used_2d = True
            else:
                sel = selected.long()
                self.absgrad_2d.index_add_(0, sel, g)
                ones = torch.ones_like(g)
                self.denom.index_add_(0, sel, ones)
                used_2d = True

        if not used_2d and model.means.grad is not None:
            g3 = model.means.grad.detach().norm(dim=-1)
            self.xyz_grad_norm += g3
            self.denom += 1.0

    def mean_grad(self) -> torch.Tensor:
        assert self.absgrad_2d is not None and self.denom is not None
        assert self.xyz_grad_norm is not None
        denom = self.denom.clamp(min=1.0)
        # Prefer 2D absgrad when it was ever accumulated
        if float(self.absgrad_2d.sum()) > 0:
            return self.absgrad_2d / denom
        return self.xyz_grad_norm / denom


def densify_phase(step: int, densify_from: int, densify_until: int) -> str:
    """Phase over the densify window: split early → balanced → clone late."""
    span = max(int(densify_until) - int(densify_from), 1)
    t = float(step - densify_from) / float(span)
    t = max(0.0, min(1.0, t))
    if t <= 0.30:
        return "split"
    if t <= 0.70:
        return "balanced"
    return "clone"


def grad_threshold_at(
    step: int,
    densify_from: int,
    densify_until: int,
    *,
    base: float = 0.0002,
) -> float:
    """Ascending threshold over the densify window (easy early, harder late)."""
    span = max(int(densify_until) - int(densify_from), 1)
    t = float(step - densify_from) / float(span)
    t = max(0.0, min(1.0, t))
    # ~0.5× → 2× base
    return float(base) * (0.5 + 1.5 * t)


def _select_clone_split_indices(
    grads: torch.Tensor,
    scale_max: torch.Tensor,
    *,
    grad_threshold: float,
    clone_scale: float,
    phase: str,
    room: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (clone_idx, split_idx). Prefers Rust helper when available and cheap."""
    n = int(grads.numel())
    if n > 0 and n <= _RUST_SELECT_MAX_N:
        try:
            from instasplat_densify import select_clone_split  # type: ignore

            c_idx, s_idx = select_clone_split(
                grads.detach().float().cpu().numpy(),
                scale_max.detach().float().cpu().numpy(),
                float(grad_threshold),
                float(clone_scale),
                str(phase),
                int(room),
            )
            device = grads.device
            return (
                torch.as_tensor(list(c_idx), device=device, dtype=torch.long),
                torch.as_tensor(list(s_idx), device=device, dtype=torch.long),
            )
        except Exception:
            pass

    high = grads >= grad_threshold
    clone_mask = high & (scale_max <= clone_scale)
    split_mask = high & (scale_max > clone_scale)
    if phase == "split":
        # Suppress most clones early (global coverage)
        clone_idx = torch.nonzero(clone_mask, as_tuple=False).view(-1)
        if clone_idx.numel() > 0:
            k = max(1, clone_idx.numel() // 4)
            clone_idx = clone_idx[torch.topk(grads[clone_idx], k).indices]
        split_idx = torch.nonzero(split_mask, as_tuple=False).view(-1)
    elif phase == "clone":
        clone_idx = torch.nonzero(clone_mask, as_tuple=False).view(-1)
        split_idx = torch.nonzero(split_mask, as_tuple=False).view(-1)
        if split_idx.numel() > 0:
            k = max(1, split_idx.numel() // 4)
            split_idx = split_idx[torch.topk(grads[split_idx], k).indices]
    else:
        clone_idx = torch.nonzero(clone_mask, as_tuple=False).view(-1)
        split_idx = torch.nonzero(split_mask, as_tuple=False).view(-1)

    # Budget: splits and clones each consume 1 net slot
    if room <= 0:
        empty = grads.new_zeros(0, dtype=torch.long)
        return empty, empty

    # Prioritize by grad magnitude
    if clone_idx.numel() > 0:
        order = torch.argsort(grads[clone_idx], descending=True)
        clone_idx = clone_idx[order]
    if split_idx.numel() > 0:
        order = torch.argsort(grads[split_idx], descending=True)
        split_idx = split_idx[order]

    # Allocate room: phase bias (1 slot each for clone or split)
    if phase == "split":
        split_budget = min(split_idx.numel(), room)
        clone_budget = min(clone_idx.numel(), max(0, room - split_budget))
    elif phase == "clone":
        clone_budget = min(clone_idx.numel(), room)
        split_budget = min(split_idx.numel(), max(0, room - clone_budget))
    else:
        clone_budget = min(clone_idx.numel(), max(0, (room + 1) // 2))
        split_budget = min(split_idx.numel(), max(0, room - clone_budget))

    return clone_idx[:clone_budget], split_idx[:split_budget]


@torch.no_grad()
def _clone(
    model: GaussianModel,
    optimizers: dict,
    idx: torch.Tensor,
) -> int:
    if idx.numel() == 0:
        return 0
    extras = {
        "means": model.means.data[idx].clone(),
        "opacities": model.opacities.data[idx].clone(),
        "scales": model.scales.data[idx].clone(),
        "quats": model.quats.data[idx].clone(),
        "f_dc": model.f_dc.data[idx].clone(),
    }
    if isinstance(model.f_rest, torch.nn.Parameter) and model.f_rest.numel() > 0:
        extras["f_rest"] = model.f_rest.data[idx].clone()
    cat_with_optimizer(model, optimizers, extras)
    return int(idx.numel())


@torch.no_grad()
def _split(
    model: GaussianModel,
    optimizers: dict,
    idx: torch.Tensor,
    *,
    scale_div: float = 1.6,
    child_opacity_frac: float = 0.6,
) -> int:
    """Split large Gaussians along a random offset; children get reduced opacity."""
    if idx.numel() == 0:
        return 0
    device = model.means.device
    dtype = model.means.dtype
    means = model.means.data[idx]
    scales = model.get_scales()[idx]
    log_scales = model.scales.data[idx] - torch.log(
        torch.tensor(scale_div, device=device, dtype=dtype)
    )
    # Revised opacity: child logit ≈ logit(σ * frac)
    opac_prob = torch.sigmoid(model.opacities.data[idx]) * child_opacity_frac
    child_logit = torch.logit(opac_prob.clamp(1e-4, 1.0 - 1e-4))
    quats = model.quats.data[idx]
    f_dc = model.f_dc.data[idx]
    f_rest = (
        model.f_rest.data[idx]
        if isinstance(model.f_rest, torch.nn.Parameter) and model.f_rest.numel()
        else None
    )

    eps = torch.randn_like(means) * (scales / scale_div)
    child_means = torch.cat([means + eps, means - eps], dim=0)
    child_opac = torch.cat([child_logit, child_logit], dim=0)
    child_scales = torch.cat([log_scales, log_scales], dim=0)
    child_quats = torch.cat([quats, quats], dim=0)
    child_fdc = torch.cat([f_dc, f_dc], dim=0)

    keep = torch.ones(model.n, dtype=torch.bool, device=device)
    keep[idx] = False
    prune_with_optimizer(model, optimizers, keep)

    extras = {
        "means": child_means,
        "opacities": child_opac,
        "scales": child_scales,
        "quats": child_quats,
        "f_dc": child_fdc,
    }
    if f_rest is not None:
        extras["f_rest"] = torch.cat([f_rest, f_rest], dim=0)
    cat_with_optimizer(model, optimizers, extras)
    return int(idx.numel())


@torch.no_grad()
def _split_no_opt(model: GaussianModel, idx: torch.Tensor, scale_div: float = 1.6) -> int:
    """Split without optimizer (tests / fallback)."""
    if idx.numel() == 0:
        return 0
    device = model.means.device
    dtype = model.means.dtype
    means = model.means.data[idx].clone()
    scales = model.get_scales()[idx]
    log_scales = model.scales.data[idx].clone() - torch.log(
        torch.tensor(scale_div, device=device, dtype=dtype)
    )
    opac_prob = torch.sigmoid(model.opacities.data[idx]) * 0.6
    child_logit = torch.logit(opac_prob.clamp(1e-4, 1.0 - 1e-4))
    quats = model.quats.data[idx].clone()
    f_dc = model.f_dc.data[idx].clone()
    f_rest = None
    if isinstance(model.f_rest, torch.nn.Parameter) and model.f_rest.numel():
        f_rest = model.f_rest.data[idx].clone()
    eps = torch.randn_like(means) * (scales / scale_div)
    keep = torch.ones(model.n, dtype=torch.bool, device=device)
    keep[idx] = False
    model.prune_mask(keep)
    for name, extra in {
        "means": torch.cat([means + eps, means - eps], dim=0),
        "opacities": torch.cat([child_logit, child_logit], dim=0),
        "scales": torch.cat([log_scales, log_scales], dim=0),
        "quats": torch.cat([quats, quats], dim=0),
        "f_dc": torch.cat([f_dc, f_dc], dim=0),
    }.items():
        param = getattr(model, name)
        setattr(model, name, torch.nn.Parameter(torch.cat([param.data, extra], dim=0)))
    if f_rest is not None:
        model.f_rest = torch.nn.Parameter(
            torch.cat([model.f_rest.data, torch.cat([f_rest, f_rest], dim=0)], dim=0)
        )
    return int(idx.numel())


@torch.no_grad()
def opacity_reset(
    model: GaussianModel,
    optimizers: dict,
    *,
    value: float = 0.01,
) -> None:
    """Periodic opacity reset (3DGS) so prune can drop dead Gaussians."""
    target = torch.full(
        (model.n,),
        float(value),
        device=model.means.device,
        dtype=model.means.dtype,
    )
    logits = torch.logit(target.clamp(1e-4, 1.0 - 1e-4))
    replace_opacity_with_optimizer(model, optimizers, logits)


@torch.no_grad()
def densify_and_prune(
    model: GaussianModel,
    state: DensifyState,
    optimizers: dict | None = None,
    *,
    grad_threshold: float = 0.0002,
    min_opacity: float = 0.005,
    max_scale: float = 0.5,
    max_gaussians: int = 80_000,
    clone_scale_frac: float = 0.01,
    scene_extent: float | None = None,
    phase: str = "balanced",
) -> dict[str, int]:
    """
    Clone / split / prune with optional Adam-preserving updates.

    When ``optimizers`` is None, falls back to Parameter realloc (tests).

    ``clone_scale_frac`` is relative to ``scene_extent`` (default: mean
    pairwise scale proxy from model means) — same idea as 3DGS percent_dense.
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
    n_keep = int(keep.sum().item())
    # Always cull dead Gaussians when at least one would survive
    if n_keep < n_before and n_keep >= 1:
        if optimizers is not None:
            prune_with_optimizer(model, optimizers, keep)
        else:
            model.prune_mask(keep)
        grads = grads[keep]
        opac = model.get_opacity()
        scales = model.get_scales()
        scale_max = scales.max(dim=-1).values
        stats["pruned"] = n_before - model.n

    if model.n >= max_gaussians:
        state.reset(model.n, model.means.device)
        return stats

    # Never densify near-transparent survivors (e.g. prune skipped when n_keep==0)
    alive = opac > min_opacity
    if not bool(alive.any()):
        state.reset(model.n, model.means.device)
        return stats
    grads = grads.clone()
    grads[~alive] = 0.0

    if scene_extent is None:
        # Robust proxy: median distance from centroid
        c = model.means.data.mean(dim=0, keepdim=True)
        dist = (model.means.data - c).norm(dim=-1)
        scene_extent = float(dist.median().clamp(min=1e-3).item())
    clone_scale = float(clone_scale_frac) * float(scene_extent)

    room = max_gaussians - model.n
    clone_idx, split_idx = _select_clone_split_indices(
        grads,
        scale_max,
        grad_threshold=grad_threshold,
        clone_scale=clone_scale,
        phase=phase,
        room=room,
    )

    if clone_idx.numel() > 0:
        if optimizers is not None:
            stats["cloned"] = _clone(model, optimizers, clone_idx)
        else:
            model.densify_clone(clone_idx, scale_div=1.0)
            stats["cloned"] = int(clone_idx.numel())
        # After clone, split indices still refer to pre-clone ids (valid: clones append)
        room = max_gaussians - model.n

    if split_idx.numel() > 0 and room > 0:
        # Re-filter split_idx still in range after prune; clones appended after n0
        split_idx = split_idx[split_idx < (model.n - stats["cloned"])]
        k = min(int(split_idx.numel()), room)
        split_idx = split_idx[:k]
        if split_idx.numel() > 0:
            if optimizers is not None:
                stats["split"] = _split(model, optimizers, split_idx)
            else:
                stats["split"] = _split_no_opt(model, split_idx)

    state.reset(model.n, model.means.device)
    return stats


# Back-compat name used by older tests that only pass model+state
def densify_and_prune_simple(model: GaussianModel, state: DensifyState, **kwargs):
    return densify_and_prune(model, state, optimizers=None, **kwargs)
