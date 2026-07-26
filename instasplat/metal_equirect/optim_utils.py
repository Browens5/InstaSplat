"""Adam state surgery for densify prune / clone / split (gsplat-style)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam

from instasplat.metal_equirect.gaussians import GaussianModel


_PARAM_NAMES = ("means", "opacities", "scales", "quats", "f_dc", "f_rest")


def build_optimizers(model: GaussianModel, lr: float) -> dict[str, Adam]:
    """One Adam per parameter — required for clean densify state updates."""
    opts: dict[str, Adam] = {
        "means": Adam([model.means], lr=lr * 0.1, eps=1e-15),
        "opacities": Adam([model.opacities], lr=lr * 2.0, eps=1e-15),
        "scales": Adam([model.scales], lr=lr * 0.5, eps=1e-15),
        "quats": Adam([model.quats], lr=lr * 0.1, eps=1e-15),
        "f_dc": Adam([model.f_dc], lr=lr, eps=1e-15),
    }
    if isinstance(model.f_rest, nn.Parameter) and model.f_rest.numel() > 0:
        opts["f_rest"] = Adam([model.f_rest], lr=lr * 0.25, eps=1e-15)
    return opts


def zero_grad(optimizers: dict[str, Adam]) -> None:
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)


def step_all(optimizers: dict[str, Adam]) -> None:
    for opt in optimizers.values():
        opt.step()


def _replace_param(model: GaussianModel, name: str, data: torch.Tensor) -> nn.Parameter:
    new_p = nn.Parameter(data)
    setattr(model, name, new_p)
    return new_p


def _migrate_state(
    opt: Adam,
    old_p: nn.Parameter,
    new_p: nn.Parameter,
    *,
    keep: torch.Tensor | None = None,
    extra_zeros: int = 0,
) -> None:
    """Move Adam moments from ``old_p`` to ``new_p`` (prune and/or append)."""
    group = opt.param_groups[0]
    old_state = opt.state.pop(old_p, None)
    group["params"] = [new_p]
    if old_state is None:
        return
    new_state: dict[str, Any] = {}
    for k, v in old_state.items():
        if not torch.is_tensor(v) or v.ndim == 0 or v.shape[0] != old_p.shape[0]:
            new_state[k] = v
            continue
        if keep is not None:
            v = v[keep]
        if extra_zeros > 0:
            z = torch.zeros(
                (extra_zeros, *v.shape[1:]), device=v.device, dtype=v.dtype
            )
            v = torch.cat([v, z], dim=0)
        new_state[k] = v
    opt.state[new_p] = new_state


@torch.no_grad()
def prune_with_optimizer(
    model: GaussianModel,
    optimizers: dict[str, Adam],
    keep: torch.Tensor,
) -> None:
    """Prune Gaussians and slice matching Adam moments."""
    for name in _PARAM_NAMES:
        if name == "f_rest" and (
            not isinstance(model.f_rest, nn.Parameter) or model.f_rest.numel() == 0
        ):
            continue
        old_p: nn.Parameter = getattr(model, name)
        new_p = _replace_param(model, name, old_p.data[keep].clone())
        opt = optimizers.get(name)
        if opt is not None:
            _migrate_state(opt, old_p, new_p, keep=keep)


@torch.no_grad()
def cat_with_optimizer(
    model: GaussianModel,
    optimizers: dict[str, Adam],
    extras: dict[str, torch.Tensor],
) -> None:
    """Append new Gaussians; Adam moments for new rows start at zero."""
    n_extra = next(iter(extras.values())).shape[0]
    for name, extra in extras.items():
        old_p: nn.Parameter = getattr(model, name)
        new_p = _replace_param(model, name, torch.cat([old_p.data, extra], dim=0))
        opt = optimizers.get(name)
        if opt is not None:
            _migrate_state(opt, old_p, new_p, extra_zeros=n_extra)


@torch.no_grad()
def replace_opacity_with_optimizer(
    model: GaussianModel,
    optimizers: dict[str, Adam],
    new_logits: torch.Tensor,
) -> None:
    """Overwrite opacity logits and clear Adam moments for opacities."""
    old_p = model.opacities
    new_p = _replace_param(model, "opacities", new_logits.clone())
    opt = optimizers.get("opacities")
    if opt is not None:
        # Drop moments so reset opacities aren't yanked back by Adam
        opt.state.pop(old_p, None)
        opt.param_groups[0]["params"] = [new_p]
