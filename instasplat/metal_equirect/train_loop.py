"""Training loop for metal_equirect backend."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from instasplat.metal_equirect.cameras import EquirectCamera
from instasplat.metal_equirect.dataset import EquirectDataset, load_view_tensors
from instasplat.metal_equirect.gaussians import (
    GaussianModel,
    export_ply,
    gaussians_from_points,
)
from instasplat.metal_equirect.rasterize import photometric_loss, rasterize_equirect


@dataclass
class TrainStats:
    steps: int
    final_loss: float
    n_gaussians: int
    device: str
    ply_path: Path | None
    elapsed_sec: float


def pick_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_equirect(
    dataset: EquirectDataset,
    export_dir: Path,
    *,
    total_steps: int = 5_000,
    export_every: int = 1_000,
    sh_degree: int = 1,
    lr: float = 0.01,
    max_gaussians_render: int = 8_000,
    max_init_points: int = 40_000,
    densify_every: int = 500,
    prefer_mps: bool = True,
    log=None,
) -> TrainStats:
    """
    Optimize Gaussians against equirect views.

    This is a Mac-native reference trainer: PyTorch MPS/CPU autodiff with a
    3DGUT-style UT equirect projector. Metal compute shaders under
    ``metal_equirect/metal/`` accelerate projection when compiled on macOS.
    """
    device = pick_device(prefer_mps=prefer_mps)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model = gaussians_from_points(
        dataset.points_xyz,
        dataset.points_rgb,
        sh_degree=sh_degree,
        max_points=max_init_points,
        device=device,
    )
    # Learning rates: means slower than appearance
    opt = torch.optim.Adam(
        [
            {"params": [model.means], "lr": lr * 0.1},
            {"params": [model.opacities], "lr": lr * 2.0},
            {"params": [model.scales], "lr": lr * 0.5},
            {"params": [model.quats], "lr": lr * 0.1},
            {"params": [model.f_dc], "lr": lr},
        ]
        + (
            [{"params": [model.f_rest], "lr": lr * 0.25}]
            if model.sh_degree >= 1 and isinstance(model.f_rest, torch.nn.Parameter)
            else []
        ),
        lr=lr,
        eps=1e-15,
    )

    if log:
        log.info(
            "metal_equirect: device=%s views=%d init_gaussians=%d steps=%d",
            device,
            len(dataset),
            model.n,
            total_steps,
        )

    last_loss = 0.0
    ply_path: Path | None = None
    n_views = max(len(dataset), 1)

    for step in range(1, total_steps + 1):
        view = dataset.views[(step - 1) % n_views]
        rgb, mask, R, t = load_view_tensors(view, device)
        cam = EquirectCamera(view.width, view.height)

        opt.zero_grad(set_to_none=True)
        pred = rasterize_equirect(
            model.means,
            model.get_quats(),
            model.get_scales(),
            model.get_opacity(),
            model.f_dc,
            model.f_rest if model.sh_degree >= 1 else model.f_dc.new_zeros(model.n, 0),
            R,
            t,
            cam,
            sh_degree=model.sh_degree,
            max_gaussians=max_gaussians_render,
        )
        loss = photometric_loss(pred, rgb, cam, mask)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

        if densify_every > 0 and step % densify_every == 0 and step < total_steps * 0.8:
            with torch.no_grad():
                opac = model.get_opacity()
                # Prune nearly transparent
                keep = opac > 0.005
                if keep.sum() < model.n and keep.sum() > 32:
                    model.prune_mask(keep)
                    opt = _rebuild_optimizer(model, lr)
                # Clone high-opacity large scales (MCMC-lite)
                score = opac * model.get_scales().mean(dim=-1)
                if model.n < max_init_points:
                    k = min(64, max(1, model.n // 50))
                    idx = torch.topk(score, k).indices
                    model.densify_clone(idx)
                    opt = _rebuild_optimizer(model, lr)

        if export_every > 0 and (step % export_every == 0 or step == total_steps):
            ply_path = export_ply(model, export_dir / f"equirect_{step:06d}.ply")
            if log:
                log.info(
                    "step %d/%d loss=%.5f gaussians=%d → %s",
                    step,
                    total_steps,
                    last_loss,
                    model.n,
                    ply_path.name,
                )
        elif log and step % 50 == 0:
            log.info("step %d/%d loss=%.5f gaussians=%d", step, total_steps, last_loss, model.n)

    if ply_path is None:
        ply_path = export_ply(model, export_dir / "equirect_final.ply")
    # Also write a stable name for export stage discovery
    final = export_dir / "scene.ply"
    export_ply(model, final)
    ply_path = final

    return TrainStats(
        steps=total_steps,
        final_loss=last_loss,
        n_gaussians=model.n,
        device=str(device),
        ply_path=ply_path,
        elapsed_sec=time.time() - t0,
    )


def _rebuild_optimizer(model: GaussianModel, lr: float) -> torch.optim.Adam:
    params = [
        {"params": [model.means], "lr": lr * 0.1},
        {"params": [model.opacities], "lr": lr * 2.0},
        {"params": [model.scales], "lr": lr * 0.5},
        {"params": [model.quats], "lr": lr * 0.1},
        {"params": [model.f_dc], "lr": lr},
    ]
    if model.sh_degree >= 1 and isinstance(model.f_rest, torch.nn.Parameter):
        params.append({"params": [model.f_rest], "lr": lr * 0.25})
    return torch.optim.Adam(params, lr=lr, eps=1e-15)
