"""Training loop for metal_equirect backend."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from instasplat.metal_equirect.cameras import EquirectCamera
from instasplat.metal_equirect.dataset import EquirectDataset, load_view_tensors
from instasplat.metal_equirect.densify import DensifyState, densify_and_prune
from instasplat.metal_equirect.gaussians import (
    GaussianModel,
    export_ply,
    gaussians_from_points,
)
from instasplat.metal_equirect.metal_runtime import compile_metallib, metal_status
from instasplat.metal_equirect.rasterize import photometric_loss, rasterize_equirect


@dataclass
class TrainStats:
    steps: int
    final_loss: float
    n_gaussians: int
    device: str
    ply_path: Path | None
    elapsed_sec: float
    preview_path: Path | None = None


ProgressCallback = Callable[[dict], None]


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
    densify_every: int = 200,
    densify_from: int = 100,
    densify_until_frac: float = 0.8,
    sh_warmup_steps: int = 500,
    with_eval3d: bool = True,
    composite: str = "tile",
    prefer_mps: bool = True,
    preview_every: int = 100,
    on_progress: ProgressCallback | None = None,
    log=None,
) -> TrainStats:
    """
    Optimize Gaussians against equirect views.

    Features vs v1:
    - Tile-based (or OIT) compositing
    - Optional 3DGUT-style eval3d opacity modulation
    - MCMC densify/prune with grad accumulation
    - SH degree warmup schedule
    - Preview JPEG + progress callback for GUI / logs
    """
    device = pick_device(prefer_mps=prefer_mps)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = export_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Best-effort Metal metallib compile on macOS (non-fatal)
    status = metal_status()
    if status["xcrun"] and not status["compiled"]:
        compile_metallib()
    if log:
        log.info("metal_runtime: %s", status)

    target_sh = max(0, min(int(sh_degree), 1))
    model = gaussians_from_points(
        dataset.points_xyz,
        dataset.points_rgb,
        sh_degree=0 if sh_warmup_steps > 0 else target_sh,
        max_points=max_init_points,
        device=device,
    )
    opt = _rebuild_optimizer(model, lr)
    densify_state = DensifyState()
    densify_state.reset(model.n, device)

    if log:
        log.info(
            "metal_equirect: device=%s views=%d init_gaussians=%d steps=%d "
            "eval3d=%s composite=%s",
            device,
            len(dataset),
            model.n,
            total_steps,
            with_eval3d,
            composite,
        )

    last_loss = 0.0
    ply_path: Path | None = None
    preview_path: Path | None = None
    n_views = max(len(dataset), 1)
    densify_until = int(total_steps * densify_until_frac)

    for step in range(1, total_steps + 1):
        # SH warmup
        if target_sh >= 1 and model.sh_degree < target_sh and step >= sh_warmup_steps:
            model.set_active_sh_degree(target_sh)
            opt = _rebuild_optimizer(model, lr)
            if log:
                log.info("Enabled SH degree %d at step %d", target_sh, step)

        view = dataset.views[(step - 1) % n_views]
        rgb, mask, R, t = load_view_tensors(view, device)
        cam = EquirectCamera(view.width, view.height)

        opt.zero_grad(set_to_none=True)
        f_rest = (
            model.f_rest
            if model.sh_degree >= 1 and isinstance(model.f_rest, torch.nn.Parameter)
            else model.f_dc.new_zeros(model.n, 0)
        )
        pred = rasterize_equirect(
            model.means,
            model.get_quats(),
            model.get_scales(),
            model.get_opacity(),
            model.f_dc,
            f_rest,
            R,
            t,
            cam,
            sh_degree=model.sh_degree,
            max_gaussians=max_gaussians_render,
            with_eval3d=with_eval3d,
            composite=composite,
        )
        loss = photometric_loss(pred, rgb, cam, mask)
        loss.backward()
        densify_state.accumulate(model)
        opt.step()
        last_loss = float(loss.detach().cpu())

        if (
            densify_every > 0
            and step >= densify_from
            and step % densify_every == 0
            and step < densify_until
        ):
            dstats = densify_and_prune(
                model,
                densify_state,
                max_gaussians=max_init_points,
            )
            opt = _rebuild_optimizer(model, lr)
            if log and (dstats["cloned"] or dstats["split"] or dstats["pruned"]):
                log.info(
                    "densify step %d: %s → %d gaussians",
                    step,
                    dstats,
                    model.n,
                )

        if preview_every > 0 and (step % preview_every == 0 or step == 1):
            preview_path = _write_preview(pred, preview_dir / f"step_{step:06d}.jpg")

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
            log.info(
                "step %d/%d loss=%.5f gaussians=%d",
                step,
                total_steps,
                last_loss,
                model.n,
            )

        if on_progress is not None:
            on_progress(
                {
                    "step": step,
                    "total_steps": total_steps,
                    "loss": last_loss,
                    "n_gaussians": model.n,
                    "sh_degree": model.sh_degree,
                    "preview": str(preview_path) if preview_path else None,
                    "ply": str(ply_path) if ply_path else None,
                }
            )

    if ply_path is None:
        ply_path = export_ply(model, export_dir / "equirect_final.ply")
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
        preview_path=preview_path,
    )


def _write_preview(pred: torch.Tensor, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = (pred.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return path


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
