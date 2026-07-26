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
    clamp_sh_degree,
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


def _model_param_devices(model: GaussianModel) -> set[str]:
    return {str(p.device) for p in model.parameters()}


def _f_rest_tensor(model: GaussianModel, n: int | None = None) -> torch.Tensor:
    if isinstance(model.f_rest, torch.nn.Parameter) and model.f_rest.numel() > 0:
        return model.f_rest if n is None else model.f_rest[:n]
    n = model.n if n is None else n
    return model.f_dc.new_zeros(n, 0)


def _probe_rasterize(
    model: GaussianModel,
    device: torch.device,
    *,
    with_eval3d: bool,
    composite: str,
) -> None:
    """One tiny forward+backward to surface MPS/device bugs before the long run."""
    cam = EquirectCamera(32, 16)
    R = torch.eye(3, device=device, dtype=model.means.dtype)
    t = torch.zeros(3, device=device, dtype=model.means.dtype)
    n = min(model.n, 64)
    pred = rasterize_equirect(
        model.means[:n],
        model.get_quats()[:n],
        model.get_scales()[:n],
        model.get_opacity()[:n],
        model.f_dc[:n],
        _f_rest_tensor(model, n),
        R,
        t,
        cam,
        sh_degree=model.sh_degree,
        max_gaussians=n,
        with_eval3d=with_eval3d,
        composite=composite,
        tile_size=8,
        max_per_tile=16,
    )
    loss = pred.mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _build_model(
    dataset: EquirectDataset,
    *,
    target_sh: int,
    sh_warmup_steps: int,
    max_gaussians: int,
    device: torch.device,
) -> GaussianModel:
    active = 0 if sh_warmup_steps > 0 and target_sh > 0 else target_sh
    return gaussians_from_points(
        dataset.points_xyz,
        dataset.points_rgb,
        sh_degree=active,
        max_sh_degree=target_sh,
        max_points=max_gaussians,
        device=device,
    )


def train_equirect(
    dataset: EquirectDataset,
    export_dir: Path,
    *,
    total_steps: int = 5_000,
    export_every: int = 500,
    viewer_every: int = 25,
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

    - Tile / OIT compositing, optional eval3d
    - SH degree 0–3 with stepwise warmup
    - Incremental PLY every ``export_every``; ``live.ply`` every ``viewer_every``
    - MPS probe with automatic CPU fallback
    """
    device = pick_device(prefer_mps=prefer_mps)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = export_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    status = metal_status()
    if status["xcrun"] and not status["compiled"]:
        compile_metallib()
    if log:
        log.info("metal_runtime: %s", status)

    target_sh = clamp_sh_degree(sh_degree)
    max_gaussians = max(1_000, int(max_init_points))
    export_every = max(0, int(export_every))
    viewer_every = max(0, int(viewer_every))

    model = _build_model(
        dataset,
        target_sh=target_sh,
        sh_warmup_steps=sh_warmup_steps,
        max_gaussians=max_gaussians,
        device=device,
    )
    devices = _model_param_devices(model)
    if len(devices) != 1:
        raise RuntimeError(
            f"GaussianModel parameters span multiple devices {devices}; "
            "all must live on the training device."
        )

    if device.type == "mps":
        try:
            _probe_rasterize(
                model, device, with_eval3d=with_eval3d, composite=composite
            )
        except RuntimeError as exc:
            msg = str(exc)
            if log:
                log.warning(
                    "MPS probe failed (%s); falling back to CPU for training",
                    msg.splitlines()[0][:160],
                )
            device = torch.device("cpu")
            model = _build_model(
                dataset,
                target_sh=target_sh,
                sh_warmup_steps=sh_warmup_steps,
                max_gaussians=max_gaussians,
                device=device,
            )

    opt = _rebuild_optimizer(model, lr)
    densify_state = DensifyState()
    densify_state.reset(model.n, device)

    if log:
        log.info(
            "metal_equirect: device=%s views=%d init_gaussians=%d steps=%d "
            "sh=%d/%d max_gaussians=%d export_every=%d viewer_every=%d "
            "eval3d=%s composite=%s",
            device,
            len(dataset),
            model.n,
            total_steps,
            model.sh_degree,
            target_sh,
            max_gaussians,
            export_every,
            viewer_every,
            with_eval3d,
            composite,
        )

    last_loss = 0.0
    ply_path: Path | None = None
    preview_path: Path | None = None
    live_path: Path | None = None
    n_views = max(len(dataset), 1)
    densify_until = int(total_steps * densify_until_frac)
    # Progressive SH: bump one degree every sh_warmup_steps
    next_sh_step = int(sh_warmup_steps) if sh_warmup_steps > 0 else 0

    for step in range(1, total_steps + 1):
        if (
            target_sh > model.sh_degree
            and next_sh_step > 0
            and step >= next_sh_step
        ):
            new_deg = min(target_sh, model.sh_degree + 1)
            model.set_active_sh_degree(new_deg)
            opt = _rebuild_optimizer(model, lr)
            next_sh_step = step + int(sh_warmup_steps)
            if log:
                log.info("Enabled SH degree %d at step %d", new_deg, step)

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
            _f_rest_tensor(model),
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
                max_gaussians=max_gaussians,
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

        # Live viewer PLY (overwrite) — GUI polls this
        if viewer_every > 0 and (step % viewer_every == 0 or step == 1):
            live_path = export_ply(model, export_dir / "live.ply")

        if export_every > 0 and (step % export_every == 0 or step == total_steps):
            ply_path = export_ply(model, export_dir / f"equirect_{step:06d}.ply")
            if log:
                log.info(
                    "step %d/%d loss=%.5f gaussians=%d sh=%d → %s",
                    step,
                    total_steps,
                    last_loss,
                    model.n,
                    model.sh_degree,
                    ply_path.name,
                )
        elif log and step % 50 == 0:
            log.info(
                "step %d/%d loss=%.5f gaussians=%d sh=%d",
                step,
                total_steps,
                last_loss,
                model.n,
                model.sh_degree,
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
                    "live_ply": str(live_path) if live_path else None,
                }
            )

    if ply_path is None:
        ply_path = export_ply(model, export_dir / "equirect_final.ply")
    final = export_dir / "scene.ply"
    export_ply(model, final)
    export_ply(model, export_dir / "live.ply")
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
    if isinstance(model.f_rest, torch.nn.Parameter) and model.f_rest.numel() > 0:
        params.append({"params": [model.f_rest], "lr": lr * 0.25})
    return torch.optim.Adam(params, lr=lr, eps=1e-15)
