"""Training loop for metal_equirect backend."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from instasplat.metal_equirect.cameras import EquirectCamera
from instasplat.metal_equirect.dataset import EquirectDataset
from instasplat.metal_equirect.densify import (
    DensifyState,
    densify_and_prune,
    densify_phase,
    grad_threshold_at,
    opacity_reset,
)
from instasplat.metal_equirect.gaussians import (
    GaussianModel,
    clamp_sh_degree,
    export_ply,
    gaussians_from_points,
)
from instasplat.metal_equirect.metal_runtime import compile_metallib, metal_status
from instasplat.metal_equirect.optim_utils import build_optimizers, step_all, zero_grad
from instasplat.metal_equirect.rasterize import photometric_loss, rasterize_equirect
from instasplat.metal_equirect.schedule import schedule_at_step
from instasplat.metal_equirect.view_cache import ViewCache


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
        prefer_metal=False,
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


class _AsyncLiveExporter:
    """Background subsampled live.ply writer — never blocks the train step."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending: tuple | None = None

    def submit(
        self,
        model: GaussianModel,
        path: Path,
        *,
        max_points: int = 8_000,
    ) -> None:
        # Snapshot on the training thread (CPU numpy) so the model can keep mutating.
        with torch.no_grad():
            n_full = model.n
            idx = None
            if n_full > max_points:
                score = model.get_opacity().detach()
                idx = torch.topk(score, max_points).indices.cpu()
            means = model.means.detach().cpu()
            scales = model.get_scales().detach().cpu()
            quats = model.get_quats().detach().cpu()
            opacity = model.opacities.detach().cpu()
            f_dc = model.f_dc.detach().cpu()
            if idx is not None:
                means = means[idx]
                scales = scales[idx]
                quats = quats[idx]
                opacity = opacity[idx]
                f_dc = f_dc[idx]
            snap = (
                means.numpy(),
                scales.numpy(),
                quats.numpy(),
                opacity.numpy(),
                f_dc.numpy(),
            )
        with self._lock:
            self._pending = (snap, Path(path))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()

    def _worker(self) -> None:
        while True:
            with self._lock:
                item = self._pending
                self._pending = None
            if item is None:
                return
            snap, path = item
            try:
                _write_ply_arrays(snap, path)
            except OSError:
                pass

    def flush(self, timeout: float = 30.0) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)


def _write_ply_arrays(
    snap: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    path: Path,
) -> Path:
    means, scales, quats, opacity, f_dc = snap
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = means.shape[0]
    props = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    )
    zeros = np.zeros((n, 3), dtype=np.float32)
    log_scales = np.log(scales.astype(np.float32) + 1e-8)
    rows = np.concatenate(
        [
            means.astype(np.float32),
            zeros,
            f_dc.astype(np.float32),
            opacity.astype(np.float32).reshape(-1, 1),
            log_scales,
            quats.astype(np.float32),
        ],
        axis=1,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.astype("<f4").tobytes())
    tmp.replace(path)
    return path


def train_equirect(
    dataset: EquirectDataset,
    export_dir: Path,
    *,
    total_steps: int = 5_000,
    export_every: int = 500,
    viewer_every: int = 100,
    sh_degree: int = 1,
    lr: float = 0.01,
    max_gaussians_render: int = 8_000,
    max_init_points: int = 40_000,
    densify_every: int = 100,
    densify_from: int = 100,
    densify_until_frac: float = 0.6,
    opacity_reset_every: int = 3_000,
    sh_warmup_steps: int = 500,
    with_eval3d: bool = True,
    composite: str = "oit",
    prefer_mps: bool = True,
    preview_every: int = 100,
    live_max_points: int = 100_000,
    cache_views: bool = True,
    use_resolution_schedule: bool = True,
    on_progress: ProgressCallback | None = None,
    log=None,
) -> TrainStats:
    """
    Optimize Gaussians against equirect views.

    Speed path:
    - Vectorized OIT / optional Metal fused forward
    - AbsGS 2D absgrad densify with Adam-preserving prune/clone/split
    - Phased split→clone over densify window + opacity reset; optional Rust index helper
    - View cache + progressive resolution schedule
    - Async subsampled ``live.ply`` for the GUI viewer
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
        status = metal_status()
    if log:
        log.info("metal_runtime: %s", status)
        if composite == "metal":
            if status.get("dispatch"):
                log.info(
                    "Fused Metal composite enabled (shared_buffers=%s); "
                    "torch OIT for backward",
                    bool(status.get("shared_buffers")),
                )
            else:
                log.info(
                    "composite=metal requested but Metal dispatch unavailable; "
                    "using vectorized torch OIT"
                )

    target_sh = clamp_sh_degree(sh_degree)
    max_gaussians = max(1_000, int(max_init_points))
    export_every = max(0, int(export_every))
    viewer_every = max(0, int(viewer_every))
    composite = composite if composite in {"tile", "oit", "metal"} else "oit"

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

    opt = build_optimizers(model, lr)
    densify_state = DensifyState()
    densify_state.reset(model.n, device)

    view_cache: ViewCache | None = None
    if cache_views:
        view_cache = ViewCache(device, max_views=max(8, len(dataset.views)))
        n_cached = view_cache.preload(dataset.views)
        if log:
            log.info("ViewCache: preloaded %d / %d panoramas on %s", n_cached, len(dataset), device)

    live_exporter = _AsyncLiveExporter()

    if log:
        log.info(
            "metal_equirect: device=%s views=%d init_gaussians=%d steps=%d "
            "sh=%d/%d max_gaussians=%d export_every=%d viewer_every=%d "
            "eval3d=%s composite=%s schedule=%s",
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
            use_resolution_schedule,
        )

    last_loss = 0.0
    ply_path: Path | None = None
    preview_path: Path | None = None
    live_path: Path | None = None
    n_views = max(len(dataset), 1)
    densify_until = int(total_steps * densify_until_frac)
    next_sh_step = int(sh_warmup_steps) if sh_warmup_steps > 0 else 0

    for step in range(1, total_steps + 1):
        if (
            target_sh > model.sh_degree
            and next_sh_step > 0
            and step >= next_sh_step
        ):
            new_deg = min(target_sh, model.sh_degree + 1)
            model.set_active_sh_degree(new_deg)
            opt = build_optimizers(model, lr)
            next_sh_step = step + int(sh_warmup_steps)
            if log:
                log.info("Enabled SH degree %d at step %d", new_deg, step)

        sched = (
            schedule_at_step(
                step, total_steps, base_max_gaussians_render=max_gaussians_render
            )
            if use_resolution_schedule
            else None
        )
        width_scale = sched.width_scale if sched else 1.0
        tile_size = sched.tile_size if sched else 16
        max_per_tile = sched.max_per_tile if sched else 64
        render_cap = sched.max_gaussians_render if sched else max_gaussians_render

        view = dataset.views[(step - 1) % n_views]
        if view_cache is not None:
            rgb, mask, R, t, w, h = view_cache.get_scaled(view, width_scale=width_scale)
        else:
            from instasplat.metal_equirect.dataset import load_view_tensors

            rgb, mask, R, t = load_view_tensors(view, device)
            w, h = view.width, view.height
            if width_scale < 1.0:
                import torch.nn.functional as F

                tw = max(16, int(round(w * width_scale)))
                th = max(8, int(round(tw * h / max(w, 1))))
                rgb = (
                    F.interpolate(
                        rgb.permute(2, 0, 1).unsqueeze(0),
                        size=(th, tw),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                if mask is not None:
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(th, tw),
                        mode="nearest",
                    ).squeeze(0).squeeze(0)
                w, h = tw, th

        cam = EquirectCamera(w, h)

        zero_grad(opt)
        pred, rinfo = rasterize_equirect(
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
            max_gaussians=render_cap,
            with_eval3d=with_eval3d,
            composite=composite,
            tile_size=tile_size,
            max_per_tile=max_per_tile,
            prefer_metal=True,
            return_info=True,
        )
        mean_2d = rinfo["mean_2d"]
        mean_2d.retain_grad()
        loss = photometric_loss(pred, rgb, cam, mask)
        loss.backward()
        densify_state.accumulate(
            model, mean_2d=mean_2d, selected=rinfo.get("selected")
        )
        step_all(opt)
        last_loss = float(loss.detach().cpu())

        if (
            opacity_reset_every > 0
            and step > densify_from
            and step % opacity_reset_every == 0
            and step < densify_until
        ):
            opacity_reset(model, opt, value=0.01)
            if log:
                log.info("opacity reset at step %d", step)

        if (
            densify_every > 0
            and step >= densify_from
            and step % densify_every == 0
            and step < densify_until
        ):
            phase = densify_phase(step, densify_from, densify_until)
            thr = grad_threshold_at(step, densify_from, densify_until)
            dstats = densify_and_prune(
                model,
                densify_state,
                opt,
                max_gaussians=max_gaussians,
                grad_threshold=thr,
                phase=phase,
            )
            if log and (
                dstats["cloned"] or dstats["split"] or dstats["pruned"] or dstats["relocated"]
            ):
                log.info(
                    "densify step %d phase=%s thr=%.5f: %s → %d gaussians",
                    step,
                    phase,
                    thr,
                    dstats,
                    model.n,
                )

        if preview_every > 0 and (step % preview_every == 0 or step == 1):
            preview_path = _write_preview(pred, preview_dir / f"step_{step:06d}.jpg")

        # Live viewer: async subsampled PLY (does not stall the step)
        if viewer_every > 0 and (step % viewer_every == 0 or step == 1):
            live_path = export_dir / "live.ply"
            live_exporter.submit(model, live_path, max_points=live_max_points)

        if export_every > 0 and (step % export_every == 0 or step == total_steps):
            ply_path = export_ply(model, export_dir / f"equirect_{step:06d}.ply")
            if log:
                phase = sched.phase if sched else "full"
                log.info(
                    "step %d/%d loss=%.5f gaussians=%d sh=%d phase=%s → %s",
                    step,
                    total_steps,
                    last_loss,
                    model.n,
                    model.sh_degree,
                    phase,
                    ply_path.name,
                )
        elif log and step % 50 == 0:
            phase = sched.phase if sched else "full"
            log.info(
                "step %d/%d loss=%.5f gaussians=%d sh=%d phase=%s res=%.2f tile=%d",
                step,
                total_steps,
                last_loss,
                model.n,
                model.sh_degree,
                phase,
                width_scale,
                tile_size,
            )

        if on_progress is not None:
            on_progress(
                {
                    "step": step,
                    "total_steps": total_steps,
                    "loss": last_loss,
                    "n_gaussians": model.n,
                    "sh_degree": model.sh_degree,
                    "phase": sched.phase if sched else "full",
                    "width_scale": width_scale,
                    "preview": str(preview_path) if preview_path else None,
                    "ply": str(ply_path) if ply_path else None,
                    "live_ply": str(live_path) if live_path else None,
                }
            )

    live_exporter.flush()
    if ply_path is None:
        ply_path = export_ply(model, export_dir / "equirect_final.ply")
    final = export_dir / "scene.ply"
    export_ply(model, final)
    export_ply(model, export_dir / "live.ply", max_points=live_max_points)
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
