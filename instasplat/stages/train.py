"""Train a 3D Gaussian splat with Brush."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from instasplat.config import PipelineConfig
from instasplat.utils.deps import check_brush
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger, run_cmd


@dataclass
class TrainResult:
    export_dir: Path
    ply_path: Path | None
    brush_bin: str


def _prepare_brush_dataset(paths: JobPaths, model_dir: Path) -> Path:
    """
    Brush expects COLMAP layout: images/ + sparse/0/ (cameras, images, points3D).
    We assemble that under 05_train/colmap_dataset.
    """
    ds = paths.train / "colmap_dataset"
    images = ds / "images"
    sparse0 = ds / "sparse" / "0"
    images.mkdir(parents=True, exist_ok=True)
    sparse0.mkdir(parents=True, exist_ok=True)

    # Link/copy images
    src_images = paths.cubemap_images
    for img in list(src_images.glob("*.jpg")) + list(src_images.glob("*.png")):
        dest = images / img.name
        if not dest.exists():
            try:
                dest.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, dest)

    # Optional masks folder (Brush: folder named 'masks')
    if any(paths.cubemap_masks.glob("*.png")):
        masks = ds / "masks"
        masks.mkdir(parents=True, exist_ok=True)
        for m in paths.cubemap_masks.glob("*.png"):
            # Brush masks should match image stems; our masks are .png vs .jpg images
            # Create mask names matching image filenames with .png
            # Also provide stem-matched copies
            dest = masks / m.name
            if not dest.exists():
                try:
                    dest.symlink_to(m.resolve())
                except OSError:
                    shutil.copy2(m, dest)
            # Match .jpg image name with .png mask of same stem — already same stem

    # Copy model files
    for name in (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    ):
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, sparse0 / name)

    return ds


def _find_latest_ply(export_dir: Path) -> Path | None:
    plys = sorted(export_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    return plys[0] if plys else None


def run_train(cfg: PipelineConfig, paths: JobPaths, model_dir: Path) -> TrainResult:
    paths.ensure()
    log = get_logger("instasplat.train", paths.logs / "train.log")
    status = check_brush(cfg.train.brush_bin)
    brush = status.path or cfg.train.brush_bin
    if not status.available and not cfg.dry_run:
        raise RuntimeError(status.notes or "Brush binary not found")

    dataset = _prepare_brush_dataset(paths, model_dir)
    export_dir = paths.brush_export
    export_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_latest_ply(export_dir)
    if existing and cfg.skip_existing:
        log.info("Skipping train; found existing splat %s", existing)
        return TrainResult(export_dir, existing, brush)

    cmd = [
        brush,
        str(dataset),
        "--total-steps",
        str(cfg.train.total_steps),
        "--max-resolution",
        str(cfg.train.max_resolution),
        "--export-path",
        str(export_dir),
        "--export-every",
        str(cfg.train.export_every),
    ]
    if cfg.train.with_viewer:
        cmd.append("--with-viewer")
    cmd.extend(cfg.train.extra_args)

    # Brush CLI flags evolve; try primary then a help-compatible fallback message
    try:
        run_cmd(cmd, log_file=paths.logs / "brush.log", dry_run=cfg.dry_run)
    except RuntimeError as exc:
        log.warning("Primary Brush invocation failed (%s); trying alternate flags", exc)
        alt = [
            brush,
            "--export-path",
            str(export_dir),
            str(dataset),
        ]
        if cfg.train.with_viewer:
            alt.append("--with-viewer")
        alt.extend(cfg.train.extra_args)
        run_cmd(alt, log_file=paths.logs / "brush_alt.log", dry_run=cfg.dry_run)

    ply = None if cfg.dry_run else _find_latest_ply(export_dir)
    if ply is None and not cfg.dry_run:
        log.warning(
            "No .ply found in %s after Brush run. Open the viewer and export manually, "
            "or adjust train.extra_args to match your Brush version CLI.",
            export_dir,
        )
    return TrainResult(export_dir, ply, brush)
