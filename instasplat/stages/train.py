"""Train a 3D Gaussian splat (Brush or OpenSplat Metal backends)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from instasplat.config import PipelineConfig
from instasplat.utils.deps import check_brush, check_opensplat
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger, run_cmd


@dataclass
class TrainResult:
    export_dir: Path
    ply_path: Path | None
    trainer_bin: str
    backend: str


def _prepare_colmap_dataset(paths: JobPaths, model_dir: Path) -> Path:
    """Assemble COLMAP layout for Brush and OpenSplat."""
    ds = paths.train / "colmap_dataset"
    images = ds / "images"
    sparse0 = ds / "sparse" / "0"
    images.mkdir(parents=True, exist_ok=True)
    sparse0.mkdir(parents=True, exist_ok=True)

    src_images = paths.cubemap_images
    for img in list(src_images.glob("*.jpg")) + list(src_images.glob("*.png")):
        dest = images / img.name
        if not dest.exists():
            try:
                dest.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, dest)

    if any(paths.cubemap_masks.glob("*.png")):
        masks = ds / "masks"
        masks.mkdir(parents=True, exist_ok=True)
        for m in paths.cubemap_masks.glob("*.png"):
            dest = masks / m.name
            if not dest.exists():
                try:
                    dest.symlink_to(m.resolve())
                except OSError:
                    shutil.copy2(m, dest)

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


def _run_brush(
    cfg: PipelineConfig,
    dataset: Path,
    export_dir: Path,
    log_dir: Path,
    log,
) -> str:
    status = check_brush(cfg.train.brush_bin)
    brush = status.path or cfg.train.brush_bin
    if not status.available and not cfg.dry_run:
        raise RuntimeError(status.notes or "Brush binary not found")

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
    try:
        run_cmd(
            cmd,
            log_file=log_dir / "brush.log",
            dry_run=cfg.dry_run,
            controllable=True,
        )
    except RuntimeError as exc:
        log.warning("Primary Brush invocation failed (%s); trying alternate flags", exc)
        alt = [brush, "--export-path", str(export_dir), str(dataset)]
        if cfg.train.with_viewer:
            alt.append("--with-viewer")
        alt.extend(cfg.train.extra_args)
        run_cmd(
            alt,
            log_file=log_dir / "brush_alt.log",
            dry_run=cfg.dry_run,
            controllable=True,
        )
    return brush


def _run_opensplat(
    cfg: PipelineConfig,
    dataset: Path,
    export_dir: Path,
    log_dir: Path,
    log,
) -> str:
    """OpenSplat C++ trainer with Metal MPS (https://github.com/pierotofy/OpenSplat)."""
    status = check_opensplat(cfg.train.opensplat_bin)
    binary = status.path or cfg.train.opensplat_bin
    if not status.available and not cfg.dry_run:
        raise RuntimeError(status.notes or "OpenSplat binary not found")

    out_ply = export_dir / "splat.ply"
    cmd = [
        binary,
        str(dataset),
        "-n",
        str(cfg.train.total_steps),
        "-o",
        str(out_ply),
    ]
    cmd.extend(cfg.train.extra_args)
    run_cmd(
        cmd,
        log_file=log_dir / "opensplat.log",
        dry_run=cfg.dry_run,
        controllable=True,
    )
    log.info("OpenSplat training finished → %s", out_ply)
    return binary


def run_train(cfg: PipelineConfig, paths: JobPaths, model_dir: Path) -> TrainResult:
    from instasplat.utils.brush_install import ensure_brush
    from instasplat.utils.control import get_controller

    paths.ensure()
    log = get_logger("instasplat.train", paths.logs / "train.log")
    get_controller().checkpoint("train")
    backend = cfg.train.backend
    if backend == "brush" and not cfg.dry_run:
        installed = ensure_brush(auto_install=True)
        if not installed.ok:
            raise RuntimeError(installed.message)
        if installed.brush_path:
            cfg.train.brush_bin = str(installed.brush_path)
        log.info("%s", installed.message)
    dataset = _prepare_colmap_dataset(paths, model_dir)
    export_dir = paths.brush_export
    export_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_latest_ply(export_dir)
    if existing and cfg.skip_existing:
        log.info("Skipping train; found existing splat %s", existing)
        return TrainResult(export_dir, existing, cfg.train.brush_bin, backend)

    if backend == "opensplat":
        trainer = _run_opensplat(cfg, dataset, export_dir, paths.logs, log)
    else:
        trainer = _run_brush(cfg, dataset, export_dir, paths.logs, log)

    ply = None if cfg.dry_run else _find_latest_ply(export_dir)
    if ply is None and not cfg.dry_run:
        log.warning(
            "No .ply found in %s after %s run. Adjust train.extra_args for your binary.",
            export_dir,
            backend,
        )
    return TrainResult(export_dir, ply, trainer, backend)
