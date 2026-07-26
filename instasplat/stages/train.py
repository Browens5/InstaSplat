"""Train a 3D Gaussian splat with the native metal_equirect trainer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger

TrainProgressCb = Callable[[dict[str, Any]], None]


@dataclass
class TrainResult:
    export_dir: Path
    ply_path: Path | None
    trainer_bin: str
    backend: str


def _find_latest_ply(export_dir: Path) -> Path | None:
    plys = sorted(export_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    return plys[0] if plys else None


def run_train(
    cfg: PipelineConfig,
    paths: JobPaths,
    model_dir: Path,
    *,
    on_progress: TrainProgressCb | None = None,
) -> TrainResult:
    from instasplat.metal_equirect import run_metal_equirect_train
    from instasplat.utils.control import get_controller

    paths.ensure()
    log = get_logger("instasplat.train", paths.logs / "train.log")
    get_controller().checkpoint("train")

    # Sole trainer — coerce legacy YAML values
    cfg.train.backend = "metal_equirect"
    export_dir = paths.train_export
    export_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_latest_ply(export_dir)
    if existing and cfg.skip_existing:
        log.info("Skipping train; found existing splat %s", existing)
        return TrainResult(export_dir, existing, "metal_equirect", "metal_equirect")

    result = run_metal_equirect_train(cfg, paths, model_dir, on_progress=on_progress)
    log.info(
        "metal_equirect finished device=%s gaussians=%d loss=%.5f",
        result.device,
        result.n_gaussians,
        result.final_loss,
    )
    ply = None if cfg.dry_run else (_find_latest_ply(export_dir) or result.ply_path)
    if ply is None and not cfg.dry_run:
        log.warning("No .ply found in %s after metal_equirect run.", export_dir)
    return TrainResult(export_dir, ply, "metal_equirect", "metal_equirect")
