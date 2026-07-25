"""Pipeline entrypoint for ``train.backend = metal_equirect``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from instasplat.config import PipelineConfig
from instasplat.metal_equirect.dataset import load_equirect_dataset
from instasplat.metal_equirect.train_loop import train_equirect
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger


@dataclass
class MetalEquirectResult:
    export_dir: Path
    ply_path: Path | None
    device: str
    n_gaussians: int
    final_loss: float
    preview_path: Path | None = None


def run_metal_equirect_train(
    cfg: PipelineConfig,
    paths: JobPaths,
    model_dir: Path,
) -> MetalEquirectResult:
    """Train Gaussians on equirect frames using COLMAP poses."""
    log = get_logger("instasplat.metal_equirect", paths.logs / "metal_equirect.log")
    export_dir = paths.brush_export
    export_dir.mkdir(parents=True, exist_ok=True)

    if cfg.dry_run:
        log.info("dry_run: would run metal_equirect on %s", model_dir)
        return MetalEquirectResult(export_dir, None, "dry_run", 0, 0.0)

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "metal_equirect requires PyTorch. "
            "Install: pip install torch  (Apple Silicon: use the MPS wheel from pytorch.org)"
        ) from exc

    max_w = int(cfg.train.max_resolution)
    max_w = max(256, min(max_w, 4096))
    dataset = load_equirect_dataset(paths, model_dir, max_width=max_w)
    log.info(
        "Loaded %d equirect views from %s (max_width=%d)",
        len(dataset),
        dataset.equirect_dir,
        max_w,
    )

    sh_degree = int(cfg.train.sh_degree)
    lr = float(cfg.train.lr)
    steps = int(cfg.train.total_steps)
    composite = str(cfg.train.composite or "tile")
    if composite not in {"tile", "oit"}:
        log.warning("Unknown composite=%s; using tile", composite)
        composite = "tile"
    if steps > 50_000:
        log.warning("total_steps=%d is high for metal_equirect; consider 10k–20k", steps)

    def _progress(ev: dict) -> None:
        # Lightweight heartbeat file for GUI / external monitors
        if ev.get("step", 0) % 25 == 0 or ev.get("step") == ev.get("total_steps"):
            hb = export_dir / "train_heartbeat.json"
            try:
                import json

                hb.write_text(json.dumps(ev), encoding="utf-8")
            except OSError:
                pass

    stats = train_equirect(
        dataset,
        export_dir,
        total_steps=steps,
        export_every=max(1, int(cfg.train.export_every)),
        sh_degree=sh_degree,
        lr=lr,
        with_eval3d=bool(cfg.train.with_eval3d),
        composite=composite,
        sh_warmup_steps=int(cfg.train.sh_warmup_steps),
        densify_every=int(cfg.train.densify_every),
        prefer_mps=bool(cfg.metal.prefer_metal),
        on_progress=_progress,
        log=log,
    )
    log.info(
        "Finished metal_equirect: loss=%.5f gaussians=%d device=%s elapsed=%.1fs → %s",
        stats.final_loss,
        stats.n_gaussians,
        stats.device,
        stats.elapsed_sec,
        stats.ply_path,
    )
    return MetalEquirectResult(
        export_dir=export_dir,
        ply_path=stats.ply_path,
        device=stats.device,
        n_gaussians=stats.n_gaussians,
        final_loss=stats.final_loss,
        preview_path=stats.preview_path,
    )
