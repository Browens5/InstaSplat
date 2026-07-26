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
    export_dir = paths.train_export
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
    by_index = int(getattr(dataset, "paired_by_index", 0) or 0)
    bad_pose = int(getattr(dataset, "skipped_bad_pose", 0) or 0)
    log.info(
        "Loaded %d equirect views from %s (max_width=%d, alias_index_pairs=%d, "
        "skipped_bad_pose=%d)",
        len(dataset),
        dataset.equirect_dir,
        max_w,
        by_index,
        bad_pose,
    )

    sh_degree = max(0, min(int(cfg.train.sh_degree), 3))
    lr = float(cfg.train.lr)
    steps = max(1_000, min(1_000_000, int(cfg.train.total_steps)))
    max_gaussians = max(1_000, min(30_000_000, int(getattr(cfg.train, "max_gaussians", 40_000))))
    export_every = max(50, min(int(cfg.train.export_every), 10_000))
    viewer_every = max(1, int(getattr(cfg.train, "viewer_every", 100)))
    composite = str(cfg.train.composite or "metal")
    if composite not in {"tile", "oit", "metal"}:
        log.warning("Unknown composite=%s; using metal", composite)
        composite = "metal"
    if steps > 100_000:
        log.warning(
            "total_steps=%d is very high; consider 10k–30k per tile unless you "
            "intentionally want a long run (max 1,000,000)",
            steps,
        )
    if max_gaussians > 1_000_000:
        log.warning(
            "max_gaussians=%d is very high for Metal/MPS memory; "
            "prefer tiled mode with smaller per-tile caps if you OOM",
            max_gaussians,
        )

    def _progress(ev: dict) -> None:
        # Heartbeat for GUI / external monitors (align with viewer_every)
        step = int(ev.get("step", 0) or 0)
        total = int(ev.get("total_steps", 0) or 0)
        if step % viewer_every == 0 or step == total or step == 1:
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
        export_every=export_every,
        viewer_every=viewer_every,
        sh_degree=sh_degree,
        lr=lr,
        max_init_points=max_gaussians,
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
