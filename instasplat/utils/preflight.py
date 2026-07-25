"""Preflight gates for local Mac long-360 tiled jobs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from instasplat.config import PipelineConfig
from instasplat.utils.chunking import ChunkManifest, probe_duration_sec
from instasplat.utils.deps import check_all
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger
from instasplat.utils.quality import validate_capture


@dataclass
class PreflightResult:
    ok: bool
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    estimated_frames: int = 0
    estimated_chunks: int = 0

    def raise_if_blocked(self) -> None:
        if self.blocking:
            raise RuntimeError(
                "Mac long-360 preflight failed:\n  - " + "\n  - ".join(self.blocking)
            )


def run_preflight(cfg: PipelineConfig, paths: JobPaths) -> PreflightResult:
    """
    Validate that a long 360 tiled job can run productively on this Mac.

    Soft-adjusts config when safe (e.g. GPS scale → none if no GPS CSV).
    """
    log = get_logger("instasplat.preflight", paths.logs / "preflight.log")
    result = PreflightResult(ok=True)

    deps = {d.name: d for d in check_all(cfg.train.brush_bin, cfg.export.splat_transform_bin)}
    for name in ("ffmpeg", "colmap"):
        if not deps.get(name, None) or not deps[name].available:
            result.blocking.append(f"Missing required tool: {name}")
    trainer_ok = (deps.get("brush") and deps["brush"].available) or (
        deps.get("opensplat") and deps["opensplat"].available
    )
    if not trainer_ok:
        result.blocking.append(
            "No Metal trainer found (brush or opensplat). "
            "Run `instasplat install-brush` (auto cargo release build) "
            "or build OpenSplat with -DGPU_RUNTIME=MPS."
        )
    if not deps.get("splat-transform") or not deps["splat-transform"].available:
        result.blocking.append("Missing splat-transform (npm i -g @playcanvas/splat-transform)")
    if not deps.get("ultralytics") or not deps["ultralytics"].available:
        if cfg.mask.enabled:
            result.warnings.append("ultralytics missing — disabling people masks for this run")
            cfg.mask.enabled = False

    # Stitch / unstitched guard
    warn_file = paths.ingest / "WARNING_UNSTITCHED.txt"
    if warn_file.exists() and not cfg.allow_unstitched:
        result.blocking.append(
            "Input looks unstitched (dual-fisheye remux). Export equirect MP4 from "
            "Insta360 Studio, or pass --allow-unstitched for testing only."
        )
    elif warn_file.exists():
        result.warnings.append("Proceeding with unstitched input (--allow-unstitched)")

    video = paths.video if paths.video.exists() else cfg.input_path
    duration = probe_duration_sec(video) if video.exists() else 0.0
    if duration <= 0:
        result.warnings.append(
            "Could not probe video duration; chunk plan may use a 60s placeholder"
        )
    elif duration < 8.0:
        result.warnings.append(
            f"Video is only {duration:.1f}s — tiled mode is overkill; consider single mode"
        )

    # Soft-fallback GPS scale when telemetry missing (common for Studio MP4-only)
    if cfg.scale.mode == "gps" and not paths.gps_csv.exists():
        result.warnings.append("No gps.csv — switching scale.mode to 'none' (visual units)")
        cfg.scale.mode = "none"
        result.notes.append("scale_soft_fallback_none")

    if not paths.gyro_csv.exists():
        result.warnings.append(
            "No gyro.csv — adaptive turn densify + pose refine blend will be weaker. "
            "Keep the sibling .insv next to the Studio MP4, or add gyro.csv / gps.csv sidecars."
        )

    issues = validate_capture(
        duration_sec=duration,
        overlap_sec=cfg.chunk.overlap_sec,
        chunk_duration_sec=cfg.chunk.duration_sec,
        min_overlap_ratio=cfg.chunk.min_overlap_ratio,
        base_fps=cfg.chunk.base_fps,
        gyro_csv=paths.gyro_csv if paths.gyro_csv.exists() else None,
        gps_csv=paths.gps_csv if paths.gps_csv.exists() else None,
    )
    for issue in issues:
        if issue.level == "error":
            result.blocking.append(f"{issue.code}: {issue.message}")
        else:
            result.warnings.append(f"{issue.code}: {issue.message}")

    # Rough resource estimate from planned chunks if present
    manifest_path = paths.chunks / "manifest.json"
    if manifest_path.exists():
        manifest = ChunkManifest.load(manifest_path)
        result.estimated_chunks = len(manifest.chunks)
        result.estimated_frames = sum(len(c.frame_times) for c in manifest.chunks)
        # Cubemap multiplies disk ~6×
        result.notes.append(
            f"plan≈{result.estimated_chunks} chunks, {result.estimated_frames} equirect frames "
            f"(~{result.estimated_frames * 6} cubemap faces)"
        )
        if result.estimated_frames > 4000:
            result.warnings.append(
                "Very large frame count — expect long COLMAP+train wall time on laptop; "
                "consider raising chunk.duration_sec or lowering max_fps"
            )

    # Disk free space soft check
    try:
        usage = shutil.disk_usage(paths.root if paths.root.exists() else Path.cwd())
        free_gb = usage.free / (1024**3)
        if free_gb < 40:
            result.warnings.append(f"Low free disk (~{free_gb:.0f} GB); 8K tiles need headroom")
        if free_gb < 15:
            result.blocking.append(f"Only ~{free_gb:.0f} GB free — need more disk for 8K tiles")
    except OSError:
        pass

    result.ok = not result.blocking
    for w in result.warnings:
        log.warning("%s", w)
    for b in result.blocking:
        log.error("%s", b)
    for n in result.notes:
        log.info("%s", n)

    summary = paths.root / "preflight.json"
    import json

    summary.write_text(
        json.dumps(
            {
                "ok": result.ok,
                "blocking": result.blocking,
                "warnings": result.warnings,
                "notes": result.notes,
                "estimated_chunks": result.estimated_chunks,
                "estimated_frames": result.estimated_frames,
                "scale_mode": cfg.scale.mode,
                "mask_enabled": cfg.mask.enabled,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
