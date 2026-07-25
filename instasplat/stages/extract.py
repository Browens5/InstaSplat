"""Extract equirectangular (or best-effort) frames from the ingest video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger, run_cmd


@dataclass
class ExtractResult:
    frame_dir: Path
    frame_count: int
    fps: float


def run_extract(cfg: PipelineConfig, paths: JobPaths) -> ExtractResult:
    paths.ensure()
    log = get_logger("instasplat.extract", paths.logs / "extract.log")
    video = paths.video
    out_dir = paths.equirect_frames
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob(f"*.{cfg.extract.image_format}"))
    if existing and cfg.skip_existing:
        log.info("Skipping extract; found %d frames", len(existing))
        return ExtractResult(out_dir, len(existing), cfg.extract.fps)

    # Clear stale frames if re-running
    for p in out_dir.glob("*"):
        if p.is_file():
            p.unlink()

    ext = cfg.extract.image_format
    pattern = str(out_dir / f"frame_%06d.{ext}")
    vf_parts = [f"fps={cfg.extract.fps}"]
    cmd = ["ffmpeg", "-y", "-i", str(video)]
    if cfg.extract.start_sec is not None:
        cmd.extend(["-ss", str(cfg.extract.start_sec)])
    if cfg.extract.end_sec is not None:
        cmd.extend(["-to", str(cfg.extract.end_sec)])
    cmd.extend(["-vf", ",".join(vf_parts)])
    if ext == "jpg":
        cmd.extend(["-q:v", str(max(1, min(31, int(round((100 - cfg.extract.jpeg_quality) * 31 / 100)))) )])
    if cfg.extract.max_frames:
        cmd.extend(["-frames:v", str(cfg.extract.max_frames)])
    cmd.append(pattern)

    run_cmd(cmd, log_file=paths.logs / "ffmpeg_extract.log", dry_run=cfg.dry_run)
    frames = sorted(out_dir.glob(f"*.{ext}"))
    log.info("Extracted %d frames @ %s fps", len(frames), cfg.extract.fps)
    return ExtractResult(out_dir, len(frames), cfg.extract.fps)
