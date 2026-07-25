"""Load and inspect existing InstaSplat job directories for resume/continue."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.stages import SINGLE_STAGES, TILED_STAGES


def is_job_dir(path: Path) -> bool:
    """True if ``path`` looks like an InstaSplat job workspace."""
    path = Path(path)
    if not path.is_dir():
        return False
    if (path / "config.yaml").exists():
        return True
    if (path / "00_ingest").is_dir():
        return True
    if (path / "10_chunks").is_dir():
        return True
    return False


def load_job_config(job: Path) -> PipelineConfig:
    """Load an existing job's config.yaml (or synthesize a minimal one)."""
    job = Path(job).resolve()
    if not job.is_dir():
        raise FileNotFoundError(f"Job directory not found: {job}")
    cfg_path = job / "config.yaml"
    if cfg_path.exists():
        cfg = PipelineConfig.load(cfg_path)
        cfg.output_dir = job.parent
        cfg.project_name = job.name
        # Prefer ingested video when original capture path is gone
        video = job / "00_ingest" / "equirect.mp4"
        if video.exists() and (not cfg.input_path.exists() or cfg.input_path.is_dir()):
            cfg.input_path = video
        return cfg
    video = job / "00_ingest" / "equirect.mp4"
    inp = video if video.exists() else job
    return PipelineConfig(input_path=inp, output_dir=job.parent, project_name=job.name)


def _chunk_dirs(chunks_root: Path) -> list[Path]:
    if not chunks_root.is_dir():
        return []
    return sorted(
        p for p in chunks_root.iterdir() if p.is_dir() and p.name.startswith("chunk_")
    )


def _chunk_has_ply(chunk_dir: Path) -> bool:
    export = chunk_dir / "06_export"
    if export.is_dir():
        for name in ("scene.ply", "scene_merged.ply"):
            if (export / name).exists():
                return True
        if any(export.glob("*.ply")):
            return True
    train = chunk_dir / "05_train" / "exports"
    if train.is_dir() and any(train.glob("*.ply")):
        return True
    return False


def _has_any(path: Path, pattern: str = "*") -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.glob(pattern))


@dataclass
class JobResumeInfo:
    """Disk inspection summary for continuing a previous run."""

    job_dir: Path
    config: PipelineConfig
    summary: str
    suggested_stages: list[str]
    details: list[str] = field(default_factory=list)
    tiled: bool = False
    tiles_total: int = 0
    tiles_done: int = 0

    @property
    def mode_label(self) -> str:
        return "tiled" if self.tiled else "single"


def inspect_job(job: Path) -> JobResumeInfo:
    """Load config and suggest which stages to continue from."""
    job = Path(job).resolve()
    if not is_job_dir(job):
        raise ValueError(
            f"Not an InstaSplat job folder (need config.yaml or 00_ingest): {job}"
        )
    cfg = load_job_config(job)
    paths = JobPaths(job)
    tiled = cfg.mode == "tiled" or cfg.chunk.enabled
    details: list[str] = []

    if paths.config.exists():
        details.append("config.yaml found")
    else:
        details.append("no config.yaml — using defaults + folder name")

    if paths.video.exists():
        details.append("ingested equirect.mp4 present")
    elif cfg.input_path.exists() and cfg.input_path.is_file():
        details.append(f"input still available: {cfg.input_path.name}")
    else:
        details.append("warning: no ingested video and original input missing")

    result_path = job / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("success"):
                details.append("last result: success")
            else:
                err = result.get("error") or "failed"
                details.append(f"last result: {err}")
        except (OSError, json.JSONDecodeError):
            details.append("result.json unreadable")

    tiles_total = 0
    tiles_done = 0
    if tiled:
        manifest = paths.chunks / "manifest.json"
        status_path = paths.chunks / "chunk_status.json"
        chunk_dirs = _chunk_dirs(paths.chunks)
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                plans = data.get("chunks") or data.get("plans") or []
                tiles_total = len(plans) if plans else len(chunk_dirs)
            except (OSError, json.JSONDecodeError):
                tiles_total = len(chunk_dirs)
            details.append(f"tile plan present ({tiles_total or '?'} tiles)")
        else:
            details.append("no tile plan yet (manifest.json)")

        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(status, dict):
                    tiles_done = sum(1 for v in status.values() if v)
                    tiles_total = tiles_total or len(status)
                    details.append(f"chunk_status: {tiles_done}/{tiles_total} ok")
            except (OSError, json.JSONDecodeError):
                pass
        elif chunk_dirs:
            tiles_done = sum(1 for c in chunk_dirs if _chunk_has_ply(c))
            tiles_total = tiles_total or len(chunk_dirs)
            details.append(f"tile exports found: {tiles_done}/{len(chunk_dirs)}")

        if (paths.chunks / "alignments.json").exists():
            details.append("tile alignments present")
        if (paths.merged / "scene_merged.ply").exists():
            details.append("merged scene_merged.ply present")
        if (job / "quality.json").exists() or (job / "cloud_job.json").exists():
            details.append("packaging outputs present")

        suggested = _suggest_tiled(paths, tiles_total, tiles_done)
    else:
        suggested = _suggest_single(paths)
        for label, ok in (
            ("frames", _has_any(paths.equirect_frames, "*.jpg") or _has_any(paths.equirect_frames, "*.png")),
            ("masks", _has_any(paths.equirect_masks, "*.png")),
            ("sfm", (paths.colmap_model / "cameras.bin").exists() or (paths.colmap_model / "cameras.txt").exists()),
            ("scaled", (paths.scaled_model / "cameras.bin").exists() or (paths.scaled_model / "cameras.txt").exists()),
            ("train export", _has_any(paths.brush_export, "*.ply")),
            ("final export", _has_any(paths.export, "*.ply") or _has_any(paths.export, "*.sog")),
        ):
            details.append(f"{label}: {'yes' if ok else 'no'}")

    summary = (
        f"{job.name} · {('tiled' if tiled else 'single')}"
        + (f" · tiles {tiles_done}/{tiles_total}" if tiled and tiles_total else "")
        + f" · continue: {', '.join(suggested)}"
    )
    return JobResumeInfo(
        job_dir=job,
        config=cfg,
        summary=summary,
        suggested_stages=suggested,
        details=details,
        tiled=tiled,
        tiles_total=tiles_total,
        tiles_done=tiles_done,
    )


def _suggest_tiled(paths: JobPaths, tiles_total: int, tiles_done: int) -> list[str]:
    order = list(TILED_STAGES)
    if not paths.video.exists() and not (paths.root / "config.yaml").exists():
        return order
    if not paths.video.exists():
        return order  # need ingest first
    # Ingested but no plan
    if not (paths.chunks / "manifest.json").exists():
        return [s for s in order if s != "ingest"]
    # Plan exists; tiles incomplete
    if tiles_total == 0 or tiles_done < tiles_total:
        return ["process_chunks", "align_chunks", "merge_chunks", "package"]
    # All tiles done
    if not (paths.chunks / "alignments.json").exists():
        return ["align_chunks", "merge_chunks", "package"]
    if not (paths.merged / "scene_merged.ply").exists():
        return ["merge_chunks", "package"]
    # Fully merged — re-package is a safe light continue
    return ["package"]


def _suggest_single(paths: JobPaths) -> list[str]:
    order = list(SINGLE_STAGES)

    def done_through(stage: str) -> bool:
        if stage == "ingest":
            return paths.video.exists()
        if stage == "extract":
            return _has_any(paths.equirect_frames, "*.jpg") or _has_any(
                paths.equirect_frames, "*.png"
            )
        if stage == "mask":
            return _has_any(paths.equirect_masks, "*.png")
        if stage == "sfm":
            return (paths.colmap_model / "cameras.bin").exists() or (
                paths.colmap_model / "cameras.txt"
            ).exists()
        if stage == "scale":
            return (paths.scaled_model / "cameras.bin").exists() or (
                paths.scaled_model / "cameras.txt"
            ).exists()
        if stage == "refine":
            refine = paths.root / "03b_refine" / "sparse" / "0"
            return (refine / "cameras.bin").exists() or (refine / "cameras.txt").exists()
        if stage == "train":
            return _has_any(paths.brush_export, "*.ply")
        if stage == "export":
            return _has_any(paths.export, "*.ply") or _has_any(paths.export, "*.sog")
        if stage == "package":
            return (paths.root / "quality.json").exists() or (
                paths.root / "cloud_job.json"
            ).exists()
        return False

    # Find first incomplete stage; include it and everything after
    start = 0
    for i, stage in enumerate(order):
        if not done_through(stage):
            start = i
            break
    else:
        return ["package"]
    # If ingest done, skip re-ingest unless nothing else
    if start == 0 and done_through("ingest"):
        start = 1
    return order[start:]
