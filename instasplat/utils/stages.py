"""Pipeline stage catalog and selection helpers."""

from __future__ import annotations

from typing import Literal

Mode = Literal["single", "tiled", "all"]

STAGE_HELP: dict[str, str] = {
    "ingest": "Resolve video + parse INSV/sidecar gyro/GPS",
    "extract": "Sample equirect frames (single mode)",
    "mask": "YOLO people masks (MPS, CPU fallback)",
    "sfm": "Cubemap/equirect COLMAP sparse reconstruction",
    "scale": "Metric scale (GPS / known distance)",
    "refine": "COLMAP BA + GPS/gyro pose blend",
    "train": "Brush or OpenSplat Gaussian training",
    "export": "splat-transform → ply/sog/spz/…",
    "package": "Nerfstudio / hierarchy / quality / cloud_job",
    "plan_chunks": "Plan overlapping tiles for long 360",
    "preflight": "Mac long-360 readiness checks",
    "process_chunks": "Per-tile mask→sfm→scale→refine→train→export",
    "align_chunks": "GPS/gyro Sim3 align tiles",
    "merge_chunks": "Merge tile splats into one scene",
}

SINGLE_STAGES: list[str] = [
    "ingest",
    "extract",
    "mask",
    "sfm",
    "scale",
    "refine",
    "train",
    "export",
    "package",
]

TILED_STAGES: list[str] = [
    "ingest",
    "plan_chunks",
    "preflight",
    "process_chunks",
    "align_chunks",
    "merge_chunks",
    "package",
]

ALL_STAGES: list[str] = list(
    dict.fromkeys([*SINGLE_STAGES, *TILED_STAGES])
)


def stage_order(mode: Mode = "all") -> list[str]:
    if mode == "single":
        return list(SINGLE_STAGES)
    if mode == "tiled":
        return list(TILED_STAGES)
    return list(ALL_STAGES)


def parse_stage_list(text: str) -> list[str]:
    parts = [p.strip() for p in text.replace(" ", "").split(",") if p.strip()]
    unknown = [p for p in parts if p not in ALL_STAGES]
    if unknown:
        raise ValueError(
            f"Unknown stage(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(ALL_STAGES)}"
        )
    return parts


def select_stages(
    *,
    mode: Mode,
    only: str | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    stages: str | None = None,
) -> list[str] | None:
    """
    Resolve which stages to run.

    Returns None when the caller should keep config defaults (full pipeline).
    """
    order = stage_order(mode)
    if only:
        chosen = parse_stage_list(only)
        # Preserve canonical order
        return [s for s in order if s in chosen] or chosen
    if stages:
        chosen = parse_stage_list(stages)
        return [s for s in order if s in chosen] or chosen
    if from_stage or to_stage:
        if from_stage and from_stage not in order:
            # Allow selecting a stage from the other mode list
            if from_stage not in ALL_STAGES:
                raise ValueError(f"Unknown --from stage: {from_stage}")
            order = stage_order("all")
        if to_stage and to_stage not in order:
            if to_stage not in ALL_STAGES:
                raise ValueError(f"Unknown --to stage: {to_stage}")
            order = stage_order("all")
        start = order.index(from_stage) if from_stage else 0
        end = order.index(to_stage) if to_stage else len(order) - 1
        if end < start:
            raise ValueError(f"--to {to_stage} is before --from {from_stage}")
        return order[start : end + 1]
    return None
