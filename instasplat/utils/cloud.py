"""Cloud worker job manifests (3DGUT / gsplat / LichtFeld / Nerfstudio)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger


CloudBackend = str  # gsplat_3dgut | lichtfeld | nerfstudio_splatfacto | brush_remote


@dataclass
class CloudJobResult:
    manifest_path: Path
    backends: list[str]


def recommended_cloud_backends(cfg: PipelineConfig) -> list[str]:
    """Pick cloud backends based on capture/train goals."""
    backends = ["nerfstudio_splatfacto", "gsplat_3dgut"]
    if cfg.sfm.mode in {"equirectangular", "auto"}:
        # Native distorted / equirect path is the GUT value-add
        backends = ["gsplat_3dgut", "lichtfeld", "nerfstudio_splatfacto"]
    if cfg.mode == "tiled" or cfg.chunk.enabled:
        backends.append("hierarchical_merge_cuda")
    return backends


def write_cloud_job_manifest(
    cfg: PipelineConfig,
    paths: JobPaths,
    out_path: Path | None = None,
) -> Path:
    """
    Write a portable cloud job description for CUDA workers.

    Local Mac keeps Brush/OpenSplat; cloud workers can pick up the Nerfstudio
    package or hierarchy anchors for 3DGUT / LichtFeld / Kerbl hierarchy merge.
    """
    out_path = out_path or (paths.root / "cloud_job.json")
    ns_dir = paths.root / "07_nerfstudio"
    hier = paths.merged / "hierarchy_manifest.json"
    transforms = ns_dir / "transforms.json"

    payload: dict[str, Any] = {
        "type": "instasplat_cloud_job_v1",
        "project_name": cfg.project_name,
        "mode": cfg.mode,
        "recommended_backends": recommended_cloud_backends(cfg),
        "dataset": {
            "nerfstudio_dir": str(ns_dir) if ns_dir.exists() else None,
            "transforms_json": str(transforms) if transforms.exists() else None,
            "hierarchy_manifest": str(hier) if hier.exists() else None,
            "colmap_model": str(paths.scaled_model if paths.scaled_model.exists() else paths.colmap_model),
            "images": str(paths.cubemap_images),
            "equirect_frames": str(paths.equirect_frames),
        },
        "local_exports": {
            "dir": str(paths.export if paths.export.exists() else paths.merged),
            "formats": list(cfg.export.formats),
        },
        "train_hints": {
            "steps": cfg.train.total_steps,
            "max_resolution": cfg.train.max_resolution,
            "prefer_equirect_gut": cfg.sfm.mode in {"equirectangular", "auto"},
            "metric_scale_mode": cfg.scale.mode,
        },
        "worker_notes": [
            "gsplat_3dgut: native fisheye/equirect without cubemap when available",
            "lichtfeld: CUDA MCMC / 3DGUT workstation path; return PLY/SOG/SPZ",
            "nerfstudio_splatfacto: use transforms.json from 07_nerfstudio",
            "hierarchical_merge_cuda: run Kerbl hierarchy merger on hierarchy_manifest anchors",
            "Do not redistribute Insta360 MediaSDK; stitch on licensed Linux workers only",
        ],
        "privacy": {
            "masks_recommended_before_upload": True,
            "mask_dir": str(paths.equirect_masks),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_cloud_manifest(cfg: PipelineConfig, paths: JobPaths) -> CloudJobResult | None:
    if not cfg.package.cloud_manifest:
        return None
    log = get_logger("instasplat.cloud", paths.logs / "cloud.log")
    if cfg.dry_run:
        return CloudJobResult(paths.root / "cloud_job.json", recommended_cloud_backends(cfg))
    path = write_cloud_job_manifest(cfg, paths)
    backends = recommended_cloud_backends(cfg)
    log.info("Cloud job manifest → %s (%s)", path, ", ".join(backends))
    return CloudJobResult(path, backends)
