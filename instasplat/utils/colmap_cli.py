"""COLMAP CLI option compatibility (3.12+ FeatureExtraction rename)."""

from __future__ import annotations

import functools
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ColmapCliCaps:
    """Detected ``feature_extractor`` / matcher option names for this COLMAP build."""

    max_image_size: str
    extract_use_gpu: str
    max_num_features: str
    match_use_gpu: str
    modern: bool  # True when FeatureExtraction.* namespace exists
    supports_equirectangular: bool = False


def _help_text(colmap: str, command: str) -> str:
    try:
        proc = subprocess.run(
            [colmap, command, "-h"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return f"{proc.stdout or ''}\n{proc.stderr or ''}"


@functools.lru_cache(maxsize=4)
def detect_colmap_caps(colmap_bin: str | None = None) -> ColmapCliCaps:
    """
    Probe COLMAP help for renamed SIFT/feature options.

    Homebrew COLMAP ≥ ~3.12 moved ``SiftExtraction.max_image_size`` and
    ``*.use_gpu`` under ``FeatureExtraction`` / ``FeatureMatching``.
    ``SiftExtraction.max_num_features`` remains under the SIFT namespace.
    """
    colmap = colmap_bin or shutil.which("colmap") or "colmap"
    extract_help = _help_text(colmap, "feature_extractor")
    match_help = _help_text(colmap, "exhaustive_matcher")
    if not match_help:
        match_help = _help_text(colmap, "sequential_matcher")

    modern = "FeatureExtraction.max_image_size" in extract_help
    legacy_size = "SiftExtraction.max_image_size" in extract_help

    if modern:
        max_image_size = "--FeatureExtraction.max_image_size"
        extract_use_gpu = "--FeatureExtraction.use_gpu"
    elif legacy_size:
        max_image_size = "--SiftExtraction.max_image_size"
        extract_use_gpu = (
            "--FeatureExtraction.use_gpu"
            if "FeatureExtraction.use_gpu" in extract_help
            else "--SiftExtraction.use_gpu"
        )
    else:
        # Help unavailable (dry-run / missing binary) — prefer modern names
        max_image_size = "--FeatureExtraction.max_image_size"
        extract_use_gpu = "--FeatureExtraction.use_gpu"
        modern = True

    if "SiftExtraction.max_num_features" in extract_help or not extract_help:
        max_num_features = "--SiftExtraction.max_num_features"
    else:
        max_num_features = "--FeatureExtraction.max_num_features"

    if "FeatureMatching.use_gpu" in match_help:
        match_use_gpu = "--FeatureMatching.use_gpu"
    elif "SiftMatching.use_gpu" in match_help:
        match_use_gpu = "--SiftMatching.use_gpu"
    else:
        match_use_gpu = (
            "--FeatureMatching.use_gpu" if modern else "--SiftMatching.use_gpu"
        )

    # Recent COLMAP builds list EQUIRECTANGULAR in ImageReader.camera_model help
    supports_eq = "EQUIRECTANGULAR" in extract_help

    return ColmapCliCaps(
        max_image_size=max_image_size,
        extract_use_gpu=extract_use_gpu,
        max_num_features=max_num_features,
        match_use_gpu=match_use_gpu,
        modern=modern or (not legacy_size),
        supports_equirectangular=supports_eq,
    )


def db_image_count(database: Path) -> int:
    """Return number of images registered in a COLMAP SQLite database."""
    if not Path(database).exists():
        return 0
    try:
        conn = sqlite3.connect(str(database))
        try:
            row = conn.execute("SELECT COUNT(*) FROM images").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def list_readable_images(image_dir: Path) -> list[Path]:
    """List JPG/PNG images that exist and are readable (skips broken symlinks)."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(image_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # Broken symlink → exists() is False
        if not p.exists():
            continue
        out.append(p)
    return out


def quality_feature_args(quality: str, caps: ColmapCliCaps | None = None) -> list[str]:
    """Return feature_extractor size/feature-count args for a quality preset."""
    caps = caps or detect_colmap_caps()
    presets = {
        "low": (1000, 2048),
        "medium": (1600, 4096),
        "high": (2400, 8192),
        "extreme": (3200, 16384),
    }
    size, nfeat = presets.get(quality, presets["high"])
    return [
        caps.max_image_size,
        str(size),
        caps.max_num_features,
        str(nfeat),
    ]
