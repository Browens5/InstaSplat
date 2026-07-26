"""COLMAP CLI option compatibility (3.12+ FeatureExtraction rename, 4.1+ EQUIRECTANGULAR)."""

from __future__ import annotations

import functools
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Native spherical equirect support landed in COLMAP 4.1.0
MIN_EQUIRECT_VERSION = (4, 1, 0)


@dataclass(frozen=True)
class ColmapCliCaps:
    """Detected ``feature_extractor`` / matcher option names for this COLMAP build."""

    max_image_size: str
    extract_use_gpu: str
    max_num_features: str
    match_use_gpu: str
    modern: bool  # True when FeatureExtraction.* namespace exists
    supports_equirectangular: bool = False
    supports_global_mapper: bool = False
    version: tuple[int, ...] | None = None
    colmap_bin: str = "colmap"
    # Optional global_mapper BA flags present in this build (name → value for equirect)
    global_ba_disable_flags: tuple[str, ...] = ()


def _help_text(colmap: str, command: str) -> str:
    try:
        proc = subprocess.run(
            [colmap, command, "-h"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return f"{proc.stdout or ''}\n{proc.stderr or ''}"


def _parse_version(text: str) -> tuple[int, ...] | None:
    """Extract a dotted version from COLMAP help / --version output."""
    # Prefer explicit "COLMAP 4.1.1" style, then any x.y.z near COLMAP
    m = re.search(r"\bCOLMAP\s+(\d+)\.(\d+)(?:\.(\d+))?", text, re.I)
    if m:
        major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        return (major, minor, patch)
    m = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", text)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _binary_mentions_equirect(colmap: str) -> bool:
    """Fallback probe when help text omits camera-model enum values."""
    path = shutil.which(colmap) if not Path(colmap).is_file() else colmap
    if not path:
        return False
    try:
        proc = subprocess.run(
            ["strings", path],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "EQUIRECTANGULAR" in (proc.stdout or "")


def _detect_equirect_support(colmap: str, extract_help: str, version: tuple[int, ...] | None) -> bool:
    if "EQUIRECTANGULAR" in extract_help:
        return True
    if version is not None and version >= MIN_EQUIRECT_VERSION:
        return True
    if _binary_mentions_equirect(colmap):
        return True
    return False


@functools.lru_cache(maxsize=4)
def detect_colmap_caps(colmap_bin: str | None = None) -> ColmapCliCaps:
    """
    Probe COLMAP help for renamed SIFT/feature options and EQUIRECTANGULAR.

    Homebrew COLMAP ≥ ~3.12 moved ``SiftExtraction.max_image_size`` and
    ``*.use_gpu`` under ``FeatureExtraction`` / ``FeatureMatching``.
    Native ``EQUIRECTANGULAR`` spherical cameras require COLMAP ≥ 4.1.
    """
    colmap = colmap_bin or shutil.which("colmap") or "colmap"
    extract_help = _help_text(colmap, "feature_extractor")
    match_help = _help_text(colmap, "exhaustive_matcher")
    if not match_help:
        match_help = _help_text(colmap, "sequential_matcher")
    # Some builds expose version only via bare -h / --version
    version_blob = extract_help + "\n" + match_help
    for args in ([colmap, "--version"], [colmap, "-h"]):
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            version_blob += f"\n{proc.stdout or ''}\n{proc.stderr or ''}"
        except (OSError, subprocess.TimeoutExpired):
            pass

    version = _parse_version(version_blob)

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

    supports_eq = _detect_equirect_support(colmap, extract_help, version)

    # GLOMAP lives as ``colmap global_mapper`` in modern COLMAP builds
    global_help = _help_text(colmap, "global_mapper")
    top_help = version_blob
    supports_global = (
        "global_mapper" in top_help.lower()
        or "GlobalMapper" in global_help
        or "database_path" in global_help
    )
    ba_flags: list[str] = []
    # Disable intrinsic refine for EQUIRECTANGULAR (no focal / PP / distortion)
    for flag in (
        "--BundleAdjustment.refine_focal_length",
        "--GlobalMapper.ba_refine_focal_length",
        "--Mapper.ba_refine_focal_length",
    ):
        key = flag.lstrip("-")
        if key in global_help or flag[2:] in global_help:
            ba_flags.extend([flag, "0"])
            break
    for flag in (
        "--BundleAdjustment.refine_principal_point",
        "--GlobalMapper.ba_refine_principal_point",
        "--Mapper.ba_refine_principal_point",
    ):
        key = flag.lstrip("-")
        if key in global_help or flag[2:] in global_help:
            ba_flags.extend([flag, "0"])
            break
    for flag in (
        "--BundleAdjustment.refine_extra_params",
        "--GlobalMapper.ba_refine_extra_params",
        "--Mapper.ba_refine_extra_params",
    ):
        key = flag.lstrip("-")
        if key in global_help or flag[2:] in global_help:
            ba_flags.extend([flag, "0"])
            break

    return ColmapCliCaps(
        max_image_size=max_image_size,
        extract_use_gpu=extract_use_gpu,
        max_num_features=max_num_features,
        match_use_gpu=match_use_gpu,
        modern=modern or (not legacy_size),
        supports_equirectangular=supports_eq,
        supports_global_mapper=supports_global,
        version=version,
        colmap_bin=colmap,
        global_ba_disable_flags=tuple(ba_flags),
    )


def equirect_requirement_message(caps: ColmapCliCaps | None = None) -> str:
    """User-facing upgrade instructions when EQUIRECTANGULAR is unavailable."""
    caps = caps or detect_colmap_caps()
    ver = ".".join(str(x) for x in caps.version) if caps.version else "unknown"
    return (
        f"COLMAP at {caps.colmap_bin} (version {ver}) does not support the "
        f"EQUIRECTANGULAR camera model (needs ≥ {MIN_EQUIRECT_VERSION[0]}."
        f"{MIN_EQUIRECT_VERSION[1]}).\n"
        "InstaSplat uses full 360 panoramas for SfM and metal_equirect training.\n"
        "Upgrade COLMAP, then re-run:\n"
        "  brew update && brew upgrade colmap\n"
        "  # or build ≥ 4.1 from https://github.com/colmap/colmap\n"
        "Verify with:  instasplat doctor\n"
        "Escape hatch only: set sfm.mode: perspective_cubemap (not recommended)."
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
