"""COLMAP sparse reconstruction from equirectangular frames."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from instasplat.config import PipelineConfig
from instasplat.utils.colmap_cli import (
    db_image_count,
    detect_colmap_caps,
    equirect_requirement_message,
    list_readable_images,
    quality_feature_args,
)
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger, run_cmd


@dataclass
class SfMResult:
    model_dir: Path
    image_dir: Path
    mask_dir: Path | None
    mode: str
    num_images: int


# Cubemap face order: front, right, back, left, up, down
FACE_YAW_PITCH = [
    (0.0, 0.0),
    (90.0, 0.0),
    (180.0, 0.0),
    (-90.0, 0.0),
    (0.0, 90.0),
    (0.0, -90.0),
]
FACE_NAMES = ["front", "right", "back", "left", "up", "down"]


def equirect_to_perspective(
    equirect: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_size: int,
) -> np.ndarray:
    """Sample a perspective view from an equirectangular panorama."""
    h, w = equirect.shape[:2]
    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Camera axes after yaw/pitch (y-up, -z forward convention for sampling)
    xs = np.linspace(-math.tan(fov / 2), math.tan(fov / 2), out_size)
    ys = np.linspace(-math.tan(fov / 2), math.tan(fov / 2), out_size)
    u, v = np.meshgrid(xs, ys)
    dirs = np.stack([u, -v, np.ones_like(u)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Rotate pitch then yaw
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    y2 = cp * y - sp * z
    z2 = sp * y + cp * z
    x3 = cy * x + sy * z2
    z3 = -sy * x + cy * z2
    y3 = y2

    lon = np.arctan2(x3, z3)
    lat = np.arcsin(np.clip(y3, -1.0, 1.0))
    px = (lon / (2 * math.pi) + 0.5) * w
    py = (0.5 - lat / math.pi) * h
    map_x = px.astype(np.float32)
    map_y = py.astype(np.float32)
    return cv2.remap(
        equirect,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def render_cubemaps(cfg: PipelineConfig, paths: JobPaths) -> tuple[Path, Path | None, int]:
    """Convert equirect frames (+ masks) into perspective cubemap faces for COLMAP."""
    log = get_logger("instasplat.sfm")
    src = paths.equirect_frames
    mask_src = paths.equirect_masks
    img_out = paths.cubemap_images
    mask_out = paths.cubemap_masks
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    frames = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")))
    if not frames:
        raise FileNotFoundError(f"No equirect frames in {src}")

    existing = list(img_out.glob("*.jpg")) + list(img_out.glob("*.png"))
    n_faces = min(cfg.sfm.cubemap_faces, 6)
    expected = len(frames) * n_faces
    if existing and cfg.skip_existing and len(existing) >= expected:
        log.info("Skipping cubemap render; found %d images", len(existing))
        has_masks = any(mask_out.glob("*.png"))
        return img_out, mask_out if has_masks else None, len(existing)

    if not cfg.dry_run:
        for p in img_out.glob("*"):
            if p.is_file():
                p.unlink()
        for p in mask_out.glob("*"):
            if p.is_file():
                p.unlink()

    count = 0
    for fr in frames:
        eq = cv2.imread(str(fr), cv2.IMREAD_COLOR)
        if eq is None:
            continue
        mask_path = mask_src / f"{fr.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        for i in range(n_faces):
            yaw, pitch = FACE_YAW_PITCH[i]
            face = equirect_to_perspective(
                eq, yaw, pitch, cfg.sfm.face_fov_deg, cfg.sfm.face_resolution
            )
            name = f"{fr.stem}_{FACE_NAMES[i]}.jpg"
            if not cfg.dry_run:
                cv2.imwrite(str(img_out / name), face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if mask is not None:
                    mface = equirect_to_perspective(
                        mask, yaw, pitch, cfg.sfm.face_fov_deg, cfg.sfm.face_resolution
                    )
                    cv2.imwrite(str(mask_out / f"{fr.stem}_{FACE_NAMES[i]}.png"), mface)
            count += 1
    log.info("Rendered %d cubemap faces", count)
    has_masks = any(mask_out.glob("*.png")) if not cfg.dry_run else mask is not None
    return img_out, mask_out if has_masks else None, count


def _is_usable_colmap_model(model_dir: Path) -> bool:
    """True if model looks like a real COLMAP reconstruction (not telemetry prior)."""
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        return False
    # fallback.py writes this marker when SfM failed and gyro/GPS poses were installed
    if (model_dir / "TELEMETRY_PRIOR.txt").exists():
        return False
    return (model_dir / "images.bin").exists() or (model_dir / "images.txt").exists()


def _empty_frames_hint(paths: JobPaths, cfg: PipelineConfig) -> str:
    chunks = paths.chunks
    has_chunks = chunks.is_dir() and any(chunks.glob("chunk_*"))
    tiled = cfg.mode == "tiled" or cfg.chunk.enabled
    lines = [
        f"No readable frames in {paths.equirect_frames}.",
    ]
    if tiled or has_chunks:
        lines.append(
            "This job looks tiled — do not run the single-mode `sfm` stage. "
            "Continue with `process_chunks` (per-tile SfM), e.g.:\n"
            "  instasplat run --job <job> --only process_chunks,align_chunks,merge_chunks,package\n"
            "or GUI: Open previous run → check process_chunks onward."
        )
    else:
        lines.append(
            "Run `extract` (and `mask`) first, or open the job and continue from extract."
        )
    lines.append(
        "Default SfM mode is equirectangular (full 360 panoramas, COLMAP ≥ 4.1)."
    )
    return "\n".join(lines)


def _prepare_equirect_images(
    cfg: PipelineConfig, paths: JobPaths
) -> tuple[Path, Path | None, int]:
    """Stage equirect frames into a dedicated folder (not mixed with cubemap faces)."""
    log = get_logger("instasplat.sfm")
    img_dir = paths.equirect_sfm_images
    mask_dir_out = paths.equirect_sfm_masks
    img_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(
        list(paths.equirect_frames.glob("*.jpg"))
        + list(paths.equirect_frames.glob("*.png"))
        + list(paths.equirect_frames.glob("*.jpeg"))
    )
    if not frames and not cfg.dry_run:
        raise FileNotFoundError(_empty_frames_hint(paths, cfg))

    # Refresh staging dir so stale cubemap faces from other modes are not mixed in
    if not cfg.dry_run:
        for p in img_dir.glob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink()
        if mask_dir_out.exists():
            for p in mask_dir_out.glob("*"):
                if p.is_file() or p.is_symlink():
                    p.unlink()

    for fr in frames:
        dest = img_dir / fr.name
        if cfg.dry_run:
            continue
        try:
            dest.symlink_to(fr.resolve())
        except OSError:
            shutil.copy2(fr, dest)

    mask_dir: Path | None = None
    if any(paths.equirect_masks.glob("*.png")):
        mask_dir = mask_dir_out
        mask_dir.mkdir(parents=True, exist_ok=True)
        for m in paths.equirect_masks.glob("*.png"):
            dest = mask_dir / m.name
            if not cfg.dry_run:
                try:
                    dest.symlink_to(m.resolve())
                except OSError:
                    shutil.copy2(m, dest)
    log.info("Staged %d equirect frames into %s", len(frames), img_dir)
    return img_dir, mask_dir, len(frames)


def run_colmap(
    cfg: PipelineConfig,
    paths: JobPaths,
    image_dir: Path,
    mask_dir: Path | None,
    camera_model: str,
) -> Path:
    log = get_logger("instasplat.sfm")
    colmap = shutil.which("colmap")
    if not colmap and not cfg.dry_run:
        raise RuntimeError(
            "COLMAP not found on PATH. Install with: brew install colmap"
        )

    db = paths.colmap_db
    sparse = paths.colmap_sparse
    sparse.mkdir(parents=True, exist_ok=True)
    model0 = sparse / "0"

    if cfg.skip_existing and _is_usable_colmap_model(model0):
        log.info("Skipping COLMAP; existing model at %s", model0)
        return model0

    images = list_readable_images(image_dir)
    if not images and not cfg.dry_run:
        raise FileNotFoundError(
            f"No readable images in {image_dir} for COLMAP.\n"
            + _empty_frames_hint(paths, cfg)
        )
    log.info("COLMAP input: %d images in %s (model=%s)", len(images), image_dir, camera_model)

    # Replace telemetry-prior / empty DB so a fixed COLMAP can re-run
    if not cfg.dry_run and model0.exists() and not _is_usable_colmap_model(model0):
        log.info("Removing non-COLMAP / telemetry prior at %s before re-run", model0)
        shutil.rmtree(model0)

    need_fresh_db = (not cfg.skip_existing) or (db_image_count(db) == 0) or (
        model0.exists() and not _is_usable_colmap_model(model0)
    )
    if db.exists() and need_fresh_db and not cfg.dry_run:
        log.info("Recreating COLMAP database (was empty or incomplete)")
        db.unlink()

    caps = detect_colmap_caps(colmap)
    log.info(
        "COLMAP feature options: %s / %s (%s)",
        caps.max_image_size,
        caps.extract_use_gpu,
        "modern" if caps.modern else "legacy",
    )

    # Equirect sequences share one camera; cubemap faces do not
    single_camera = "1" if camera_model.upper() == "EQUIRECTANGULAR" else "0"

    extract_cmd = [
        colmap or "colmap",
        "feature_extractor",
        "--database_path",
        str(db),
        "--image_path",
        str(image_dir),
        "--ImageReader.single_camera",
        single_camera,
        "--ImageReader.camera_model",
        camera_model,
        *quality_feature_args(cfg.sfm.quality, caps),
        caps.extract_use_gpu,
        "1" if cfg.sfm.use_gpu else "0",
    ]
    if mask_dir is not None and any(mask_dir.glob("*.png")):
        extract_cmd.extend(["--ImageReader.mask_path", str(mask_dir)])

    run_cmd(extract_cmd, log_file=paths.logs / "colmap_features.log", dry_run=cfg.dry_run)

    n_db = 0 if cfg.dry_run else db_image_count(db)
    if not cfg.dry_run and n_db == 0:
        raise RuntimeError(
            f"COLMAP feature_extractor registered 0 images from {image_dir} "
            f"(camera_model={camera_model}). "
            "Check colmap_features.log. Ensure camera_model=EQUIRECTANGULAR and "
            "COLMAP ≥ 4.1; for tiled jobs run process_chunks instead of top-level sfm."
        )
    log.info("Feature extraction registered %d images", n_db or len(images))

    gpu_flag = "1" if cfg.sfm.use_gpu else "0"
    if cfg.sfm.matcher == "exhaustive":
        match_cmd = [
            colmap or "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(db),
            caps.match_use_gpu,
            gpu_flag,
        ]
    else:
        match_cmd = [
            colmap or "colmap",
            "sequential_matcher",
            "--database_path",
            str(db),
            "--SequentialMatching.overlap",
            str(cfg.sfm.sequential_overlap),
            caps.match_use_gpu,
            gpu_flag,
        ]
    run_cmd(match_cmd, log_file=paths.logs / "colmap_match.log", dry_run=cfg.dry_run)

    # Clean previous sparse outputs when re-running a full COLMAP pass
    if not cfg.dry_run and (not cfg.skip_existing or not _is_usable_colmap_model(model0)):
        for child in list(sparse.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)

    map_cmd = [
        colmap or "colmap",
        "mapper",
        "--database_path",
        str(db),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse),
    ]
    run_cmd(map_cmd, log_file=paths.logs / "colmap_mapper.log", dry_run=cfg.dry_run)

    # Prefer model 0; if only others exist, pick largest
    if cfg.dry_run:
        model0.mkdir(parents=True, exist_ok=True)
        return model0

    models = [p for p in sparse.iterdir() if p.is_dir()]
    if not models:
        raise RuntimeError("COLMAP mapper produced no models")
    if model0 not in models:
        # Use first model
        model0 = min(models)
    # Export text model for scale stage convenience
    txt_dir = sparse / f"{model0.name}_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            colmap or "colmap",
            "model_converter",
            "--input_path",
            str(model0),
            "--output_path",
            str(txt_dir),
            "--output_type",
            "TXT",
        ],
        log_file=paths.logs / "colmap_to_txt.log",
        dry_run=cfg.dry_run,
        check=False,
    )
    return model0


def _resolve_sfm_mode(cfg: PipelineConfig, caps) -> str:
    """
    Choose SfM mode. Default product path is full EQUIRECTANGULAR panoramas.

    ``auto`` → equirectangular when COLMAP supports it, else raise (no silent
    cubemap downgrade). Cubemap remains an explicit opt-in.
    """
    mode = cfg.sfm.mode
    if mode == "auto":
        if caps is not None and caps.supports_equirectangular:
            return "equirectangular"
        if caps is None:
            # dry-run / missing binary — still prefer equirect staging
            return "equirectangular"
        raise RuntimeError(equirect_requirement_message(caps))
    return mode


def run_sfm(cfg: PipelineConfig, paths: JobPaths) -> SfMResult:
    paths.ensure()
    log = get_logger("instasplat.sfm", paths.logs / "sfm.log")
    colmap = shutil.which("colmap")
    caps = detect_colmap_caps(colmap) if colmap else None
    mode = _resolve_sfm_mode(cfg, caps)

    if mode == "equirectangular":
        if caps is not None and not caps.supports_equirectangular and not cfg.dry_run:
            raise RuntimeError(equirect_requirement_message(caps))
        if not any(paths.equirect_frames.glob("*")) and (
            cfg.mode == "tiled" or cfg.chunk.enabled or any(paths.chunks.glob("chunk_*"))
        ):
            raise FileNotFoundError(_empty_frames_hint(paths, cfg))
        log.info(
            "Running COLMAP with EQUIRECTANGULAR camera model "
            "(full 360 panoramas → SfM → metal_equirect)"
        )
        img_dir, mask_dir, n = _prepare_equirect_images(cfg, paths)
        # Force EQUIRECTANGULAR regardless of sfm.camera_model (pinhole is cubemap-only)
        model = run_colmap(cfg, paths, img_dir, mask_dir, "EQUIRECTANGULAR")
        return SfMResult(model, img_dir, mask_dir, mode, n)

    if mode != "perspective_cubemap":
        raise ValueError(f"Unknown sfm.mode: {mode}")

    # Explicit opt-in: cubemap perspective rig (older COLMAP / debugging)
    log.warning(
        "sfm.mode=perspective_cubemap — remapping panoramas to pinhole faces. "
        "Prefer equirectangular (COLMAP ≥ 4.1) for a full-360 SfM+train path."
    )
    if not any(paths.equirect_frames.glob("*")) and not cfg.dry_run:
        raise FileNotFoundError(_empty_frames_hint(paths, cfg))
    img_dir, mask_dir, n = render_cubemaps(cfg, paths)
    cam = cfg.sfm.camera_model
    if cam.upper() == "EQUIRECTANGULAR":
        cam = "SIMPLE_PINHOLE"
    model = run_colmap(cfg, paths, img_dir, mask_dir, cam)
    return SfMResult(model, img_dir, mask_dir, "perspective_cubemap", n)
