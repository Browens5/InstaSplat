"""Gyro/GPS pose prior fallback when COLMAP fails (LongSplat / on-the-fly inspired)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger
from instasplat.utils.telemetry import (
    integrate_orientation,
    interpolate_rotation,
    interpolate_xyz,
    load_gps,
    load_gyro,
)


@dataclass
class FallbackResult:
    model_dir: Path
    n_poses: int
    method: str


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    m = R
    tr = float(np.trace(m))
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def write_telemetry_colmap_model(
    *,
    image_dir: Path,
    out_dir: Path,
    frame_times: dict[str, float],
    gyro_csv: Path | None,
    gps_csv: Path | None,
    image_size: tuple[int, int] = (1024, 1024),
    focal: float | None = None,
) -> int:
    """
    Synthesize a minimal COLMAP text model from GPS centers + gyro rotations.

    Enables continuing the splat pipeline when SfM collapses (unposed-style prior).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if not images:
        return 0

    gyro = load_gyro(gyro_csv) if gyro_csv and gyro_csv.exists() else None
    gps = load_gps(gps_csv) if gps_csv and gps_csv.exists() else None
    gyro_rots = integrate_orientation(gyro) if gyro is not None else None

    w, h = image_size
    f = focal if focal is not None else float(0.9 * w)
    cameras = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"1 SIMPLE_PINHOLE {w} {h} {f} {w/2.0} {h/2.0}",
    ]
    (out_dir / "cameras.txt").write_text("\n".join(cameras) + "\n", encoding="utf-8")

    img_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    pts_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]

    for i, img in enumerate(images, start=1):
        stem = img.stem
        base = stem
        for face in ("_front", "_right", "_back", "_left", "_up", "_down"):
            if base.endswith(face):
                base = base[: -len(face)]
                break
        t_sec = float(frame_times.get(base, frame_times.get(stem, float(i - 1))))

        if gps is not None:
            center = interpolate_xyz(gps, t_sec)
        else:
            # Unit-speed forward along +Z if no GPS
            center = np.array([0.0, 0.0, t_sec], dtype=np.float64)

        if gyro_rots is not None and gyro is not None:
            R_w_b = interpolate_rotation(gyro_rots, gyro.t_sec, t_sec)
            # world-from-body; COLMAP R is world-to-camera
            R = R_w_b.T
        else:
            R = np.eye(3, dtype=np.float64)

        tvec = -R @ center
        q = _rotmat_to_qvec(R)
        img_lines.append(
            f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {tvec[0]} {tvec[1]} {tvec[2]} 1 {img.name}"
        )
        img_lines.append("")  # empty POINTS2D
        # Seed a sparse point at each camera center for Brush init
        pts_lines.append(
            f"{i} {center[0]} {center[1]} {center[2]} 200 200 200 1.0"
        )

    (out_dir / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")
    (out_dir / "points3D.txt").write_text("\n".join(pts_lines) + "\n", encoding="utf-8")
    (out_dir / "TELEMETRY_PRIOR.txt").write_text(
        "Synthetic COLMAP model from GPS/gyro priors. "
        "Use only when SfM failed; quality will be lower than real COLMAP.\n",
        encoding="utf-8",
    )
    return len(images)


def run_fallback_poses(cfg: PipelineConfig, paths: JobPaths) -> FallbackResult | None:
    if not cfg.sfm.telemetry_fallback:
        return None
    log = get_logger("instasplat.fallback", paths.logs / "fallback_poses.log")
    model = paths.colmap_model
    # If a real model already exists, skip
    if (model / "images.bin").exists() or (model / "images.txt").exists():
        return None

    ft_path = paths.root / "01_frames" / "frame_times.json"
    frame_times = {}
    if ft_path.exists():
        frame_times = {k: float(v) for k, v in json.loads(ft_path.read_text()).items()}

    out = paths.colmap_sparse / "0"
    n = 0 if cfg.dry_run else write_telemetry_colmap_model(
        image_dir=paths.cubemap_images,
        out_dir=out,
        frame_times=frame_times,
        gyro_csv=paths.gyro_csv if paths.gyro_csv.exists() else None,
        gps_csv=paths.gps_csv if paths.gps_csv.exists() else None,
        image_size=(cfg.sfm.face_resolution, cfg.sfm.face_resolution),
    )
    if cfg.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        n = 0
    if n <= 0 and not cfg.dry_run:
        log.warning(
            "Telemetry fallback produced 0 poses (no images in %s and/or no frame_times) — not usable",
            paths.cubemap_images,
        )
        return None
    log.warning("Installed telemetry pose prior with %d images (SfM fallback)", n)
    return FallbackResult(out, n, "gyro_gps_prior")
