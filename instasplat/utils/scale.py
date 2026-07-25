"""Metric scale helpers for COLMAP models."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def read_points3d_txt(path: Path) -> dict[int, np.ndarray]:
    """Parse COLMAP points3D.txt → {point_id: xyz}."""
    points: dict[int, np.ndarray] = {}
    if not path.exists():
        return points
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pid = int(parts[0])
        xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        points[pid] = xyz
    return points


def read_images_txt(path: Path) -> list[dict]:
    """Parse COLMAP images.txt (pose lines only)."""
    images: list[dict] = []
    if not path.exists():
        return images
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    # Each image = 2 lines
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        images.append(
            {
                "image_id": int(parts[0]),
                "qw": float(parts[1]),
                "qx": float(parts[2]),
                "qy": float(parts[3]),
                "qz": float(parts[4]),
                "tx": float(parts[5]),
                "ty": float(parts[6]),
                "tz": float(parts[7]),
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
        )
    return images


def camera_centers(images: list[dict]) -> np.ndarray:
    """World-space camera centers from COLMAP qwxyz + tvec."""
    centers = []
    for im in images:
        q = np.array([im["qw"], im["qx"], im["qy"], im["qz"]], dtype=np.float64)
        R = qvec_to_rotmat(q)
        t = np.array([im["tx"], im["ty"], im["tz"]], dtype=np.float64)
        # x_cam = R * x_world + t  →  center = -R^T t
        centers.append(-R.T @ t)
    return np.asarray(centers, dtype=np.float64)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def scale_factor_from_known_distance(
    points: dict[int, np.ndarray],
    id_a: int,
    id_b: int,
    known_m: float,
) -> float:
    if id_a not in points or id_b not in points:
        raise KeyError(f"Point ids {id_a}, {id_b} not both in reconstruction")
    dist = float(np.linalg.norm(points[id_a] - points[id_b]))
    if dist < 1e-12:
        raise ValueError("Selected points are coincident")
    return known_m / dist


def scale_factor_from_gps_path(
    centers: np.ndarray,
    gps_xyz_m: np.ndarray,
) -> float:
    """Estimate scale by comparing COLMAP path length to GPS path length."""
    if len(centers) < 2 or len(gps_xyz_m) < 2:
        raise ValueError("Need at least 2 poses for GPS scale")
    colmap_len = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    gps_len = float(np.sum(np.linalg.norm(np.diff(gps_xyz_m, axis=0), axis=1)))
    if colmap_len < 1e-12:
        raise ValueError("COLMAP path length is zero")
    return gps_len / colmap_len


def scale_factor_from_stereo_baseline(
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    baseline_m: float,
) -> float:
    """
    If dual-lens poses are available as paired centers, estimate scale from
    mean inter-lens distance vs known physical baseline.
    """
    if len(centers_a) != len(centers_b) or len(centers_a) == 0:
        raise ValueError("Paired stereo centers required")
    dists = np.linalg.norm(centers_a - centers_b, axis=1)
    mean_d = float(np.mean(dists))
    if mean_d < 1e-12:
        raise ValueError("Estimated stereo separation is zero")
    return baseline_m / mean_d


def apply_scale_to_model(src: Path, dst: Path, scale: float) -> None:
    """
    Copy a COLMAP text/binary model and scale translations + points by ``scale``.
    Prefers text model if present; otherwise shells out is left to caller.
    """
    dst.mkdir(parents=True, exist_ok=True)
    cameras = src / "cameras.txt"
    images = src / "images.txt"
    points = src / "points3D.txt"
    if not (cameras.exists() and images.exists() and points.exists()):
        # Binary models: write a scale.txt sidecar and instruct conversion
        (dst / "scale_factor.txt").write_text(f"{scale}\n", encoding="utf-8")
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            src_f = src / name
            if src_f.exists():
                (dst / name).write_bytes(src_f.read_bytes())
        (dst / "README_SCALE.txt").write_text(
            "Binary model copied; apply scale with "
            "`colmap model_transformer --transform_path scale_transform.txt` "
            "or re-export to TXT and re-run scale stage.\n"
            f"scale_factor={scale}\n",
            encoding="utf-8",
        )
        # Write a 4x4 similarity transform for COLMAP model_transformer
        # S * I, translation scaled separately is handled by scaling points/images
        transform = [
            f"{scale} 0 0 0",
            f"0 {scale} 0 0",
            f"0 0 {scale} 0",
            "0 0 0 1",
        ]
        (dst / "scale_transform.txt").write_text("\n".join(transform) + "\n", encoding="utf-8")
        return

    (dst / "cameras.txt").write_text(cameras.read_text(encoding="utf-8"), encoding="utf-8")

    out_images: list[str] = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    lines = [ln for ln in images.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        tx, ty, tz = float(parts[5]) * scale, float(parts[6]) * scale, float(parts[7]) * scale
        parts[5], parts[6], parts[7] = f"{tx}", f"{ty}", f"{tz}"
        out_images.append(" ".join(parts))
        out_images.append(lines[i + 1] if i + 1 < len(lines) else "")
    (dst / "images.txt").write_text("\n".join(out_images) + "\n", encoding="utf-8")

    out_pts = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for line in points.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        x, y, z = float(parts[1]) * scale, float(parts[2]) * scale, float(parts[3]) * scale
        parts[1], parts[2], parts[3] = f"{x}", f"{y}", f"{z}"
        out_pts.append(" ".join(parts))
    (dst / "points3D.txt").write_text("\n".join(out_pts) + "\n", encoding="utf-8")
    (dst / "scale_factor.txt").write_text(f"{scale}\n", encoding="utf-8")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def gps_latlon_to_local_xyz(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray | None = None) -> np.ndarray:
    """Convert lat/lon(/alt) to local ENU meters relative to first sample."""
    if alt is None:
        alt = np.zeros_like(lat)
    origin_lat, origin_lon, origin_alt = float(lat[0]), float(lon[0]), float(alt[0])
    xyz = np.zeros((len(lat), 3), dtype=np.float64)
    for i, (la, lo, al) in enumerate(zip(lat, lon, alt, strict=True)):
        # Approximate local ENU
        north = haversine_m(origin_lat, origin_lon, float(la), origin_lon)
        east = haversine_m(origin_lat, origin_lon, origin_lat, float(lo))
        if la < origin_lat:
            north = -north
        if lo < origin_lon:
            east = -east
        xyz[i] = [east, north, float(al) - origin_alt]
    return xyz
