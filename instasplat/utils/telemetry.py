"""Gyro / GPS telemetry loading, integration, and adaptive sampling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from instasplat.utils.scale import gps_latlon_to_local_xyz, haversine_m


@dataclass
class GyroSeries:
    t_sec: np.ndarray  # shape (N,)
    omega: np.ndarray  # shape (N, 3) rad/s approx (gx, gy, gz)


@dataclass
class GpsSeries:
    t_sec: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    xyz_m: np.ndarray  # local ENU meters


def _load_numeric_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned: dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    cleaned[k] = float(v)
                except ValueError:
                    continue
            if cleaned:
                rows.append(cleaned)
    return rows


def load_gyro(path: Path) -> GyroSeries | None:
    rows = _load_numeric_csv(path)
    if len(rows) < 2:
        return None
    # Prefer timestamp_ms; fall back to first column-ish keys
    ts = np.array(
        [r.get("timestamp_ms", r.get("t", 0.0)) for r in rows],
        dtype=np.float64,
    )
    # Normalize to seconds from start
    if np.nanmax(ts) > 1e6:  # likely epoch ms
        t_sec = (ts - ts[0]) / 1000.0
    elif np.nanmax(ts) > 1000:  # relative ms
        t_sec = (ts - ts[0]) / 1000.0
    else:
        t_sec = ts - ts[0]
    omega = np.array(
        [[r.get("gx", 0.0), r.get("gy", 0.0), r.get("gz", 0.0)] for r in rows],
        dtype=np.float64,
    )
    return GyroSeries(t_sec=t_sec, omega=omega)


def load_gps(path: Path) -> GpsSeries | None:
    rows = _load_numeric_csv(path)
    if len(rows) < 2:
        return None
    ts = np.array(
        [r.get("timestamp_ms", r.get("t", float(i))) for i, r in enumerate(rows)],
        dtype=np.float64,
    )
    if np.nanmax(ts) > 1000:
        t_sec = (ts - ts[0]) / 1000.0
    else:
        t_sec = ts - ts[0]
    lat = np.array([r.get("lat", r.get("latitude", 0.0)) for r in rows], dtype=np.float64)
    lon = np.array([r.get("lon", r.get("longitude", 0.0)) for r in rows], dtype=np.float64)
    alt = np.array([r.get("alt", r.get("altitude", 0.0)) for r in rows], dtype=np.float64)
    xyz = gps_latlon_to_local_xyz(lat, lon, alt)
    return GpsSeries(t_sec=t_sec, lat=lat, lon=lon, alt=alt, xyz_m=xyz)


def angular_speed(gyro: GyroSeries) -> np.ndarray:
    return np.linalg.norm(gyro.omega, axis=1)


def integrate_orientation(gyro: GyroSeries) -> np.ndarray:
    """
    Crude orientation integration → rotation matrices world←body over time.

    Uses small-angle SO(3) exponential map; adequate as a prior for chunk
    alignment, not as a full AHRS substitute.
    """
    n = len(gyro.t_sec)
    R = np.eye(3, dtype=np.float64)
    mats = np.zeros((n, 3, 3), dtype=np.float64)
    mats[0] = R
    for i in range(1, n):
        dt = float(gyro.t_sec[i] - gyro.t_sec[i - 1])
        if dt <= 0:
            mats[i] = R
            continue
        w = gyro.omega[i - 1] * dt
        theta = float(np.linalg.norm(w))
        if theta < 1e-12:
            mats[i] = R
            continue
        k = w / theta
        K = np.array(
            [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
            dtype=np.float64,
        )
        dR = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        R = R @ dR
        # Re-orthonormalize occasionally
        if i % 64 == 0:
            u, _, vt = np.linalg.svd(R)
            R = u @ vt
        mats[i] = R
    return mats


def interpolate_xyz(gps: GpsSeries, t_sec: float) -> np.ndarray:
    if t_sec <= float(gps.t_sec[0]):
        return gps.xyz_m[0].copy()
    if t_sec >= float(gps.t_sec[-1]):
        return gps.xyz_m[-1].copy()
    idx = int(np.searchsorted(gps.t_sec, t_sec) - 1)
    idx = max(0, min(idx, len(gps.t_sec) - 2))
    t0, t1 = float(gps.t_sec[idx]), float(gps.t_sec[idx + 1])
    a = 0.0 if t1 <= t0 else (t_sec - t0) / (t1 - t0)
    return (1 - a) * gps.xyz_m[idx] + a * gps.xyz_m[idx + 1]


def interpolate_rotation(rots: np.ndarray, t_src: np.ndarray, t_sec: float) -> np.ndarray:
    if t_sec <= float(t_src[0]):
        return rots[0].copy()
    if t_sec >= float(t_src[-1]):
        return rots[-1].copy()
    idx = int(np.searchsorted(t_src, t_sec) - 1)
    idx = max(0, min(idx, len(t_src) - 2))
    # Nearest for simplicity (SLERP would be nicer)
    return rots[idx].copy() if abs(t_src[idx] - t_sec) < abs(t_src[idx + 1] - t_sec) else rots[idx + 1].copy()


def path_length_m(gps: GpsSeries, t0: float, t1: float) -> float:
    mask = (gps.t_sec >= t0) & (gps.t_sec <= t1)
    pts = gps.xyz_m[mask]
    if len(pts) < 2:
        # fall back to endpoints
        a = interpolate_xyz(gps, t0)
        b = interpolate_xyz(gps, t1)
        return float(np.linalg.norm(b - a))
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def adaptive_frame_times(
    duration_sec: float,
    *,
    base_fps: float,
    max_fps: float,
    gyro: GyroSeries | None,
    turn_omega_threshold: float = 0.35,
    boost_fps: float | None = None,
) -> np.ndarray:
    """
    Build a non-uniform frame timestamp schedule that densifies on turns.

    For 8K@30 source, base_fps might be 4–8 and boost_fps 12–15 so we keep
    much more data than a flat 1–2 fps pipeline without ingesting every frame.
    """
    boost = boost_fps if boost_fps is not None else min(max_fps, max(base_fps * 2.5, base_fps + 4))
    if duration_sec <= 0:
        return np.array([0.0], dtype=np.float64)
    if gyro is None or len(gyro.t_sec) < 2:
        n = max(1, int(duration_sec * base_fps) + 1)
        return np.linspace(0.0, duration_sec, n)

    speed = angular_speed(gyro)
    # Sample candidate times at max_fps grid, keep based on local omega
    dt = 1.0 / max(max_fps, 0.1)
    times: list[float] = []
    t = 0.0
    while t <= duration_sec + 1e-9:
        # Local omega
        gi = int(np.searchsorted(gyro.t_sec, t) - 1)
        gi = max(0, min(gi, len(speed) - 1))
        omega = float(speed[gi])
        target_fps = boost if omega >= turn_omega_threshold else base_fps
        step = 1.0 / max(target_fps, 0.1)
        if not times or (t - times[-1]) >= step - 1e-9:
            times.append(t)
        t += dt
    if not times or times[-1] < duration_sec - 1e-3:
        times.append(duration_sec)
    return np.asarray(times, dtype=np.float64)


def gps_span_m(gps: GpsSeries) -> float:
    if len(gps.lat) < 2:
        return 0.0
    return haversine_m(float(gps.lat[0]), float(gps.lon[0]), float(gps.lat[-1]), float(gps.lon[-1]))
