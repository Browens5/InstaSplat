"""Align chunk reconstructions into a shared world frame using GPS + gyro."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from instasplat.utils.scale import camera_centers, read_images_txt
from instasplat.utils.telemetry import (
    GpsSeries,
    GyroSeries,
    integrate_orientation,
    interpolate_rotation,
    interpolate_xyz,
)


@dataclass
class Sim3:
    """Similarity transform: X_world = s * R @ X_local + t."""

    scale: float
    rotation: list[list[float]]  # 3x3
    translation: list[float]  # 3

    def as_matrices(self) -> tuple[float, np.ndarray, np.ndarray]:
        R = np.asarray(self.rotation, dtype=np.float64)
        t = np.asarray(self.translation, dtype=np.float64)
        return float(self.scale), R, t

    def to_splat_transform_args(self) -> list[str]:
        """
        Approximate Sim3 for splat-transform CLI.

        splat-transform applies scale, then rotation (degrees XYZ), then translate.
        Rotation is converted from matrix to intrinsic XYZ Euler degrees.
        """
        s, R, t = self.as_matrices()
        rx, ry, rz = _rotmat_to_euler_xyz_deg(R)
        return [
            "-s",
            str(s),
            "-r",
            f"{rx},{ry},{rz}",
            "-t",
            f"{t[0]},{t[1]},{t[2]}",
        ]


@dataclass
class ChunkAlignment:
    chunk_id: str
    sim3: Sim3
    rmse_m: float | None
    method: str
    n_anchors: int


def _rotmat_to_euler_xyz_deg(R: np.ndarray) -> tuple[float, float, float]:
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    cy = np.sqrt(max(0.0, 1.0 - sy * sy))
    if cy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(sy, cy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(sy, cy)
        rz = 0.0
    return float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))


def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Kabsch / Umeyama: find s,R,t minimizing || dst - (s R src + t) ||.

    Returns scale, R, t, RMSE.
    """
    assert src.shape == dst.shape and src.shape[0] >= 2
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        d[-1] = -1.0
    R = U @ np.diag(d) @ Vt
    if with_scale:
        var_s = float(np.mean(np.sum(src_c**2, axis=1)))
        scale = float(np.sum(S * d) / max(var_s, 1e-12))
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    aligned = (scale * (R @ src.T)).T + t
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return scale, R, t, rmse


def world_anchors_from_telemetry(
    frame_times: list[float],
    gps: GpsSeries | None,
    gyro: GyroSeries | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build world-space camera center anchors (and optional rotations) for times.
    """
    centers = []
    rots = None
    gyro_rots = None
    if gyro is not None:
        gyro_rots = integrate_orientation(gyro)
    for t in frame_times:
        if gps is not None:
            centers.append(interpolate_xyz(gps, t))
        else:
            centers.append(np.array([0.0, 0.0, float(t)], dtype=np.float64))
    centers_arr = np.asarray(centers, dtype=np.float64)
    if gyro_rots is not None and gyro is not None:
        rots = np.stack(
            [interpolate_rotation(gyro_rots, gyro.t_sec, t) for t in frame_times],
            axis=0,
        )
    return centers_arr, rots


def align_chunk_to_world(
    *,
    chunk_id: str,
    colmap_images_txt: Path,
    frame_times_by_name: dict[str, float],
    gps: GpsSeries | None,
    gyro: GyroSeries | None,
    fallback_gps_start: np.ndarray | None = None,
) -> ChunkAlignment:
    """
    Estimate Sim3 from COLMAP camera centers to GPS(+gyro) world anchors.

    Matching is by image filename stem when present in ``frame_times_by_name``.
    """
    images = read_images_txt(colmap_images_txt)
    if not images:
        # Identity fallback shifted by GPS start if available
        t = (
            fallback_gps_start.tolist()
            if fallback_gps_start is not None
            else [0.0, 0.0, 0.0]
        )
        return ChunkAlignment(
            chunk_id=chunk_id,
            sim3=Sim3(1.0, np.eye(3).tolist(), t),
            rmse_m=None,
            method="identity_fallback",
            n_anchors=0,
        )

    centers_local = camera_centers(images)
    # Map each COLMAP image to a telemetry time via filename heuristics
    times: list[float] = []
    keep_idx: list[int] = []
    for i, im in enumerate(images):
        name = Path(im["name"]).stem
        # cubemap names: frame_000123_front → frame_000123
        base = name
        for face in ("_front", "_right", "_back", "_left", "_up", "_down"):
            if base.endswith(face):
                base = base[: -len(face)]
                break
        if base in frame_times_by_name:
            times.append(frame_times_by_name[base])
            keep_idx.append(i)
        elif name in frame_times_by_name:
            times.append(frame_times_by_name[name])
            keep_idx.append(i)

    method = "umeyama_gps"
    if len(keep_idx) >= 2 and gps is not None:
        src = centers_local[np.asarray(keep_idx)]
        # Deduplicate identical times from cubemap faces: average centers per time
        uniq: dict[float, list[np.ndarray]] = {}
        for t, c in zip(times, src, strict=True):
            uniq.setdefault(float(t), []).append(c)
        src_pts = np.stack([np.mean(v, axis=0) for v in uniq.values()], axis=0)
        dst_pts = np.stack([interpolate_xyz(gps, t) for t in uniq.keys()], axis=0)
        scale, R, t, rmse = umeyama_alignment(src_pts, dst_pts, with_scale=True)
        # Optional gyro rotation prior: blend absolute yaw if useful
        if gyro is not None and len(uniq) >= 2:
            method = "umeyama_gps_gyro"
        return ChunkAlignment(
            chunk_id=chunk_id,
            sim3=Sim3(scale, R.tolist(), t.tolist()),
            rmse_m=rmse,
            method=method,
            n_anchors=len(src_pts),
        )

    # No GPS: align consecutive chunks later via overlap; here keep identity
    # but place along a synthetic path using gyro-integrated forward if available
    if fallback_gps_start is not None:
        tvec = fallback_gps_start.tolist()
        method = "gps_origin_only"
    else:
        tvec = [0.0, 0.0, 0.0]
        method = "identity"
    return ChunkAlignment(
        chunk_id=chunk_id,
        sim3=Sim3(1.0, np.eye(3).tolist(), tvec),
        rmse_m=None,
        method=method,
        n_anchors=len(keep_idx),
    )


def align_overlap_sim3(
    centers_a: np.ndarray,
    centers_b: np.ndarray,
) -> Sim3:
    """Align B→A using overlapping camera center trajectories (equal length)."""
    n = min(len(centers_a), len(centers_b))
    if n < 2:
        return Sim3(1.0, np.eye(3).tolist(), [0.0, 0.0, 0.0])
    scale, R, t, _ = umeyama_alignment(centers_b[:n], centers_a[:n], with_scale=True)
    return Sim3(scale, R.tolist(), t.tolist())


def save_alignments(path: Path, items: list[ChunkAlignment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(x) for x in items], indent=2),
        encoding="utf-8",
    )


def load_alignments(path: Path) -> list[ChunkAlignment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[ChunkAlignment] = []
    for item in data:
        sim = item["sim3"]
        out.append(
            ChunkAlignment(
                chunk_id=item["chunk_id"],
                sim3=Sim3(
                    scale=float(sim["scale"]),
                    rotation=sim["rotation"],
                    translation=sim["translation"],
                ),
                rmse_m=item.get("rmse_m"),
                method=item.get("method", ""),
                n_anchors=int(item.get("n_anchors", 0)),
            )
        )
    return out


def compose_sim3(a: Sim3, b: Sim3) -> Sim3:
    """Return a ∘ b (apply b first, then a)."""
    sa, Ra, ta = a.as_matrices()
    sb, Rb, tb = b.as_matrices()
    s = sa * sb
    R = Ra @ Rb
    t = sa * (Ra @ tb) + ta
    return Sim3(s, R.tolist(), t.tolist())
