"""Pose / camera refine (Self-Cali-GS inspired, Mac-friendly)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger, run_cmd
from instasplat.utils.scale import (
    iter_images_txt_rows,
    qvec_to_rotmat,
    read_images_txt,
)
from instasplat.utils.telemetry import (
    integrate_orientation,
    interpolate_rotation,
    interpolate_xyz,
    load_gps,
    load_gyro,
)


@dataclass
class RefineResult:
    model_dir: Path
    method: str
    notes: list[str]


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → COLMAP qw,qx,qy,qz."""
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


def _find_text_model(model_dir: Path) -> Path | None:
    parent = model_dir.parent
    txt = parent / f"{model_dir.name}_txt"
    if (txt / "images.txt").exists():
        return txt
    if (model_dir / "images.txt").exists():
        return model_dir
    return None


def _run_colmap_ba(cfg: PipelineConfig, paths: JobPaths, model_dir: Path, out_dir: Path) -> bool:
    colmap = shutil.which("colmap")
    if not colmap:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap,
        "bundle_adjuster",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(out_dir),
        "--BundleAdjustment.refine_focal_length",
        "1" if cfg.refine.refine_intrinsics else "0",
        "--BundleAdjustment.refine_principal_point",
        "0",
        "--BundleAdjustment.refine_extra_params",
        "1" if cfg.refine.refine_distortion else "0",
    ]
    try:
        run_cmd(cmd, log_file=paths.logs / "colmap_ba.log", dry_run=cfg.dry_run)
        return cfg.dry_run or any(out_dir.iterdir())
    except RuntimeError:
        return False


def _blend_poses_with_gyro_gps(
    images_txt: Path,
    out_images_txt: Path,
    *,
    gyro_csv: Path | None,
    gps_csv: Path | None,
    frame_times: dict[str, float],
    pose_blend: float,
) -> list[str]:
    """
    Soft-blend COLMAP camera centers toward GPS, and yaw toward gyro.

    This is a lightweight Self-Cali-style extrinsic prior — not full joint GS
    optimization, but improves metric consistency before splat training.
    """
    notes: list[str] = []
    images = read_images_txt(images_txt)
    if not images:
        shutil.copy2(images_txt, out_images_txt)
        return ["no images to refine"]

    gyro = load_gyro(gyro_csv) if gyro_csv and gyro_csv.exists() else None
    gps = load_gps(gps_csv) if gps_csv and gps_csv.exists() else None
    gyro_rots = integrate_orientation(gyro) if gyro is not None else None
    alpha = float(np.clip(pose_blend, 0.0, 1.0))

    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    # Preserve POINTS2D lines from original (handles empty POINTS2D rows)
    pts_by_name = {
        " ".join(pose.split()[9:]): pts for pose, pts in iter_images_txt_rows(images_txt)
    }

    for im in images:
        name = Path(im["name"]).stem
        base = name
        for face in ("_front", "_right", "_back", "_left", "_up", "_down"):
            if base.endswith(face):
                base = base[: -len(face)]
                break
        t_sec = frame_times.get(base, frame_times.get(name))

        q = np.array([im["qw"], im["qx"], im["qy"], im["qz"]], dtype=np.float64)
        R = qvec_to_rotmat(q)
        tvec = np.array([im["tx"], im["ty"], im["tz"]], dtype=np.float64)
        center = -R.T @ tvec

        if gps is not None and t_sec is not None:
            target = interpolate_xyz(gps, float(t_sec))
            center = (1 - alpha) * center + alpha * target
            notes.append("gps_center_blend")

        if gyro_rots is not None and gyro is not None and t_sec is not None:
            Rg = interpolate_rotation(gyro_rots, gyro.t_sec, float(t_sec))
            # Blend rotations in tangent space (simple lerp + reortho)
            Rb = (1 - alpha) * R + alpha * Rg
            u, _, vt = np.linalg.svd(Rb)
            R = u @ vt
            if np.linalg.det(R) < 0:
                u[:, -1] *= -1
                R = u @ vt
            notes.append("gyro_rot_blend")

        t_new = -R @ center
        q_new = _rotmat_to_qvec(R)
        pose_line = (
            f"{im['image_id']} {q_new[0]} {q_new[1]} {q_new[2]} {q_new[3]} "
            f"{t_new[0]} {t_new[1]} {t_new[2]} {im['camera_id']} {im['name']}"
        )
        lines.append(pose_line)
        lines.append(pts_by_name.get(im["name"], ""))

    out_images_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sorted(set(notes)) or ["pose_copy"]


def run_refine(cfg: PipelineConfig, paths: JobPaths, model_dir: Path) -> RefineResult:
    """
    Refine COLMAP poses/intrinsics before training.

    Steps:
    1) Optional COLMAP bundle_adjuster (intrinsics / distortion flags)
    2) Soft GPS/gyro extrinsic blend (Self-Cali-inspired prior)
    """
    paths.ensure()
    log = get_logger("instasplat.refine", paths.logs / "refine.log")
    out = paths.root / "03b_refine" / "sparse" / "0"
    out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    if not cfg.refine.enabled:
        if not cfg.dry_run and model_dir.exists():
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(model_dir, out)
        return RefineResult(out, "disabled_copy", ["refine.disabled"])

    ba_out = paths.root / "03b_refine" / "ba" / "0"
    method = "telemetry_blend"
    src_model = model_dir
    if cfg.refine.run_colmap_ba:
        ok = _run_colmap_ba(cfg, paths, model_dir, ba_out)
        if ok:
            src_model = ba_out
            method = "colmap_ba+telemetry"
            notes.append("colmap_bundle_adjuster")
        else:
            notes.append("colmap_ba_skipped")

    txt = _find_text_model(src_model)
    if txt is None and not cfg.dry_run:
        # copy binary model as-is
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(src_model, out)
        notes.append("no_text_model")
        return RefineResult(out, method, notes)

    if cfg.dry_run:
        return RefineResult(out, method, notes + ["dry_run"])

    # Start from text model copy
    for name in ("cameras.txt", "points3D.txt"):
        src = (txt / name) if txt else None
        if src and src.exists():
            shutil.copy2(src, out / name)
    # Also copy bins if present from src_model
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        src = src_model / name
        if src.exists():
            shutil.copy2(src, out / name)

    frame_times: dict[str, float] = {}
    candidates = [
        paths.root / "01_frames" / "frame_times.json",
        paths.root / "frame_times.json",
    ]
    for c in candidates:
        if c.exists():
            frame_times = {k: float(v) for k, v in json.loads(c.read_text()).items()}
            break

    blend_notes = _blend_poses_with_gyro_gps(
        txt / "images.txt" if txt else out / "images.txt",
        out / "images.txt",
        gyro_csv=paths.gyro_csv if paths.gyro_csv.exists() else None,
        gps_csv=paths.gps_csv if paths.gps_csv.exists() else None,
        frame_times=frame_times,
        pose_blend=cfg.refine.pose_blend,
    )
    notes.extend(blend_notes)

    # Write summary
    (out.parent.parent / "refine_summary.json").write_text(
        json.dumps({"method": method, "notes": notes}, indent=2),
        encoding="utf-8",
    )
    log.info("Refine complete (%s): %s", method, ", ".join(notes))
    return RefineResult(out, method, notes)
