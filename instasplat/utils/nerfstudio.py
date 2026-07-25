"""Nerfstudio / cloud-trainer dataset packaging."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger
from instasplat.utils.scale import qvec_to_rotmat, read_images_txt


@dataclass
class PackageResult:
    nerfstudio_dir: Path | None
    hierarchy_manifest: Path | None
    notes: list[str]


def _read_cameras_txt(path: Path) -> dict[int, dict]:
    cams: dict[int, dict] = {}
    if not path.exists():
        return cams
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        cam_id = int(parts[0])
        model = parts[1]
        w, h = int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        cams[cam_id] = {"model": model, "width": w, "height": h, "params": params}
    return cams


def _colmap_image_to_c2w(im: dict) -> list[list[float]]:
    q = np.array([im["qw"], im["qx"], im["qy"], im["qz"]], dtype=np.float64)
    R = qvec_to_rotmat(q)  # world to camera
    t = np.array([im["tx"], im["ty"], im["tz"]], dtype=np.float64)
    # c2w
    R_c2w = R.T
    c = -R.T @ t
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = c
    # Nerfstudio often expects OpenGL-ish (flip YZ). Apply common COLMAP→NS convert.
    conv = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w = c2w @ conv
    return c2w.tolist()


def write_nerfstudio_transforms(
    model_dir: Path,
    image_dir: Path,
    out_dir: Path,
    *,
    copy_images: bool = False,
) -> Path:
    """
    Export a Nerfstudio-style transforms.json from a COLMAP text/bin model.

    Useful for cloud splatfacto / gsplat / 3DGUT workers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images_txt = model_dir / "images.txt"
    cameras_txt = model_dir / "cameras.txt"
    # Prefer sibling _txt
    if not images_txt.exists():
        alt = model_dir.parent / f"{model_dir.name}_txt"
        if (alt / "images.txt").exists():
            images_txt = alt / "images.txt"
            cameras_txt = alt / "cameras.txt"

    images = read_images_txt(images_txt)
    cams = _read_cameras_txt(cameras_txt)
    if not images or not cams:
        raise FileNotFoundError(f"Need cameras.txt + images.txt near {model_dir}")

    # Use first camera for global intrinsics (typical single-camera jobs)
    cam0 = next(iter(cams.values()))
    w, h = cam0["width"], cam0["height"]
    params = cam0["params"]
    if cam0["model"] in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
        fl_x = fl_y = params[0]
        cx, cy = w / 2, h / 2
        if len(params) >= 3 and cam0["model"] == "SIMPLE_PINHOLE":
            cx, cy = params[1], params[2]
    elif cam0["model"] in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE"}:
        fl_x, fl_y = params[0], params[1]
        cx, cy = params[2], params[3]
    else:
        fl_x = fl_y = params[0] if params else float(w)
        cx, cy = w / 2.0, h / 2.0

    frames = []
    img_out = out_dir / "images"
    if copy_images:
        img_out.mkdir(parents=True, exist_ok=True)

    for im in images:
        src = image_dir / im["name"]
        rel = f"images/{im['name']}"
        if copy_images and src.exists():
            dest = img_out / im["name"]
            if not dest.exists():
                try:
                    dest.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dest)
        frames.append(
            {
                "file_path": rel if copy_images else str(src),
                "transform_matrix": _colmap_image_to_c2w(im),
            }
        )

    fov_x = 2 * math.atan(w / (2 * fl_x)) if fl_x else 0.0
    payload = {
        "camera_model": cam0["model"],
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "camera_angle_x": fov_x,
        "frames": frames,
        "ply_file_path": "sparse_pc.ply",
    }
    # Optional sparse point cloud copy
    pts = model_dir / "points3D.ply"
    if not pts.exists():
        # leave placeholder note
        (out_dir / "SPARSE_PC.txt").write_text(
            "Export points3D to PLY with colmap model_converter if needed.\n",
            encoding="utf-8",
        )
    else:
        shutil.copy2(pts, out_dir / "sparse_pc.ply")

    out = out_dir / "transforms.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def write_hierarchy_manifest(
    chunk_root: Path,
    alignments_path: Path,
    out_path: Path,
) -> Path:
    """
    Write an on-the-fly / hierarchical-3DGS style anchor manifest for tiles.

    Does not run the CUDA hierarchy merger; packages chunk PLYs + Sim3 as
    anchors for later LOD tooling or cloud hierarchical merge.
    """
    anchors = []
    aligns = []
    if alignments_path.exists():
        aligns = json.loads(alignments_path.read_text(encoding="utf-8"))
    by_id = {a["chunk_id"]: a for a in aligns}
    for child in sorted(chunk_root.glob("chunk_*")):
        if not child.is_dir():
            continue
        cid = child.name
        ply = child / "06_export" / "scene.ply"
        if not ply.exists():
            exports = list((child / "05_train" / "exports").rglob("*.ply")) if (child / "05_train" / "exports").exists() else []
            ply_s = str(exports[0]) if exports else None
        else:
            ply_s = str(ply)
        plan = {}
        pp = child / "chunk_plan.json"
        if pp.exists():
            plan = json.loads(pp.read_text(encoding="utf-8"))
        al = by_id.get(cid, {})
        sim = al.get("sim3", {})
        center = plan.get("gps_start_xyz") or sim.get("translation") or [0, 0, 0]
        anchors.append(
            {
                "id": cid,
                "center": center,
                "ply": ply_s,
                "start_sec": plan.get("start_sec"),
                "end_sec": plan.get("end_sec"),
                "sim3": sim,
                "align_method": al.get("method"),
                "align_rmse_m": al.get("rmse_m"),
            }
        )
    payload = {
        "type": "instasplat_hierarchy_v1",
        "inspired_by": [
            "graphdeco-inria/hierarchical-3d-gaussians",
            "graphdeco-inria/on-the-fly-nvs",
        ],
        "anchors": anchors,
        "note": (
            "Anchors are per-tile Gaussian PLYs in a shared world frame after Sim3. "
            "Run Kerbl hierarchy merger or splat-transform streamed LOD for viewer LOD."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_package(cfg: PipelineConfig, paths: JobPaths, model_dir: Path) -> PackageResult:
    log = get_logger("instasplat.package", paths.logs / "package.log")
    notes: list[str] = []
    ns_dir = None
    hier = None

    if cfg.package.nerfstudio:
        ns_dir = paths.root / "07_nerfstudio"
        try:
            if not cfg.dry_run:
                write_nerfstudio_transforms(
                    model_dir,
                    paths.cubemap_images,
                    ns_dir,
                    copy_images=cfg.package.copy_images,
                )
            notes.append("nerfstudio_transforms")
            log.info("Wrote Nerfstudio dataset → %s", ns_dir)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"nerfstudio_failed:{exc}")
            log.warning("Nerfstudio export failed: %s", exc)
            ns_dir = None

    if cfg.package.hierarchy_manifest:
        chunk_root = paths.root / "10_chunks"
        if chunk_root.exists():
            hier = write_hierarchy_manifest(
                chunk_root,
                chunk_root / "alignments.json",
                paths.root / "11_merged" / "hierarchy_manifest.json",
            )
            notes.append("hierarchy_manifest")
            log.info("Wrote hierarchy manifest → %s", hier)

    return PackageResult(ns_dir, hier, notes)
