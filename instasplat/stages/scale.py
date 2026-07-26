"""Apply metric scale to a COLMAP sparse model."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger
from instasplat.utils.scale import (
    apply_scale_to_model,
    camera_centers,
    gps_latlon_to_local_xyz,
    read_images_txt,
    read_points3d_txt,
    scale_factor_from_gps_path,
    scale_factor_from_known_distance,
)


@dataclass
class ScaleResult:
    model_dir: Path
    scale_factor: float
    mode: str


def _find_text_model(sparse_model: Path) -> Path | None:
    parent = sparse_model.parent
    txt = parent / f"{sparse_model.name}_txt"
    if (txt / "images.txt").exists():
        return txt
    if (sparse_model / "images.txt").exists():
        return sparse_model
    return None


def _load_gps(path: Path) -> np.ndarray:
    lats, lons, alts = [], [], []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Support several header styles
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lon") or row.get("longitude")
            alt = row.get("alt") or row.get("altitude") or "0"
            if lat is None or lon is None:
                continue
            lats.append(float(lat))
            lons.append(float(lon))
            alts.append(float(alt))
    if len(lats) < 2:
        raise ValueError(f"Need >=2 GPS samples in {path}")
    return gps_latlon_to_local_xyz(np.array(lats), np.array(lons), np.array(alts))


def run_scale(cfg: PipelineConfig, paths: JobPaths, sfm_model: Path) -> ScaleResult:
    paths.ensure()
    log = get_logger("instasplat.scale", paths.logs / "scale.log")
    out = paths.scaled_model
    out.parent.mkdir(parents=True, exist_ok=True)

    mode = cfg.scale.mode
    if mode == "none":
        log.info("Scale mode=none; copying sparse model unchanged (arbitrary COLMAP units)")
        if out.exists():
            shutil.rmtree(out)
        if not cfg.dry_run:
            shutil.copytree(sfm_model, out)
            (paths.scale / "scale_factor.txt").write_text("1.0\n", encoding="utf-8")
            (paths.scale / "SCALE_NOTE.txt").write_text(
                "Reconstruction is NOT metric. Set scale.mode to known_distance, gps, "
                "or stereo_baseline for true-to-scale output.\n",
                encoding="utf-8",
            )
        return ScaleResult(out, 1.0, mode)

    txt_model = _find_text_model(sfm_model)
    if txt_model is None and not cfg.dry_run:
        raise FileNotFoundError(
            f"Text COLMAP model not found near {sfm_model}. Re-run sfm stage."
        )

    scale = 1.0
    if mode == "known_distance":
        if cfg.scale.known_distance_m is None:
            raise ValueError("scale.known_distance_m is required for known_distance mode")
        # Allow specifying point ids via point_a_name / point_b_name as integer strings
        if not cfg.scale.point_a_name or not cfg.scale.point_b_name:
            raise ValueError("point_a_name and point_b_name (COLMAP point3D ids) required")
        id_a, id_b = int(cfg.scale.point_a_name), int(cfg.scale.point_b_name)
        points = read_points3d_txt(txt_model / "points3D.txt") if txt_model else {}
        scale = scale_factor_from_known_distance(points, id_a, id_b, cfg.scale.known_distance_m)
    elif mode == "gps":
        gps_path = cfg.scale.gps_csv or paths.gps_csv
        if not gps_path.exists():
            # Soft fallback — Studio MP4 jobs often lack trailer GPS until sidecar/INSV pair
            log.warning(
                "GPS CSV not found (%s); falling back to scale.mode=none",
                gps_path,
            )
            if out.exists():
                shutil.rmtree(out)
            if not cfg.dry_run:
                shutil.copytree(sfm_model, out)
                (paths.scale / "scale_factor.txt").write_text("1.0\n", encoding="utf-8")
                (paths.scale / "SCALE_NOTE.txt").write_text(
                    "Requested GPS scale but gps.csv was missing. Output is NOT metric.\n",
                    encoding="utf-8",
                )
            return ScaleResult(out, 1.0, "none_gps_missing")
        images = read_images_txt(txt_model / "images.txt") if txt_model else []
        centers = camera_centers(images)
        gps_xyz = _load_gps(gps_path)
        # Resample GPS path length vs camera path — simple global scale
        scale = scale_factor_from_gps_path(centers, gps_xyz)
    elif mode == "stereo_baseline":
        log.warning(
            "stereo_baseline scale is approximate and expects paired dual-lens poses; "
            "falling back to documenting baseline for manual use unless paired models exist"
        )
        # Without a dual-pose reconstruction, we cannot auto-scale; record intent
        scale = 1.0
        (paths.scale / "STEREO_BASELINE_TODO.txt").write_text(
            f"Configured baseline={cfg.scale.stereo_baseline_m} m. "
            "Auto scale requires a dual-lens SfM graph; use known_distance or gps for now.\n",
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Unknown scale mode: {mode}")

    log.info("Applying scale factor %.8f (mode=%s)", scale, mode)
    if out.exists() and not cfg.dry_run:
        shutil.rmtree(out)
    if cfg.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    else:
        src = txt_model if txt_model is not None else sfm_model
        apply_scale_to_model(src, out, scale)
        # Point at the SfM image set actually used (equirect by default)
        img_dir = paths.resolve_sfm_image_dir(cfg.sfm.mode)
        (paths.scale / "image_path.txt").write_text(str(img_dir), encoding="utf-8")
    return ScaleResult(out, scale, mode)
