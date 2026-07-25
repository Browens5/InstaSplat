"""Tests for refine, fallback, nerfstudio package, robust alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.stages.fallback import write_telemetry_colmap_model
from instasplat.utils.align import icp_se3, umeyama_ransac
from instasplat.utils.nerfstudio import write_hierarchy_manifest, write_nerfstudio_transforms
from instasplat.utils.telemetry import GyroSeries


def test_umeyama_ransac_rejects_outliers() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    dst = (R @ src.T).T + t
    # Corrupt a few points heavily
    dst[0] += 50
    dst[1] -= 40
    s, Rr, tt, rmse, nin = umeyama_ransac(src, dst, thresh_m=1.0, iters=80)
    assert abs(s - 1.0) < 0.05
    assert nin >= 20
    assert rmse < 0.5


def test_icp_refines_small_offset() -> None:
    rng = np.random.default_rng(2)
    dst = rng.normal(size=(40, 3))
    # src is dst with small rigid offset
    angle = np.deg2rad(5)
    R = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    t = np.array([0.2, -0.1, 0.05])
    src = (R.T @ (dst - t).T).T  # inverse apply
    Rd, td, rmse = icp_se3(src, dst, max_iters=30)
    aligned = (Rd @ src.T).T + td
    assert np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))) < 0.05


def test_telemetry_fallback_model(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        (img_dir / f"frame_{i:06d}_front.jpg").write_bytes(b"x")
    # synthetic gyro/gps csv
    gyro = tmp_path / "gyro.csv"
    gps = tmp_path / "gps.csv"
    gyro.write_text(
        "timestamp_ms,gx,gy,gz\n"
        + "\n".join(f"{i*100},0,0,0.1" for i in range(20))
        + "\n",
        encoding="utf-8",
    )
    gps.write_text(
        "timestamp_ms,lat,lon,alt\n"
        + "\n".join(f"{i*1000},37.0,{-122.0 + i*0.0001},10" for i in range(10))
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "sparse"
    times = {f"frame_{i:06d}": float(i) for i in range(5)}
    n = write_telemetry_colmap_model(
        image_dir=img_dir,
        out_dir=out,
        frame_times=times,
        gyro_csv=gyro,
        gps_csv=gps,
        image_size=(512, 512),
    )
    assert n == 5
    assert (out / "images.txt").exists()
    assert (out / "cameras.txt").exists()
    assert (out / "points3D.txt").exists()


def test_nerfstudio_transforms(tmp_path: Path) -> None:
    model = tmp_path / "0"
    model.mkdir()
    imgs = tmp_path / "images"
    imgs.mkdir()
    (imgs / "a.jpg").write_bytes(b"x")
    (model / "cameras.txt").write_text(
        "#\n1 SIMPLE_PINHOLE 100 100 80 50 50\n",
        encoding="utf-8",
    )
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 a.jpg\n\n",
        encoding="utf-8",
    )
    (model / "points3D.txt").write_text("#\n", encoding="utf-8")
    out = tmp_path / "ns"
    path = write_nerfstudio_transforms(model, imgs, out, copy_images=True)
    data = path.read_text(encoding="utf-8")
    assert "transform_matrix" in data
    assert (out / "images" / "a.jpg").exists()


def test_hierarchy_manifest(tmp_path: Path) -> None:
    chunks = tmp_path / "10_chunks"
    c0 = chunks / "chunk_000"
    (c0 / "06_export").mkdir(parents=True)
    (c0 / "06_export" / "scene.ply").write_text("ply\n", encoding="utf-8")
    (c0 / "chunk_plan.json").write_text(
        '{"chunk_id":"chunk_000","start_sec":0,"end_sec":10,"gps_start_xyz":[1,2,3]}',
        encoding="utf-8",
    )
    aligns = chunks / "alignments.json"
    aligns.write_text(
        '[{"chunk_id":"chunk_000","sim3":{"scale":1,"rotation":[[1,0,0],[0,1,0],[0,0,1]],'
        '"translation":[1,2,3]},"rmse_m":0.1,"method":"test","n_anchors":3}]',
        encoding="utf-8",
    )
    out = tmp_path / "hierarchy_manifest.json"
    write_hierarchy_manifest(chunks, aligns, out)
    text = out.read_text(encoding="utf-8")
    assert "instasplat_hierarchy_v1" in text
    assert "chunk_000" in text


def test_refine_config_defaults() -> None:
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=Path("r"), project_name="t")
    cfg.enable_large_8k_defaults()
    assert cfg.refine.enabled
    assert cfg.package.hierarchy_manifest
    assert cfg.export.streamed_lod
    assert "package" in cfg.stages
