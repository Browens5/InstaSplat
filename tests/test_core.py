"""Unit tests that do not require COLMAP/Brush binaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from instasplat.config import PipelineConfig
from instasplat.pipeline import Pipeline
from instasplat.stages.sfm import equirect_to_perspective
from instasplat.utils.paths import JobPaths
from instasplat.utils.scale import (
    apply_scale_to_model,
    gps_latlon_to_local_xyz,
    qvec_to_rotmat,
    scale_factor_from_known_distance,
)


def test_config_roundtrip(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        input_path=tmp_path / "a.insv",
        output_dir=tmp_path / "runs",
        project_name="t",
    )
    cfg.export.translate = (1.0, 2.0, 3.0)
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    loaded = PipelineConfig.load(path)
    assert loaded.project_name == "t"
    assert loaded.export.translate == (1.0, 2.0, 3.0)
    assert loaded.mask.classes == [0]


def test_job_paths(tmp_path: Path) -> None:
    paths = JobPaths(tmp_path / "job")
    paths.ensure()
    assert paths.equirect_frames.is_dir()
    assert paths.export.is_dir()


def test_equirect_to_perspective_shape() -> None:
    eq = np.zeros((100, 200, 3), dtype=np.uint8)
    eq[:, :, 2] = 255
    face = equirect_to_perspective(eq, 0, 0, 90, 64)
    assert face.shape == (64, 64, 3)


def test_scale_known_distance(tmp_path: Path) -> None:
    points = {1: np.array([0.0, 0.0, 0.0]), 2: np.array([2.0, 0.0, 0.0])}
    s = scale_factor_from_known_distance(points, 1, 2, known_m=4.0)
    assert abs(s - 2.0) < 1e-9

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "cameras.txt").write_text("# cameras\n", encoding="utf-8")
    (src / "images.txt").write_text(
        "1 1 0 0 0 1 0 0 1 frame.jpg\n\n",
        encoding="utf-8",
    )
    (src / "points3D.txt").write_text(
        "1 1 2 3 255 0 0 0\n",
        encoding="utf-8",
    )
    apply_scale_to_model(src, dst, 2.0)
    pts = (dst / "points3D.txt").read_text(encoding="utf-8")
    assert "2.0 4.0 6.0" in pts or "2.0 4.0 6.0" in pts.replace("2.000", "2.0")


def test_gps_local_xyz() -> None:
    lat = np.array([37.0, 37.001])
    lon = np.array([-122.0, -122.0])
    xyz = gps_latlon_to_local_xyz(lat, lon)
    assert xyz.shape == (2, 3)
    assert xyz[0].tolist() == [0.0, 0.0, 0.0]
    assert abs(xyz[1, 1]) > 50  # ~111m per degree latitude


def test_qvec_identity() -> None:
    R = qvec_to_rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(R, np.eye(3))


def test_dry_run_pipeline(tmp_path: Path) -> None:
    # Create a tiny fake mp4-like file; ffmpeg may fail — we only dry-run.
    inp = tmp_path / "clip.mp4"
    inp.write_bytes(b"not-a-real-video")
    cfg = PipelineConfig(
        input_path=inp,
        output_dir=tmp_path / "runs",
        project_name="dry",
        dry_run=True,
        stages=["ingest"],
    )
    result = Pipeline(cfg).run()
    assert result.success
    assert (tmp_path / "runs" / "dry" / "result.json").exists()


def test_example_yaml_parses() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "instasplat.example.yaml"
    data = yaml.safe_load(example.read_text(encoding="utf-8"))
    cfg = PipelineConfig.from_dict(data)
    assert cfg.export.formats[0] == "ply"
