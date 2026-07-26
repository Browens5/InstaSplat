"""Tests for chunking, telemetry, alignment, and tiled large-splat mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.pipeline import Pipeline
from instasplat.utils.align import Sim3, compose_sim3, umeyama_alignment
from instasplat.utils.chunking import plan_chunks
from instasplat.utils.metal import detect_metal, metal_report
from instasplat.utils.telemetry import (
    GyroSeries,
    adaptive_frame_times,
    integrate_orientation,
)


def test_adaptive_frame_times_densifies_on_turns() -> None:
    t = np.linspace(0, 10, 200)
    omega = np.zeros((200, 3))
    omega[100:130, 1] = 1.0  # turn burst
    gyro = GyroSeries(t_sec=t, omega=omega)
    times = adaptive_frame_times(10.0, base_fps=2.0, max_fps=10.0, gyro=gyro, boost_fps=8.0)
    # More samples overall than pure 2fps (21), denser near the turn
    assert len(times) > 20
    mid = times[(times > 4.5) & (times < 7.0)]
    early = times[times < 2.0]
    assert len(mid) >= len(early)


def test_plan_chunks_temporal_overlap() -> None:
    manifest = plan_chunks(
        duration_sec=100.0,
        chunk_duration_sec=25.0,
        overlap_sec=5.0,
        base_fps=4.0,
        max_fps=8.0,
        max_frames_per_chunk=200,
        target_path_length_m=None,
    )
    assert len(manifest.chunks) >= 4
    # Overlap: next start < prev end
    for a, b in zip(manifest.chunks, manifest.chunks[1:], strict=False):
        assert b.start_sec < a.end_sec
        assert b.start_sec >= a.end_sec - 5.0 - 1e-6


def test_plan_chunks_preselected_count() -> None:
    """num_chunks forces an exact tile count and covers the full timeline."""
    manifest = plan_chunks(
        duration_sec=100.0,
        chunk_duration_sec=25.0,  # ignored when num_chunks is set
        overlap_sec=5.0,
        base_fps=4.0,
        max_fps=8.0,
        max_frames_per_chunk=200,
        num_chunks=8,
        target_path_length_m=40.0,  # must not override fixed count
    )
    assert len(manifest.chunks) == 8
    assert manifest.strategy == "fixed_count_temporal"
    assert manifest.chunks[0].start_sec == 0.0
    assert abs(manifest.chunks[-1].end_sec - 100.0) < 1e-6
    for a, b in zip(manifest.chunks, manifest.chunks[1:], strict=False):
        assert b.start_sec < a.end_sec  # overlapping
    # Single-chunk covers the whole video
    one = plan_chunks(duration_sec=42.0, num_chunks=1, overlap_sec=5.0, target_path_length_m=None)
    assert len(one.chunks) == 1
    assert one.chunks[0].start_sec == 0.0
    assert abs(one.chunks[0].end_sec - 42.0) < 1e-6


def test_umeyama_recovers_sim3() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    s_true, angle = 2.5, np.deg2rad(30)
    R_true = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    t_true = np.array([10.0, -3.0, 1.5])
    dst = (s_true * (R_true @ src.T)).T + t_true
    s, R, t, rmse = umeyama_alignment(src, dst)
    assert abs(s - s_true) < 1e-6
    assert rmse < 1e-6
    assert np.allclose(t, t_true, atol=1e-6)


def test_compose_sim3() -> None:
    a = Sim3(2.0, np.eye(3).tolist(), [1, 0, 0])
    b = Sim3(3.0, np.eye(3).tolist(), [0, 1, 0])
    c = compose_sim3(a, b)
    assert abs(c.scale - 6.0) < 1e-9
    assert abs(c.translation[0] - 1.0) < 1e-9
    assert abs(c.translation[1] - 2.0) < 1e-9


def test_integrate_orientation_shape() -> None:
    t = np.linspace(0, 1, 50)
    omega = np.zeros((50, 3))
    omega[:, 2] = 0.5
    mats = integrate_orientation(GyroSeries(t, omega))
    assert mats.shape == (50, 3, 3)
    # Nearly orthonormal
    assert abs(np.linalg.det(mats[-1]) - 1.0) < 1e-3


def test_metal_report_keys() -> None:
    r = metal_report()
    assert "mps_available" in r
    assert "torch_device" in r
    status = detect_metal()
    assert status.torch_device in {"cpu", "mps"}


def test_large_8k_defaults_and_dry_run(tmp_path: Path) -> None:
    inp = tmp_path / "clip.mp4"
    inp.write_bytes(b"fake")
    cfg = PipelineConfig(input_path=inp, output_dir=tmp_path / "runs", project_name="big")
    cfg.enable_large_8k_defaults()
    cfg.dry_run = True
    assert cfg.mode == "tiled"
    assert cfg.chunk.enabled
    assert "plan_chunks" in cfg.stages
    assert "spz" in cfg.export.formats
    assert cfg.export.min_opacity == 0.05
    result = Pipeline(cfg).run()
    assert result.success
    manifest = tmp_path / "runs" / "big" / "10_chunks" / "manifest.json"
    assert manifest.exists()


def test_trainer_backend_roundtrip(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        input_path=tmp_path / "a.mp4",
        output_dir=tmp_path / "out",
        project_name="x",
    )
    assert cfg.train.backend == "metal_equirect"
    # Legacy brush/opensplat YAML coerces to metal_equirect
    path = tmp_path / "c.yaml"
    path.write_text(
        "input_path: a.mp4\noutput_dir: out\nproject_name: x\n"
        "train:\n  backend: brush\n  brush_bin: brush\n",
        encoding="utf-8",
    )
    loaded = PipelineConfig.load(path)
    assert loaded.train.backend == "metal_equirect"


def test_init_large_config_roundtrip(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        input_path=tmp_path / "a.mp4",
        output_dir=tmp_path / "out",
        project_name="x",
    )
    cfg.enable_large_8k_defaults()
    path = tmp_path / "c.yaml"
    cfg.save(path)
    loaded = PipelineConfig.load(path)
    assert loaded.mode == "tiled"
    assert loaded.chunk.max_fps == 15.0
    assert loaded.metal.prefer_metal is True
    assert "spz" in loaded.export.formats
