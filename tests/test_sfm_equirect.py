"""EQUIRECTANGULAR-first SfM path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from instasplat.config import PipelineConfig
from instasplat.stages.fallback import write_telemetry_colmap_model
from instasplat.stages.sfm import _empty_frames_hint, _resolve_sfm_mode, run_sfm
from instasplat.utils.colmap_cli import ColmapCliCaps, detect_colmap_caps
from instasplat.utils.paths import JobPaths


def test_default_sfm_mode_is_equirectangular() -> None:
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=Path("runs"), project_name="t")
    assert cfg.sfm.mode == "equirectangular"
    assert cfg.sfm.camera_model == "EQUIRECTANGULAR"
    cfg.enable_mac_long_360_defaults()
    assert cfg.sfm.mode == "equirectangular"


def test_empty_frames_hint_mentions_equirect(tmp_path: Path) -> None:
    cfg = PipelineConfig(input_path=tmp_path / "a.mp4", output_dir=tmp_path, project_name="j")
    cfg.enable_mac_long_360_defaults()
    paths = JobPaths(tmp_path / "j")
    paths.ensure()
    (paths.chunks / "chunk_000").mkdir(parents=True)
    hint = _empty_frames_hint(paths, cfg)
    assert "process_chunks" in hint
    assert "equirectangular" in hint


def test_resolve_auto_prefers_equirect() -> None:
    caps = ColmapCliCaps(
        max_image_size="--FeatureExtraction.max_image_size",
        extract_use_gpu="--FeatureExtraction.use_gpu",
        max_num_features="--SiftExtraction.max_num_features",
        match_use_gpu="--FeatureMatching.use_gpu",
        modern=True,
        supports_equirectangular=True,
        version=(4, 1, 1),
    )
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=Path("runs"), project_name="t")
    cfg.sfm.mode = "auto"
    assert _resolve_sfm_mode(cfg, caps) == "equirectangular"


def test_resolve_auto_raises_without_equirect() -> None:
    caps = ColmapCliCaps(
        max_image_size="--SiftExtraction.max_image_size",
        extract_use_gpu="--SiftExtraction.use_gpu",
        max_num_features="--SiftExtraction.max_num_features",
        match_use_gpu="--SiftMatching.use_gpu",
        modern=False,
        supports_equirectangular=False,
        version=(3, 9, 1),
        colmap_bin="/opt/homebrew/bin/colmap",
    )
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=Path("runs"), project_name="t")
    cfg.sfm.mode = "auto"
    with pytest.raises(RuntimeError, match="EQUIRECTANGULAR"):
        _resolve_sfm_mode(cfg, caps)


def test_run_sfm_no_silent_cubemap_fallback(tmp_path: Path, monkeypatch) -> None:
    detect_colmap_caps.cache_clear()
    inp = tmp_path / "cap.mp4"
    inp.write_bytes(b"x" * 32)
    cfg = PipelineConfig(input_path=inp, output_dir=tmp_path / "out", project_name="job")
    cfg.sfm.mode = "equirectangular"
    paths = JobPaths(cfg.work_dir())
    paths.ensure()
    # Minimal equirect frame so we pass the empty-frames check
    import cv2

    img = np.zeros((64, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(paths.equirect_frames / "f000.jpg"), img)

    monkeypatch.setattr(
        "instasplat.stages.sfm.shutil.which", lambda _n: "/usr/bin/colmap-fake"
    )

    def fake_caps(_bin=None):
        return ColmapCliCaps(
            max_image_size="--FeatureExtraction.max_image_size",
            extract_use_gpu="--FeatureExtraction.use_gpu",
            max_num_features="--SiftExtraction.max_num_features",
            match_use_gpu="--FeatureMatching.use_gpu",
            modern=True,
            supports_equirectangular=False,
            version=(3, 8, 0),
            colmap_bin="/usr/bin/colmap-fake",
        )

    monkeypatch.setattr("instasplat.stages.sfm.detect_colmap_caps", fake_caps)
    with pytest.raises(RuntimeError, match="brew upgrade colmap"):
        run_sfm(cfg, paths)
    detect_colmap_caps.cache_clear()


def test_telemetry_fallback_writes_equirect_camera(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "f000.jpg").write_bytes(b"jpg")
    out = tmp_path / "model"
    n = write_telemetry_colmap_model(
        image_dir=img_dir,
        out_dir=out,
        frame_times={"f000": 0.0},
        gyro_csv=None,
        gps_csv=None,
        image_size=(3840, 1920),
        camera_model="EQUIRECTANGULAR",
    )
    assert n == 1
    cam = (out / "cameras.txt").read_text(encoding="utf-8")
    assert "EQUIRECTANGULAR" in cam
    assert "SIMPLE_PINHOLE" not in cam


def test_resolve_sfm_image_dir_prefers_equirect(tmp_path: Path) -> None:
    paths = JobPaths(tmp_path)
    paths.ensure()
    (paths.equirect_sfm_images / "a.jpg").write_bytes(b"x")
    assert paths.resolve_sfm_image_dir("equirectangular") == paths.equirect_sfm_images
    assert paths.resolve_sfm_image_dir(None) == paths.equirect_sfm_images
