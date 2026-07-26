"""Guards around empty SfM inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from instasplat.config import PipelineConfig
from instasplat.stages.sfm import _empty_frames_hint, run_sfm
from instasplat.utils.paths import JobPaths


def test_empty_frames_hint_mentions_process_chunks(tmp_path: Path) -> None:
    cfg = PipelineConfig(input_path=tmp_path / "a.mp4", output_dir=tmp_path, project_name="j")
    cfg.enable_mac_long_360_defaults()
    paths = JobPaths(tmp_path / "j")
    paths.ensure()
    (paths.chunks / "chunk_000").mkdir(parents=True)
    hint = _empty_frames_hint(paths, cfg)
    assert "process_chunks" in hint
    assert "equirectangular" in hint


def test_run_sfm_fails_fast_without_frames(tmp_path: Path) -> None:
    inp = tmp_path / "cap.mp4"
    inp.write_bytes(b"x" * 32)
    cfg = PipelineConfig(input_path=inp, output_dir=tmp_path / "out", project_name="job")
    cfg.sfm.mode = "equirectangular"
    cfg.mode = "tiled"
    cfg.chunk.enabled = True
    paths = JobPaths(cfg.work_dir())
    paths.ensure()
    (paths.chunks / "chunk_000").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="process_chunks"):
        run_sfm(cfg, paths)
