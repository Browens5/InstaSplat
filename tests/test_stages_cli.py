"""Tests for individual stage selection helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from instasplat.config import PipelineConfig
from instasplat.pipeline import Pipeline
from instasplat.utils.stages import parse_stage_list, select_stages


def test_parse_stage_list() -> None:
    assert parse_stage_list("mask, sfm") == ["mask", "sfm"]
    with pytest.raises(ValueError):
        parse_stage_list("mask,not_a_stage")


def test_select_only_and_range() -> None:
    only = select_stages(mode="tiled", only="plan_chunks,process_chunks")
    assert only == ["plan_chunks", "process_chunks"]
    rng = select_stages(mode="single", from_stage="sfm", to_stage="train")
    assert rng == ["sfm", "scale", "refine", "train"]
    assert select_stages(mode="single") is None


def test_pipeline_honors_explicit_single_stage_on_tiled_job(tmp_path: Path) -> None:
    inp = tmp_path / "cap.mp4"
    inp.write_bytes(b"x" * 64)
    cfg = PipelineConfig(input_path=inp, output_dir=tmp_path / "out", project_name="j")
    cfg.enable_mac_long_360_defaults()
    cfg.stages = ["preflight"]
    cfg.dry_run = True
    cfg.preflight = True
    result = Pipeline(cfg).run()
    # dry_run preflight should not block
    assert result.success
    assert list(result.stage_timings.keys()) == ["preflight"]


def test_pipeline_run_only_mask_stage_name(tmp_path: Path) -> None:
    inp = tmp_path / "cap.mp4"
    inp.write_bytes(b"x" * 64)
    cfg = PipelineConfig(input_path=inp, output_dir=tmp_path / "out", project_name="j")
    cfg.dry_run = True
    cfg.stages = ["ingest"]
    result = Pipeline(cfg).run()
    assert result.success
    assert "ingest" in result.stage_timings
    assert "mask" not in result.stage_timings
