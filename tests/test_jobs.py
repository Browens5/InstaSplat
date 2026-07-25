"""Tests for job load / continue helpers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from instasplat.config import PipelineConfig
from instasplat.utils.jobs import inspect_job, is_job_dir, load_job_config


def _write_cfg(job: Path, *, tiled: bool = True) -> PipelineConfig:
    inp = job.parent / "cap.mp4"
    inp.write_bytes(b"x" * 32)
    cfg = PipelineConfig(
        input_path=inp,
        output_dir=job.parent,
        project_name=job.name,
    )
    if tiled:
        cfg.enable_mac_long_360_defaults()
    job.mkdir(parents=True, exist_ok=True)
    cfg.save(job / "config.yaml")
    return cfg


def test_is_job_dir_and_load(tmp_path: Path) -> None:
    job = tmp_path / "walk"
    assert not is_job_dir(job)
    _write_cfg(job)
    assert is_job_dir(job)
    cfg = load_job_config(job)
    assert cfg.project_name == "walk"
    assert cfg.output_dir == tmp_path
    assert cfg.mode == "tiled"


def test_load_job_prefers_ingested_video(tmp_path: Path) -> None:
    job = tmp_path / "walk"
    cfg = _write_cfg(job)
    # Point config at a missing original path
    data = yaml.safe_load((job / "config.yaml").read_text(encoding="utf-8"))
    data["input_path"] = str(tmp_path / "missing_original.mp4")
    (job / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    ingest = job / "00_ingest"
    ingest.mkdir(parents=True)
    video = ingest / "equirect.mp4"
    video.write_bytes(b"video")
    loaded = load_job_config(job)
    assert loaded.input_path == video.resolve() or loaded.input_path == video
    assert loaded.input_path.exists()
    assert cfg.project_name == "walk"


def test_inspect_suggests_process_chunks_when_tiles_incomplete(tmp_path: Path) -> None:
    job = tmp_path / "walk"
    _write_cfg(job)
    (job / "00_ingest").mkdir(parents=True)
    (job / "00_ingest" / "equirect.mp4").write_bytes(b"v")
    chunks = job / "10_chunks"
    chunks.mkdir()
    (chunks / "manifest.json").write_text(
        json.dumps(
            {
                "duration_sec": 60,
                "source_fps_hint": 30,
                "strategy": "temporal",
                "notes": [],
                "chunks": [
                    {
                        "chunk_id": "chunk_000",
                        "index": 0,
                        "start_sec": 0,
                        "end_sec": 25,
                        "overlap_prev_sec": 0,
                        "frame_times": [0.0],
                    },
                    {
                        "chunk_id": "chunk_001",
                        "index": 1,
                        "start_sec": 20,
                        "end_sec": 45,
                        "overlap_prev_sec": 5,
                        "frame_times": [20.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (chunks / "chunk_status.json").write_text(
        json.dumps({"chunk_000": True, "chunk_001": False}),
        encoding="utf-8",
    )
    info = inspect_job(job)
    assert info.tiled
    assert info.tiles_done == 1
    assert info.tiles_total == 2
    assert info.suggested_stages[0] == "process_chunks"
    assert "package" in info.suggested_stages


def test_inspect_suggests_package_when_merged(tmp_path: Path) -> None:
    job = tmp_path / "walk"
    _write_cfg(job)
    (job / "00_ingest").mkdir(parents=True)
    (job / "00_ingest" / "equirect.mp4").write_bytes(b"v")
    chunks = job / "10_chunks"
    chunks.mkdir()
    (chunks / "manifest.json").write_text(
        json.dumps(
            {
                "duration_sec": 30,
                "source_fps_hint": 30,
                "strategy": "temporal",
                "notes": [],
                "chunks": [
                    {
                        "chunk_id": "chunk_000",
                        "index": 0,
                        "start_sec": 0,
                        "end_sec": 25,
                        "overlap_prev_sec": 0,
                        "frame_times": [0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (chunks / "chunk_status.json").write_text(
        json.dumps({"chunk_000": True}), encoding="utf-8"
    )
    (chunks / "alignments.json").write_text("[]", encoding="utf-8")
    merged = job / "11_merged"
    merged.mkdir()
    (merged / "scene_merged.ply").write_bytes(b"ply")
    info = inspect_job(job)
    assert info.suggested_stages == ["package"]


def test_inspect_single_skips_done_prefix(tmp_path: Path) -> None:
    job = tmp_path / "single_job"
    _write_cfg(job, tiled=False)
    (job / "00_ingest").mkdir(parents=True)
    (job / "00_ingest" / "equirect.mp4").write_bytes(b"v")
    frames = job / "01_frames" / "equirect"
    frames.mkdir(parents=True)
    (frames / "0001.jpg").write_bytes(b"jpg")
    info = inspect_job(job)
    assert not info.tiled
    assert "ingest" not in info.suggested_stages
    assert "extract" not in info.suggested_stages
    assert info.suggested_stages[0] == "mask"
