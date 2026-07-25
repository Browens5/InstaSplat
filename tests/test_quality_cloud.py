"""Tests for quality report, cloud manifests, and CPU LOD packaging."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.utils.chunking import plan_chunks
from instasplat.utils.cloud import recommended_cloud_backends, write_cloud_job_manifest
from instasplat.utils.hierarchy import build_cpu_lod_levels, _write_xyz_opacity_ply
from instasplat.utils.paths import JobPaths
from instasplat.utils.quality import build_quality_report, validate_capture


def test_validate_capture_flags_low_overlap() -> None:
    issues = validate_capture(
        duration_sec=60.0,
        overlap_sec=1.0,
        chunk_duration_sec=25.0,
        min_overlap_ratio=0.15,
        base_fps=6.0,
    )
    codes = {i.code for i in issues}
    assert "low_overlap" in codes


def test_plan_chunks_notes_low_overlap() -> None:
    manifest = plan_chunks(
        duration_sec=50.0,
        chunk_duration_sec=25.0,
        overlap_sec=2.0,
        base_fps=4.0,
        max_fps=8.0,
        target_path_length_m=None,
    )
    assert any("overlap_ratio" in n for n in manifest.notes)


def test_quality_report_from_job(tmp_path: Path) -> None:
    root = tmp_path / "job"
    paths = JobPaths(root)
    paths.ensure()
    # Minimal chunk manifest
    (paths.chunks / "manifest.json").write_text(
        """
        {
          "duration_sec": 80.0,
          "source_fps_hint": 30.0,
          "strategy": "temporal",
          "notes": [],
          "chunks": [
            {
              "chunk_id": "chunk_000",
              "index": 0,
              "start_sec": 0,
              "end_sec": 25,
              "overlap_prev_sec": 0,
              "frame_times": [0,1,2,3,4,5,6,7,8,9,10,11,12]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    (paths.export / "scene.ply").write_text("ply\n", encoding="utf-8")
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=tmp_path, project_name="job")
    report = build_quality_report(cfg, paths)
    assert report.grade in {"excellent", "good", "fair", "poor"}
    assert report.metrics["n_chunks"] == 1
    assert "scene.ply" in report.metrics["export_bytes"]


def test_cloud_job_manifest(tmp_path: Path) -> None:
    root = tmp_path / "job"
    paths = JobPaths(root)
    paths.ensure()
    (paths.root / "07_nerfstudio").mkdir()
    (paths.root / "07_nerfstudio" / "transforms.json").write_text("{}", encoding="utf-8")
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=tmp_path, project_name="job")
    cfg.enable_large_8k_defaults()
    cfg.sfm.mode = "equirectangular"
    out = write_cloud_job_manifest(cfg, paths)
    text = out.read_text(encoding="utf-8")
    assert "instasplat_cloud_job_v1" in text
    assert "gsplat_3dgut" in text
    backends = recommended_cloud_backends(cfg)
    assert backends[0] == "gsplat_3dgut"


def test_cpu_lod_from_ascii_ply(tmp_path: Path) -> None:
    chunks = tmp_path / "10_chunks"
    c0 = chunks / "chunk_000" / "06_export"
    c0.mkdir(parents=True)
    xyz = np.random.default_rng(0).normal(size=(200, 3)).astype(np.float32)
    op = np.linspace(0.01, 0.9, 200).astype(np.float32)
    ply = c0 / "scene.ply"
    _write_xyz_opacity_ply(ply, xyz, op)
    hier = tmp_path / "hierarchy_manifest.json"
    hier.write_text(
        '{"type":"instasplat_hierarchy_v1","anchors":[{"id":"chunk_000","ply":"%s"}]}'
        % str(ply),
        encoding="utf-8",
    )
    out = tmp_path / "lod"
    result = build_cpu_lod_levels(hier, out)
    assert result.lod_manifest.exists()
    assert len(result.levels) >= 2
    assert (out / "lod_level_1.ply").exists()


def test_package_defaults_include_cloud_quality() -> None:
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=Path("r"), project_name="t")
    cfg.enable_large_8k_defaults()
    assert cfg.package.cloud_manifest
    assert cfg.package.quality_report
    assert cfg.package.cpu_lod
