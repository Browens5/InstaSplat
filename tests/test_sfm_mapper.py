"""SfM mapper selection (incremental vs global)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from instasplat.config import PipelineConfig
from instasplat.stages.sfm import run_colmap
from instasplat.utils.colmap_cli import ColmapCliCaps, detect_colmap_caps
from instasplat.utils.paths import JobPaths


def _caps(*, global_ok: bool = True) -> ColmapCliCaps:
    return ColmapCliCaps(
        max_image_size="--FeatureExtraction.max_image_size",
        extract_use_gpu="--FeatureExtraction.use_gpu",
        max_num_features="--SiftExtraction.max_num_features",
        match_use_gpu="--FeatureMatching.use_gpu",
        modern=True,
        supports_equirectangular=True,
        supports_global_mapper=global_ok,
        version=(4, 1, 1),
        colmap_bin="colmap",
        global_ba_disable_flags=(
            "--BundleAdjustment.refine_focal_length",
            "0",
            "--BundleAdjustment.refine_principal_point",
            "0",
            "--BundleAdjustment.refine_extra_params",
            "0",
        ),
    )


def test_default_mapper_is_incremental() -> None:
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=Path("runs"), project_name="t")
    assert cfg.sfm.mapper == "incremental"
    cfg.enable_mac_long_360_defaults()
    assert cfg.sfm.mapper == "incremental"


def test_run_colmap_global_uses_global_mapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    detect_colmap_caps.cache_clear()
    inp = tmp_path / "a.mp4"
    inp.write_bytes(b"x")
    cfg = PipelineConfig(input_path=inp, output_dir=tmp_path / "out", project_name="job")
    cfg.sfm.mapper = "global"
    cfg.dry_run = True
    paths = JobPaths(cfg.work_dir())
    paths.ensure()
    img_dir = paths.equirect_sfm_images
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "frame_000001.jpg").write_bytes(b"jpg")

    monkeypatch.setattr("instasplat.stages.sfm.shutil.which", lambda _n: "/usr/bin/colmap")
    monkeypatch.setattr("instasplat.stages.sfm.detect_colmap_caps", lambda _c=None: _caps())
    monkeypatch.setattr("instasplat.stages.sfm.db_image_count", lambda _p: 1)
    monkeypatch.setattr(
        "instasplat.stages.sfm.list_readable_images",
        lambda d: list(d.glob("*.jpg")),
    )

    calls: list[list[str]] = []

    def fake_run(cmd, **_kw):
        calls.append(list(cmd))
        return MagicMock(returncode=0)

    monkeypatch.setattr("instasplat.stages.sfm.run_cmd", fake_run)
    run_colmap(cfg, paths, img_dir, None, "EQUIRECTANGULAR")
    map_calls = [c for c in calls if len(c) > 1 and c[1] in {"mapper", "global_mapper"}]
    assert map_calls
    assert map_calls[0][1] == "global_mapper"
    assert "--BundleAdjustment.refine_focal_length" in map_calls[0]
    detect_colmap_caps.cache_clear()


def test_run_colmap_global_falls_back_without_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detect_colmap_caps.cache_clear()
    inp = tmp_path / "a.mp4"
    inp.write_bytes(b"x")
    cfg = PipelineConfig(input_path=inp, output_dir=tmp_path / "out", project_name="job")
    cfg.sfm.mapper = "global"
    cfg.dry_run = False
    paths = JobPaths(cfg.work_dir())
    paths.ensure()
    img_dir = paths.equirect_sfm_images
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "frame_000001.jpg").write_bytes(b"jpg")

    monkeypatch.setattr("instasplat.stages.sfm.shutil.which", lambda _n: "/usr/bin/colmap")
    monkeypatch.setattr(
        "instasplat.stages.sfm.detect_colmap_caps", lambda _c=None: _caps(global_ok=False)
    )
    monkeypatch.setattr("instasplat.stages.sfm.db_image_count", lambda _p: 1)
    monkeypatch.setattr(
        "instasplat.stages.sfm.list_readable_images",
        lambda d: list(d.glob("*.jpg")),
    )

    calls: list[list[str]] = []

    def fake_run(cmd, **_kw):
        calls.append(list(cmd))
        # Create a fake model so run_colmap can finish
        (paths.colmap_sparse / "0").mkdir(parents=True, exist_ok=True)
        (paths.colmap_sparse / "0" / "images.bin").write_bytes(b"x")
        return MagicMock(returncode=0)

    monkeypatch.setattr("instasplat.stages.sfm.run_cmd", fake_run)
    run_colmap(cfg, paths, img_dir, None, "EQUIRECTANGULAR")
    map_calls = [c for c in calls if len(c) > 1 and c[1] in {"mapper", "global_mapper"}]
    assert map_calls
    assert map_calls[0][1] == "mapper"
    detect_colmap_caps.cache_clear()
