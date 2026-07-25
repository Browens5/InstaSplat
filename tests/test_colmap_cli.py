"""Tests for COLMAP CLI option compatibility."""

from __future__ import annotations

from pathlib import Path

from instasplat.stages.sfm import _is_usable_colmap_model
from instasplat.utils.colmap_cli import (
    ColmapCliCaps,
    detect_colmap_caps,
    quality_feature_args,
)


MODERN_EXTRACT_HELP = """
--FeatureExtraction.max_image_size arg (=3200)
--FeatureExtraction.use_gpu arg (=1)
--SiftExtraction.max_num_features arg (=8192)
"""

LEGACY_EXTRACT_HELP = """
--SiftExtraction.max_image_size arg (=3200)
--SiftExtraction.use_gpu arg (=1)
--SiftExtraction.max_num_features arg (=8192)
"""

MODERN_MATCH_HELP = """
--FeatureMatching.use_gpu arg (=1)
"""

LEGACY_MATCH_HELP = """
--SiftMatching.use_gpu arg (=1)
"""


def test_detect_modern_caps(monkeypatch) -> None:
    detect_colmap_caps.cache_clear()

    def fake_help(colmap: str, command: str) -> str:
        if command == "feature_extractor":
            return MODERN_EXTRACT_HELP
        return MODERN_MATCH_HELP

    monkeypatch.setattr("instasplat.utils.colmap_cli._help_text", fake_help)
    caps = detect_colmap_caps("colmap-fake")
    assert caps.modern
    assert caps.max_image_size == "--FeatureExtraction.max_image_size"
    assert caps.extract_use_gpu == "--FeatureExtraction.use_gpu"
    assert caps.max_num_features == "--SiftExtraction.max_num_features"
    assert caps.match_use_gpu == "--FeatureMatching.use_gpu"
    args = quality_feature_args("high", caps)
    assert "--FeatureExtraction.max_image_size" in args
    assert "2400" in args
    assert "--SiftExtraction.max_image_size" not in args
    detect_colmap_caps.cache_clear()


def test_detect_legacy_caps(monkeypatch) -> None:
    detect_colmap_caps.cache_clear()

    def fake_help(colmap: str, command: str) -> str:
        if command == "feature_extractor":
            return LEGACY_EXTRACT_HELP
        return LEGACY_MATCH_HELP

    monkeypatch.setattr("instasplat.utils.colmap_cli._help_text", fake_help)
    caps = detect_colmap_caps("colmap-legacy")
    assert not caps.modern
    assert caps.max_image_size == "--SiftExtraction.max_image_size"
    assert caps.extract_use_gpu == "--SiftExtraction.use_gpu"
    assert caps.match_use_gpu == "--SiftMatching.use_gpu"
    detect_colmap_caps.cache_clear()


def test_quality_args_with_explicit_caps() -> None:
    caps = ColmapCliCaps(
        max_image_size="--FeatureExtraction.max_image_size",
        extract_use_gpu="--FeatureExtraction.use_gpu",
        max_num_features="--SiftExtraction.max_num_features",
        match_use_gpu="--FeatureMatching.use_gpu",
        modern=True,
    )
    assert quality_feature_args("low", caps) == [
        "--FeatureExtraction.max_image_size",
        "1000",
        "--SiftExtraction.max_num_features",
        "2048",
    ]


def test_telemetry_prior_not_usable_model(tmp_path: Path) -> None:
    model = tmp_path / "0"
    model.mkdir()
    (model / "images.txt").write_text("x", encoding="utf-8")
    (model / "TELEMETRY_PRIOR.txt").write_text("prior", encoding="utf-8")
    assert not _is_usable_colmap_model(model)

    real = tmp_path / "1"
    real.mkdir()
    (real / "images.bin").write_bytes(b"bin")
    assert _is_usable_colmap_model(real)
