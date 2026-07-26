"""Tests for COLMAP CLI option compatibility."""

from __future__ import annotations

from pathlib import Path

from instasplat.stages.sfm import _is_usable_colmap_model
from instasplat.utils.colmap_cli import (
    ColmapCliCaps,
    db_image_count,
    detect_colmap_caps,
    equirect_requirement_message,
    list_readable_images,
    quality_feature_args,
)


MODERN_EXTRACT_HELP = """
COLMAP 4.1.1
--FeatureExtraction.max_image_size arg (=3200)
--FeatureExtraction.use_gpu arg (=1)
--SiftExtraction.max_num_features arg (=8192)
--ImageReader.camera_model arg (=SIMPLE_RADIAL)
                              {SIMPLE_PINHOLE, PINHOLE, EQUIRECTANGULAR, ...}
"""

LEGACY_EXTRACT_HELP = """
COLMAP 3.9.1
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
    monkeypatch.setattr(
        "instasplat.utils.colmap_cli._binary_mentions_equirect", lambda _c: False
    )
    caps = detect_colmap_caps("colmap-fake")
    assert caps.modern
    assert caps.supports_equirectangular
    assert caps.version == (4, 1, 1)
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
    monkeypatch.setattr(
        "instasplat.utils.colmap_cli._binary_mentions_equirect", lambda _c: False
    )
    # Swallow --version / -h probes
    import subprocess

    def fake_run(*_a, **_k):
        class P:
            stdout = ""
            stderr = ""

        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Re-apply help stub used inside detect — version comes from extract_help
    monkeypatch.setattr("instasplat.utils.colmap_cli._help_text", fake_help)
    caps = detect_colmap_caps("colmap-legacy")
    assert not caps.modern
    assert not caps.supports_equirectangular
    assert caps.version == (3, 9, 1)
    assert caps.max_image_size == "--SiftExtraction.max_image_size"
    assert caps.extract_use_gpu == "--SiftExtraction.use_gpu"
    assert caps.match_use_gpu == "--SiftMatching.use_gpu"
    msg = equirect_requirement_message(caps)
    assert "brew upgrade colmap" in msg
    assert "4.1" in msg
    detect_colmap_caps.cache_clear()


def test_version_alone_enables_equirect(monkeypatch) -> None:
    """COLMAP ≥ 4.1 supports EQUIRECTANGULAR even if help omits the enum."""
    detect_colmap_caps.cache_clear()

    def fake_help(colmap: str, command: str) -> str:
        if command == "feature_extractor":
            return (
                "COLMAP 4.1.0\n"
                "--FeatureExtraction.max_image_size arg (=3200)\n"
                "--FeatureExtraction.use_gpu arg (=1)\n"
                "--SiftExtraction.max_num_features arg (=8192)\n"
            )
        return MODERN_MATCH_HELP

    monkeypatch.setattr("instasplat.utils.colmap_cli._help_text", fake_help)
    monkeypatch.setattr(
        "instasplat.utils.colmap_cli._binary_mentions_equirect", lambda _c: False
    )
    caps = detect_colmap_caps("colmap-4.1")
    assert caps.supports_equirectangular
    assert caps.version == (4, 1, 0)
    detect_colmap_caps.cache_clear()


def test_quality_args_with_explicit_caps() -> None:
    caps = ColmapCliCaps(
        max_image_size="--FeatureExtraction.max_image_size",
        extract_use_gpu="--FeatureExtraction.use_gpu",
        max_num_features="--SiftExtraction.max_num_features",
        match_use_gpu="--FeatureMatching.use_gpu",
        modern=True,
        supports_equirectangular=True,
        version=(4, 1, 1),
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


def test_list_readable_images_skips_broken_symlink(tmp_path: Path) -> None:
    good = tmp_path / "a.jpg"
    good.write_bytes(b"jpg")
    bad = tmp_path / "b.jpg"
    bad.symlink_to(tmp_path / "missing.jpg")
    found = list_readable_images(tmp_path)
    assert found == [good]


def test_db_image_count_empty_and_missing(tmp_path: Path) -> None:
    assert db_image_count(tmp_path / "no.db") == 0
    import sqlite3

    db = tmp_path / "db.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO images VALUES (1, 'a.jpg')")
    conn.execute("INSERT INTO images VALUES (2, 'b.jpg')")
    conn.commit()
    conn.close()
    assert db_image_count(db) == 2
