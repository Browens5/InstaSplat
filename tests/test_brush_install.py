"""Tests for Brush install helpers."""

from __future__ import annotations

import tarfile
from pathlib import Path

from instasplat.utils.brush_install import (
    _extract_brush_binary,
    _host_release_asset_names,
    _parse_rust_version,
    _tail,
)


def test_parse_rust_version() -> None:
    assert _parse_rust_version("rustc 1.88.0 (abc)") == (1, 88, 0)
    assert _parse_rust_version("cargo 1.83.0") == (1, 83, 0)
    assert _parse_rust_version("nope") is None


def test_tail() -> None:
    text = "\n".join(f"line{i}" for i in range(100))
    assert "line99" in _tail(text, 5)
    assert "line0" not in _tail(text, 5)


def test_host_release_asset_names_darwin_arm(monkeypatch) -> None:
    monkeypatch.setattr("instasplat.utils.brush_install.platform.system", lambda: "Darwin")
    monkeypatch.setattr("instasplat.utils.brush_install.platform.machine", lambda: "arm64")
    assert _host_release_asset_names() == ["brush-app-aarch64-apple-darwin.tar.xz"]


def test_extract_brush_binary_from_tar(tmp_path: Path) -> None:
    archive = tmp_path / "brush.tar.xz"
    # Use gz for broader tarfile support in tests
    archive = tmp_path / "brush.tar.gz"
    root = tmp_path / "staging"
    nested = root / "brush-app-aarch64-apple-darwin"
    nested.mkdir(parents=True)
    binary = nested / "brush_app"
    binary.write_bytes(b"#!/bin/sh\necho brush\n")
    binary.chmod(0o755)
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(nested, arcname="brush-app-aarch64-apple-darwin")
    out = tmp_path / "out"
    found = _extract_brush_binary(archive, out)
    assert found is not None
    assert found.name == "brush_app"
    assert found.exists()
