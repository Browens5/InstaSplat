"""Tests for environment setup helpers."""

from __future__ import annotations

from pathlib import Path

from instasplat.utils.setup_env import (
    format_setup_report,
    repo_root,
    run_setup,
)


def test_run_setup_verify_only_structure() -> None:
    report = run_setup(install_system=False, dry_run=True)
    names = {s.name for s in report.steps}
    assert "python" in names
    assert "pytorch" in names
    assert "instasplat" in names
    assert "ffmpeg" in names
    assert "colmap" in names
    assert "splat-transform" in names
    assert isinstance(report.ready, bool)
    assert report.next_commands


def test_format_setup_report_contains_status() -> None:
    text = format_setup_report(run_setup(install_system=False))
    assert "InstaSplat environment" in text
    assert "python" in text
    assert "pytorch" in text


def test_run_setup_dry_run_does_not_require_brew() -> None:
    """Dry-run on non-Mac still returns a structured report."""
    report = run_setup(install_system=True, dry_run=True)
    assert report.steps
    assert any(s.name == "python" for s in report.steps)


def test_repo_root_points_at_workspace() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "docs" / "MAC_360_PIPELINE.md").is_file()
    assert (root / "docs" / "METAL_SPLAT_WORKFLOW.md").is_file()


def test_setup_macos_script_exists() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "setup_macos.sh"
    assert script.is_file()
    body = script.read_text(encoding="utf-8")
    assert "pip install" in body
    assert "instasplat setup" in body
    assert "MAC_360_PIPELINE" in body
    assert "METAL_SPLAT_WORKFLOW" in body
