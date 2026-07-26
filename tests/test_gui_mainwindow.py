"""Smoke test: MainWindow constructs control buttons (needs PySide6)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_mainwindow_has_run_pause_stop_buttons() -> None:
    from PySide6.QtWidgets import QApplication

    from instasplat.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    assert win.run_btn is not None
    assert win.pause_btn is not None
    assert win.stop_btn is not None
    assert not win.pause_btn.isEnabled()
    assert not win.stop_btn.isEnabled()
    assert win.run_btn.isEnabled()
    # Training controls are in thousands (1 k = 1000)
    assert win.steps.value() >= 1
    assert win.steps.suffix().strip() == "k"
    assert 1 <= win.max_gaussians.value() <= 30_000
    assert win.max_gaussians.suffix().strip() == "k"
    assert 0 <= win.sh_degree.value() <= 3
    assert 50 <= win.export_every.value() <= 1000
    win.close()
    del win
    _ = app


def test_build_config_includes_train_options(tmp_path) -> None:
    from PySide6.QtWidgets import QApplication

    from instasplat.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    # Minimal single-mode job so _build_config does not need tiled stages only
    win.large_8k_cb.setChecked(False)
    win._select_mode_stages()
    video = tmp_path / "eq.mp4"
    video.write_bytes(b"fake")
    win.input_edit.setText(str(video))
    win.output_edit.setText(str(tmp_path / "runs"))
    win.name_edit.setText("gui_train")
    win.steps.setValue(8)  # 8 k → 8000 steps
    win.max_gaussians.setValue(25)  # 25 k → 25_000
    win.sh_degree.setValue(2)
    win.export_every.setValue(200)
    cfg = win._build_config()
    assert cfg.train.total_steps == 8_000
    assert cfg.train.max_gaussians == 25_000
    assert cfg.train.sh_degree == 2
    assert cfg.train.export_every == 200
    assert cfg.train.viewer_every == 100
    assert cfg.train.resolution_schedule is False
    win.resolution_schedule_cb.setChecked(True)
    cfg2 = win._build_config()
    assert cfg2.train.resolution_schedule is True
    win._apply_config_to_form(cfg2)
    assert win.resolution_schedule_cb.isChecked()
    win.close()
    del win
    _ = app


def test_build_config_preselects_chunk_count(tmp_path) -> None:
    from PySide6.QtWidgets import QApplication

    from instasplat.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.large_8k_cb.setChecked(True)
    win._select_mode_stages()
    video = tmp_path / "eq.mp4"
    video.write_bytes(b"fake")
    win.input_edit.setText(str(video))
    win.output_edit.setText(str(tmp_path / "runs"))
    win.name_edit.setText("gui_chunks")
    win.num_chunks.setValue(6)
    cfg = win._build_config()
    assert cfg.mode == "tiled"
    assert cfg.chunk.enabled
    assert cfg.chunk.num_chunks == 6
    win._apply_config_to_form(cfg)
    assert win.num_chunks.value() == 6
    win.close()
    del win
    _ = app


def test_gui_max_steps_allows_1m_in_k() -> None:
    from pathlib import Path
    import tempfile

    from PySide6.QtWidgets import QApplication

    from instasplat.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    assert win.steps.maximum() == 1_000  # 1000 k = 1,000,000 steps
    win.steps.setValue(1_000)
    win.large_8k_cb.setChecked(False)
    win._select_mode_stages()
    td = Path(tempfile.mkdtemp())
    (td / "eq.mp4").write_bytes(b"x")
    win.input_edit.setText(str(td / "eq.mp4"))
    win.output_edit.setText(str(td / "runs"))
    win.name_edit.setText("long")
    win.max_gaussians.setValue(40)
    cfg = win._build_config()
    assert cfg.train.total_steps == 1_000_000
    win.close()
    del win
    _ = app


def test_gui_max_gaussians_allows_30m_in_k() -> None:
    from PySide6.QtWidgets import QApplication

    from instasplat.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    assert win.max_gaussians.maximum() == 30_000
    win.max_gaussians.setValue(30_000)
    win.large_8k_cb.setChecked(False)
    win._select_mode_stages()
    # Need paths for _build_config — use empty and catch, or set paths
    from pathlib import Path
    import tempfile

    td = Path(tempfile.mkdtemp())
    (td / "eq.mp4").write_bytes(b"x")
    win.input_edit.setText(str(td / "eq.mp4"))
    win.output_edit.setText(str(td / "runs"))
    win.name_edit.setText("big")
    win.steps.setValue(15)
    cfg = win._build_config()
    assert cfg.train.max_gaussians == 30_000_000
    assert cfg.train.total_steps == 15_000
    win.close()
    del win
    _ = app
