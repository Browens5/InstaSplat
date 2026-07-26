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
    win.close()
    del win
    _ = app
