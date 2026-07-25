"""Tests for pause control, progress ETA, and Brush install helpers."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from instasplat.utils.brush_install import find_built_brush
from instasplat.utils.control import PipelineStopped, RunController
from instasplat.utils.progress import StageProgressTracker, format_duration


def test_format_duration() -> None:
    assert format_duration(45) == "45s"
    assert "m" in format_duration(125)
    assert format_duration(None) == "—"


def test_stage_tracker_eta() -> None:
    tr = StageProgressTracker(stages=["ingest", "train", "export"])
    ev = tr.start_stage("ingest", 0)
    assert ev.stage_eta_sec is not None and ev.stage_eta_sec > 0
    assert ev.overall_eta_sec is not None and ev.overall_eta_sec > ev.stage_eta_sec
    time.sleep(0.05)
    done = tr.finish_stage("ingest", 0)
    assert done.stage_elapsed_sec >= 0.05
    assert "Finished ingest" in done.message
    line = done.terminal_line()
    assert "elapsed" in line


def test_controller_pause_resume() -> None:
    ctrl = RunController()
    assert not ctrl.paused
    ctrl.pause()
    assert ctrl.paused
    released = threading.Event()

    def waiter() -> None:
        ctrl.wait_if_paused()
        released.set()

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.2)
    assert not released.is_set()
    ctrl.resume()
    t.join(timeout=2)
    assert released.is_set()


def test_controller_stop_raises() -> None:
    ctrl = RunController()
    ctrl.stop()
    try:
        ctrl.wait_if_paused()
        raised = False
    except PipelineStopped:
        raised = True
    assert raised


def test_find_built_brush_missing(tmp_path: Path) -> None:
    assert find_built_brush(tmp_path) is None
    release = tmp_path / "target" / "release"
    release.mkdir(parents=True)
    fake = release / "brush"
    fake.write_text("x", encoding="utf-8")
    fake.chmod(0o755)
    assert find_built_brush(tmp_path) == fake
