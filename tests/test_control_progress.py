"""Tests for pause control and progress ETA helpers."""

from __future__ import annotations

import threading
import time

from instasplat.utils.control import PipelineStopped, RunController
from instasplat.utils.progress import (
    HeartbeatPublisher,
    StageProgressTracker,
    format_duration,
    live_progress_from_event,
    stage_fraction,
)


def test_format_duration() -> None:
    assert format_duration(45) == "45s"
    assert "m" in format_duration(125)
    assert format_duration(None) == "—"


def test_stage_fraction() -> None:
    assert stage_fraction(0, 100) == 0.0
    assert 0.4 < stage_fraction(50, 50) < 0.6
    assert stage_fraction(100, 0) == 0.99


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


def test_heartbeat_is_quiet_and_advances_elapsed() -> None:
    tr = StageProgressTracker(stages=["mask"])
    tr.start_stage("mask", 0)
    time.sleep(0.05)
    hb = tr.heartbeat()
    assert hb.quiet
    assert hb.status == "running"
    assert hb.stage_elapsed_sec >= 0.05


def test_activity_announcement_and_heartbeat() -> None:
    tr = StageProgressTracker(stages=["sfm"])
    tr.start_stage("sfm", 0)
    ev = tr.announce_activity("COLMAP: feature extraction (120 images, high)")
    assert not ev.quiet
    assert "feature extraction" in ev.message
    hb = tr.heartbeat()
    assert hb.quiet
    assert hb.message == ev.message
    tr.set_activity("COLMAP: sparse reconstruction (incremental mapper)")
    hb2 = tr.heartbeat()
    assert "sparse reconstruction" in hb2.message


def test_heartbeat_publisher_emits() -> None:
    tr = StageProgressTracker(stages=["mask"])
    tr.start_stage("mask", 0)
    events: list = []
    done = threading.Event()

    def on_ev(ev) -> None:
        events.append(ev)
        if len(events) >= 2:
            done.set()

    pub = HeartbeatPublisher(tr, on_ev, interval_sec=0.05)
    pub.start()
    assert done.wait(timeout=2.0), f"only got {len(events)} heartbeats"
    pub.stop()
    assert len(events) >= 2
    assert all(e.quiet for e in events)


def test_live_progress_from_event_moves_fraction() -> None:
    tr = StageProgressTracker(stages=["sfm", "train"])
    ev = tr.start_stage("sfm", 0)
    t0 = time.time() - 120.0
    live = live_progress_from_event(ev, stage_wall_t0=t0, run_wall_t0=t0 - 10.0)
    assert live.stage_elapsed_sec >= 119.0
    assert live.overall_frac > ev.overall_frac
    assert stage_fraction(live.stage_elapsed_sec, live.stage_eta_sec) > 0.05


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

