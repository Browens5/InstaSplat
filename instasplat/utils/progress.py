"""Stage timing, elapsed display, and ETA estimates."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# Heuristic defaults (seconds) used until real timings are observed
DEFAULT_STAGE_SEC: dict[str, float] = {
    "ingest": 45,
    "extract": 120,
    "mask": 180,
    "sfm": 900,
    "scale": 20,
    "refine": 120,
    "train": 2400,
    "export": 90,
    "package": 30,
    "plan_chunks": 20,
    "preflight": 10,
    "process_chunks": 7200,
    "align_chunks": 60,
    "merge_chunks": 180,
}


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


@dataclass
class ProgressEvent:
    stage: str
    overall_frac: float
    message: str
    stage_index: int
    stage_count: int
    stage_elapsed_sec: float
    stage_eta_sec: float | None
    overall_elapsed_sec: float
    overall_eta_sec: float | None
    status: str = "running"  # running | paused | finished | stopped
    # Heartbeats are frequent — UIs should update bars but skip log spam
    quiet: bool = False

    def terminal_line(self) -> str:
        eta = format_duration(self.stage_eta_sec)
        el = format_duration(self.stage_elapsed_sec)
        oeta = format_duration(self.overall_eta_sec)
        return (
            f"[{self.stage_index + 1}/{self.stage_count}] {self.stage}: {self.message} "
            f"| elapsed {el} | ETA {eta} | overall ETA {oeta}"
        )


def stage_fraction(elapsed_sec: float, eta_sec: float | None) -> float:
    """0–0.99 in-stage progress from elapsed + remaining estimate."""
    rem = max(0.0, float(eta_sec)) if eta_sec is not None else 0.0
    denom = max(0.0, float(elapsed_sec)) + rem
    if denom <= 0:
        return 0.0
    return min(0.99, float(elapsed_sec) / denom)


@dataclass
class StageProgressTracker:
    stages: list[str]
    prior_timings: dict[str, float] = field(default_factory=dict)
    chunk_count: int | None = None

    _t0: float = field(default_factory=time.time)
    _stage_t0: float | None = None
    _current: str | None = None
    _completed: dict[str, float] = field(default_factory=dict)
    _idx: int = 0
    # Fine-grained phase inside the current stage (e.g. COLMAP feature extraction)
    _activity: str | None = None

    def start_stage(self, name: str, index: int) -> ProgressEvent:
        self._current = name
        self._idx = index
        self._stage_t0 = time.time()
        self._activity = f"Starting {name}"
        return self.event(name, index, self._activity, frac_override=index / max(len(self.stages), 1))

    def finish_stage(self, name: str, index: int) -> ProgressEvent:
        elapsed = 0.0
        if self._stage_t0 is not None:
            elapsed = time.time() - self._stage_t0
        self._completed[name] = elapsed
        self._current = None
        self._stage_t0 = None
        self._activity = None
        frac = (index + 1) / max(len(self.stages), 1)
        now = time.time()
        overall_elapsed = now - self._t0
        # Remaining after this finished stage = sum of future stages only
        overall_eta = sum(self._stage_estimate(n) for n in self.stages[index + 1 :])
        return ProgressEvent(
            stage=name,
            overall_frac=frac,
            message=f"Finished {name} in {format_duration(elapsed)}",
            stage_index=index,
            stage_count=len(self.stages),
            stage_elapsed_sec=elapsed,
            stage_eta_sec=0.0,
            overall_elapsed_sec=overall_elapsed,
            overall_eta_sec=overall_eta,
            status="finished",
        )

    def set_activity(self, message: str) -> None:
        """Update the in-stage phase string (picked up by heartbeats)."""
        text = (message or "").strip()
        if text:
            self._activity = text

    def announce_activity(self, message: str) -> ProgressEvent:
        """Set activity and return a non-quiet event (log + GUI status)."""
        self.set_activity(message)
        name = self._current or (self.stages[self._idx] if self.stages else "pipeline")
        return self.event(name, self._idx, self._activity or message, quiet=False)

    def heartbeat(self, message: str | None = None, status: str = "running") -> ProgressEvent:
        name = self._current or (self.stages[self._idx] if self.stages else "pipeline")
        msg = message or self._activity or f"Running {name}"
        return self.event(
            name,
            self._idx,
            msg,
            status=status,
            quiet=True,
        )

    def event(
        self,
        stage: str,
        index: int,
        message: str,
        *,
        frac_override: float | None = None,
        status: str = "running",
        quiet: bool = False,
    ) -> ProgressEvent:
        now = time.time()
        overall_elapsed = now - self._t0
        stage_elapsed = (now - self._stage_t0) if self._stage_t0 else 0.0
        stage_eta = self._estimate_stage_remaining(stage, stage_elapsed)
        overall_eta = self._estimate_overall_remaining(index, stage, stage_elapsed)
        frac = frac_override
        if frac is None:
            base = index / max(len(self.stages), 1)
            # blend in-stage progress from elapsed/estimate
            est = self._stage_estimate(stage)
            inner = min(0.95, stage_elapsed / est) if est > 0 else 0.0
            frac = min(0.999, base + inner / max(len(self.stages), 1))
        return ProgressEvent(
            stage=stage,
            overall_frac=frac,
            message=message,
            stage_index=index,
            stage_count=len(self.stages),
            stage_elapsed_sec=stage_elapsed,
            stage_eta_sec=stage_eta,
            overall_elapsed_sec=overall_elapsed,
            overall_eta_sec=overall_eta,
            status=status,
            quiet=quiet,
        )

    def estimate_seconds(self, stage: str) -> float:
        return self._stage_estimate(stage)

    def _stage_estimate(self, stage: str) -> float:
        if stage in self.prior_timings:
            return max(1.0, self.prior_timings[stage])
        if stage in self._completed:
            return max(1.0, self._completed[stage])
        base = DEFAULT_STAGE_SEC.get(stage, 300.0)
        if stage == "process_chunks" and self.chunk_count:
            # Rough: each chunk ≈ train+sfm default
            per = DEFAULT_STAGE_SEC["sfm"] + DEFAULT_STAGE_SEC["train"] + DEFAULT_STAGE_SEC["mask"]
            return max(base, self.chunk_count * per * 0.5)
        return base

    def _estimate_stage_remaining(self, stage: str, elapsed: float) -> float | None:
        est = self._stage_estimate(stage)
        rem = est - elapsed
        if rem < 0:
            # Over estimate — grow gently
            return max(5.0, elapsed * 0.15)
        return rem

    def _estimate_overall_remaining(self, index: int, stage: str, stage_elapsed: float) -> float | None:
        rem = self._estimate_stage_remaining(stage, stage_elapsed) or 0.0
        for name in self.stages[index + 1 :]:
            rem += self._stage_estimate(name)
        return rem

    def to_timings_dict(self) -> dict[str, Any]:
        return dict(self._completed)


class HeartbeatPublisher:
    """Emit quiet progress heartbeats on a background thread while a stage runs."""

    def __init__(
        self,
        tracker: StageProgressTracker,
        on_progress: Callable[[ProgressEvent], None],
        *,
        interval_sec: float = 1.0,
    ) -> None:
        self._tracker = tracker
        self._on_progress = on_progress
        self._interval = max(0.25, float(interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="instasplat-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None and t.is_alive():
            t.join(timeout=self._interval + 1.0)

    def _emit(self) -> None:
        if self._tracker._current is None:
            return
        try:
            self._on_progress(self._tracker.heartbeat())
        except Exception:  # noqa: BLE001 — never kill a stage over UI progress
            pass

    def _loop(self) -> None:
        # Emit once immediately so bars move without waiting a full interval
        self._emit()
        while not self._stop.wait(self._interval):
            self._emit()


def live_progress_from_event(
    ev: ProgressEvent,
    *,
    stage_wall_t0: float,
    run_wall_t0: float,
    now: float | None = None,
) -> ProgressEvent:
    """
    Recompute elapsed/ETA/fraction for a running stage using wall-clock time.

    Used by the GUI timer so bars keep moving even if a heartbeat is delayed.
    """
    if ev.status != "running":
        return ev
    now = time.time() if now is None else now
    stage_elapsed = max(0.0, now - stage_wall_t0)
    overall_elapsed = max(0.0, now - run_wall_t0)
    est = DEFAULT_STAGE_SEC.get(ev.stage, 300.0)
    # Prefer scaling from the last reported remaining+elapsed when available
    prior_total = (ev.stage_elapsed_sec or 0.0) + (
        ev.stage_eta_sec if ev.stage_eta_sec is not None else est
    )
    if prior_total > 1.0:
        est = max(est, prior_total)
    rem = est - stage_elapsed
    stage_eta = max(5.0, rem) if rem > 0 else max(5.0, stage_elapsed * 0.15)
    # Remaining later stages ≈ previous overall ETA minus previous stage ETA
    later = 0.0
    if ev.overall_eta_sec is not None and ev.stage_eta_sec is not None:
        later = max(0.0, ev.overall_eta_sec - ev.stage_eta_sec)
    overall_eta = stage_eta + later
    base = ev.stage_index / max(ev.stage_count, 1)
    inner = min(0.95, stage_elapsed / est) if est > 0 else 0.0
    frac = min(0.999, base + inner / max(ev.stage_count, 1))
    return ProgressEvent(
        stage=ev.stage,
        overall_frac=frac,
        message=ev.message,
        stage_index=ev.stage_index,
        stage_count=ev.stage_count,
        stage_elapsed_sec=stage_elapsed,
        stage_eta_sec=stage_eta,
        overall_elapsed_sec=overall_elapsed,
        overall_eta_sec=overall_eta,
        status=ev.status,
        quiet=True,
    )
