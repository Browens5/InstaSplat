"""Pause / resume / stop control for pipeline + long-running trainers."""

from __future__ import annotations

import contextvars
import os
import signal
import threading
import time
from dataclasses import dataclass, field


class PipelineStopped(RuntimeError):
    """Raised when the user requests a hard stop."""


@dataclass
class RunController:
    """
    Cooperative pause between stages + SIGSTOP/SIGCONT for trainer subprocesses.

    Thread-safe: GUI worker thread owns the pipeline; UI thread toggles pause.
    """

    _pause: threading.Event = field(default_factory=threading.Event)
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _procs: list[int] = field(default_factory=list)
    _status_msg: str = "running"

    def __post_init__(self) -> None:
        # Start unpaused
        self._pause.set()

    @classmethod
    def noop(cls) -> RunController:
        return cls()

    @property
    def paused(self) -> bool:
        return not self._pause.is_set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def status(self) -> str:
        if self.stopped:
            return "stopped"
        if self.paused:
            return "paused"
        return self._status_msg

    def pause(self) -> None:
        with self._lock:
            self._pause.clear()
            self._status_msg = "paused"
            for pid in list(self._procs):
                _signal_pid(pid, signal.SIGSTOP)

    def resume(self) -> None:
        with self._lock:
            for pid in list(self._procs):
                _signal_pid(pid, signal.SIGCONT)
            self._pause.set()
            self._status_msg = "running"

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            self._pause.set()  # unblock waiters
            self._status_msg = "stopped"
            for pid in list(self._procs):
                _signal_pid(pid, signal.SIGTERM)

    def register_pid(self, pid: int) -> None:
        with self._lock:
            if pid not in self._procs:
                self._procs.append(pid)
            if self.paused:
                _signal_pid(pid, signal.SIGSTOP)

    def unregister_pid(self, pid: int) -> None:
        with self._lock:
            if pid in self._procs:
                self._procs.remove(pid)

    def wait_if_paused(self, poll_sec: float = 0.25) -> None:
        """Block while paused; raise if stop requested."""
        if self._stop.is_set():
            raise PipelineStopped("Pipeline stopped by user")
        while not self._pause.is_set():
            if self._stop.is_set():
                raise PipelineStopped("Pipeline stopped by user")
            time.sleep(poll_sec)
        if self._stop.is_set():
            raise PipelineStopped("Pipeline stopped by user")

    def checkpoint(self, label: str = "") -> None:
        """Call at stage / chunk boundaries."""
        if label:
            self._status_msg = label
        self.wait_if_paused()


_CTRL: contextvars.ContextVar[RunController | None] = contextvars.ContextVar(
    "instasplat_run_controller", default=None
)


def set_controller(ctrl: RunController | None) -> contextvars.Token:
    return _CTRL.set(ctrl)


def reset_controller(token: contextvars.Token) -> None:
    _CTRL.reset(token)


def get_controller() -> RunController:
    return _CTRL.get() or RunController.noop()


def _signal_pid(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass
