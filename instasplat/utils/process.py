"""Subprocess helpers and logging utilities."""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from instasplat.utils.control import PipelineStopped, get_controller


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    log_file: Path | None = None,
    dry_run: bool = False,
    check: bool = True,
    env: dict[str, str] | None = None,
    controllable: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Run a command.

    When ``controllable=True`` (trainers), the process is launched with Popen so
    the RunController can SIGSTOP/SIGCONT it for GUI pause/resume.
    """
    logger = get_logger("instasplat.cmd")
    rendered = " ".join(str(c) for c in cmd)
    logger.info("$ %s", rendered)
    ctrl = get_controller()
    ctrl.wait_if_paused()
    if dry_run:
        return subprocess.CompletedProcess(list(cmd), 0, stdout="", stderr="")

    if not controllable:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text(
                f"CMD: {rendered}\nEXIT: {proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n",
                encoding="utf-8",
            )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({proc.returncode}): {rendered}\n{proc.stderr or proc.stdout}"
            )
        return proc

    # Controllable long-running process (Brush / OpenSplat)
    popen = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    ctrl.register_pid(popen.pid)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    try:
        while True:
            if ctrl.stopped:
                popen.terminate()
                try:
                    popen.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    popen.kill()
                raise PipelineStopped("Trainer stopped by user")
            # Drain without blocking forever
            ret = popen.poll()
            if ret is not None:
                out, err = popen.communicate()
                if out:
                    stdout_chunks.append(out)
                if err:
                    stderr_chunks.append(err)
                break
            # Cooperative pause is handled via SIGSTOP on the PID; just sleep
            time.sleep(0.5)
    finally:
        ctrl.unregister_pid(popen.pid)

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    code = popen.returncode if popen.returncode is not None else -1
    result = subprocess.CompletedProcess(list(cmd), code, stdout=stdout, stderr=stderr)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            f"CMD: {rendered}\nEXIT: {result.returncode}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n",
            encoding="utf-8",
        )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {rendered}\n{stderr or stdout}"
        )
    return result
