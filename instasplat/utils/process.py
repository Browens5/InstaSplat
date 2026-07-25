"""Subprocess helpers and logging utilities."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path


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
) -> subprocess.CompletedProcess[str]:
    logger = get_logger("instasplat.cmd")
    rendered = " ".join(str(c) for c in cmd)
    logger.info("$ %s", rendered)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
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
