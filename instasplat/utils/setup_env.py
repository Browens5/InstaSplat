"""Streamlined environment setup / readiness checks for InstaSplat."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SetupStep:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SetupReport:
    steps: list[SetupStep] = field(default_factory=list)
    ready: bool = False
    next_commands: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append(SetupStep(name, ok, detail))


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(cmd: list[str], timeout: float = 120.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 1, str(exc)


def ensure_brew_tools(*, dry_run: bool = False) -> list[SetupStep]:
    """Install ffmpeg / exiftool / colmap via Homebrew when missing."""
    steps: list[SetupStep] = []
    brew = _which("brew")
    if not brew:
        steps.append(
            SetupStep(
                "homebrew",
                False,
                "Not found — install from https://brew.sh then re-run setup",
            )
        )
        return steps
    steps.append(SetupStep("homebrew", True, brew))

    pkgs = ["ffmpeg", "exiftool", "colmap", "git"]
    missing = [p for p in pkgs if not _which(p)]
    if not missing:
        steps.append(SetupStep("brew-tools", True, "ffmpeg, exiftool, colmap, git on PATH"))
        return steps
    if dry_run:
        steps.append(SetupStep("brew-tools", False, f"Would install: {', '.join(missing)}"))
        return steps
    code, out = _run([brew, "install", *missing], timeout=600.0)
    ok = code == 0 or all(_which(p) for p in pkgs)
    steps.append(
        SetupStep(
            "brew-tools",
            ok,
            "installed " + ", ".join(missing) if ok else (out[-400:] or "brew install failed"),
        )
    )
    return steps


def ensure_splat_transform(*, dry_run: bool = False) -> SetupStep:
    if _which("splat-transform"):
        return SetupStep("splat-transform", True, _which("splat-transform") or "")
    npm = _which("npm")
    if not npm:
        return SetupStep(
            "splat-transform",
            False,
            "npm missing — install Node 18+ then: npm i -g @playcanvas/splat-transform",
        )
    if dry_run:
        return SetupStep("splat-transform", False, "Would run npm i -g @playcanvas/splat-transform")
    code, out = _run([npm, "install", "-g", "@playcanvas/splat-transform"], timeout=300.0)
    path = _which("splat-transform")
    return SetupStep(
        "splat-transform",
        bool(path),
        path or (out[-300:] if out else "npm global install failed"),
    )


def check_tool(name: str, *, hint: str = "") -> SetupStep:
    path = _which(name)
    if path:
        return SetupStep(name, True, path)
    return SetupStep(name, False, hint or f"missing — install {name}")


def check_pytorch() -> SetupStep:
    try:
        import torch

        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        detail = f"{torch.__version__}; MPS={'yes' if mps else 'no'}"
        if sys.platform == "darwin" and os.uname().machine == "arm64" and not mps:
            detail += " — install MPS wheel from https://pytorch.org if training is slow/CPU-only"
        return SetupStep("pytorch", True, detail)
    except ImportError:
        return SetupStep("pytorch", False, "pip install -e .  (torch is a core dependency)")


def check_instasplat_import() -> SetupStep:
    try:
        import instasplat  # noqa: F401

        return SetupStep("instasplat", True, f"import ok ({getattr(instasplat, '__version__', '?')})")
    except Exception as exc:  # noqa: BLE001
        return SetupStep("instasplat", False, f"pip install -e '.[gui]' — {exc}")


def _mark_ready(report: SetupReport) -> None:
    by_name = {s.name: s.ok for s in report.steps}
    tools_ok = by_name.get("brew-tools", False) or all(
        by_name.get(n, False) for n in ("ffmpeg", "colmap")
    )
    report.ready = (
        tools_ok
        and by_name.get("splat-transform", False)
        and by_name.get("pytorch", False)
        and by_name.get("instasplat", False)
    )
    report.next_commands = [
        "instasplat doctor",
        "instasplat gui",
        "instasplat mac-360 -i ./capture_equirect.mp4 -o ./runs -n walk",
        "# Full metal splat workflow: docs/METAL_SPLAT_WORKFLOW.md",
    ]


def run_setup(
    *,
    install_system: bool = True,
    dry_run: bool = False,
) -> SetupReport:
    """
    Verify (and optionally install) tools needed for the metal_equirect pipeline.

    Does not create a venv — callers should activate one first. System packages
    use Homebrew / npm when ``install_system`` is True.
    """
    report = SetupReport()
    report.add("python", True, sys.version.split()[0])

    if not install_system:
        for name, hint in (
            ("ffmpeg", "brew install ffmpeg"),
            ("exiftool", "brew install exiftool"),
            ("colmap", "brew install colmap"),
            ("splat-transform", "npm i -g @playcanvas/splat-transform"),
        ):
            report.steps.append(check_tool(name, hint=hint))
    elif sys.platform == "darwin":
        for step in ensure_brew_tools(dry_run=dry_run):
            report.steps.append(step)
        report.steps.append(ensure_splat_transform(dry_run=dry_run))
    else:
        for name, hint in (
            ("ffmpeg", "install ffmpeg"),
            ("exiftool", "install exiftool"),
            ("colmap", "install colmap"),
        ):
            report.steps.append(check_tool(name, hint=hint))
        report.steps.append(ensure_splat_transform(dry_run=dry_run))

    report.steps.append(check_instasplat_import())
    report.steps.append(check_pytorch())
    _mark_ready(report)
    return report


def format_setup_report(report: SetupReport) -> str:
    """Plain-text checklist (for logs / tests)."""
    lines = ["InstaSplat environment", "----------------------"]
    for step in report.steps:
        mark = "OK" if step.ok else "MISSING"
        detail = f" — {step.detail}" if step.detail else ""
        lines.append(f"[{mark}] {step.name}{detail}")
    lines.append("")
    lines.append("Ready" if report.ready else "Not fully ready")
    return "\n".join(lines)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
