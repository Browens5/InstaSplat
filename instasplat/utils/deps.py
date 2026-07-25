"""External dependency discovery and capability report."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class DepStatus:
    name: str
    available: bool
    path: str | None = None
    version: str | None = None
    notes: str | None = None
    required_for: list[str] | None = None


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(cmd: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 1, "", str(exc)


def check_ffmpeg() -> DepStatus:
    path = _which("ffmpeg")
    if not path:
        return DepStatus(
            "ffmpeg",
            False,
            notes="Install via Homebrew: brew install ffmpeg",
            required_for=["extract"],
        )
    code, out, err = _run([path, "-version"])
    ver = (out or err).splitlines()[0] if (out or err) else None
    return DepStatus("ffmpeg", code == 0, path, ver, required_for=["extract"])


def check_exiftool() -> DepStatus:
    path = _which("exiftool")
    if not path:
        return DepStatus(
            "exiftool",
            False,
            notes="Install via Homebrew: brew install exiftool",
            required_for=["ingest", "gyro"],
        )
    code, out, _ = _run([path, "-ver"])
    return DepStatus("exiftool", code == 0, path, out or None, required_for=["ingest", "gyro"])


def check_colmap() -> DepStatus:
    path = _which("colmap")
    if not path:
        return DepStatus(
            "colmap",
            False,
            notes="Install via Homebrew: brew install colmap (or build from source)",
            required_for=["sfm"],
        )
    code, out, err = _run([path, "-h"])
    # COLMAP prints help to stderr sometimes
    available = code == 0 or "COLMAP" in (out + err)
    return DepStatus(
        "colmap",
        available,
        path,
        notes="CPU mapping is typical on macOS Apple Silicon",
        required_for=["sfm"],
    )


def check_brush(bin_name: str = "brush") -> DepStatus:
    path = _which(bin_name)
    if not path:
        # Also check common cargo install location
        cargo = Path.home() / ".cargo" / "bin" / bin_name
        path = str(cargo) if cargo.exists() else None
    if not path:
        return DepStatus(
            "brush",
            False,
            notes=(
                "Build from https://github.com/ArthurBrussee/brush "
                "(cargo install / cargo run --release). Ideal for Mac Metal/WebGPU."
            ),
            required_for=["train"],
        )
    code, out, err = _run([path, "--help"])
    return DepStatus(
        "brush",
        code == 0 or "brush" in (out + err).lower(),
        path,
        required_for=["train"],
    )


def check_opensplat(bin_name: str = "opensplat") -> DepStatus:
    path = _which(bin_name)
    if not path:
        return DepStatus(
            "opensplat",
            False,
            notes=(
                "Build from https://github.com/pierotofy/OpenSplat with "
                "-DGPU_RUNTIME=MPS for Apple Silicon Metal training (AGPL)."
            ),
            required_for=["train"],
        )
    code, out, err = _run([path, "--help"])
    return DepStatus(
        "opensplat",
        code == 0 or "opensplat" in (out + err).lower() or "splat" in (out + err).lower(),
        path,
        notes="Metal MPS trainer alternative to Brush",
        required_for=["train"],
    )


def check_splat_transform(bin_name: str = "splat-transform") -> DepStatus:
    path = _which(bin_name)
    if not path:
        return DepStatus(
            "splat-transform",
            False,
            notes="Install via npm: npm install -g @playcanvas/splat-transform",
            required_for=["export"],
        )
    code, out, err = _run([path, "--help"])
    return DepStatus(
        "splat-transform",
        code == 0 or "splat" in (out + err).lower(),
        path,
        required_for=["export"],
    )


def check_mediasdk() -> DepStatus:
    path = _which("MediaSDKTest")
    if not path:
        return DepStatus(
            "MediaSDKTest",
            False,
            notes=(
                "Official Insta360 MediaSDK targets Windows/Ubuntu, not macOS. "
                "On Mac, export equirectangular MP4 from Insta360 Studio, or run "
                "MediaSDK in a Linux Docker/cloud worker."
            ),
            required_for=["stitch"],
        )
    return DepStatus("MediaSDKTest", True, path, required_for=["stitch"])


def check_python_ml() -> list[DepStatus]:
    results: list[DepStatus] = []
    try:
        import torch

        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        results.append(
            DepStatus(
                "pytorch",
                True,
                version=torch.__version__,
                notes=f"MPS available: {mps}",
                required_for=["mask"],
            )
        )
    except ImportError:
        results.append(
            DepStatus(
                "pytorch",
                False,
                notes="pip install torch (MPS builds for Apple Silicon)",
                required_for=["mask"],
            )
        )
    try:
        import ultralytics

        results.append(
            DepStatus(
                "ultralytics",
                True,
                version=getattr(ultralytics, "__version__", "unknown"),
                required_for=["mask"],
            )
        )
    except ImportError:
        results.append(
            DepStatus(
                "ultralytics",
                False,
                notes="pip install ultralytics",
                required_for=["mask"],
            )
        )
    return results


def check_all(
    brush_bin: str = "brush",
    splat_transform_bin: str = "splat-transform",
) -> list[DepStatus]:
    return [
        check_ffmpeg(),
        check_exiftool(),
        check_colmap(),
        check_brush(brush_bin),
        check_opensplat(),
        check_splat_transform(splat_transform_bin),
        check_mediasdk(),
        *check_python_ml(),
    ]


def report_dict(
    brush_bin: str = "brush",
    splat_transform_bin: str = "splat-transform",
) -> dict[str, Any]:
    deps = check_all(brush_bin, splat_transform_bin)
    return {
        "platform": os.uname().sysname,
        "machine": os.uname().machine,
        "deps": [asdict(d) for d in deps],
        "ready_stages": _ready_stages(deps),
    }


def _ready_stages(deps: list[DepStatus]) -> dict[str, bool]:
    by_name = {d.name: d for d in deps}
    return {
        "ingest": by_name.get("exiftool", DepStatus("exiftool", False)).available
        or by_name.get("ffmpeg", DepStatus("ffmpeg", False)).available,
        "extract": by_name.get("ffmpeg", DepStatus("ffmpeg", False)).available,
        "mask": by_name.get("ultralytics", DepStatus("ultralytics", False)).available,
        "sfm": by_name.get("colmap", DepStatus("colmap", False)).available,
        "train": by_name.get("brush", DepStatus("brush", False)).available
        or by_name.get("opensplat", DepStatus("opensplat", False)).available,
        "export": by_name.get("splat-transform", DepStatus("splat-transform", False)).available,
        "official_stitch": by_name.get("MediaSDKTest", DepStatus("MediaSDKTest", False)).available,
    }


def print_report(rich_console: Any | None = None) -> dict[str, Any]:
    data = report_dict()
    if rich_console is not None:
        from rich.table import Table

        table = Table(title="InstaSplat dependency check")
        table.add_column("Tool")
        table.add_column("OK")
        table.add_column("Path / Version")
        table.add_column("Notes")
        for d in data["deps"]:
            table.add_row(
                d["name"],
                "✓" if d["available"] else "✗",
                d.get("path") or d.get("version") or "—",
                d.get("notes") or "",
            )
        rich_console.print(table)
        rich_console.print("Ready stages:", data["ready_stages"])
    else:
        print(json.dumps(data, indent=2))
    return data
