"""Optional Metal acceleration for equirect projection (macOS)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


_METAL_DIR = Path(__file__).resolve().parent / "metal"
_SOURCE = _METAL_DIR / "EquirectProject.metal"
_LIB = _METAL_DIR / "EquirectProject.metallib"


def metal_available() -> bool:
    return sys.platform == "darwin" and shutil.which("xcrun") is not None


def metallib_path() -> Path | None:
    if _LIB.exists():
        return _LIB
    return None


def compile_metallib(*, force: bool = False) -> Path | None:
    """Compile EquirectProject.metal → metallib on macOS. Returns path or None."""
    if not metal_available():
        return None
    if _LIB.exists() and not force:
        return _LIB
    if not _SOURCE.exists():
        return None
    air = _METAL_DIR / "EquirectProject.air"
    try:
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(_SOURCE), "-o", str(air)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(_LIB)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return _LIB if _LIB.exists() else None


def metal_status() -> dict:
    return {
        "platform": sys.platform,
        "xcrun": bool(shutil.which("xcrun")),
        "source": str(_SOURCE) if _SOURCE.exists() else None,
        "metallib": str(_LIB) if _LIB.exists() else None,
        "compiled": _LIB.exists(),
        # Projection still runs in PyTorch; metallib is prepared for native dispatch.
        "active_backend": "torch_ut",
        "note": (
            "Metallib compiles on macOS for future GPU dispatch; "
            "training uses PyTorch MPS/CPU UT path today."
        ),
    }
