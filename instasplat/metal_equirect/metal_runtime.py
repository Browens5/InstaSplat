"""Optional Metal acceleration for equirect soft-OIT (macOS + PyObjC)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import torch


_METAL_DIR = Path(__file__).resolve().parent / "metal"
_SOURCE = _METAL_DIR / "EquirectProject.metal"
_LIB = _METAL_DIR / "EquirectProject.metallib"

_LOCK = threading.Lock()
_PIPELINE = None  # lazy MetalPipeline | False
_ACTIVE = "torch_oit"


def metal_available() -> bool:
    return sys.platform == "darwin" and shutil.which("xcrun") is not None


def metallib_path() -> Path | None:
    return _LIB if _LIB.exists() else None


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


def _try_load_pipeline():
    """Return a MetalPipeline instance, False if unavailable, or None if not tried."""
    global _PIPELINE
    with _LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        if sys.platform != "darwin":
            _PIPELINE = False
            return _PIPELINE
        lib = compile_metallib()
        if lib is None:
            _PIPELINE = False
            return _PIPELINE
        try:
            from instasplat.metal_equirect._metal_dispatch import MetalPipeline

            pipe = MetalPipeline(lib)
            if not pipe.ok:
                _PIPELINE = False
            else:
                _PIPELINE = pipe
        except Exception:
            _PIPELINE = False
        return _PIPELINE


def metal_raster_available() -> bool:
    """True when fused soft-OIT Metal kernels can be dispatched."""
    pipe = _try_load_pipeline()
    return bool(pipe)


def metal_status() -> dict:
    pipe = _try_load_pipeline()
    active = "metal_oit" if pipe else "torch_oit"
    return {
        "platform": sys.platform,
        "xcrun": bool(shutil.which("xcrun")),
        "source": str(_SOURCE) if _SOURCE.exists() else None,
        "metallib": str(_LIB) if _LIB.exists() else None,
        "compiled": _LIB.exists(),
        "dispatch": bool(pipe),
        "active_backend": active,
        "note": (
            "Fused soft-OIT metallib dispatch when PyObjC + MPS available; "
            "otherwise vectorized PyTorch OIT (MPS/CPU)."
            if pipe
            else (
                "Metallib compiles on macOS for native soft-OIT; "
                "training uses vectorized PyTorch OIT until PyObjC dispatch loads."
            )
        ),
    }


def _metal_soft_oit_numpy(
    mean_2d: np.ndarray,
    cov_2d: np.ndarray,
    radius: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    height: int,
    width: int,
    *,
    footprint: int = 24,
) -> np.ndarray | None:
    pipe = _try_load_pipeline()
    if not pipe:
        return None
    try:
        return pipe.soft_oit(
            mean_2d,
            cov_2d,
            radius,
            opacities,
            colors,
            depth,
            valid,
            height,
            width,
            footprint=footprint,
        )
    except Exception:
        return None


def metal_fused_oit_ste(
    torch_img: torch.Tensor,
    mean_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    radius: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    camera,
) -> torch.Tensor:
    """
    Straight-through estimator: Metal forward values, torch autograd via ``torch_img``.

    If Metal dispatch fails, returns ``torch_img`` unchanged.
    """
    global _ACTIVE
    with torch.no_grad():
        metal_np = _metal_soft_oit_numpy(
            mean_2d.detach().float().cpu().numpy(),
            cov_2d.detach().float().cpu().numpy(),
            radius.detach().float().cpu().numpy(),
            opacities.detach().float().cpu().numpy(),
            colors.detach().float().cpu().numpy(),
            depth.detach().float().cpu().numpy(),
            valid.detach().to(torch.uint8).cpu().numpy(),
            int(camera.height),
            int(camera.width),
        )
    if metal_np is None:
        _ACTIVE = "torch_oit"
        return torch_img
    _ACTIVE = "metal_oit"
    metal_t = torch.from_numpy(metal_np).to(
        device=torch_img.device, dtype=torch_img.dtype
    )
    # STE: forward = Metal, backward flows through torch_img
    return torch_img + (metal_t - torch_img).detach()
