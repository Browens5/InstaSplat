"""Fused soft-OIT composite: Metal forward (or CPU ref), torch backward."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import torch

from instasplat.metal_equirect.soft_oit_ref import soft_oit_reference


_METAL_DIR = Path(__file__).resolve().parent / "metal"
_SOURCE = _METAL_DIR / "EquirectProject.metal"
_LIB = _METAL_DIR / "EquirectProject.metallib"

_LOCK = threading.Lock()
_PIPELINE = None  # MetalPipeline | False after first probe
_ACTIVE = "torch_oit"
_FORCE_REF = False  # tests: skip Metal, use CPU reference forward


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
    global _PIPELINE
    with _LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        if _FORCE_REF or sys.platform != "darwin":
            _PIPELINE = False
            return _PIPELINE
        lib = compile_metallib()
        if lib is None:
            _PIPELINE = False
            return _PIPELINE
        try:
            from instasplat.metal_equirect._metal_dispatch import MetalPipeline

            pipe = MetalPipeline(lib)
            _PIPELINE = pipe if pipe.ok else False
        except Exception:
            _PIPELINE = False
        return _PIPELINE


def set_force_reference(enabled: bool = True) -> None:
    """Test helper: force CPU reference forward (ignore Metal)."""
    global _FORCE_REF, _PIPELINE, _ACTIVE
    _FORCE_REF = bool(enabled)
    _PIPELINE = None
    _ACTIVE = "ref_oit" if enabled else "torch_oit"


def force_reference_enabled() -> bool:
    """True when tests forced the CPU reference forward path."""
    return bool(_FORCE_REF)


def metal_raster_available() -> bool:
    """True when real Metal soft-OIT kernels can be dispatched."""
    return bool(_try_load_pipeline())


def fused_composite_available() -> bool:
    """True when fused composite forward can run (Metal or CPU reference)."""
    return True


def active_composite_backend() -> str:
    return _ACTIVE


def metal_status() -> dict:
    pipe = _try_load_pipeline()
    shared = bool(pipe and getattr(pipe, "shared_buffers", False))
    if pipe:
        active = "metal_oit_shared" if shared else "metal_oit"
    elif _FORCE_REF:
        active = "ref_oit"
    else:
        active = "torch_oit"
    return {
        "platform": sys.platform,
        "xcrun": bool(shutil.which("xcrun")),
        "source": str(_SOURCE) if _SOURCE.exists() else None,
        "metallib": str(_LIB) if _LIB.exists() else None,
        "compiled": _LIB.exists(),
        "dispatch": bool(pipe),
        "shared_buffers": shared,
        "force_reference": _FORCE_REF,
        "active_backend": active,
        "fused_composite": True,
        "note": (
            "Fused soft-OIT with pooled MTL shared buffers (one D2H/H2D per step) "
            "when PyObjC metallib loads; else CPU reference or torch OIT."
            if shared
            else (
                "Fused soft-OIT: Metal forward + torch backward when available; "
                "otherwise CPU reference forward or torch OIT."
            )
        ),
    }


def soft_oit_forward_numpy(
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
) -> tuple[np.ndarray, str]:
    """NumPy fused forward (legacy / tests). Prefers Metal; else CPU reference."""
    global _ACTIVE
    pipe = _try_load_pipeline()
    if pipe and not _FORCE_REF:
        try:
            out = pipe.soft_oit(
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
            _ACTIVE = "metal_oit_shared" if getattr(pipe, "shared_buffers", False) else "metal_oit"
            return out, _ACTIVE
        except Exception:
            pass
    out = soft_oit_reference(
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
    _ACTIVE = "ref_oit"
    return out, "ref_oit"


def soft_oit_forward_tensors(
    mean_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    radius: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    height: int,
    width: int,
    *,
    footprint: int = 24,
) -> tuple[torch.Tensor, str]:
    """
    Tensor fused forward.

    On Metal: shared-buffer path (one copy in, one copy out, pooled MTLBuffers).
    With ``set_force_reference(True)``: CPU reference (tests only).
    Otherwise raises so callers can fall back to vectorized torch OIT.
    """
    global _ACTIVE
    pipe = _try_load_pipeline()
    if pipe and not _FORCE_REF:
        try:
            out = pipe.soft_oit_from_tensors(
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
                out_device=mean_2d.device,
                out_dtype=mean_2d.dtype,
            )
            _ACTIVE = (
                "metal_oit_shared" if getattr(pipe, "shared_buffers", False) else "metal_oit"
            )
            return out, _ACTIVE
        except Exception as exc:
            raise RuntimeError(f"Metal soft-OIT dispatch failed: {exc}") from exc

    if not _FORCE_REF:
        raise RuntimeError(
            "Fused Metal soft-OIT unavailable (no metallib/PyObjC dispatch). "
            "Fall back to composite=oit."
        )

    arrays = (
        mean_2d.detach().float().cpu().numpy(),
        cov_2d.detach().float().cpu().numpy(),
        radius.detach().float().cpu().numpy(),
        opacities.detach().float().cpu().numpy(),
        colors.detach().float().cpu().numpy(),
        depth.detach().float().cpu().numpy(),
        valid.detach().to(torch.uint8).cpu().numpy(),
    )
    rgb_np = soft_oit_reference(
        *arrays,
        height,
        width,
        footprint=footprint,
    )
    _ACTIVE = "ref_oit"
    return (
        torch.from_numpy(rgb_np).to(device=mean_2d.device, dtype=mean_2d.dtype),
        "ref_oit",
    )


class FusedSoftOIT(torch.autograd.Function):
    """
    Metal shared-buffer (or CPU-ref) soft-OIT forward; torch OIT for backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        mean_2d: torch.Tensor,
        cov_2d: torch.Tensor,
        radius: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        depth: torch.Tensor,
        valid: torch.Tensor,
        height: int,
        width: int,
        footprint: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(mean_2d, cov_2d, radius, opacities, colors, depth, valid)
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.footprint = int(footprint)

        out, _backend = soft_oit_forward_tensors(
            mean_2d,
            cov_2d,
            radius,
            opacities,
            colors,
            depth,
            valid,
            ctx.height,
            ctx.width,
            footprint=ctx.footprint,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        mean_2d, cov_2d, radius, opacities, colors, depth, valid = ctx.saved_tensors
        from instasplat.metal_equirect.cameras import EquirectCamera
        from instasplat.metal_equirect.rasterize import _composite_oit_batched

        cam = EquirectCamera(ctx.width, ctx.height)
        with torch.enable_grad():
            mean_g = mean_2d.detach().requires_grad_(True)
            cov_g = cov_2d.detach().requires_grad_(True)
            rad_g = radius.detach().requires_grad_(True)
            op_g = opacities.detach().requires_grad_(True)
            col_g = colors.detach().requires_grad_(True)
            pred = _composite_oit_batched(
                mean_g,
                cov_g,
                rad_g,
                op_g,
                col_g,
                depth.detach(),
                valid.detach(),
                cam,
                max_footprint=ctx.footprint,
            )
            pred.backward(grad_output)

        return (
            mean_g.grad,
            cov_g.grad,
            rad_g.grad,
            op_g.grad,
            col_g.grad,
            None,
            None,
            None,
            None,
            None,
        )


def fused_soft_oit(
    mean_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    radius: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    camera,
    *,
    footprint: int = 24,
) -> torch.Tensor:
    """Public entry: fused Metal/ref soft-OIT with torch backward."""
    return FusedSoftOIT.apply(
        mean_2d,
        cov_2d,
        radius,
        opacities,
        colors,
        depth,
        valid,
        int(camera.height),
        int(camera.width),
        int(footprint),
    )


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
    """Legacy STE wrapper — ignores ``torch_img``, runs fused path."""
    _ = torch_img
    return fused_soft_oit(
        mean_2d, cov_2d, radius, opacities, colors, depth, valid, camera
    )
