"""Metal / Apple Silicon device helpers."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any


@dataclass
class MetalStatus:
    is_macos: bool
    is_apple_silicon: bool
    mps_available: bool
    torch_device: str
    notes: str


def detect_metal() -> MetalStatus:
    is_macos = platform.system() == "Darwin"
    machine = platform.machine().lower()
    is_apple_silicon = is_macos and machine in {"arm64", "aarch64"}
    mps_available = False
    notes = []
    try:
        import torch

        mps_available = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        if mps_available:
            notes.append("PyTorch MPS (Metal) available for YOLO masking")
        elif is_macos:
            notes.append("PyTorch installed but MPS unavailable — using CPU for YOLO")
    except ImportError:
        notes.append("PyTorch not installed")

    if is_apple_silicon:
        notes.append("metal_equirect should use PyTorch MPS on Apple Silicon")
    elif is_macos:
        notes.append("Intel Mac: prefer Apple Silicon for metal_equirect / 8K jobs")

    device = "mps" if mps_available else "cpu"
    return MetalStatus(
        is_macos=is_macos,
        is_apple_silicon=is_apple_silicon,
        mps_available=mps_available,
        torch_device=device,
        notes="; ".join(notes),
    )


def apply_metal_env(prefer_metal: bool = True) -> dict[str, str]:
    """Return env updates that favor Metal-friendly runtimes."""
    env = dict(os.environ)
    if prefer_metal:
        # Allow ops that lack MPS kernels to fall back instead of crashing.
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # Discourage CUDA assumptions in mixed tooling.
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return env


def metal_report() -> dict[str, Any]:
    status = detect_metal()
    return {
        "is_macos": status.is_macos,
        "is_apple_silicon": status.is_apple_silicon,
        "mps_available": status.mps_available,
        "torch_device": status.torch_device,
        "notes": status.notes,
    }
