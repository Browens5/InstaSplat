"""CPU reference for the Metal soft-OIT kernel (parity / Linux CI)."""

from __future__ import annotations

import math

import numpy as np


def soft_oit_reference(
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
    depth_tau: float = 0.15,
) -> np.ndarray:
    """
    NumPy soft-OIT matching ``soft_oit_accumulate`` + ``soft_oit_normalize``.

    Used when Metal/PyObjC is unavailable and for kernel parity tests.
    """
    n = int(mean_2d.shape[0])
    H, W = int(height), int(width)
    color_acc = np.zeros((H, W, 3), dtype=np.float64)
    weight_acc = np.zeros((H, W), dtype=np.float64)
    fp = int(max(1, footprint))

    mean_2d = np.asarray(mean_2d, dtype=np.float64)
    cov_2d = np.asarray(cov_2d, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid).astype(bool)

    for i in range(n):
        if not valid[i]:
            continue
        mu = mean_2d[i]
        if not (np.isfinite(mu[0]) and np.isfinite(mu[1])):
            continue
        r = float(np.clip(radius[i], 1.0, float(fp)))
        if not (r > 0.5):
            continue
        # Match Metal: +eps toward SPD before det (same as torch safe path)
        a = float(cov_2d[i, 0, 0]) + 1e-4
        b01 = 0.5 * (float(cov_2d[i, 0, 1]) + float(cov_2d[i, 1, 0]))
        c = float(cov_2d[i, 1, 1]) + 1e-4
        det = a * c - b01 * b01
        if det < 1e-12:
            continue
        inv00 = c / det
        inv11 = a / det
        inv01 = -b01 / det
        op = float(opacities[i]) * math.exp(-depth_tau * max(float(depth[i]), 0.01))
        col = colors[i]
        half = min(int(math.ceil(r)), fp)

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px = mu[0] + dx
                py = mu[1] + dy
                du = px - mu[0]
                du = du - W * round(du / W)
                dv = py - mu[1]
                if du * du + dv * dv > r * r:
                    continue
                py_r = round(py)
                if py_r < 0.0 or py_r >= H:
                    continue
                maha = inv00 * du * du + 2.0 * inv01 * du * dv + inv11 * dv * dv
                if not np.isfinite(maha) or maha < 0.0:
                    maha = 1e6 if not np.isfinite(maha) else maha
                alpha = op * math.exp(-0.5 * maha)
                alpha = 0.0 if alpha < 0.0 else (0.99 if alpha > 0.99 else alpha)
                if alpha < 1e-5:
                    continue
                ix = int(round(px)) % W
                if ix < 0:
                    ix += W
                iy = int(py_r)
                if iy < 0:
                    iy = 0
                elif iy >= H:
                    iy = H - 1
                color_acc[iy, ix] += col * alpha
                weight_acc[iy, ix] += alpha

    w = weight_acc[..., None] + 1e-6
    rgb = np.clip(color_acc / w, 0.0, 1.0)
    return rgb.astype(np.float32)
