"""Equirectangular camera model + 3DGUT-style unscented projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EquirectCamera:
    """Equirectangular panorama camera (longitude × latitude → pixels)."""

    width: int
    height: int

    def project_dirs(self, dirs: torch.Tensor) -> torch.Tensor:
        """
        Project camera-space direction vectors to pixel coords (u, v).

        Convention: +Z forward, +X right, +Y down (COLMAP-like).
        lon ∈ [-π, π] → u ∈ [0, W); lat ∈ [-π/2, π/2] → v ∈ [0, H).
        """
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        lon = torch.atan2(x, z)
        lat = torch.asin(torch.clamp(-y / (torch.linalg.norm(dirs, dim=-1) + 1e-8), -1.0, 1.0))
        u = (lon / (2.0 * math.pi) + 0.5) * self.width
        v = (0.5 - lat / math.pi) * self.height
        return torch.stack([u, v], dim=-1)

    def latitude_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Per-row weight ≈ cos(lat) for equirect area compensation."""
        ys = torch.arange(self.height, device=device, dtype=dtype)
        lat = (0.5 - (ys + 0.5) / self.height) * math.pi
        return torch.cos(lat).clamp(min=0.05).view(self.height, 1)


def yaw_pitch_to_rotmat(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation taking equirect camera rays into a cubemap face frame."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    return Rx @ Ry


def quat_to_rotmat_torch(q: torch.Tensor) -> torch.Tensor:
    """
    (…, 4) wxyz → (…, 3, 3) Hamilton / COLMAP convention.

    Equivalent to ``instasplat.utils.scale.qvec_to_rotmat`` after unit normalize:
    ``R = [[1-2(y²+z²), 2(xy-zw), 2(xz+yw)], …]``.
    """
    n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
    q = q / n
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    # Unit-quat form: ww+xx-yy-zz == 1 - 2(yy+zz), etc.
    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def build_covariance(
    scales: torch.Tensor, quats: torch.Tensor
) -> torch.Tensor:
    """(N,3) scales + (N,4) wxyz → (N,3,3) Σ."""
    R = quat_to_rotmat_torch(quats)
    S = torch.diag_embed(scales.clamp(min=1e-6))
    return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)


def unscented_equirect_project(
    means_cam: torch.Tensor,
    cov_cam: torch.Tensor,
    camera: EquirectCamera,
    alpha: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    3DGUT-style Unscented Transform through equirectangular projection.

    Returns:
        mean_2d: (N, 2) pixel coords
        cov_2d: (N, 2, 2) screen-space covariance
        valid: (N,) bool — in front of camera / finite
    """
    n = means_cam.shape[0]
    device = means_cam.device
    dtype = means_cam.dtype
    # Merwe / Julier sigma points: mean + ± sqrt((n+λ)Σ) columns
    dim = 3
    lam = alpha**2 * (dim + 2) - dim  # κ=2, β unused for mean/cov here
    scale = dim + lam
    eye = torch.eye(dim, device=device, dtype=dtype).expand(n, dim, dim)
    # Cholesky of scale * cov
    cov_s = cov_cam + eye * 1e-8
    try:
        chol = torch.linalg.cholesky(cov_s * scale)
    except RuntimeError:
        # Fallback: diagonal
        diag = torch.diagonal(cov_s, dim1=-2, dim2=-1).clamp(min=1e-8)
        chol = torch.diag_embed(torch.sqrt(diag * scale))

    # (N, 7, 3) sigma points
    zeros = torch.zeros(n, 1, 3, device=device, dtype=dtype)
    pos = chol.transpose(-1, -2)  # rows as offsets
    neg = -pos
    offsets = torch.cat([zeros, pos, neg], dim=1)
    sigmas = means_cam.unsqueeze(1) + offsets  # (N, 7, 3)

    # Validity: finite camera-space means with non-zero radius from origin.
    # Equirect has no single "near plane"; reject only degenerate / NaN means.
    depths = means_cam[:, 2]
    valid = (
        torch.isfinite(depths)
        & torch.isfinite(means_cam).all(dim=-1)
        & (torch.linalg.norm(means_cam, dim=-1) > 1e-6)
    )

    flat = sigmas.reshape(-1, 3)
    uv = camera.project_dirs(flat).reshape(n, 7, 2)
    valid = valid & torch.isfinite(uv).all(dim=-1).all(dim=-1)

    # Wrap-aware mean for longitude (u): work in relative offsets from sigma0
    u0 = uv[:, 0, 0]
    du = uv[..., 0] - u0.unsqueeze(-1)
    w = float(camera.width)
    du = du - w * torch.round(du / w)
    uv_adj = uv.clone()
    uv_adj[..., 0] = u0.unsqueeze(-1) + du

    w0 = lam / scale
    wi = 1.0 / (2.0 * scale)
    weights = torch.full((7,), wi, device=device, dtype=dtype)
    weights[0] = w0

    mean_2d = (uv_adj * weights.view(1, 7, 1)).sum(dim=1)
    # Wrap u into image
    mean_2d = mean_2d.clone()
    mean_2d[:, 0] = torch.remainder(mean_2d[:, 0], w)

    diff = uv_adj - mean_2d.unsqueeze(1)
    diff_u = diff[..., 0:1]
    diff_u = diff_u - w * torch.round(diff_u / w)
    diff = torch.cat([diff_u, diff[..., 1:2]], dim=-1)
    cov_2d = torch.einsum("i,nij,nik->njk", weights, diff, diff)
    # Floor for numerical stability + PSD nudge
    eye2 = torch.eye(2, device=device, dtype=dtype).expand(n, 2, 2)
    cov_2d = cov_2d + eye2 * 0.25
    # Symmetrize (floating error)
    cov_2d = 0.5 * (cov_2d + cov_2d.transpose(-1, -2))
    return mean_2d, cov_2d, valid
