"""Gaussian parameterizations and PLY I/O (3DGS / gsplat-compatible)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


_SH_C0 = 0.28209479177387814


class GaussianModel(nn.Module):
    """3D Gaussians with DC + optional higher SH (degree ≤ 1 in v1)."""

    def __init__(
        self,
        means: torch.Tensor,
        rgbs: torch.Tensor,
        *,
        sh_degree: int = 1,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        n = means.shape[0]
        self.sh_degree = max(0, min(int(sh_degree), 1))
        self.means = nn.Parameter(means.float().clone())
        # opacity logit
        self.opacities = nn.Parameter(torch.logit(torch.full((n,), 0.1)))
        # log scales
        self.scales = nn.Parameter(torch.log(torch.full((n, 3), init_scale)))
        # identity quats wxyz
        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)
        # SH: f_dc (3) + optional f_rest (9 for degree 1)
        f_dc = (rgbs.float().clamp(0, 1) - 0.5) / _SH_C0
        self.f_dc = nn.Parameter(f_dc)
        if self.sh_degree >= 1:
            self.f_rest = nn.Parameter(torch.zeros(n, 9))
        else:
            self.register_buffer("f_rest", torch.zeros(n, 0))

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    def get_scales(self) -> torch.Tensor:
        return torch.exp(self.scales).clamp(min=1e-6, max=50.0)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities)

    def get_quats(self) -> torch.Tensor:
        return self.quats / (torch.linalg.norm(self.quats, dim=-1, keepdim=True) + 1e-8)

    def colors_dc(self) -> torch.Tensor:
        return (0.5 + _SH_C0 * self.f_dc).clamp(0, 1)

    def prune_mask(self, keep: torch.Tensor) -> None:
        """In-place prune by boolean keep mask (no grad)."""
        with torch.no_grad():
            for name in ("means", "opacities", "scales", "quats", "f_dc"):
                param = getattr(self, name)
                new = nn.Parameter(param.data[keep].clone())
                setattr(self, name, new)
            if self.sh_degree >= 1 and self.f_rest.numel() > 0:
                self.f_rest = nn.Parameter(self.f_rest.data[keep].clone())

    def densify_clone(self, idx: torch.Tensor, scale_div: float = 1.6) -> None:
        """Clone selected Gaussians (MCMC-lite)."""
        if idx.numel() == 0:
            return
        with torch.no_grad():
            extras = {
                "means": self.means.data[idx].clone(),
                "opacities": self.opacities.data[idx].clone(),
                "scales": self.scales.data[idx].clone() - np.log(scale_div),
                "quats": self.quats.data[idx].clone(),
                "f_dc": self.f_dc.data[idx].clone(),
            }
            if self.sh_degree >= 1 and isinstance(self.f_rest, nn.Parameter) and self.f_rest.numel():
                extras["f_rest"] = self.f_rest.data[idx].clone()
            for name, extra in extras.items():
                param = getattr(self, name)
                merged = torch.cat([param.data, extra], dim=0)
                setattr(self, name, nn.Parameter(merged))

    def set_active_sh_degree(self, degree: int) -> None:
        """Enable higher SH bands mid-training (allocates f_rest if needed)."""
        degree = max(0, min(int(degree), 1))
        if degree <= self.sh_degree:
            self.sh_degree = degree
            return
        with torch.no_grad():
            if degree >= 1 and (
                not isinstance(self.f_rest, nn.Parameter) or self.f_rest.numel() == 0
            ):
                self.f_rest = nn.Parameter(
                    torch.zeros(self.n, 9, device=self.means.device, dtype=self.means.dtype)
                )
        self.sh_degree = degree


def gaussians_from_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    sh_degree: int = 1,
    max_points: int = 80_000,
    device: torch.device | None = None,
) -> GaussianModel:
    if len(xyz) == 0:
        # Minimal random cloud so training can still run smoke tests
        xyz = np.random.randn(64, 3).astype(np.float32) * 0.5
        rgb = np.full((64, 3), 0.6, dtype=np.float32)
    if len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz, rgb = xyz[idx], rgb[idx]
    # Init scale from nearest-neighbor spacing
    if len(xyz) >= 2:
        # subsample for speed
        sample = xyz[:: max(1, len(xyz) // 2000)]
        d = np.linalg.norm(sample[None, :, :] - sample[:, None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = float(np.median(d.min(axis=1)))
        init_scale = max(nn * 0.5, 1e-3)
    else:
        init_scale = 0.05
    means = torch.from_numpy(xyz.astype(np.float32))
    rgbs = torch.from_numpy(rgb.astype(np.float32))
    if device is not None:
        means = means.to(device)
        rgbs = rgbs.to(device)
    return GaussianModel(means, rgbs, sh_degree=sh_degree, init_scale=init_scale)


def export_ply(model: GaussianModel, path: Path) -> Path:
    """Write a Gaussian PLY readable by splat-transform / InstaSplat viewers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    means = model.means.detach().cpu().numpy()
    scales = model.get_scales().detach().cpu().numpy()
    quats = model.get_quats().detach().cpu().numpy()
    opacity = model.opacities.detach().cpu().numpy()  # logit space like 3DGS
    f_dc = model.f_dc.detach().cpu().numpy()
    n = means.shape[0]
    # Standard 3DGS / Brush-ish properties
    props = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    )
    # rot as wxyz to match many exporters
    zeros = np.zeros((n, 3), dtype=np.float32)
    log_scales = np.log(scales.astype(np.float32) + 1e-8)
    rows = np.concatenate(
        [
            means.astype(np.float32),
            zeros,
            f_dc.astype(np.float32),
            opacity.astype(np.float32).reshape(-1, 1),
            log_scales,
            quats.astype(np.float32),
        ],
        axis=1,
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.astype("<f4").tobytes())
    return path
