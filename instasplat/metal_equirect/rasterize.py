"""Differentiable equirectangular Gaussian rasterizer (3DGUT UT projection)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from instasplat.metal_equirect.cameras import (
    EquirectCamera,
    build_covariance,
    unscented_equirect_project,
)


def _eval_sh_color(
    f_dc: torch.Tensor,
    f_rest: torch.Tensor,
    dirs: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """View-dependent color from DC (+ degree-1 SH). dirs: (N,3) camera-space unit."""
    # DC
    c0 = 0.28209479177387814
    rgb = 0.5 + c0 * f_dc
    if sh_degree < 1 or f_rest.numel() == 0:
        return rgb.clamp(0, 1)
    # Degree 1 SH constants
    c1 = 0.4886025119029199
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    # f_rest layout: 3 bands × 3 channels → (N, 9) as [1_0_r,g,b, 1_1_r,g,b, 1_2_r,g,b]
    rest = f_rest.view(-1, 3, 3)
    rgb = rgb + c1 * (-y).unsqueeze(-1) * rest[:, 0]
    rgb = rgb + c1 * z.unsqueeze(-1) * rest[:, 1]
    rgb = rgb + c1 * x.unsqueeze(-1) * rest[:, 2]
    return rgb.clamp(0, 1)


def rasterize_equirect(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    f_dc: torch.Tensor,
    f_rest: torch.Tensor,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    camera: EquirectCamera,
    *,
    sh_degree: int = 1,
    bbox_scale: float = 3.0,
    max_gaussians: int | None = 12_000,
) -> torch.Tensor:
    """
    Soft-alpha composite of Gaussians into an equirect RGB image (H, W, 3).

    Projection uses the Unscented Transform (3DGUT-inspired). Blending is a
    differentiable PyTorch implementation suitable for MPS/CPU; Metal kernels
    can replace the projection stage later without changing this API.
    """
    device = means.device
    dtype = means.dtype
    n = means.shape[0]
    if max_gaussians is not None and n > max_gaussians:
        # Keep densest / most opaque
        score = opacities.detach() * scales.detach().mean(dim=-1)
        idx = torch.topk(score, max_gaussians).indices
        means, quats, scales = means[idx], quats[idx], scales[idx]
        opacities, f_dc = opacities[idx], f_dc[idx]
        if f_rest.numel() > 0:
            f_rest = f_rest[idx]
        n = max_gaussians

    # World → camera
    means_cam = (R_w2c @ means.T).T + t_w2c
    cov_w = build_covariance(scales, quats)
    cov_cam = torch.einsum("ij,njk,lk->nil", R_w2c, cov_w, R_w2c)

    mean_2d, cov_2d, valid = unscented_equirect_project(means_cam, cov_cam, camera)
    dirs = means_cam / (torch.linalg.norm(means_cam, dim=-1, keepdim=True) + 1e-8)
    colors = _eval_sh_color(f_dc, f_rest, dirs, sh_degree)

    H, W = camera.height, camera.width
    image = torch.zeros(H, W, 3, device=device, dtype=dtype)
    T = torch.ones(H, W, device=device, dtype=dtype)  # transmittance

    # Sort near → far for front-to-back transmittance compositing
    depth = torch.linalg.norm(means_cam, dim=-1)
    order = torch.argsort(depth)
    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    c = cov_2d[:, 1, 1]
    tr = a + c
    det = a * c - b * b
    disc = torch.sqrt((tr * tr - 4 * det).clamp(min=0))
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    radius = bbox_scale * torch.sqrt(torch.maximum(l1, l2).clamp(min=1e-4))

    for i in order.tolist():
        if not bool(valid[i]):
            continue
        u0 = float(mean_2d[i, 0].detach())
        v0 = float(mean_2d[i, 1].detach())
        rad = float(radius[i].detach())
        if not math.isfinite(u0) or not math.isfinite(v0) or rad > max(H, W):
            continue
        u_min = int(max(0, math.floor(u0 - rad)))
        u_max = int(min(W, math.ceil(u0 + rad)))
        v_min = int(max(0, math.floor(v0 - rad)))
        v_max = int(min(H, math.ceil(v0 + rad)))
        if u_min >= u_max or v_min >= v_max:
            continue
        ys = torch.arange(v_min, v_max, device=device, dtype=dtype)
        xs = torch.arange(u_min, u_max, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        du = xx - mean_2d[i, 0]
        du = du - W * torch.round(du / W)
        dv = yy - mean_2d[i, 1]
        cov = cov_2d[i]
        det_i = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        if float(det_i.detach()) <= 1e-12:
            continue
        inv00 = cov[1, 1] / det_i
        inv11 = cov[0, 0] / det_i
        inv01 = -cov[0, 1] / det_i
        maha = inv00 * du * du + 2 * inv01 * du * dv + inv11 * dv * dv
        alpha = (opacities[i] * torch.exp(-0.5 * maha)).clamp(0, 0.99)

        # Out-of-place patch update (avoids autograd inplace errors)
        img_new = image.clone()
        T_new = T.clone()
        patch_T = T[v_min:v_max, u_min:u_max]
        weight = alpha * patch_T
        img_new[v_min:v_max, u_min:u_max] = (
            image[v_min:v_max, u_min:u_max] + weight.unsqueeze(-1) * colors[i]
        )
        T_new[v_min:v_max, u_min:u_max] = patch_T * (1.0 - alpha)
        image = img_new
        T = T_new

    return image


def photometric_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    camera: EquirectCamera,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Latitude-weighted L1 + coarse SSIM-style term."""
    w = camera.latitude_weights(pred.device, pred.dtype).expand_as(pred[..., 0])
    if mask is not None:
        # white = keep
        w = w * mask
    w = w / (w.mean() + 1e-8)
    l1 = (w.unsqueeze(-1) * (pred - gt).abs()).mean()
    # Simple local structure: downsample MSE
    pred_s = F.avg_pool2d(pred.permute(2, 0, 1).unsqueeze(0), 4)
    gt_s = F.avg_pool2d(gt.permute(2, 0, 1).unsqueeze(0), 4)
    ssim_like = 1.0 - F.cosine_similarity(pred_s.flatten(1), gt_s.flatten(1)).mean()
    return l1 + 0.2 * ssim_like
