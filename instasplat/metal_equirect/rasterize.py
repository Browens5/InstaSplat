"""Differentiable equirectangular Gaussian rasterizer (3DGUT UT + tile composite)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from instasplat.metal_equirect.cameras import (
    EquirectCamera,
    build_covariance,
    rotate_covariances,
    unscented_equirect_project,
)


def _eval_sh_color(
    f_dc: torch.Tensor,
    f_rest: torch.Tensor,
    dirs: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """View-dependent color from DC (+ degree-1 SH). dirs: (N,3) camera-space unit."""
    c0 = 0.28209479177387814
    rgb = 0.5 + c0 * f_dc
    if sh_degree < 1 or f_rest.numel() == 0:
        return rgb.clamp(0, 1)
    c1 = 0.4886025119029199
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    rest = f_rest.view(-1, 3, 3)
    rgb = rgb + c1 * (-y).unsqueeze(-1) * rest[:, 0]
    rgb = rgb + c1 * z.unsqueeze(-1) * rest[:, 1]
    rgb = rgb + c1 * x.unsqueeze(-1) * rest[:, 2]
    return rgb.clamp(0, 1)


def _eval3d_response(
    means_cam: torch.Tensor,
    cov_cam: torch.Tensor,
    opacities: torch.Tensor,
) -> torch.Tensor:
    """
    3DGUT-inspired 3D particle response along the view ray through the mean.

    Approximates the peak response of the 3D Gaussian on the ray from the
    camera origin through ``means_cam`` (used to modulate 2D splat opacity).
    """
    # Ray direction to mean
    d = means_cam / (torch.linalg.norm(means_cam, dim=-1, keepdim=True) + 1e-8)
    # Mahalanobis of mean under cov (self-response at center = 1); use
    # perpendicular scale: opacity * (det Σ)^{-1/6} proxy for peak density
    det = torch.linalg.det(cov_cam).clamp(min=1e-12)
    peak = opacities * torch.pow(det, -1.0 / 6.0)
    # Soften with distance so far particles don't dominate
    dist = torch.linalg.norm(means_cam, dim=-1).clamp(min=1e-3)
    return (peak / dist).clamp(0, 1.5) * (opacities / (opacities + 1e-6))


def _radii_from_cov2d(cov_2d: torch.Tensor, bbox_scale: float) -> torch.Tensor:
    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    c = cov_2d[:, 1, 1]
    tr = a + c
    det = a * c - b * b
    disc = torch.sqrt((tr * tr - 4 * det).clamp(min=0))
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    return bbox_scale * torch.sqrt(torch.maximum(l1, l2).clamp(min=1e-4))


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
    tile_size: int = 16,
    max_per_tile: int = 64,
    with_eval3d: bool = False,
    composite: str = "tile",  # tile | oit
) -> torch.Tensor:
    """
    Soft-alpha composite of Gaussians into an equirect RGB image (H, W, 3).

    - Projection: Unscented Transform (3DGUT-inspired)
    - ``composite='tile'``: depth-sorted blend inside tiles (default, quality)
    - ``composite='oit'``: order-independent weighted sum (faster)
    - ``with_eval3d``: modulate opacity by 3D particle response
    """
    device = means.device
    dtype = means.dtype
    n = means.shape[0]
    if max_gaussians is not None and n > max_gaussians:
        score = opacities.detach() * scales.detach().mean(dim=-1)
        idx = torch.topk(score, max_gaussians).indices
        means, quats, scales = means[idx], quats[idx], scales[idx]
        opacities, f_dc = opacities[idx], f_dc[idx]
        if f_rest.numel() > 0:
            f_rest = f_rest[idx]
        n = max_gaussians

    # Ensure camera pose tensors live with Gaussians (mixed CPU/MPS → MPS crash)
    if R_w2c.device != means.device or R_w2c.dtype != means.dtype:
        R_w2c = R_w2c.to(device=means.device, dtype=means.dtype)
    if t_w2c.device != means.device or t_w2c.dtype != means.dtype:
        t_w2c = t_w2c.to(device=means.device, dtype=means.dtype)

    means_cam = means @ R_w2c.transpose(0, 1) + t_w2c
    cov_w = build_covariance(scales, quats)
    cov_cam = rotate_covariances(R_w2c, cov_w)

    mean_2d, cov_2d, valid = unscented_equirect_project(means_cam, cov_cam, camera)
    dirs = means_cam / (torch.linalg.norm(means_cam, dim=-1, keepdim=True) + 1e-8)
    colors = _eval_sh_color(f_dc, f_rest, dirs, sh_degree)
    depth = torch.linalg.norm(means_cam, dim=-1)
    radius = _radii_from_cov2d(cov_2d, bbox_scale)

    opac = opacities
    if with_eval3d:
        opac = (opac * _eval3d_response(means_cam, cov_cam, opacities)).clamp(0, 0.99)

    if composite == "oit":
        return _composite_oit(
            mean_2d, cov_2d, radius, opac, colors, depth, valid, camera
        )
    return _composite_tiled(
        mean_2d,
        cov_2d,
        radius,
        opac,
        colors,
        depth,
        valid,
        camera,
        tile_size=tile_size,
        max_per_tile=max_per_tile,
    )


def _gaussian_alpha_patch(
    mean_2d_i: torch.Tensor,
    cov_2d_i: torch.Tensor,
    opacity_i: torch.Tensor,
    ys: torch.Tensor,
    xs: torch.Tensor,
    width: int,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    du = xx - mean_2d_i[0]
    du = du - width * torch.round(du / width)
    dv = yy - mean_2d_i[1]
    det_i = cov_2d_i[0, 0] * cov_2d_i[1, 1] - cov_2d_i[0, 1] * cov_2d_i[1, 0]
    if float(det_i.detach()) <= 1e-12:
        return torch.zeros_like(du)
    inv00 = cov_2d_i[1, 1] / det_i
    inv11 = cov_2d_i[0, 0] / det_i
    inv01 = -cov_2d_i[0, 1] / det_i
    maha = inv00 * du * du + 2 * inv01 * du * dv + inv11 * dv * dv
    return (opacity_i * torch.exp(-0.5 * maha)).clamp(0, 0.99)


def _composite_oit(
    mean_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    radius: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    camera: EquirectCamera,
) -> torch.Tensor:
    """Order-independent transparency: depth-weighted sum / normalize."""
    H, W = camera.height, camera.width
    device, dtype = mean_2d.device, mean_2d.dtype
    color_acc = torch.zeros(H * W, 3, device=device, dtype=dtype)
    weight_acc = torch.zeros(H * W, device=device, dtype=dtype)
    # Closer particles get higher weight
    depth_w = torch.exp(-0.15 * depth.clamp(min=0.01))

    n = mean_2d.shape[0]
    for i in range(n):
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
        alpha = _gaussian_alpha_patch(mean_2d[i], cov_2d[i], opacities[i], ys, xs, W)
        w = alpha * depth_w[i]
        yy, xx = torch.meshgrid(
            torch.arange(v_min, v_max, device=device),
            torch.arange(u_min, u_max, device=device),
            indexing="ij",
        )
        flat = (yy * W + xx).reshape(-1)
        w_flat = w.reshape(-1)
        col = w_flat.unsqueeze(-1) * colors[i].unsqueeze(0)
        color_acc = color_acc.index_add(0, flat, col)
        weight_acc = weight_acc.index_add(0, flat, w_flat)

    img = color_acc / (weight_acc.unsqueeze(-1) + 1e-6)
    return img.view(H, W, 3).clamp(0, 1)


def _composite_tiled(
    mean_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    radius: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    camera: EquirectCamera,
    *,
    tile_size: int,
    max_per_tile: int,
) -> torch.Tensor:
    """Depth-sorted alpha compositing inside tiles; assemble via cat (autograd-safe)."""
    H, W = camera.height, camera.width
    device, dtype = mean_2d.device, mean_2d.dtype

    u0 = mean_2d[:, 0].detach()
    v0 = mean_2d[:, 1].detach()
    rad = radius.detach()
    u_min = torch.clamp((u0 - rad).floor().long(), 0, max(W - 1, 0))
    u_max = torch.clamp((u0 + rad).ceil().long(), 0, W)
    v_min = torch.clamp((v0 - rad).floor().long(), 0, max(H - 1, 0))
    v_max = torch.clamp((v0 + rad).ceil().long(), 0, H)

    n_tiles_x = max(1, math.ceil(W / tile_size))
    n_tiles_y = max(1, math.ceil(H / tile_size))
    row_tensors: list[torch.Tensor] = []

    for ty in range(n_tiles_y):
        col_tensors: list[torch.Tensor] = []
        for tx in range(n_tiles_x):
            x0 = tx * tile_size
            y0 = ty * tile_size
            x1 = min(W, x0 + tile_size)
            y1 = min(H, y0 + tile_size)
            hit = (
                valid
                & (u_max > x0)
                & (u_min < x1)
                & (v_max > y0)
                & (v_min < y1)
            )
            ids = torch.nonzero(hit, as_tuple=False).view(-1)
            th, tw = y1 - y0, x1 - x0
            tile_img = torch.zeros(th, tw, 3, device=device, dtype=dtype)
            if ids.numel() > 0:
                if ids.numel() > max_per_tile:
                    dsel = depth[ids]
                    keep = torch.topk(dsel, max_per_tile, largest=False).indices
                    ids = ids[keep]
                ids = ids[torch.argsort(depth[ids])]
                tile_T = torch.ones(th, tw, device=device, dtype=dtype)
                ys = torch.arange(y0, y1, device=device, dtype=dtype)
                xs = torch.arange(x0, x1, device=device, dtype=dtype)
                for i in ids.tolist():
                    alpha = _gaussian_alpha_patch(
                        mean_2d[i], cov_2d[i], opacities[i], ys, xs, W
                    )
                    weight = alpha * tile_T
                    tile_img = tile_img + weight.unsqueeze(-1) * colors[i]
                    tile_T = tile_T * (1.0 - alpha)
            col_tensors.append(tile_img)
        row_tensors.append(torch.cat(col_tensors, dim=1))

    image = torch.cat(row_tensors, dim=0)
    return image[:H, :W].clamp(0, 1)


def photometric_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    camera: EquirectCamera,
    mask: torch.Tensor | None = None,
    *,
    ssim_weight: float = 0.2,
) -> torch.Tensor:
    """Latitude-weighted L1 + coarse SSIM-style term."""
    w = camera.latitude_weights(pred.device, pred.dtype).expand_as(pred[..., 0])
    if mask is not None:
        w = w * mask
    w = w / (w.mean() + 1e-8)
    l1 = (w.unsqueeze(-1) * (pred - gt).abs()).mean()
    pred_s = F.avg_pool2d(pred.permute(2, 0, 1).unsqueeze(0), 4)
    gt_s = F.avg_pool2d(gt.permute(2, 0, 1).unsqueeze(0), 4)
    ssim_like = 1.0 - F.cosine_similarity(pred_s.flatten(1), gt_s.flatten(1)).mean()
    return l1 + ssim_weight * ssim_like
