"""Differentiable equirectangular Gaussian rasterizer (vectorized + optional Metal)."""

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


# Inria / 3DGS SH constants
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890651424703142,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445325712351571,
    -0.5900435899266435,
)

# Max half-extent (px) for vectorized soft splats — caps memory & work
_MAX_FOOTPRINT = 24
_OIT_CHUNK = 48


def _eval_sh_color(
    f_dc: torch.Tensor,
    f_rest: torch.Tensor,
    dirs: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """
    View-dependent color from DC + SH rest (degree 0–3).

    ``f_rest`` is flat ``(N, ((d+1)²−1)*3)`` laid out as bands-major RGB.
    """
    rgb = 0.5 + _SH_C0 * f_dc
    if sh_degree < 1 or f_rest.numel() == 0:
        return rgb.clamp(0, 1)

    n = f_dc.shape[0]
    bands = f_rest.shape[-1] // 3
    rest = f_rest.view(n, bands, 3)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    def _band(i: int) -> torch.Tensor:
        if i >= bands:
            return f_dc.new_zeros(n, 3)
        return rest[:, i]

    rgb = rgb + _SH_C1 * (-y).unsqueeze(-1) * _band(0)
    rgb = rgb + _SH_C1 * z.unsqueeze(-1) * _band(1)
    rgb = rgb + _SH_C1 * x.unsqueeze(-1) * _band(2)
    if sh_degree < 2:
        return rgb.clamp(0, 1)

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    rgb = rgb + _SH_C2[0] * xy.unsqueeze(-1) * _band(3)
    rgb = rgb + _SH_C2[1] * yz.unsqueeze(-1) * _band(4)
    rgb = rgb + _SH_C2[2] * (2.0 * zz - xx - yy).unsqueeze(-1) * _band(5)
    rgb = rgb + _SH_C2[3] * xz.unsqueeze(-1) * _band(6)
    rgb = rgb + _SH_C2[4] * (xx - yy).unsqueeze(-1) * _band(7)
    if sh_degree < 3:
        return rgb.clamp(0, 1)

    rgb = rgb + _SH_C3[0] * y.unsqueeze(-1) * (3.0 * xx - yy).unsqueeze(-1) * _band(8)
    rgb = rgb + _SH_C3[1] * xy.unsqueeze(-1) * z.unsqueeze(-1) * _band(9)
    rgb = rgb + _SH_C3[2] * y.unsqueeze(-1) * (4.0 * zz - xx - yy).unsqueeze(-1) * _band(10)
    rgb = rgb + _SH_C3[3] * z.unsqueeze(-1) * (2.0 * zz - 3.0 * xx - 3.0 * yy).unsqueeze(
        -1
    ) * _band(11)
    rgb = rgb + _SH_C3[4] * x.unsqueeze(-1) * (4.0 * zz - xx - yy).unsqueeze(-1) * _band(12)
    rgb = rgb + _SH_C3[5] * z.unsqueeze(-1) * (xx - yy).unsqueeze(-1) * _band(13)
    rgb = rgb + _SH_C3[6] * x.unsqueeze(-1) * (xx - 3.0 * yy).unsqueeze(-1) * _band(14)
    return rgb.clamp(0, 1)


def _eval3d_response(
    means_cam: torch.Tensor,
    cov_cam: torch.Tensor,
    opacities: torch.Tensor,
) -> torch.Tensor:
    det = torch.linalg.det(cov_cam).clamp(min=1e-12)
    peak = opacities * torch.pow(det, -1.0 / 6.0)
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


def _inv_cov2d_elements(
    cov_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Safe 2×2 inverse factors (inv00, inv01, inv11) + positive-det mask.

    UT projection can yield non-SPD 2D covariances near poles / wrap; those
    Gaussians contribute zero alpha instead of NaN grads.
    """
    # Symmetrize + jitter toward SPD
    a = cov_2d[:, 0, 0]
    b01 = 0.5 * (cov_2d[:, 0, 1] + cov_2d[:, 1, 0])
    c = cov_2d[:, 1, 1]
    eps = 1e-4
    a = a + eps
    c = c + eps
    det = a * c - b01 * b01
    ok = det > 1e-12
    det_safe = torch.where(ok, det, torch.ones_like(det))
    inv00 = torch.where(ok, c / det_safe, torch.zeros_like(det))
    inv11 = torch.where(ok, a / det_safe, torch.zeros_like(det))
    inv01 = torch.where(ok, -b01 / det_safe, torch.zeros_like(det))
    return inv00, inv01, inv11, ok


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
    composite: str = "oit",  # metal | oit | tile
    prefer_metal: bool = True,
) -> torch.Tensor:
    """
    Soft-alpha composite of Gaussians into an equirect RGB image (H, W, 3).

    - ``metal`` — fused soft-OIT (Metal forward / CPU-ref fallback; torch backward)
    - ``oit`` — vectorized torch soft-OIT; upgrades to ``metal`` when
      ``prefer_metal`` and a Metal (or forced-ref) fused path is requested
    - ``tile`` — tiled soft-OIT with per-tile top-K
    """
    if R_w2c.device != means.device or R_w2c.dtype != means.dtype:
        R_w2c = R_w2c.to(device=means.device, dtype=means.dtype)
    if t_w2c.device != means.device or t_w2c.dtype != means.dtype:
        t_w2c = t_w2c.to(device=means.device, dtype=means.dtype)

    n = means.shape[0]
    if max_gaussians is not None and n > max_gaussians:
        score = opacities.detach() * scales.detach().mean(dim=-1)
        idx = torch.topk(score, max_gaussians).indices
        means, quats, scales = means[idx], quats[idx], scales[idx]
        opacities, f_dc = opacities[idx], f_dc[idx]
        if f_rest.numel() > 0:
            f_rest = f_rest[idx]
        n = max_gaussians

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

    use_fused = composite == "metal" or (
        prefer_metal and composite == "oit" and means.device.type in {"mps", "cpu"}
    )
    # Only escalate oit→fused when real Metal is up, or composite explicitly metal.
    if use_fused and composite != "metal":
        try:
            from instasplat.metal_equirect.metal_runtime import metal_raster_available

            use_fused = metal_raster_available()
        except Exception:
            use_fused = False

    if use_fused or composite == "metal":
        try:
            from instasplat.metal_equirect.metal_runtime import fused_soft_oit

            return fused_soft_oit(
                mean_2d,
                cov_2d,
                radius,
                opac,
                colors,
                depth,
                valid,
                camera,
                footprint=_MAX_FOOTPRINT,
            )
        except Exception:
            if composite == "metal":
                # Explicit metal request: fall through to torch OIT rather than crash.
                pass

    if composite == "tile":
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
    return _composite_oit_batched(
        mean_2d, cov_2d, radius, opac, colors, depth, valid, camera
    )


def _composite_oit_batched(
    mean_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    radius: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    camera: EquirectCamera,
    *,
    max_footprint: int = _MAX_FOOTPRINT,
    chunk: int = _OIT_CHUNK,
) -> torch.Tensor:
    """
    Vectorized order-independent soft splat — no per-Gaussian Python loops,
    no ``.item()`` / ``float(tensor)`` GPU syncs in the inner path.
    """
    H, W = camera.height, camera.width
    device, dtype = mean_2d.device, mean_2d.dtype
    color_acc = torch.zeros(H * W, 3, device=device, dtype=dtype)
    weight_acc = torch.zeros(H * W, device=device, dtype=dtype)
    depth_w = torch.exp(-0.15 * depth.clamp(min=0.01))

    rad = radius.clamp(min=1.0, max=float(max_footprint))
    keep = valid & torch.isfinite(mean_2d).all(dim=-1) & torch.isfinite(rad) & (rad > 0.5)
    ids = torch.nonzero(keep, as_tuple=False).view(-1)
    if ids.numel() == 0:
        return color_acc.view(H, W, 3)

    span = torch.arange(-max_footprint, max_footprint + 1, device=device, dtype=dtype)

    for s in range(0, int(ids.numel()), chunk):
        sel = ids[s : s + chunk]
        b = int(sel.numel())
        mu = mean_2d[sel]
        cov = cov_2d[sel]
        op = opacities[sel] * depth_w[sel]
        col = colors[sel]
        r = rad[sel]

        cx = mu[:, 0]
        cy = mu[:, 1]
        py = cy[:, None, None] + span[None, :, None]
        px = cx[:, None, None] + span[None, None, :]

        du = px - cx[:, None, None]
        du = du - W * torch.round(du / W)
        dv = py - cy[:, None, None]

        inside = (du * du + dv * dv) <= (r[:, None, None] ** 2)
        py_round = torch.round(py)
        in_v = (py_round >= 0) & (py_round < H)
        inside = inside & in_v

        inv00, inv01, inv11, cov_ok = _inv_cov2d_elements(cov)
        maha = (
            inv00[:, None, None] * du * du
            + 2 * inv01[:, None, None] * du * dv
            + inv11[:, None, None] * dv * dv
        )
        maha = torch.nan_to_num(maha, nan=1e6, posinf=1e6, neginf=1e6).clamp(min=0)
        alpha = (op[:, None, None] * torch.exp(-0.5 * maha)).clamp(0, 0.99)
        alpha = torch.where(
            inside & cov_ok[:, None, None], alpha, torch.zeros_like(alpha)
        )

        px_i = torch.remainder(torch.round(px), W).long()
        py_i = py_round.long().clamp(0, H - 1)
        flat = (py_i * W + px_i).reshape(b, -1)
        w_flat = alpha.reshape(b, -1)
        color_acc = color_acc.index_add(
            0, flat.reshape(-1), (w_flat.unsqueeze(-1) * col[:, None, :]).reshape(-1, 3)
        )
        weight_acc = weight_acc.index_add(0, flat.reshape(-1), w_flat.reshape(-1))

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
    """
    Per-tile top-K soft-OIT (vectorized ``index_add``).

    Tile loops remain (few when ``tile_size`` is large early in training);
    Gaussian work inside each tile is batched — no per-splat Python / GPU syncs.
    """
    H, W = camera.height, camera.width
    device, dtype = mean_2d.device, mean_2d.dtype

    u0 = mean_2d[:, 0].detach()
    v0 = mean_2d[:, 1].detach()
    rad = radius.detach().clamp(max=float(_MAX_FOOTPRINT))
    u_min = torch.clamp((u0 - rad).floor().long(), 0, max(W - 1, 0))
    u_max = torch.clamp((u0 + rad).ceil().long(), 0, W)
    v_min = torch.clamp((v0 - rad).floor().long(), 0, max(H - 1, 0))
    v_max = torch.clamp((v0 + rad).ceil().long(), 0, H)

    n_tiles_x = max(1, math.ceil(W / tile_size))
    n_tiles_y = max(1, math.ceil(H / tile_size))
    # Depth weighting approximates front-to-back without a Python blend loop.
    depth_w = torch.exp(-0.15 * depth.clamp(min=0.01))
    row_tensors: list[torch.Tensor] = []
    span = torch.arange(-_MAX_FOOTPRINT, _MAX_FOOTPRINT + 1, device=device, dtype=dtype)

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
            color_acc = torch.zeros(th * tw, 3, device=device, dtype=dtype)
            weight_acc = torch.zeros(th * tw, device=device, dtype=dtype)
            if ids.numel() > 0:
                if ids.numel() > max_per_tile:
                    keep = torch.topk(depth[ids], max_per_tile, largest=False).indices
                    ids = ids[keep]
                b = int(ids.numel())
                mu = mean_2d[ids]
                cov = cov_2d[ids]
                op = opacities[ids] * depth_w[ids]
                col = colors[ids]
                r = radius[ids].clamp(min=1.0, max=float(_MAX_FOOTPRINT))

                py = mu[:, 1:2].unsqueeze(-1) + span[None, :, None]
                px = mu[:, 0:1].unsqueeze(-1) + span[None, None, :]
                du = px - mu[:, 0:1].unsqueeze(-1)
                du = du - W * torch.round(du / W)
                dv = py - mu[:, 1:2].unsqueeze(-1)
                inside = (du * du + dv * dv) <= (r[:, None, None] ** 2)
                py_round = torch.round(py)
                px_i = torch.remainder(torch.round(px), W).long()
                py_i = py_round.long()
                in_tile = (
                    inside
                    & (px_i >= x0)
                    & (px_i < x1)
                    & (py_i >= y0)
                    & (py_i < y1)
                )

                inv00, inv01, inv11, cov_ok = _inv_cov2d_elements(cov)
                maha = (
                    inv00[:, None, None] * du * du
                    + 2 * inv01[:, None, None] * du * dv
                    + inv11[:, None, None] * dv * dv
                )
                maha = torch.nan_to_num(maha, nan=1e6, posinf=1e6, neginf=1e6).clamp(
                    min=0
                )
                alpha = (op[:, None, None] * torch.exp(-0.5 * maha)).clamp(0, 0.99)
                alpha = torch.where(
                    in_tile & cov_ok[:, None, None], alpha, torch.zeros_like(alpha)
                )

                local = ((py_i - y0).clamp(0, th - 1) * tw + (px_i - x0).clamp(0, tw - 1))
                local = local.reshape(b, -1)
                w_flat = alpha.reshape(b, -1)
                # Zeroed alpha outside the tile; clamped local index is harmless.
                color_acc = color_acc.index_add(
                    0,
                    local.reshape(-1),
                    (w_flat.unsqueeze(-1) * col[:, None, :]).reshape(-1, 3),
                )
                weight_acc = weight_acc.index_add(0, local.reshape(-1), w_flat.reshape(-1))

            tile_img = (color_acc / (weight_acc.unsqueeze(-1) + 1e-6)).view(th, tw, 3)
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
