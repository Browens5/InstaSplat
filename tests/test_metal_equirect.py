"""Tests for metal_equirect cameras, dataset lifting, and short train smoke."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from instasplat.metal_equirect.cameras import (
    EquirectCamera,
    unscented_equirect_project,
    yaw_pitch_to_rotmat,
)
from instasplat.metal_equirect.dataset import (
    _lift_cubemap_pose_to_equirect,
    load_equirect_dataset,
)
from instasplat.metal_equirect.gaussians import export_ply, gaussians_from_points
from instasplat.metal_equirect.rasterize import photometric_loss, rasterize_equirect
from instasplat.metal_equirect.train_loop import train_equirect
from instasplat.utils.paths import JobPaths


def test_equirect_project_center() -> None:
    cam = EquirectCamera(200, 100)
    # +Z forward → image center
    dirs = torch.tensor([[0.0, 0.0, 1.0]])
    uv = cam.project_dirs(dirs)
    assert abs(float(uv[0, 0]) - 100.0) < 1.0
    assert abs(float(uv[0, 1]) - 50.0) < 1.0


def test_unscented_project_finite() -> None:
    cam = EquirectCamera(128, 64)
    means = torch.tensor([[0.1, 0.0, 2.0], [-0.5, 0.2, 1.5]])
    cov = torch.eye(3).unsqueeze(0).expand(2, 3, 3) * 0.01
    mean_2d, cov_2d, valid = unscented_equirect_project(means, cov, cam)
    assert mean_2d.shape == (2, 2)
    assert cov_2d.shape == (2, 2, 2)
    assert bool(valid.all())
    assert torch.isfinite(mean_2d).all()


def test_yaw_pitch_rot_front_is_identity() -> None:
    R = yaw_pitch_to_rotmat(0.0, 0.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


def test_lift_front_pose_identity_face() -> None:
    R = np.eye(3, dtype=np.float64)
    t = np.array([0.0, 0.0, 1.0])
    R_eq, t_eq = _lift_cubemap_pose_to_equirect(R, t, "front")
    np.testing.assert_allclose(R_eq, R, atol=1e-6)
    np.testing.assert_allclose(t_eq, t, atol=1e-6)


def _write_mini_job(root: Path, *, n_pts: int = 20) -> JobPaths:
    paths = JobPaths(root)
    paths.equirect_frames.mkdir(parents=True)
    paths.colmap_model.mkdir(parents=True)
    # Tiny equirect PNG via numpy + cv2
    import cv2

    img = np.zeros((32, 64, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 120)
    cv2.imwrite(str(paths.equirect_frames / "frame_0001.jpg"), img)
    # Cubemap-named COLMAP image (front)
    (paths.colmap_model / "cameras.txt").write_text(
        "# Camera list\n1 SIMPLE_PINHOLE 64 64 50 32 32\n",
        encoding="utf-8",
    )
    (paths.colmap_model / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 frame_0001_front.jpg\n"
        "\n",
        encoding="utf-8",
    )
    lines = ["# points"]
    for i in range(n_pts):
        lines.append(
            f"{i} {0.1 * i} 0.0 {1.0 + 0.05 * i} 128 128 128 0.1"
        )
    (paths.colmap_model / "points3D.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def test_load_equirect_dataset_from_cubemap_names(tmp_path: Path) -> None:
    paths = _write_mini_job(tmp_path / "job")
    ds = load_equirect_dataset(paths, paths.colmap_model, max_width=64)
    assert len(ds) == 1
    assert ds.views[0].name == "frame_0001"
    assert ds.views[0].width == 64
    assert ds.points_xyz.shape[0] == 20


def test_rasterize_and_loss_backward() -> None:
    cam = EquirectCamera(48, 24)
    n = 12
    means = torch.randn(n, 3) * 0.3 + torch.tensor([0.0, 0.0, 2.0])
    means.requires_grad_(True)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.05)
    opac = torch.full((n,), 0.4)
    f_dc = torch.zeros(n, 3)
    f_rest = torch.zeros(n, 0)
    R = torch.eye(3)
    t = torch.zeros(3)
    pred = rasterize_equirect(
        means, quats, scales, opac, f_dc, f_rest, R, t, cam, sh_degree=0, max_gaussians=n
    )
    assert pred.shape == (24, 48, 3)
    gt = torch.zeros_like(pred)
    loss = photometric_loss(pred, gt, cam)
    loss.backward()
    assert means.grad is not None
    assert torch.isfinite(means.grad).all()


def test_train_smoke_exports_ply(tmp_path: Path) -> None:
    paths = _write_mini_job(tmp_path / "job", n_pts=16)
    ds = load_equirect_dataset(paths, paths.colmap_model, max_width=64)
    # Force very small views for speed
    for v in ds.views:
        v.width, v.height = 32, 16
    export_dir = tmp_path / "exports"
    stats = train_equirect(
        ds,
        export_dir,
        total_steps=3,
        export_every=3,
        sh_degree=0,
        lr=0.05,
        max_gaussians_render=16,
        max_init_points=16,
        densify_every=0,
        prefer_mps=False,
        log=None,
    )
    assert stats.ply_path is not None and stats.ply_path.exists()
    assert stats.ply_path.stat().st_size > 64
    assert stats.n_gaussians > 0


def test_export_ply_roundtrip(tmp_path: Path) -> None:
    xyz = np.random.randn(8, 3).astype(np.float32)
    rgb = np.random.rand(8, 3).astype(np.float32)
    model = gaussians_from_points(xyz, rgb, sh_degree=0, max_points=8, device=torch.device("cpu"))
    path = export_ply(model, tmp_path / "t.ply")
    data = path.read_bytes()
    assert b"end_header" in data
    assert b"f_dc_0" in data
