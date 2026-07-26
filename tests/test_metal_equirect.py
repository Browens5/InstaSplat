"""Tests for metal_equirect cameras, dataset, densify, and train smoke."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

from instasplat.metal_equirect.cameras import (
    EquirectCamera,
    unscented_equirect_project,
    yaw_pitch_to_rotmat,
)
from instasplat.metal_equirect.colmap_bin import ensure_images_txt, read_images_bin
from instasplat.metal_equirect.dataset import (
    _lift_cubemap_pose_to_equirect,
    load_equirect_dataset,
)
from instasplat.metal_equirect.densify import DensifyState, densify_and_prune
from instasplat.metal_equirect.gaussians import export_ply, gaussians_from_points
from instasplat.metal_equirect.metal_runtime import metal_status
from instasplat.metal_equirect.rasterize import photometric_loss, rasterize_equirect
from instasplat.metal_equirect.train_loop import train_equirect
from instasplat.utils.paths import JobPaths


def test_equirect_project_center() -> None:
    cam = EquirectCamera(200, 100)
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
    import cv2

    img = np.zeros((32, 64, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 120)
    cv2.imwrite(str(paths.equirect_frames / "frame_0001.jpg"), img)
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
        lines.append(f"{i} {0.1 * i} 0.0 {1.0 + 0.05 * i} 128 128 128 0.1")
    (paths.colmap_model / "points3D.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def test_load_equirect_dataset_from_cubemap_names(tmp_path: Path) -> None:
    paths = _write_mini_job(tmp_path / "job")
    ds = load_equirect_dataset(paths, paths.colmap_model, max_width=64)
    assert len(ds) == 1
    assert ds.views[0].name == "frame_0001"
    assert ds.views[0].width == 64
    assert ds.points_xyz.shape[0] == 20


def test_rasterize_tile_and_oit_backward() -> None:
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
    for mode in ("tile", "oit"):
        m = means.clone().detach().requires_grad_(True)
        pred = rasterize_equirect(
            m,
            quats,
            scales,
            opac,
            f_dc,
            f_rest,
            R,
            t,
            cam,
            sh_degree=0,
            max_gaussians=n,
            with_eval3d=True,
            composite=mode,
            tile_size=8,
            max_per_tile=16,
        )
        assert pred.shape == (24, 48, 3)
        loss = photometric_loss(pred, torch.zeros_like(pred), cam)
        loss.backward()
        assert m.grad is not None
        assert torch.isfinite(m.grad).all()


def test_train_smoke_exports_ply_and_preview(tmp_path: Path) -> None:
    paths = _write_mini_job(tmp_path / "job", n_pts=16)
    ds = load_equirect_dataset(paths, paths.colmap_model, max_width=64)
    for v in ds.views:
        v.width, v.height = 32, 16
    export_dir = tmp_path / "exports"
    events: list[dict] = []
    stats = train_equirect(
        ds,
        export_dir,
        total_steps=3,
        export_every=3,
        sh_degree=1,
        sh_warmup_steps=2,
        lr=0.05,
        max_gaussians_render=16,
        max_init_points=16,
        densify_every=0,
        with_eval3d=True,
        composite="oit",
        prefer_mps=False,
        preview_every=1,
        on_progress=events.append,
        log=None,
    )
    assert stats.ply_path is not None and stats.ply_path.exists()
    assert stats.preview_path is not None and stats.preview_path.exists()
    assert events and events[-1]["step"] == 3
    assert (export_dir / "previews").is_dir()


def test_densify_clone_split_prune() -> None:
    xyz = np.random.randn(40, 3).astype(np.float32)
    rgb = np.random.rand(40, 3).astype(np.float32)
    model = gaussians_from_points(xyz, rgb, sh_degree=1, max_points=40, device=torch.device("cpu"))
    state = DensifyState()
    state.reset(model.n, model.means.device)
    # Fake grads so densify triggers
    model.means.grad = torch.ones_like(model.means) * 0.01
    state.accumulate(model)
    stats = densify_and_prune(
        model,
        state,
        grad_threshold=0.001,
        max_gaussians=200,
        clone_scale_frac=0.05,
    )
    assert model.n >= 1
    assert isinstance(stats["cloned"], int)


def test_read_images_bin_roundtrip(tmp_path: Path) -> None:
    # Official COLMAP layout: int32 image_id + 7 doubles + int32 camera_id
    name = b"frame_0001_front.jpg\x00"
    body = b"".join(
        [
            struct.pack("<Q", 1),  # n_images
            struct.pack(
                "<idddddddi",
                1,  # image_id (int32)
                1.0,
                0.0,
                0.0,
                0.0,  # q
                0.0,
                0.0,
                1.0,  # t
                1,  # camera_id (int32)
            ),
            name,
            struct.pack("<Q", 0),  # n_points2D
        ]
    )
    model = tmp_path / "sparse"
    model.mkdir()
    (model / "images.bin").write_bytes(body)
    imgs = read_images_bin(model / "images.bin")
    assert len(imgs) == 1
    assert imgs[0]["name"] == "frame_0001_front.jpg"
    assert abs(imgs[0]["qw"] - 1.0) < 1e-12
    assert abs(imgs[0]["tz"] - 1.0) < 1e-12
    txt = ensure_images_txt(model)
    assert txt is not None and txt.exists()
    assert "frame_0001_front.jpg" in txt.read_text(encoding="utf-8")


def test_read_images_bin_with_points2d_and_e_names(tmp_path: Path) -> None:
    """Multi-image official bin must keep full names and finite quats."""
    images = [
        {
            "image_id": 1,
            "qw": 0.9,
            "qx": 0.1,
            "qy": 0.2,
            "qz": 0.3,
            "tx": 1.0,
            "ty": 2.0,
            "tz": 3.0,
            "camera_id": 1,
            "name": "e_000001.jpg",
        },
        {
            "image_id": 4,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "tz": 1.0,
            "camera_id": 1,
            "name": "e_000004.jpg",
        },
    ]
    path = tmp_path / "images.bin"
    parts = [struct.pack("<Q", 2)]
    for i, im in enumerate(images):
        parts.append(
            struct.pack(
                "<idddddddi",
                im["image_id"],
                im["qw"],
                im["qx"],
                im["qy"],
                im["qz"],
                im["tx"],
                im["ty"],
                im["tz"],
                im["camera_id"],
            )
        )
        parts.append(im["name"].encode() + b"\x00")
        n2d = 3 if i == 0 else 0
        parts.append(struct.pack("<Q", n2d))
        for _ in range(n2d):
            parts.append(struct.pack("<ddq", 12.0, 34.0, -1))
    path.write_bytes(b"".join(parts))
    imgs = read_images_bin(path)
    assert [im["name"] for im in imgs] == ["e_000001.jpg", "e_000004.jpg"]
    assert abs(imgs[0]["qw"] - 0.9) < 1e-12
    assert abs(imgs[0]["tz"] - 3.0) < 1e-12


def test_numpy_and_torch_quat_agree() -> None:
    from instasplat.metal_equirect.cameras import quat_to_rotmat_torch
    from instasplat.utils.scale import qvec_to_rotmat

    q = np.array([0.7, 0.1, -0.2, 0.4], dtype=np.float64)
    Rn = qvec_to_rotmat(q)
    Rt = quat_to_rotmat_torch(torch.tensor(q, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(Rn, Rt, atol=1e-9)
    np.testing.assert_allclose(Rn @ Rn.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(Rn), 1.0, atol=1e-9)


def test_export_ply_roundtrip(tmp_path: Path) -> None:
    xyz = np.random.randn(8, 3).astype(np.float32)
    rgb = np.random.rand(8, 3).astype(np.float32)
    model = gaussians_from_points(xyz, rgb, sh_degree=0, max_points=8, device=torch.device("cpu"))
    path = export_ply(model, tmp_path / "t.ply")
    data = path.read_bytes()
    assert b"end_header" in data
    assert b"f_dc_0" in data


def test_metal_status_dict() -> None:
    st = metal_status()
    assert "active_backend" in st
    assert st["active_backend"] == "torch_ut"
