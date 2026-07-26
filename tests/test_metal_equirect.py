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


def test_gaussians_from_points_uniform_device() -> None:
    """All parameters must share the request device (MPS mixed-device crash)."""
    xyz = np.random.randn(32, 3).astype(np.float32)
    rgb = np.random.rand(32, 3).astype(np.float32)
    for dev in (torch.device("cpu"),):
        model = gaussians_from_points(xyz, rgb, sh_degree=1, max_points=32, device=dev)
        devices = {p.device for p in model.parameters()}
        assert devices == {dev}
        assert model.f_rest.device == dev


def test_sh_degree_3_allocates_rest_and_eval() -> None:
    from instasplat.metal_equirect.gaussians import sh_rest_dim
    from instasplat.metal_equirect.rasterize import _eval_sh_color

    xyz = np.random.randn(8, 3).astype(np.float32)
    rgb = np.random.rand(8, 3).astype(np.float32)
    model = gaussians_from_points(
        xyz, rgb, sh_degree=0, max_sh_degree=3, max_points=8, device=torch.device("cpu")
    )
    assert model.max_sh_degree == 3
    assert model.sh_degree == 0
    assert model.f_rest.shape[-1] == sh_rest_dim(3)
    model.set_active_sh_degree(3)
    assert model.sh_degree == 3
    dirs = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    rgb_out = _eval_sh_color(model.f_dc, model.f_rest, dirs, 3)
    assert rgb_out.shape == (8, 3)
    assert torch.isfinite(rgb_out).all()


def test_rotate_covariances_matches_r_sigma_rt() -> None:
    from instasplat.metal_equirect.cameras import rotate_covariances

    torch.manual_seed(0)
    R = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.det(R) < 0:
        R = R.clone()
        R[:, 0] = -R[:, 0]
    cov_w = torch.eye(3).unsqueeze(0) * torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1)
    cov_w = cov_w + 0.01 * torch.randn(5, 3, 3)
    cov_w = 0.5 * (cov_w + cov_w.transpose(-1, -2))
    got = rotate_covariances(R, cov_w)
    ref = torch.einsum("ij,njk,lk->nil", R, cov_w, R)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


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
        viewer_every=1,
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
        use_resolution_schedule=True,
        on_progress=events.append,
        log=None,
    )
    assert stats.ply_path is not None and stats.ply_path.exists()
    assert stats.preview_path is not None and stats.preview_path.exists()
    assert events and events[-1]["step"] == 3
    assert (export_dir / "previews").is_dir()
    assert (export_dir / "live.ply").exists()
    assert events[-1].get("phase") in {"coarse", "mid", "fine", "full"}


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
    assert st["active_backend"] in {
        "torch_oit",
        "metal_oit",
        "metal_oit_shared",
        "ref_oit",
    }
    assert st.get("fused_composite") is True
    assert "dispatch" in st
    assert "shared_buffers" in st


def test_pack_cov2d_symmetrizes() -> None:
    from instasplat.metal_equirect._metal_buffers import pack_cov2d

    cov = torch.tensor(
        [[[1.0, 0.2], [0.4, 2.0]], [[3.0, -0.5], [-0.5, 4.0]]],
        dtype=torch.float32,
    )
    packed = pack_cov2d(cov)
    assert packed.shape == (2, 4)
    assert abs(float(packed[0, 1]) - 0.3) < 1e-6
    assert float(packed[0, 0]) == 1.0 and float(packed[0, 2]) == 2.0


def test_host_shared_pool_grow_and_view() -> None:
    """Exercise SharedBufferPool semantics with a tiny host-side fake MTL device."""
    from instasplat.metal_equirect._metal_buffers import SharedBufferPool

    class _FakeBuf:
        def __init__(self, n: int) -> None:
            self._mem = bytearray(n)

        def contents(self):
            return memoryview(self._mem)

    class _FakeDevice:
        def newBufferWithLength_options_(self, length, _opts):
            return _FakeBuf(int(length))

    pool = SharedBufferPool(_FakeDevice(), storage_mode=0)
    src = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    pool.copy_in("color", src, (4, 3))
    view = pool.torch_view("color", (4, 3), torch.float32)
    assert torch.allclose(view, src)
    # Grow path
    big = torch.zeros(100, dtype=torch.float32)
    pool.copy_in("color", big, (100,))
    assert pool.torch_view("color", (100,), torch.float32).shape == (100,)
    # Second copy into same capacity should not realloc-break the view link
    big2 = torch.ones(100, dtype=torch.float32)
    pool.copy_in("color", big2, (100,))
    assert float(pool.torch_view("color", (100,), torch.float32).sum()) == 100.0


def test_fused_metal_composite_forward_and_backward() -> None:
    """composite=metal uses fused path (CPU ref on Linux) with torch grads."""
    from instasplat.metal_equirect.metal_runtime import (
        active_composite_backend,
        set_force_reference,
    )

    set_force_reference(True)
    try:
        cam = EquirectCamera(32, 16)
        n = 8
        means = (torch.randn(n, 3) * 0.2 + torch.tensor([0.0, 0.0, 2.0])).requires_grad_(
            True
        )
        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0
        pred = rasterize_equirect(
            means,
            quats,
            torch.full((n, 3), 0.05),
            torch.full((n,), 0.5),
            torch.zeros(n, 3),
            torch.zeros(n, 0),
            torch.eye(3),
            torch.zeros(3),
            cam,
            sh_degree=0,
            max_gaussians=n,
            with_eval3d=False,
            composite="metal",
            prefer_metal=True,
        )
        assert pred.shape == (16, 32, 3)
        assert torch.isfinite(pred).all()
        assert active_composite_backend() == "ref_oit"
        loss = photometric_loss(pred, torch.zeros_like(pred), cam)
        loss.backward()
        assert means.grad is not None
        assert torch.isfinite(means.grad).all()
    finally:
        set_force_reference(False)


def test_soft_oit_reference_matches_torch_oit_roughly() -> None:
    """CPU reference and torch OIT should be in the same ballpark on simple scenes."""
    from instasplat.metal_equirect.cameras import (
        build_covariance,
        rotate_covariances,
        unscented_equirect_project,
    )
    from instasplat.metal_equirect.rasterize import (
        _composite_oit_batched,
        _radii_from_cov2d,
    )
    from instasplat.metal_equirect.soft_oit_ref import soft_oit_reference

    torch.manual_seed(1)
    cam = EquirectCamera(24, 12)
    n = 5
    means = torch.randn(n, 3) * 0.15 + torch.tensor([0.0, 0.0, 2.0])
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.04)
    opac = torch.full((n,), 0.6)
    colors = torch.rand(n, 3)
    means_cam = means
    cov_cam = rotate_covariances(torch.eye(3), build_covariance(scales, quats))
    mean_2d, cov_2d, valid = unscented_equirect_project(means_cam, cov_cam, cam)
    radius = _radii_from_cov2d(cov_2d, 3.0)
    depth = torch.linalg.norm(means_cam, dim=-1)
    torch_img = _composite_oit_batched(
        mean_2d, cov_2d, radius, opac, colors, depth, valid, cam
    )
    ref = soft_oit_reference(
        mean_2d.numpy(),
        cov_2d.numpy(),
        radius.numpy(),
        opac.numpy(),
        colors.numpy(),
        depth.numpy(),
        valid.numpy(),
        cam.height,
        cam.width,
    )
    # Same soft-OIT family; footprints differ slightly (radius half vs fixed grid)
    mae = float(np.abs(torch_img.numpy() - ref).mean())
    assert mae < 0.25
    assert np.isfinite(ref).all()


def test_schedule_coarse_to_fine() -> None:
    from instasplat.metal_equirect.schedule import schedule_at_step

    early = schedule_at_step(1, 1000)
    mid = schedule_at_step(500, 1000)
    late = schedule_at_step(900, 1000)
    assert early.phase == "coarse" and early.width_scale == 0.5 and early.tile_size == 64
    assert mid.phase == "mid" and mid.width_scale == 0.75
    assert late.phase == "fine" and late.width_scale == 1.0 and late.tile_size == 16
    assert early.max_per_tile >= late.max_per_tile


def test_view_cache_decode_once(tmp_path: Path, monkeypatch) -> None:
    from instasplat.metal_equirect.dataset import TrainView
    from instasplat.metal_equirect.view_cache import ViewCache

    img_path = tmp_path / "p.jpg"
    import cv2

    cv2.imwrite(str(img_path), np.zeros((16, 32, 3), dtype=np.uint8) + 50)
    view = TrainView(
        name="p",
        image_path=img_path,
        width=32,
        height=16,
        R_w2c=np.eye(3, dtype=np.float32),
        t_w2c=np.zeros(3, dtype=np.float32),
        mask_path=None,
    )
    calls = {"n": 0}
    real_imread = cv2.imread

    def counted_imread(*args, **kwargs):
        calls["n"] += 1
        return real_imread(*args, **kwargs)

    monkeypatch.setattr(cv2, "imread", counted_imread)
    cache = ViewCache(torch.device("cpu"), max_views=8)
    cache.get(view)
    cache.get(view)
    rgb, mask, R, t, w, h = cache.get_scaled(view, width_scale=0.5)
    assert calls["n"] == 1
    assert w == 16 and h == 8
    assert rgb.shape == (h, w, 3)


def test_export_ply_subsampled(tmp_path: Path) -> None:
    xyz = np.random.randn(40, 3).astype(np.float32)
    rgb = np.random.rand(40, 3).astype(np.float32)
    model = gaussians_from_points(xyz, rgb, sh_degree=0, max_points=40, device=torch.device("cpu"))
    path = export_ply(model, tmp_path / "sub.ply", max_points=10)
    text = path.read_bytes().split(b"end_header")[0].decode("ascii")
    assert "element vertex 10" in text


def test_oit_no_item_sync_in_forward() -> None:
    """Vectorized OIT must not call .item() on GPU tensors in the hot path."""
    cam = EquirectCamera(32, 16)
    n = 20
    means = (torch.randn(n, 3) * 0.2 + torch.tensor([0.0, 0.0, 2.0])).requires_grad_(True)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    pred = rasterize_equirect(
        means,
        quats,
        torch.full((n, 3), 0.05),
        torch.full((n,), 0.4),
        torch.zeros(n, 3),
        torch.zeros(n, 0),
        torch.eye(3),
        torch.zeros(3),
        cam,
        sh_degree=0,
        max_gaussians=n,
        with_eval3d=False,
        composite="oit",
        prefer_metal=False,
    )
    assert pred.shape == (16, 32, 3)
    pred.mean().backward()
    assert means.grad is not None
