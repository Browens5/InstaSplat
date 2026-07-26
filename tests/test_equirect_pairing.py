"""COLMAP ↔ equirect training view pairing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from instasplat.metal_equirect.dataset import load_equirect_dataset
from instasplat.utils.paths import JobPaths
from instasplat.utils.scale import apply_scale_to_model, read_images_txt


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((32, 64, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    cv2.imwrite(str(path), img)


def test_read_images_txt_with_empty_points2d(tmp_path: Path) -> None:
    """Empty POINTS2D lines must not drop every other image."""
    txt = tmp_path / "images.txt"
    txt.write_text(
        "# Image list with two lines of data per image:\n"
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "# POINTS2D[] as (X, Y, POINT3D_ID)\n"
        "1 1 0 0 0 0 0 0 1 frame_000001.jpg\n"
        "\n"
        "2 1 0 0 0 0 0 1 1 frame_000002.jpg\n"
        "\n"
        "3 1 0 0 0 0 0 2 1 frame_000003.jpg\n"
        "\n",
        encoding="utf-8",
    )
    images = read_images_txt(txt)
    assert [im["name"] for im in images] == [
        "frame_000001.jpg",
        "frame_000002.jpg",
        "frame_000003.jpg",
    ]


def test_load_dataset_from_images_equirect_only(tmp_path: Path) -> None:
    """Pairing works when panoramas live only under 03_sfm/images_equirect."""
    paths = JobPaths(tmp_path / "job")
    paths.ensure()
    # Leave 01_frames/equirect empty (common after ensure()); stage SfM copies only
    for i in (1, 2):
        _write_jpg(paths.equirect_sfm_images / f"frame_{i:06d}.jpg")
    paths.colmap_model.mkdir(parents=True, exist_ok=True)
    (paths.colmap_model / "cameras.txt").write_text(
        "1 EQUIRECTANGULAR 64 32 64 32\n",
        encoding="utf-8",
    )
    (paths.colmap_model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame_000001.jpg\n\n"
        "2 1 0 0 0 0 0 1 1 frame_000002.jpg\n\n",
        encoding="utf-8",
    )
    (paths.colmap_model / "points3D.txt").write_text(
        "1 0 0 1 128 128 128 0.1\n",
        encoding="utf-8",
    )
    ds = load_equirect_dataset(paths, paths.colmap_model, max_width=64)
    assert len(ds) == 2
    assert {v.name for v in ds.views} == {"frame_000001", "frame_000002"}


def test_load_dataset_with_path_prefix_in_colmap_name(tmp_path: Path) -> None:
    paths = JobPaths(tmp_path / "job")
    paths.ensure()
    _write_jpg(paths.equirect_frames / "frame_000001.jpg")
    paths.colmap_model.mkdir(parents=True, exist_ok=True)
    (paths.colmap_model / "cameras.txt").write_text(
        "1 EQUIRECTANGULAR 64 32 64 32\n",
        encoding="utf-8",
    )
    (paths.colmap_model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 images_equirect/frame_000001.jpg\n\n",
        encoding="utf-8",
    )
    (paths.colmap_model / "points3D.txt").write_text("1 0 0 1 1 1 1 0\n", encoding="utf-8")
    ds = load_equirect_dataset(paths, paths.colmap_model, max_width=64)
    assert len(ds) == 1


def test_load_dataset_pairs_e_prefix_with_frame_files(tmp_path: Path) -> None:
    """COLMAP e_000001.jpg must pair with on-disk frame_000001.jpg."""
    paths = JobPaths(tmp_path / "job")
    paths.ensure()
    for i in (1, 2, 3):
        _write_jpg(paths.equirect_frames / f"frame_{i:06d}.jpg")
        _write_jpg(paths.equirect_sfm_images / f"frame_{i:06d}.jpg")
    paths.colmap_model.mkdir(parents=True, exist_ok=True)
    (paths.colmap_model / "cameras.txt").write_text(
        "1 EQUIRECTANGULAR 64 32 64 32\n",
        encoding="utf-8",
    )
    (paths.colmap_model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 e_000001.jpg\n\n"
        "2 0.9 0.1 0.2 0.3 0 0 1 1 e_000002.jpg\n\n"
        "3 1 0 0 0 0 0 2 1 e_000003.jpg\n\n",
        encoding="utf-8",
    )
    (paths.colmap_model / "points3D.txt").write_text(
        "1 0 0 1 128 128 128 0.1\n",
        encoding="utf-8",
    )
    ds = load_equirect_dataset(paths, paths.colmap_model, max_width=64)
    assert len(ds) == 3
    assert {v.name for v in ds.views} == {
        "frame_000001",
        "frame_000002",
        "frame_000003",
    }
    assert all(v.image_path.name.startswith("frame_") for v in ds.views)
    assert int(getattr(ds, "paired_by_index", 0)) == 3


def test_load_dataset_from_official_images_bin(tmp_path: Path) -> None:
    """Real COLMAP images.bin (int32 ids) must yield usable poses + pairing."""
    import struct

    from instasplat.metal_equirect.colmap_bin import read_images_bin

    paths = JobPaths(tmp_path / "job")
    paths.ensure()
    _write_jpg(paths.equirect_frames / "frame_000001.jpg")
    paths.colmap_model.mkdir(parents=True, exist_ok=True)
    (paths.colmap_model / "cameras.txt").write_text(
        "1 EQUIRECTANGULAR 64 32 64 32\n",
        encoding="utf-8",
    )
    (paths.colmap_model / "points3D.txt").write_text("1 0 0 1 1 1 1 0\n", encoding="utf-8")
    name = b"e_000001.jpg\x00"
    body = b"".join(
        [
            struct.pack("<Q", 1),
            struct.pack(
                "<idddddddi",
                1,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1,
            ),
            name,
            struct.pack("<Q", 2),
            struct.pack("<ddq", 1.0, 2.0, -1),
            struct.pack("<ddq", 3.0, 4.0, -1),
        ]
    )
    (paths.colmap_model / "images.bin").write_bytes(body)
    imgs = read_images_bin(paths.colmap_model / "images.bin")
    assert imgs[0]["name"] == "e_000001.jpg"
    assert abs(imgs[0]["qw"] - 1.0) < 1e-12
    ds = load_equirect_dataset(paths, paths.colmap_model, max_width=64)
    assert len(ds) == 1
    assert ds.views[0].name == "frame_000001"
    np.testing.assert_allclose(ds.views[0].t_w2c, [0.0, 0.0, 1.0], atol=1e-5)


def test_qvec_normalize_prevents_overflow() -> None:
    from instasplat.utils.scale import qvec_to_rotmat

    # Huge unnormalized quat previously overflowed 2*x*x products
    R = qvec_to_rotmat(np.array([1e200, 0.0, 0.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
    R2 = qvec_to_rotmat(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(R2, np.eye(3), atol=1e-9)


def test_scale_preserves_all_images_with_empty_points(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "cameras.txt").write_text("1 EQUIRECTANGULAR 64 32 64 32\n", encoding="utf-8")
    (src / "images.txt").write_text(
        "1 1 0 0 0 1 0 0 1 a.jpg\n\n"
        "2 1 0 0 0 2 0 0 1 b.jpg\n\n",
        encoding="utf-8",
    )
    (src / "points3D.txt").write_text("1 1 0 0 0 0 0 0\n", encoding="utf-8")
    apply_scale_to_model(src, dst, 2.0)
    images = read_images_txt(dst / "images.txt")
    assert [im["name"] for im in images] == ["a.jpg", "b.jpg"]
    assert abs(float(images[0]["tx"]) - 2.0) < 1e-9
    assert abs(float(images[1]["tx"]) - 4.0) < 1e-9
