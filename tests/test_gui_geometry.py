"""Tests for GUI COLMAP / splat geometry loaders."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from instasplat.gui.geometry import (
    PointCloud,
    discover_colmap_model,
    discover_splat_ply,
    load_colmap_sparse,
    load_splat_ply,
)


def _write_points3d_txt(path: Path, n: int = 5) -> None:
    lines = [
        "# Point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for i in range(n):
        lines.append(
            f"{i} {float(i)} {float(i) * 0.5} {float(i) * -0.25} "
            f"{10 * i} {20 * i} {30 * i} 0.1 1 0 0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_points3d_bin(path: Path, n: int = 3) -> None:
    chunks = [struct.pack("<Q", n)]
    for i in range(n):
        chunks.append(struct.pack("<Q", i))
        chunks.append(struct.pack("<ddd", float(i), float(i + 1), float(i + 2)))
        chunks.append(struct.pack("<BBB", 255, 128, 64))
        chunks.append(struct.pack("<d", 0.01))
        chunks.append(struct.pack("<Q", 0))  # empty track
    path.write_bytes(b"".join(chunks))


def _write_ascii_gaussian_ply(path: Path, n: int = 4) -> None:
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "end_header\n"
    )
    body = "".join(
        f"{i}.0 {i * 0.1} {-i * 0.2} 1.0 0.0 -0.5 0.0\n" for i in range(n)
    )
    path.write_text(header + body, encoding="utf-8")


def test_load_colmap_sparse_txt(tmp_path: Path) -> None:
    model = tmp_path / "sparse" / "0"
    model.mkdir(parents=True)
    _write_points3d_txt(model / "points3D.txt", n=6)
    cloud = load_colmap_sparse(model)
    assert cloud is not None
    assert cloud.n == 6
    assert cloud.xyz.shape == (6, 3)
    assert cloud.rgb.shape == (6, 3)
    assert 0.0 <= float(cloud.rgb.min()) and float(cloud.rgb.max()) <= 1.0


def test_load_colmap_sparse_bin(tmp_path: Path) -> None:
    model = tmp_path / "sparse" / "0"
    model.mkdir(parents=True)
    _write_points3d_bin(model / "points3D.bin", n=4)
    cloud = load_colmap_sparse(model)
    assert cloud is not None
    assert cloud.n == 4
    np.testing.assert_allclose(cloud.xyz[1], [1.0, 2.0, 3.0], rtol=1e-5)


def test_load_splat_ply_ascii(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    _write_ascii_gaussian_ply(ply, n=5)
    cloud = load_splat_ply(ply)
    assert cloud is not None
    assert cloud.n == 5
    assert cloud.kind == "splat"
    assert cloud.xyz[2, 0] == 2.0


def test_discover_colmap_and_splat(tmp_path: Path) -> None:
    job = tmp_path / "job"
    sparse = job / "03_sfm" / "sparse" / "0"
    sparse.mkdir(parents=True)
    _write_points3d_txt(sparse / "points3D.txt", n=2)
    export = job / "06_export"
    export.mkdir(parents=True)
    _write_ascii_gaussian_ply(export / "export_final.ply", n=3)

    assert discover_colmap_model(job) == sparse
    assert discover_splat_ply(job) == export / "export_final.ply"


def test_subsample(tmp_path: Path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    _write_points3d_txt(model / "points3D.txt", n=50)
    cloud = load_colmap_sparse(model, max_points=10)
    assert cloud is not None
    assert cloud.n == 10


def test_for_display_flips_y() -> None:
    xyz = np.array([[1.0, 2.0, 3.0], [0.0, -4.0, 5.0]], dtype=np.float32)
    rgb = np.ones((2, 3), dtype=np.float32)
    cloud = PointCloud(xyz, rgb, source="t", kind="points")
    disp = cloud.for_display()
    np.testing.assert_allclose(disp.xyz[:, 0], xyz[:, 0])
    np.testing.assert_allclose(disp.xyz[:, 1], -xyz[:, 1])
    np.testing.assert_allclose(disp.xyz[:, 2], xyz[:, 2])
    # Original unchanged (training / reload path)
    np.testing.assert_allclose(cloud.xyz[:, 1], [2.0, -4.0])
