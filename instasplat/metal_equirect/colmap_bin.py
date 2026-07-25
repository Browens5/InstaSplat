"""Minimal COLMAP binary readers (images.bin / cameras.bin) for metal_equirect."""

from __future__ import annotations

import struct
from pathlib import Path


def read_images_bin(path: Path) -> list[dict]:
    """
    Parse COLMAP images.bin → list of pose dicts (same keys as read_images_txt).

    See https://colmap.github.io/format.html#binary-file-format
    """
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if len(data) < 8:
        return []
    (n_images,) = struct.unpack_from("<Q", data, 0)
    off = 8
    images: list[dict] = []
    for _ in range(int(n_images)):
        # Format: uint64 image_id, 4×double q, 3×double t, uint32 camera_id,
        # null-terminated name, uint64 n_points2D, then points2D…
        if off + 8 + 32 + 24 + 4 > len(data):
            break
        (image_id,) = struct.unpack_from("<Q", data, off)
        off += 8
        qw, qx, qy, qz = struct.unpack_from("<dddd", data, off)
        off += 32
        tx, ty, tz = struct.unpack_from("<ddd", data, off)
        off += 24
        camera_id, = struct.unpack_from("<I", data, off)
        off += 4
        # name: null-terminated
        end = data.find(b"\x00", off)
        if end < 0:
            break
        name = data[off:end].decode("utf-8", errors="replace")
        off = end + 1
        (n2d,) = struct.unpack_from("<Q", data, off)
        off += 8
        # each point2D: double x, double y, uint64 point3D_id
        off += int(n2d) * 24
        if off > len(data):
            break
        images.append(
            {
                "image_id": int(image_id),
                "qw": float(qw),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "tx": float(tx),
                "ty": float(ty),
                "tz": float(tz),
                "camera_id": int(camera_id),
                "name": name,
            }
        )
    return images


def ensure_images_txt(model_dir: Path) -> Path | None:
    """Return path to images.txt, converting from images.bin when needed."""
    model_dir = Path(model_dir)
    txt = model_dir / "images.txt"
    if txt.exists():
        return txt
    bin_path = model_dir / "images.bin"
    if not bin_path.exists():
        return None
    images = read_images_bin(bin_path)
    if not images:
        return None
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for im in images:
        lines.append(
            f"{im['image_id']} {im['qw']} {im['qx']} {im['qy']} {im['qz']} "
            f"{im['tx']} {im['ty']} {im['tz']} {im['camera_id']} {im['name']}"
        )
        lines.append("")  # empty points2D line
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt
