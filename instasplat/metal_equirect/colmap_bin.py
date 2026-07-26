"""Minimal COLMAP binary readers (images.bin / cameras.bin) for metal_equirect."""

from __future__ import annotations

import struct
from pathlib import Path


def read_images_bin(path: Path) -> list[dict]:
    """
    Parse COLMAP images.bin → list of pose dicts (same keys as read_images_txt).

    Layout matches COLMAP ``ReadImagesBinary`` / ``scripts/python/read_write_model.py``:
    ``uint64 n_images``, then per image ``idddddddi`` (int32 image_id, 4×double q,
    3×double t, int32 camera_id), null-terminated name, ``uint64 n_points2D``,
    then ``n_points2D × (double x, double y, uint64 point3D_id)``.

    NOTE: ``image_id`` is **int32**, not uint64. Reading it as 8 bytes misaligns
    the stream and produces garbage quaternions (overflows in ``qvec_to_rotmat``)
    and truncated names (e.g. ``e_000001.jpg`` → ``0001.jpg``).
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
        # 4 + 7*8 + 4 = 64 bytes of fixed properties
        if off + 64 > len(data):
            break
        (
            image_id,
            qw,
            qx,
            qy,
            qz,
            tx,
            ty,
            tz,
            camera_id,
        ) = struct.unpack_from("<idddddddi", data, off)
        off += 64
        end = data.find(b"\x00", off)
        if end < 0:
            break
        name = data[off:end].decode("utf-8", errors="replace")
        off = end + 1
        if off + 8 > len(data):
            break
        (n2d,) = struct.unpack_from("<Q", data, off)
        off += 8
        # each point2D: double x, double y, uint64 point3D_id
        need = int(n2d) * 24
        if off + need > len(data):
            break
        off += need
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


def write_images_bin(path: Path, images: list[dict]) -> None:
    """Write a minimal COLMAP-compatible images.bin (empty POINTS2D)."""
    path = Path(path)
    parts = [struct.pack("<Q", len(images))]
    for im in images:
        parts.append(
            struct.pack(
                "<idddddddi",
                int(im["image_id"]),
                float(im["qw"]),
                float(im["qx"]),
                float(im["qy"]),
                float(im["qz"]),
                float(im["tx"]),
                float(im["ty"]),
                float(im["tz"]),
                int(im["camera_id"]),
            )
        )
        parts.append(str(im["name"]).encode("utf-8") + b"\x00")
        parts.append(struct.pack("<Q", 0))
    path.write_bytes(b"".join(parts))


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
