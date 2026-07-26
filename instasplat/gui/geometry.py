"""Load COLMAP sparse clouds and Gaussian PLY centers for the GUI viewer."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from instasplat.utils.hierarchy import _parse_gaussian_ply_xyz_opacity
from instasplat.utils.scale import read_points3d_txt


# 0-degree SH constant for converting f_dc_* → approximate RGB
_SH_C0 = 0.28209479177387814


@dataclass
class PointCloud:
    """XYZ + RGB in [0,1], optionally with per-point sizes."""

    xyz: np.ndarray  # (N, 3) float32
    rgb: np.ndarray  # (N, 3) float32
    source: str = ""
    kind: str = "points"  # points | splat

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0]) if self.xyz is not None else 0

    def subsample(self, max_points: int) -> PointCloud:
        if self.n <= max_points:
            return self
        idx = np.linspace(0, self.n - 1, max_points, dtype=np.int64)
        return PointCloud(self.xyz[idx], self.rgb[idx], self.source, self.kind)

    def for_display(self) -> PointCloud:
        """
        Flip for the GUI viewer.

        COLMAP / 3DGS world frames are Y-down relative to our OpenGL-style
        viewer (Y-up), so sparse clouds and splat centers appear upside down
        unless we negate Y at display time. Does not mutate training data.
        """
        if self.n == 0:
            return self
        xyz = np.asarray(self.xyz, dtype=np.float32).copy()
        xyz[:, 1] *= -1.0
        return PointCloud(xyz, self.rgb, self.source, self.kind)


def _read_points3d_txt_rgb(path: Path) -> PointCloud | None:
    """points3D.txt with XYZ + RGB columns."""
    if not path.exists():
        return None
    xyzs: list[list[float]] = []
    rgbs: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            # Fall back to xyz-only helper
            continue
        xyzs.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgbs.append([float(parts[4]) / 255.0, float(parts[5]) / 255.0, float(parts[6]) / 255.0])
    if not xyzs:
        # xyz-only
        pts = read_points3d_txt(path)
        if not pts:
            return None
        xyz = np.stack(list(pts.values()), axis=0).astype(np.float32)
        rgb = np.full((len(xyz), 3), 0.75, dtype=np.float32)
        return PointCloud(xyz, rgb, str(path), "points")
    xyz = np.asarray(xyzs, dtype=np.float32)
    rgb = np.asarray(rgbs, dtype=np.float32)
    return PointCloud(xyz, rgb, str(path), "points")


def _read_points3d_bin(path: Path, max_points: int = 2_000_000) -> PointCloud | None:
    """Parse COLMAP points3D.bin (little-endian)."""
    if not path.exists() or path.stat().st_size < 8:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 8:
        return None
    (n_points,) = struct.unpack_from("<Q", data, 0)
    off = 8
    n_points = min(int(n_points), max_points)
    xyz = np.zeros((n_points, 3), dtype=np.float32)
    rgb = np.zeros((n_points, 3), dtype=np.float32)
    kept = 0
    for _ in range(n_points):
        need = off + 8 + 24 + 3 + 8 + 8
        if need > len(data):
            break
        # point3D_id
        off += 8
        x, y, z = struct.unpack_from("<ddd", data, off)
        off += 24
        r, g, b = struct.unpack_from("<BBB", data, off)
        off += 3
        off += 8  # error
        (track_len,) = struct.unpack_from("<Q", data, off)
        off += 8
        off += int(track_len) * 8  # image_id + point2D_idx
        if off > len(data) + 1:
            break
        xyz[kept] = (x, y, z)
        rgb[kept] = (r / 255.0, g / 255.0, b / 255.0)
        kept += 1
    if kept == 0:
        return None
    return PointCloud(xyz[:kept], rgb[:kept], str(path), "points")


def load_colmap_sparse(model_dir: Path, max_points: int = 500_000) -> PointCloud | None:
    """Load sparse reconstruction from a COLMAP model folder."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return None
    # Prefer TXT (includes telemetry fallback); then BIN
    txt = model_dir / "points3D.txt"
    cloud = _read_points3d_txt_rgb(txt)
    if cloud is None:
        cloud = _read_points3d_bin(model_dir / "points3D.bin", max_points=max_points)
    if cloud is None:
        # Sibling *_txt export from model_converter
        parent = model_dir.parent
        for cand in sorted(parent.glob("*_txt")):
            cloud = _read_points3d_txt_rgb(cand / "points3D.txt")
            if cloud is not None:
                break
    if cloud is None:
        return None
    return cloud.subsample(max_points)


def _parse_gaussian_ply_xyz_rgb(path: Path, max_points: int = 800_000) -> PointCloud | None:
    """Load Gaussian PLY centers with approximate RGB from f_dc_* or rgb props."""
    if not path.exists() or path.stat().st_size < 64:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    header_end = raw.find(b"end_header\n")
    hdr_len = len(b"end_header\n")
    if header_end < 0:
        header_end = raw.find(b"end_header\r\n")
        if header_end < 0:
            # Fall back to xyz+opacity helper, gray color
            parsed = _parse_gaussian_ply_xyz_opacity(path, max_points=max_points)
            if parsed is None:
                return None
            xyz, opacity = parsed
            t = 1 / (1 + np.exp(-np.clip(opacity, -20, 20)))
            rgb = np.stack([t, t, t], axis=1).astype(np.float32)
            return PointCloud(xyz, rgb, str(path), "splat")
        hdr_len = len(b"end_header\r\n")
    header = raw[: header_end + hdr_len]
    body = raw[header_end + hdr_len :]
    header_txt = header.decode("utf-8", errors="replace")
    fmt = "ascii"
    n_verts = 0
    props: list[str] = []
    for line in header_txt.splitlines():
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        elif line.startswith("property "):
            props.append(line.split()[-1])
    if n_verts <= 0 or "x" not in props or "y" not in props or "z" not in props:
        return None
    n_verts = min(n_verts, max_points)
    ix, iy, iz = props.index("x"), props.index("y"), props.index("z")

    def _color_from_row(parts_or_arr) -> tuple[float, float, float]:
        if "red" in props and "green" in props and "blue" in props:
            r = float(parts_or_arr[props.index("red")])
            g = float(parts_or_arr[props.index("green")])
            b = float(parts_or_arr[props.index("blue")])
            if r > 1 or g > 1 or b > 1:
                return r / 255.0, g / 255.0, b / 255.0
            return r, g, b
        if "f_dc_0" in props and "f_dc_1" in props and "f_dc_2" in props:
            r = 0.5 + _SH_C0 * float(parts_or_arr[props.index("f_dc_0")])
            g = 0.5 + _SH_C0 * float(parts_or_arr[props.index("f_dc_1")])
            b = 0.5 + _SH_C0 * float(parts_or_arr[props.index("f_dc_2")])
            return (
                float(np.clip(r, 0, 1)),
                float(np.clip(g, 0, 1)),
                float(np.clip(b, 0, 1)),
            )
        if "opacity" in props:
            o = float(parts_or_arr[props.index("opacity")])
            t = 1 / (1 + np.exp(-np.clip(o, -20, 20)))
            return t, t, t
        return 0.85, 0.8, 0.65

    xyz = np.zeros((n_verts, 3), dtype=np.float32)
    rgb = np.zeros((n_verts, 3), dtype=np.float32)

    if fmt == "ascii":
        text = body.decode("utf-8", errors="replace").splitlines()
        for i in range(n_verts):
            if i >= len(text):
                break
            parts = text[i].split()
            if len(parts) <= max(ix, iy, iz):
                continue
            xyz[i] = (float(parts[ix]), float(parts[iy]), float(parts[iz]))
            rgb[i] = _color_from_row(parts)
        return PointCloud(xyz, rgb, str(path), "splat")

    if fmt.startswith("binary"):
        n_props = len(props)
        stride = 4 * n_props
        if stride <= 0 or len(body) < stride:
            return None
        n_verts = min(n_verts, len(body) // stride)
        arr = np.frombuffer(body[: n_verts * stride], dtype="<f4").reshape(n_verts, n_props)
        xyz = arr[:, [ix, iy, iz]].astype(np.float32, copy=False)
        rgb = np.zeros((n_verts, 3), dtype=np.float32)
        if "f_dc_0" in props and "f_dc_1" in props and "f_dc_2" in props:
            rgb[:, 0] = np.clip(0.5 + _SH_C0 * arr[:, props.index("f_dc_0")], 0, 1)
            rgb[:, 1] = np.clip(0.5 + _SH_C0 * arr[:, props.index("f_dc_1")], 0, 1)
            rgb[:, 2] = np.clip(0.5 + _SH_C0 * arr[:, props.index("f_dc_2")], 0, 1)
        elif "red" in props:
            rgb[:, 0] = arr[:, props.index("red")]
            rgb[:, 1] = arr[:, props.index("green")]
            rgb[:, 2] = arr[:, props.index("blue")]
            if rgb.max() > 1.5:
                rgb /= 255.0
        elif "opacity" in props:
            o = arr[:, props.index("opacity")]
            t = 1 / (1 + np.exp(-np.clip(o, -20, 20)))
            rgb[:, 0] = rgb[:, 1] = rgb[:, 2] = t
        else:
            rgb[:] = (0.85, 0.8, 0.65)
        return PointCloud(xyz, rgb.astype(np.float32), str(path), "splat")

    return None


def load_splat_ply(path: Path, max_points: int = 800_000) -> PointCloud | None:
    cloud = _parse_gaussian_ply_xyz_rgb(Path(path), max_points=max_points)
    if cloud is None:
        return None
    return cloud.subsample(max_points)


def find_latest_ply(root: Path) -> Path | None:
    if not root.exists():
        return None
    plys = sorted(root.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    return plys[0] if plys else None


def discover_colmap_model(job_root: Path) -> Path | None:
    """Pick the best available sparse model under a job."""
    from instasplat.utils.paths import JobPaths

    paths = JobPaths(job_root)
    candidates = [
        paths.root / "03b_refine" / "sparse" / "0",
        paths.scaled_model,
        paths.colmap_model,
        paths.colmap_sparse / "0_txt",
    ]
    # Also scan chunk models during tiled process
    if paths.chunks.is_dir():
        for c in sorted(paths.chunks.glob("chunk_*")):
            candidates.append(c / "03_sfm" / "sparse" / "0")
            candidates.append(c / "04_scale" / "sparse" / "0")
    for cand in candidates:
        if (cand / "points3D.bin").exists() or (cand / "points3D.txt").exists():
            return cand
    return None


def discover_splat_ply(job_root: Path) -> Path | None:
    from instasplat.utils.paths import JobPaths

    paths = JobPaths(job_root)
    for root in (paths.export, paths.merged, paths.brush_export):
        ply = find_latest_ply(root)
        if ply is not None:
            return ply
    if paths.chunks.is_dir():
        newest: Path | None = None
        newest_m = -1.0
        for c in paths.chunks.glob("chunk_*"):
            for sub in (c / "06_export", c / "05_train" / "exports"):
                ply = find_latest_ply(sub)
                if ply is not None and ply.stat().st_mtime > newest_m:
                    newest, newest_m = ply, ply.stat().st_mtime
        return newest
    return None
