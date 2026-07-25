"""CPU-side hierarchy / LOD packaging (Kerbl-inspired, no CUDA merger)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger


@dataclass
class HierarchyLodResult:
    lod_manifest: Path
    levels: list[dict[str, Any]]


def _parse_gaussian_ply_xyz_opacity(path: Path, max_points: int = 2_000_000) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Best-effort PLY reader for Gaussian xyz + opacity.

    Supports ASCII and binary_little_endian with common 3DGS property names.
    Returns None if the file is missing or unreadable.
    """
    if not path.exists() or path.stat().st_size < 64:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    header_end = raw.find(b"end_header\n")
    if header_end < 0:
        header_end = raw.find(b"end_header\r\n")
        if header_end < 0:
            return None
        header = raw[: header_end + len(b"end_header\r\n")]
        body = raw[header_end + len(b"end_header\r\n") :]
    else:
        header = raw[: header_end + len(b"end_header\n")]
        body = raw[header_end + len(b"end_header\n") :]

    header_txt = header.decode("utf-8", errors="replace")
    lines = header_txt.splitlines()
    fmt = "ascii"
    n_verts = 0
    props: list[str] = []
    for line in lines:
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        elif line.startswith("property "):
            props.append(line.split()[-1])
    if n_verts <= 0 or not props:
        return None
    n_verts = min(n_verts, max_points)

    name_x = "x" if "x" in props else None
    name_y = "y" if "y" in props else None
    name_z = "z" if "z" in props else None
    op_name = "opacity" if "opacity" in props else ("o" if "o" in props else None)
    if not (name_x and name_y and name_z):
        return None

    ix, iy, iz = props.index(name_x), props.index(name_y), props.index(name_z)
    io = props.index(op_name) if op_name else None

    if fmt == "ascii":
        text = body.decode("utf-8", errors="replace").splitlines()
        xyz = np.zeros((n_verts, 3), dtype=np.float32)
        opacity = np.ones(n_verts, dtype=np.float32)
        for i in range(n_verts):
            if i >= len(text):
                break
            parts = text[i].split()
            if len(parts) <= max(ix, iy, iz):
                continue
            xyz[i] = (float(parts[ix]), float(parts[iy]), float(parts[iz]))
            if io is not None and len(parts) > io:
                opacity[i] = float(parts[io])
        return xyz, opacity

    if fmt.startswith("binary"):
        # Assume float32 properties (standard 3DGS PLY)
        n_props = len(props)
        stride = 4 * n_props
        need = n_verts * stride
        if len(body) < need:
            n_verts = len(body) // stride
            if n_verts <= 0:
                return None
        arr = np.frombuffer(body[: n_verts * stride], dtype="<f4").reshape(n_verts, n_props)
        xyz = arr[:, [ix, iy, iz]].astype(np.float32, copy=False)
        opacity = arr[:, io].astype(np.float32, copy=False) if io is not None else np.ones(n_verts, np.float32)
        return xyz, opacity

    return None


def _write_xyz_opacity_ply(path: Path, xyz: np.ndarray, opacity: np.ndarray) -> None:
    """Write a minimal ASCII PLY (preview / coarse LOD, not full SH)."""
    n = len(xyz)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float opacity",
        "end_header",
    ]
    for i in range(n):
        lines.append(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {opacity[i]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cpu_lod_levels(
    hierarchy_manifest: Path,
    out_dir: Path,
    *,
    opacity_thresholds: list[float] | None = None,
    max_points_per_level: list[int] | None = None,
) -> HierarchyLodResult:
    """
    Build coarse LOD previews from hierarchy anchors.

    Level 0 = finest available tile PLYs (references only).
    Higher levels = opacity-pruned + subsampled XYZ previews for streaming UI.
    """
    opacity_thresholds = opacity_thresholds or [0.05, 0.15, 0.35]
    max_points_per_level = max_points_per_level or [500_000, 120_000, 30_000]
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(hierarchy_manifest.read_text(encoding="utf-8"))
    anchors = data.get("anchors") or []

    # Aggregate points from tile PLYs (best effort)
    all_xyz: list[np.ndarray] = []
    all_op: list[np.ndarray] = []
    for a in anchors:
        ply = a.get("ply")
        if not ply:
            continue
        parsed = _parse_gaussian_ply_xyz_opacity(Path(ply))
        if parsed is None:
            continue
        xyz, op = parsed
        # Softmax-ish: 3DGS opacity is often logit; map through sigmoid if |op|>1.5
        if np.nanmax(np.abs(op)) > 1.5:
            op = 1.0 / (1.0 + np.exp(-op))
        all_xyz.append(xyz)
        all_op.append(op)

    levels: list[dict[str, Any]] = [
        {
            "level": 0,
            "kind": "anchor_refs",
            "opacity_min": 0.0,
            "n_anchors": len(anchors),
            "anchors": [a.get("id") for a in anchors],
            "note": "Full tile PLYs — use splat-transform / Kerbl merger for production LOD",
        }
    ]

    if all_xyz:
        xyz = np.concatenate(all_xyz, axis=0)
        op = np.concatenate(all_op, axis=0)
        for i, (thr, cap) in enumerate(zip(opacity_thresholds, max_points_per_level), start=1):
            mask = op >= thr
            sel_xyz = xyz[mask]
            sel_op = op[mask]
            if len(sel_xyz) > cap:
                idx = np.linspace(0, len(sel_xyz) - 1, cap).astype(int)
                sel_xyz = sel_xyz[idx]
                sel_op = sel_op[idx]
            rel = f"lod_level_{i}.ply"
            _write_xyz_opacity_ply(out_dir / rel, sel_xyz, sel_op)
            levels.append(
                {
                    "level": i,
                    "kind": "cpu_preview",
                    "opacity_min": thr,
                    "max_points": cap,
                    "n_points": int(len(sel_xyz)),
                    "ply": rel,
                }
            )

    payload = {
        "type": "instasplat_lod_v1",
        "inspired_by": [
            "graphdeco-inria/hierarchical-3d-gaussians",
            "graphdeco-inria/on-the-fly-nvs",
        ],
        "source_hierarchy": str(hierarchy_manifest),
        "levels": levels,
        "note": (
            "CPU preview LODs are XYZ+opacity only for fast inspect. "
            "For full SH LOD trees, run the CUDA Kerbl hierarchy merger on cloud."
        ),
    }
    lod_path = out_dir / "lod_levels.json"
    lod_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return HierarchyLodResult(lod_path, levels)


def run_hierarchy_lod(cfg: PipelineConfig, paths: JobPaths) -> HierarchyLodResult | None:
    if not cfg.package.cpu_lod:
        return None
    hier = paths.merged / "hierarchy_manifest.json"
    if not hier.exists():
        return None
    log = get_logger("instasplat.hierarchy", paths.logs / "hierarchy_lod.log")
    out = paths.merged / "lod"
    if cfg.dry_run:
        return HierarchyLodResult(out / "lod_levels.json", [])
    result = build_cpu_lod_levels(hier, out)
    log.info("CPU LOD levels → %s (%d levels)", result.lod_manifest, len(result.levels))
    return result
