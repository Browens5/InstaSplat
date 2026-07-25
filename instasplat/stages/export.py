"""Export Gaussian splats via PlayCanvas splat-transform."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from instasplat.config import ExportFormat, PipelineConfig
from instasplat.utils.deps import check_splat_transform
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger, run_cmd


@dataclass
class ExportResult:
    outputs: dict[str, Path]
    source_ply: Path


def _build_transform_args(cfg: PipelineConfig) -> list[str]:
    args: list[str] = []
    ex = cfg.export
    if ex.filter_nan:
        args.append("-N")
    if ex.translate is not None:
        tx, ty, tz = ex.translate
        args.extend(["-t", f"{tx},{ty},{tz}"])
    if ex.rotate_deg is not None:
        rx, ry, rz = ex.rotate_deg
        args.extend(["-r", f"{rx},{ry},{rz}"])
    if ex.scale is not None:
        args.extend(["-s", str(ex.scale)])
    if ex.filter_bands is not None:
        args.extend(["--filterBands", str(ex.filter_bands)])
    if ex.min_opacity is not None:
        args.extend(["-c", f"opacity,gt,{ex.min_opacity}"])
    return args


def run_export(cfg: PipelineConfig, paths: JobPaths, ply_path: Path | None) -> ExportResult:
    paths.ensure()
    log = get_logger("instasplat.export", paths.logs / "export.log")
    status = check_splat_transform(cfg.export.splat_transform_bin)
    st_bin = status.path or cfg.export.splat_transform_bin
    if not status.available and not cfg.dry_run:
        raise RuntimeError(status.notes or "splat-transform not found")

    if ply_path is None:
        # Search brush exports
        candidates = sorted(
            paths.brush_export.rglob("*.ply"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No PLY found under {paths.brush_export}. Run train stage first."
            )
        ply_path = candidates[0]

    # Always keep a canonical PLY copy in export dir
    out_dir = paths.export
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = out_dir / "scene.ply"
    outputs: dict[str, Path] = {}

    if not cfg.dry_run:
        if ply_path.resolve() != canonical.resolve():
            shutil.copy2(ply_path, canonical)
        outputs["ply"] = canonical
    else:
        outputs["ply"] = canonical

    formats = list(cfg.export.formats)
    if "ply" not in formats:
        formats = ["ply", *formats]

    for fmt in formats:
        if fmt == "ply":
            continue
        dest = _destination_for_format(out_dir, fmt)
        if dest.exists() and cfg.skip_existing and fmt != "ply":
            outputs[fmt] = dest
            continue
        cmd = [st_bin, str(canonical), *_build_transform_args(cfg), str(dest)]
        run_cmd(cmd, log_file=paths.logs / f"splat_transform_{fmt}.log", dry_run=cfg.dry_run)
        outputs[fmt] = dest
        log.info("Wrote %s", dest)

    # Hierarchical / streamed LOD (inspired by hierarchical-3d-gaussians + splat-transform)
    if cfg.export.streamed_lod:
        lod_dest = out_dir / "lod-meta.json"
        cmd = [st_bin, str(canonical), *_build_transform_args(cfg), str(lod_dest)]
        run_cmd(
            cmd,
            log_file=paths.logs / "splat_transform_lod.log",
            dry_run=cfg.dry_run,
            check=False,
        )
        outputs["lod-meta.json"] = lod_dest
        log.info("Wrote streamed LOD bundle → %s", lod_dest)

    if "spz" in outputs and cfg.export.spz_coordinate_note:
        (out_dir / "SPZ_COORDINATES.txt").write_text(
            "Niantic SPZ defaults to RUB (OpenGL/three.js). "
            "If viewers look rotated vs COLMAP/PLY (often RDF), convert axes in "
            "splat-transform or the SPZ pack/unpack options. See nianticlabs/spz.\n",
            encoding="utf-8",
        )

    # Manifest
    manifest = out_dir / "exports.txt"
    manifest.write_text(
        "\n".join(f"{k}: {v}" for k, v in outputs.items()) + "\n",
        encoding="utf-8",
    )
    return ExportResult(outputs, canonical)


def _destination_for_format(out_dir: Path, fmt: ExportFormat | str) -> Path:
    if fmt == "sog":
        return out_dir / "scene.sog"
    if fmt == "compressed.ply":
        return out_dir / "scene.compressed.ply"
    if fmt == "spz":
        return out_dir / "scene.spz"
    if fmt == "glb":
        return out_dir / "scene.glb"
    if fmt == "html":
        return out_dir / "scene.html"
    if fmt == "csv":
        return out_dir / "scene.csv"
    if fmt == "ply":
        return out_dir / "scene.ply"
    return out_dir / f"scene.{fmt}"
