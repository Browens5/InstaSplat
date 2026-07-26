"""Large-splat chunk extract/process and aligned merge stages."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.utils.align import (
    ChunkAlignment,
    align_chunk_to_world,
    align_overlap_sim3,
    compose_sim3,
    load_alignments,
    save_alignments,
)
from instasplat.utils.chunking import (
    ChunkManifest,
    ChunkPlan,
    load_telemetry_pair,
    plan_chunks,
    probe_duration_sec,
)
from instasplat.utils.metal import apply_metal_env, detect_metal
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger, run_cmd
from instasplat.utils.scale import camera_centers, read_images_txt
from instasplat.utils.telemetry import interpolate_xyz

if TYPE_CHECKING:
    from instasplat.utils.progress import ProgressEvent

ChunkProgressCb = Callable[[str, "ProgressEvent"], None]


@dataclass
class TiledResult:
    success: bool
    root: Path
    manifest: ChunkManifest | None
    chunk_results: dict[str, bool]
    merged_outputs: dict[str, Path]
    error: str | None = None


def chunks_root(paths: JobPaths) -> Path:
    return paths.root / "10_chunks"


def chunk_dir(paths: JobPaths, chunk_id: str) -> Path:
    return chunks_root(paths) / chunk_id


def run_plan_chunks(cfg: PipelineConfig, paths: JobPaths) -> ChunkManifest:
    paths.ensure()
    log = get_logger("instasplat.chunk", paths.logs / "chunk_plan.log")
    metal = detect_metal()
    log.info("Metal status: %s", metal.notes)

    video = paths.video if paths.video.exists() else cfg.input_path
    duration = probe_duration_sec(video)
    gyro, gps = load_telemetry_pair(
        paths.gyro_csv if paths.gyro_csv.exists() else None,
        paths.gps_csv if paths.gps_csv.exists() else None,
    )
    cc = cfg.chunk
    n_chunks = max(0, int(getattr(cc, "num_chunks", 0) or 0))
    manifest = plan_chunks(
        duration_sec=duration,
        chunk_duration_sec=cc.duration_sec,
        overlap_sec=cc.overlap_sec,
        base_fps=cc.base_fps,
        max_fps=cc.max_fps,
        max_frames_per_chunk=cc.max_frames_per_chunk,
        target_path_length_m=None if n_chunks > 0 else cc.target_path_length_m,
        num_chunks=n_chunks if n_chunks > 0 else None,
        gyro=gyro,
        gps=gps,
        source_fps_hint=cc.source_fps_hint,
    )
    out = chunks_root(paths) / "manifest.json"
    manifest.save(out)
    log.info(
        "Planned %d chunks over %.1fs (%s)%s",
        len(manifest.chunks),
        manifest.duration_sec,
        manifest.strategy,
        f" [num_chunks={n_chunks}]" if n_chunks > 0 else "",
    )
    return manifest


def _extract_chunk_window(
    cfg: PipelineConfig,
    video: Path,
    plan: ChunkPlan,
    window_path: Path,
) -> Path | None:
    """Extract one continuous chunk window so per-frame seeks stay local (8K-critical)."""
    if window_path.exists() and cfg.skip_existing:
        return window_path
    if cfg.dry_run:
        return window_path
    duration = max(0.1, plan.end_sec - plan.start_sec)
    # Prefer stream copy when keyframes allow; fall back to fast re-encode.
    copy_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{plan.start_sec:.4f}",
        "-i",
        str(video),
        "-t",
        f"{duration:.4f}",
        "-c",
        "copy",
        "-an",
        str(window_path),
    ]
    run_cmd(
        copy_cmd,
        log_file=window_path.parent / "ffmpeg_window.log",
        dry_run=False,
        check=False,
    )
    if window_path.exists() and window_path.stat().st_size > 1024:
        return window_path
    # Re-encode fallback (accurate trim for 8K Studio exports)
    enc_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{plan.start_sec:.4f}",
        "-i",
        str(video),
        "-t",
        f"{duration:.4f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-an",
        str(window_path),
    ]
    run_cmd(
        enc_cmd,
        log_file=window_path.parent / "ffmpeg_window_reencode.log",
        dry_run=False,
        check=False,
    )
    return window_path if window_path.exists() else None


def _extract_chunk_frames(
    cfg: PipelineConfig,
    video: Path,
    plan: ChunkPlan,
    out_dir: Path,
) -> list[Path]:
    """
    Extract adaptive frame times for one tile.

    Strategy for long 8K on Mac:
    1) Cut a short chunk window once (copy or veryfast x264)
    2) Seek within that window for each sample time (avoids full-file 8K seeks)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.jpg")) + sorted(out_dir.glob("*.png"))
    if existing and cfg.skip_existing and len(existing) >= max(1, int(0.8 * len(plan.frame_times))):
        mapping = {f"frame_{i:06d}": t for i, t in enumerate(plan.frame_times)}
        (out_dir.parent / "frame_times.json").write_text(
            json.dumps(mapping, indent=2),
            encoding="utf-8",
        )
        return existing

    frames: list[Path] = []
    ext = cfg.extract.image_format
    if cfg.dry_run:
        for i, t in enumerate(plan.frame_times):
            frames.append(out_dir / f"frame_{i:06d}.{ext}")
        mapping = {f"frame_{i:06d}": t for i, t in enumerate(plan.frame_times)}
        (out_dir.parent / "frame_times.json").write_text(
            json.dumps(mapping, indent=2),
            encoding="utf-8",
        )
        return frames

    window = out_dir.parent / "chunk_window.mp4"
    source = _extract_chunk_window(cfg, video, plan, window) or video
    use_relative = source == window

    for i, t in enumerate(plan.frame_times):
        out = out_dir / f"frame_{i:06d}.{ext}"
        if out.exists() and cfg.skip_existing:
            frames.append(out)
            continue
        seek = max(0.0, t - plan.start_sec) if use_relative else t
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{seek:.4f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ]
        run_cmd(cmd, check=False, dry_run=False)
        if out.exists():
            frames.append(out)

    mapping = {f"frame_{i:06d}": t for i, t in enumerate(plan.frame_times)}
    (out_dir.parent / "frame_times.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )
    return frames


def _chunk_pipeline_config(
    cfg: PipelineConfig,
    plan: ChunkPlan,
    parent_paths: JobPaths,
) -> PipelineConfig:
    """Build a per-chunk config that reuses parent video but writes into chunk dir."""
    cdir = chunk_dir(parent_paths, plan.chunk_id)
    child = replace(
        cfg,
        output_dir=cdir.parent,  # 10_chunks
        project_name=plan.chunk_id,
        stages=["extract", "mask", "sfm", "scale", "train", "export"],
    )
    # Dense adaptive sampling already chosen; disable flat fps extract path by
    # marking extract to use pre-extracted frames (handled outside Pipeline).
    child.extract.start_sec = plan.start_sec
    child.extract.end_sec = plan.end_sec
    child.extract.fps = cfg.chunk.base_fps
    child.extract.max_frames = len(plan.frame_times)
    # Prefer GPS scale when telemetry exists; otherwise keep none
    if parent_paths.gps_csv.exists():
        if child.scale.mode == "none":
            child.scale.mode = "gps"
        child.scale.gps_csv = parent_paths.gps_csv
    elif child.scale.mode == "gps":
        child.scale.mode = "none"
    # Metal defaults
    if cfg.metal.prefer_metal:
        child.mask.device = detect_metal().torch_device
        child.train.backend = "metal_equirect"
    return child


def _chunk_already_done(cdir: Path) -> bool:
    ply = _find_chunk_ply(cdir)
    return ply is not None and ply.exists()


def process_one_chunk(
    cfg: PipelineConfig,
    parent_paths: JobPaths,
    plan: ChunkPlan,
    *,
    on_progress: ChunkProgressCb | None = None,
) -> tuple[str, bool, str | None]:
    from instasplat.utils.control import get_controller

    log = get_logger("instasplat.chunk")
    get_controller().checkpoint(f"chunk:{plan.chunk_id}")
    cdir = chunk_dir(parent_paths, plan.chunk_id)
    cdir.mkdir(parents=True, exist_ok=True)
    if cfg.skip_existing and _chunk_already_done(cdir):
        log.info("Skipping %s — existing export PLY found", plan.chunk_id)
        return plan.chunk_id, True, None

    # Copy/link ingest telemetry + video pointer into chunk workspace
    chunk_paths = JobPaths(cdir)
    chunk_paths.ensure()
    video = parent_paths.video if parent_paths.video.exists() else cfg.input_path
    if not chunk_paths.video.exists() and video.exists() and not cfg.dry_run:
        try:
            chunk_paths.video.symlink_to(video.resolve())
        except OSError:
            # Avoid copying huge 8K files — write pointer instead
            (chunk_paths.ingest / "video_pointer.txt").write_text(str(video.resolve()), encoding="utf-8")
            if video.suffix.lower() in {".mp4", ".mov", ".insv"}:
                # For extract we need a readable path; symlink failure → use absolute
                pass

    # Pre-extract adaptive frames into chunk equirect folder
    source_video = video
    frames = _extract_chunk_frames(cfg, source_video, plan, chunk_paths.equirect_frames)
    if not frames and not cfg.dry_run:
        return plan.chunk_id, False, "no frames extracted"

    # Copy gyro/gps references
    for src, dst_name in (
        (parent_paths.gyro_csv, "gyro.csv"),
        (parent_paths.gps_csv, "gps.csv"),
        (parent_paths.metadata_json, "metadata.json"),
    ):
        if src.exists():
            dst = chunk_paths.ingest / dst_name
            if not dst.exists():
                shutil.copy2(src, dst)

    child_cfg = _chunk_pipeline_config(cfg, plan, parent_paths)
    # Scale before refine so GPS/gyro pose blend operates in metric-ish units
    child_cfg.stages = ["mask", "sfm", "scale", "refine", "train", "export"]
    # Ensure ingest video path exists for any tool that probes it
    if not chunk_paths.video.exists():
        (chunk_paths.ingest / "equirect_source.txt").write_text(str(source_video), encoding="utf-8")

    if cfg.dry_run:
        (cdir / "chunk_plan.json").write_text(
            json.dumps(
                {
                    "chunk_id": plan.chunk_id,
                    "start_sec": plan.start_sec,
                    "end_sec": plan.end_sec,
                    "frame_times": plan.frame_times,
                    "gps_start_xyz": plan.gps_start_xyz,
                    "gps_end_xyz": plan.gps_end_xyz,
                    "dry_run": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return plan.chunk_id, True, None

    env = apply_metal_env(cfg.metal.prefer_metal)
    # Pipeline subprocesses inherit via os.environ update in worker
    import os

    os.environ.update(
        {
            k: v
            for k, v in env.items()
            if k in ("PYTORCH_ENABLE_MPS_FALLBACK", "CUDA_VISIBLE_DEVICES")
        }
    )

    from instasplat.pipeline import Pipeline
    from instasplat.utils.progress import ProgressEvent

    def _forward(ev: ProgressEvent) -> None:
        if on_progress is None:
            return
        try:
            on_progress(plan.chunk_id, ev)
        except Exception:  # noqa: BLE001
            pass

    result = Pipeline(child_cfg, on_progress=_forward).run(stages=child_cfg.stages)
    # Persist plan alongside result
    (cdir / "chunk_plan.json").write_text(
        json.dumps(
            {
                "chunk_id": plan.chunk_id,
                "start_sec": plan.start_sec,
                "end_sec": plan.end_sec,
                "frame_times": plan.frame_times,
                "gps_start_xyz": plan.gps_start_xyz,
                "gps_end_xyz": plan.gps_end_xyz,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not result.success:
        log.error("Chunk %s failed: %s", plan.chunk_id, result.error)
        return plan.chunk_id, False, result.error
    return plan.chunk_id, True, None


def run_process_chunks(
    cfg: PipelineConfig,
    paths: JobPaths,
    manifest: ChunkManifest,
    *,
    on_chunk_progress: ChunkProgressCb | None = None,
) -> dict[str, bool]:
    log = get_logger("instasplat.chunk", paths.logs / "chunk_process.log")
    workers = max(1, cfg.chunk.max_parallel_chunks)
    if cfg.metal.prefer_metal and cfg.metal.serialize_train:
        # metal_equirect on MPS is memory-heavy; keep train serialized via workers=1
        # unless user raised max_parallel_chunks and disabled serialize_train.
        workers = min(workers, cfg.chunk.max_parallel_chunks)
        if cfg.metal.serialize_train:
            workers = 1
            log.info("Serializing chunk processing for Metal equirect train stability")

    results: dict[str, bool] = {}
    if workers == 1 or cfg.dry_run:
        for plan in manifest.chunks:
            cid, ok, err = process_one_chunk(
                cfg, paths, plan, on_progress=on_chunk_progress
            )
            results[cid] = ok
            if not ok:
                log.warning("%s failed: %s", cid, err)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    process_one_chunk,
                    cfg,
                    paths,
                    plan,
                    on_progress=on_chunk_progress,
                ): plan.chunk_id
                for plan in manifest.chunks
            }
            for fut in as_completed(futs):
                cid, ok, err = fut.result()
                results[cid] = ok
                if not ok:
                    log.warning("%s failed: %s", cid, err)
    (chunks_root(paths) / "chunk_status.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    ok_n = sum(1 for v in results.values() if v)
    fail_n = len(results) - ok_n
    log.info("Chunk processing: %d ok, %d failed (of %d)", ok_n, fail_n, len(results))
    if fail_n and not cfg.allow_partial_merge and not cfg.dry_run:
        failed = [k for k, v in results.items() if not v]
        raise RuntimeError(
            f"{fail_n} chunk(s) failed: {', '.join(failed)}. "
            "Re-run to resume successful tiles, or pass --allow-partial-merge."
        )
    return results


def _frame_times_map(chunk_path: Path) -> dict[str, float]:
    p = chunk_path / "01_frames" / "frame_times.json"
    if p.exists():
        return {k: float(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    # fallback from chunk_plan
    plan = chunk_path / "chunk_plan.json"
    if plan.exists():
        data = json.loads(plan.read_text(encoding="utf-8"))
        return {f"frame_{i:06d}": float(t) for i, t in enumerate(data.get("frame_times", []))}
    return {}


def _find_images_txt(chunk_path: Path) -> Path | None:
    candidates = [
        chunk_path / "04_scale" / "sparse" / "0" / "images.txt",
        chunk_path / "03_sfm" / "sparse" / "0_txt" / "images.txt",
        chunk_path / "03_sfm" / "sparse" / "0" / "images.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_chunk_ply(chunk_path: Path) -> Path | None:
    export = chunk_path / "06_export" / "scene.ply"
    if export.exists():
        return export
    plys = sorted(
        (chunk_path / "05_train" / "exports").rglob("*.ply"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if (chunk_path / "05_train" / "exports").exists() else []
    return plys[0] if plys else None


def run_align_chunks(cfg: PipelineConfig, paths: JobPaths, manifest: ChunkManifest) -> list[ChunkAlignment]:
    log = get_logger("instasplat.align", paths.logs / "align.log")
    gyro, gps = load_telemetry_pair(
        paths.gyro_csv if paths.gyro_csv.exists() else None,
        paths.gps_csv if paths.gps_csv.exists() else None,
    )
    alignments: list[ChunkAlignment] = []
    prev_centers_world: np.ndarray | None = None
    prev_align: ChunkAlignment | None = None

    for plan in manifest.chunks:
        cpath = chunk_dir(paths, plan.chunk_id)
        images_txt = _find_images_txt(cpath)
        fmap = _frame_times_map(cpath)
        fallback = (
            np.array(plan.gps_start_xyz, dtype=np.float64)
            if plan.gps_start_xyz
            else None
        )
        if images_txt is None:
            align = align_chunk_to_world(
                chunk_id=plan.chunk_id,
                colmap_images_txt=cpath / "missing_images.txt",
                frame_times_by_name=fmap,
                gps=gps,
                gyro=gyro,
                fallback_gps_start=fallback,
            )
        else:
            align = align_chunk_to_world(
                chunk_id=plan.chunk_id,
                colmap_images_txt=images_txt,
                frame_times_by_name=fmap,
                gps=gps,
                gyro=gyro,
                fallback_gps_start=fallback,
            )
            # Overlap refinement against previous chunk when GPS weak
            if prev_centers_world is not None and images_txt.exists():
                images = read_images_txt(images_txt)
                local = camera_centers(images)
                if len(local) >= 2 and align.n_anchors < 2:
                    # Map previous world centers into a Sim3 for this chunk via overlap heuristic
                    overlap_sim = align_overlap_sim3(prev_centers_world[-min(20, len(prev_centers_world)) :], local[: min(20, len(local))])
                    if prev_align is not None:
                        align = ChunkAlignment(
                            chunk_id=plan.chunk_id,
                            sim3=compose_sim3(prev_align.sim3, overlap_sim),
                            rmse_m=None,
                            method="overlap_chain",
                            n_anchors=min(20, len(local)),
                        )

        alignments.append(align)
        # Quality gate: if GPS alignment RMSE is terrible, fall back to overlap chain
        max_rmse = cfg.refine.max_align_rmse_m
        if (
            align.rmse_m is not None
            and align.rmse_m > max_rmse
            and prev_align is not None
            and images_txt is not None
            and images_txt.exists()
        ):
            log.warning(
                "%s align RMSE %.2fm > %.2fm — falling back to overlap chain",
                plan.chunk_id,
                align.rmse_m,
                max_rmse,
            )
            images = read_images_txt(images_txt)
            local = camera_centers(images)
            if prev_centers_world is not None and len(local) >= 2:
                overlap_sim = align_overlap_sim3(
                    prev_centers_world[-min(20, len(prev_centers_world)) :],
                    local[: min(20, len(local))],
                )
                align = ChunkAlignment(
                    chunk_id=plan.chunk_id,
                    sim3=compose_sim3(prev_align.sim3, overlap_sim),
                    rmse_m=None,
                    method="overlap_chain_rmse_reject",
                    n_anchors=min(20, len(local)),
                )
                alignments[-1] = align

        # Update prev world centers estimate
        if images_txt is not None and images_txt.exists():
            s, R, t = align.sim3.as_matrices()
            local = camera_centers(read_images_txt(images_txt))
            prev_centers_world = (s * (R @ local.T)).T + t
        elif gps is not None:
            prev_centers_world = np.stack(
                [interpolate_xyz(gps, t) for t in plan.frame_times[:20]],
                axis=0,
            )
        prev_align = align
        log.info(
            "Aligned %s method=%s anchors=%d rmse=%s",
            plan.chunk_id,
            align.method,
            align.n_anchors,
            align.rmse_m,
        )

    out = chunks_root(paths) / "alignments.json"
    save_alignments(out, alignments)
    return alignments


def run_merge_chunks(
    cfg: PipelineConfig,
    paths: JobPaths,
    manifest: ChunkManifest,
    alignments: list[ChunkAlignment] | None = None,
) -> dict[str, Path]:
    log = get_logger("instasplat.merge", paths.logs / "merge.log")
    if alignments is None:
        ap = chunks_root(paths) / "alignments.json"
        alignments = load_alignments(ap) if ap.exists() else []
    by_id = {a.chunk_id: a for a in alignments}

    merge_dir = paths.root / "11_merged"
    merge_dir.mkdir(parents=True, exist_ok=True)
    paths.export.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {"ply": paths.export / "scene.ply"}

    if cfg.dry_run:
        for fmt in cfg.export.formats:
            if fmt == "ply":
                continue
            outputs[fmt] = (
                paths.export / "scene.compressed.ply"
                if fmt == "compressed.ply"
                else paths.export / f"scene.{fmt}"
            )
        (merge_dir / "merged_manifest.json").write_text(
            json.dumps({k: str(v) for k, v in outputs.items()}, indent=2),
            encoding="utf-8",
        )
        log.info("Dry-run merge of %d chunks", len(manifest.chunks))
        return outputs

    # Warn if overlap is thin (on-the-fly-nvs / LongSplat capture guidance)
    if cfg.chunk.duration_sec > 0:
        ratio = cfg.chunk.overlap_sec / cfg.chunk.duration_sec
        if ratio < cfg.chunk.min_overlap_ratio:
            log.warning(
                "chunk.overlap_sec/duration_sec=%.2f < min_overlap_ratio=%.2f — "
                "tile seams may be hard to align",
                ratio,
                cfg.chunk.min_overlap_ratio,
            )

    transformed: list[Path] = []
    missing: list[str] = []

    st_bin = cfg.export.splat_transform_bin
    for plan in manifest.chunks:
        ply = _find_chunk_ply(chunk_dir(paths, plan.chunk_id))
        if ply is None:
            log.warning("No PLY for %s — skipping", plan.chunk_id)
            missing.append(plan.chunk_id)
            continue
        align = by_id.get(plan.chunk_id)
        dest = merge_dir / f"{plan.chunk_id}_world.ply"
        if dest.exists() and cfg.skip_existing:
            transformed.append(dest)
            continue
        if align is None:
            shutil.copy2(ply, dest)
            transformed.append(dest)
            continue
        args = [st_bin, str(ply), "-N", *align.sim3.to_splat_transform_args(), str(dest)]
        if not cfg.export.filter_nan:
            args = [st_bin, str(ply), *align.sim3.to_splat_transform_args(), str(dest)]
        run_cmd(args, log_file=paths.logs / f"merge_{plan.chunk_id}.log", dry_run=False, check=True)
        if not dest.exists():
            raise RuntimeError(f"splat-transform failed to write {dest}")
        transformed.append(dest)

    if missing and not cfg.allow_partial_merge:
        raise RuntimeError(
            f"Missing PLYs for chunks: {', '.join(missing)}. "
            "Fix failed tiles or pass --allow-partial-merge."
        )
    if not transformed:
        raise RuntimeError("No chunk PLYs available to merge")

    merged_ply = merge_dir / "scene_merged.ply"
    merge_cmd = [st_bin, *[str(p) for p in transformed]]
    # LongSplat-style prune: drop near-transparent Gaussians during merge
    prune = cfg.export.min_opacity or cfg.chunk.merge_prune_opacity
    if cfg.export.filter_nan:
        merge_cmd.append("-N")
    if prune and prune > 0:
        merge_cmd.extend(["-c", f"opacity,gt,{prune}"])
    merge_cmd.append(str(merged_ply))
    run_cmd(merge_cmd, log_file=paths.logs / "merge_all.log", dry_run=False, check=True)
    if not merged_ply.exists():
        raise RuntimeError(f"Merge failed — expected {merged_ply}")

    canonical = paths.export / "scene.ply"
    shutil.copy2(merged_ply, canonical)
    outputs["ply"] = canonical

    for fmt in cfg.export.formats:
        if fmt == "ply":
            continue
        dest = (
            paths.export / "scene.compressed.ply"
            if fmt == "compressed.ply"
            else paths.export / f"scene.{fmt}"
        )
        cmd = [st_bin, str(canonical)]
        if cfg.export.filter_nan:
            cmd.append("-N")
        if prune and prune > 0:
            cmd.extend(["-c", f"opacity,gt,{prune}"])
        cmd.append(str(dest))
        run_cmd(cmd, log_file=paths.logs / f"export_merged_{fmt}.log", dry_run=False, check=True)
        outputs[fmt] = dest

    if cfg.export.streamed_lod:
        lod_dest = paths.export / "lod-meta.json"
        cmd = [st_bin, str(canonical)]
        if cfg.export.filter_nan:
            cmd.append("-N")
        cmd.append(str(lod_dest))
        run_cmd(cmd, log_file=paths.logs / "export_merged_lod.log", dry_run=False, check=False)
        outputs["lod-meta.json"] = lod_dest

    if "spz" in outputs and cfg.export.spz_coordinate_note:
        (paths.export / "SPZ_COORDINATES.txt").write_text(
            "Niantic SPZ defaults to RUB (OpenGL/three.js). "
            "See docs/RESEARCH_STRATEGIES.md and nianticlabs/spz.\n",
            encoding="utf-8",
        )

    (merge_dir / "merged_manifest.json").write_text(
        json.dumps({k: str(v) for k, v in outputs.items()}, indent=2),
        encoding="utf-8",
    )
    log.info("Merged %d chunk splats → %s", len(transformed), canonical)
    return outputs
