"""End-to-end pipeline orchestration (single-scene and tiled large-splat)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from instasplat.config import PipelineConfig
from instasplat.stages.export import ExportResult, run_export
from instasplat.stages.extract import ExtractResult, run_extract
from instasplat.stages.fallback import run_fallback_poses
from instasplat.stages.ingest import IngestResult, run_ingest
from instasplat.stages.mask import MaskResult, run_mask
from instasplat.stages.refine import RefineResult, run_refine
from instasplat.stages.scale import ScaleResult, run_scale
from instasplat.stages.sfm import SfMResult, run_sfm
from instasplat.stages.tiled import (
    ChunkManifest,
    TiledResult,
    chunks_root,
    run_align_chunks,
    run_merge_chunks,
    run_plan_chunks,
    run_process_chunks,
)
from instasplat.stages.train import TrainResult, run_train
from instasplat.utils.chunking import ChunkManifest as _ChunkManifest
from instasplat.utils.control import (
    PipelineStopped,
    RunController,
    reset_controller,
    set_controller,
)
from instasplat.utils.metal import detect_metal
from instasplat.utils.nerfstudio import PackageResult, run_package
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger
from instasplat.utils.progress import (
    HeartbeatPublisher,
    ProgressEvent,
    StageProgressTracker,
    format_duration,
)

ProgressCb = Callable[[ProgressEvent], None]


@dataclass
class PipelineResult:
    success: bool
    paths: JobPaths
    stage_timings: dict[str, float] = field(default_factory=dict)
    ingest: IngestResult | None = None
    extract: ExtractResult | None = None
    mask: MaskResult | None = None
    sfm: SfMResult | None = None
    refine: RefineResult | None = None
    scale: ScaleResult | None = None
    train: TrainResult | None = None
    export: ExportResult | None = None
    package: PackageResult | None = None
    tiled: TiledResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        def _safe(obj: Any) -> Any:
            if obj is None:
                return None
            if hasattr(obj, "__dataclass_fields__"):
                raw = asdict(obj)
                return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}
            return str(obj)

        return {
            "success": self.success,
            "root": str(self.paths.root),
            "stage_timings": self.stage_timings,
            "error": self.error,
            "ingest": _safe(self.ingest),
            "extract": _safe(self.extract),
            "mask": _safe(self.mask),
            "sfm": _safe(self.sfm),
            "refine": _safe(self.refine),
            "scale": _safe(self.scale),
            "train": _safe(self.train),
            "export": {
                "outputs": {k: str(v) for k, v in self.export.outputs.items()},
                "source_ply": str(self.export.source_ply),
            }
            if self.export
            else None,
            "package": _safe(self.package),
            "tiled": {
                "success": self.tiled.success,
                "root": str(self.tiled.root),
                "chunk_results": self.tiled.chunk_results,
                "merged_outputs": {k: str(v) for k, v in self.tiled.merged_outputs.items()},
                "error": self.tiled.error,
                "manifest_chunks": (
                    len(self.tiled.manifest.chunks) if self.tiled.manifest else 0
                ),
            }
            if self.tiled
            else None,
        }


class Pipeline:
    STAGE_ORDER: ClassVar[list[str]] = [
        "ingest",
        "extract",
        "mask",
        "sfm",
        "scale",
        "refine",
        "train",
        "export",
        "package",
        "plan_chunks",
        "preflight",
        "process_chunks",
        "align_chunks",
        "merge_chunks",
    ]

    TILED_ORDER: ClassVar[list[str]] = [
        "ingest",
        "plan_chunks",
        "preflight",
        "process_chunks",
        "align_chunks",
        "merge_chunks",
        "package",
    ]

    def __init__(
        self,
        cfg: PipelineConfig,
        on_progress: ProgressCb | None = None,
        controller: RunController | None = None,
    ):
        self.cfg = cfg
        self.paths = JobPaths(cfg.work_dir())
        self.on_progress = on_progress or (lambda _ev: None)
        self.controller = controller or RunController()
        self.log = get_logger("instasplat.pipeline")
        self._manifest: ChunkManifest | _ChunkManifest | None = None
        self._chunk_status: dict[str, bool] = {}
        self._alignments = None
        self._tracker: StageProgressTracker | None = None

    def run(self, stages: list[str] | None = None) -> PipelineResult:
        token = set_controller(self.controller)
        try:
            return self._run_inner(stages)
        finally:
            reset_controller(token)

    def _run_inner(self, stages: list[str] | None = None) -> PipelineResult:
        self._apply_metal_defaults()
        self.paths.ensure()
        self.cfg.save(self.paths.config)

        if stages is None:
            wanted = list(self.cfg.stages) if self.cfg.stages else None
            if wanted:
                # Explicit stage list (from --only / stage / --from--to): honor order
                # even if a single-mode stage is requested on a tiled job.
                selected = [s for s in self.STAGE_ORDER if s in wanted]
                if not selected:
                    selected = wanted
            elif self.cfg.mode == "tiled" or self.cfg.chunk.enabled:
                selected = list(self.TILED_ORDER)
            else:
                selected = [
                    "ingest",
                    "extract",
                    "mask",
                    "sfm",
                    "scale",
                    "refine",
                    "train",
                    "export",
                    "package",
                ]
        else:
            selected = [s for s in self.STAGE_ORDER if s in stages]
            if not selected:
                selected = list(stages)

        result = PipelineResult(success=False, paths=self.paths)
        self._tracker = StageProgressTracker(stages=selected)
        try:
            for i, name in enumerate(selected):
                self.controller.checkpoint(name)
                ev = self._tracker.start_stage(name, i)
                self.on_progress(ev)
                self.log.info(
                    "▶ %s (%d/%d) — ETA %s",
                    name,
                    i + 1,
                    len(selected),
                    format_duration(ev.stage_eta_sec),
                )
                hb = HeartbeatPublisher(self._tracker, self.on_progress, interval_sec=1.0)
                hb.start()
                try:
                    self._run_stage(name, result)
                finally:
                    hb.stop()
                # Refresh chunk count after plan for better ETAs
                if name == "plan_chunks" and self._manifest is not None:
                    self._tracker.chunk_count = len(self._manifest.chunks)
                done = self._tracker.finish_stage(name, i)
                result.stage_timings[name] = done.stage_elapsed_sec
                self.on_progress(done)
                self.log.info(
                    "■ %s finished in %s (overall elapsed %s, overall ETA %s)",
                    name,
                    format_duration(done.stage_elapsed_sec),
                    format_duration(done.overall_elapsed_sec),
                    format_duration(done.overall_eta_sec),
                )
            result.success = True
        except PipelineStopped as exc:
            self.log.warning("Pipeline stopped: %s", exc)
            result.error = str(exc)
            result.success = False
        except Exception as exc:
            self.log.exception("Pipeline failed")
            result.error = str(exc)
            result.success = False

        summary = self.paths.root / "result.json"
        summary.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    def _apply_metal_defaults(self) -> None:
        if not self.cfg.metal.prefer_metal:
            return
        status = detect_metal()
        if self.cfg.metal.torch_device:
            self.cfg.mask.device = self.cfg.metal.torch_device
        elif status.mps_available:
            self.cfg.mask.device = "mps"
        else:
            self.cfg.mask.device = "cpu"
        self.log.info("Metal defaults: %s (YOLO device=%s)", status.notes, self.cfg.mask.device)

    def _active_model(self, result: PipelineResult) -> Path:
        if result.refine is not None and result.refine.model_dir.exists():
            return result.refine.model_dir
        if result.scale is not None and result.scale.model_dir.exists():
            return result.scale.model_dir
        if result.sfm is not None:
            return result.sfm.model_dir
        refined = self.paths.root / "03b_refine" / "sparse" / "0"
        if refined.exists():
            return refined
        return self.paths.colmap_model

    def _run_stage(self, name: str, result: PipelineResult) -> None:
        cfg, paths = self.cfg, self.paths
        if name == "ingest":
            result.ingest = run_ingest(cfg, paths)
        elif name == "extract":
            result.extract = run_extract(cfg, paths)
        elif name == "mask":
            result.mask = run_mask(cfg, paths)
        elif name == "sfm":
            try:
                result.sfm = run_sfm(cfg, paths)
            except Exception as exc:
                self.log.warning("SfM failed (%s); trying telemetry fallback", exc)
                fb = run_fallback_poses(cfg, paths)
                if fb is None or fb.n_poses <= 0:
                    raise
                result.sfm = SfMResult(
                    model_dir=fb.model_dir,
                    image_dir=paths.cubemap_images,
                    mask_dir=paths.cubemap_masks if any(paths.cubemap_masks.glob("*.png")) else None,
                    mode="telemetry_fallback",
                    num_images=fb.n_poses,
                )
            else:
                # If mapper produced nothing, still try fallback
                model = result.sfm.model_dir
                has = (model / "images.bin").exists() or (model / "images.txt").exists()
                if not has and not cfg.dry_run:
                    fb = run_fallback_poses(cfg, paths)
                    if fb is not None and fb.n_poses > 0:
                        result.sfm = SfMResult(
                            model_dir=fb.model_dir,
                            image_dir=paths.cubemap_images,
                            mask_dir=None,
                            mode="telemetry_fallback",
                            num_images=fb.n_poses,
                        )
                    elif not has:
                        raise RuntimeError(
                            "SfM produced no usable model (and telemetry fallback had 0 poses). "
                            "For tiled jobs run process_chunks; for single-mode use "
                            "perspective_cubemap after extract/mask."
                        )
        elif name == "scale":
            # Scale before refine so GPS/gyro blend uses metric-ish units
            model = result.sfm.model_dir if result.sfm else paths.colmap_model
            if not model.exists() and not cfg.dry_run:
                candidates = list(paths.colmap_sparse.glob("*"))
                dirs = [c for c in candidates if c.is_dir() and not c.name.endswith("_txt")]
                if not dirs:
                    raise FileNotFoundError("No COLMAP model found; run sfm first")
                model = dirs[0]
            result.scale = run_scale(cfg, paths, model)
        elif name == "refine":
            model = (
                result.scale.model_dir
                if result.scale is not None and result.scale.model_dir.exists()
                else (result.sfm.model_dir if result.sfm else paths.colmap_model)
            )
            result.refine = run_refine(cfg, paths, model)
        elif name == "train":
            model = self._active_model(result)
            result.train = run_train(cfg, paths, model)
        elif name == "preflight":
            from instasplat.utils.preflight import run_preflight

            if cfg.preflight:
                pf = run_preflight(cfg, paths)
                if not cfg.dry_run:
                    pf.raise_if_blocked()
            else:
                self.log.info("Preflight skipped (cfg.preflight=false)")
        elif name == "export":
            ply = result.train.ply_path if result.train else None
            result.export = run_export(cfg, paths, ply)
        elif name == "package":
            model = self._active_model(result)
            if paths.scaled_model.exists():
                model = paths.scaled_model
            # Tiled parent jobs often have no root COLMAP — package still writes
            # hierarchy / cloud / quality from chunk artifacts.
            result.package = run_package(cfg, paths, model)
        elif name == "plan_chunks":
            self._manifest = run_plan_chunks(cfg, paths)
        elif name == "process_chunks":
            manifest = self._load_manifest()
            self._chunk_status = run_process_chunks(cfg, paths, manifest)
            if not any(self._chunk_status.values()) and not cfg.dry_run:
                raise RuntimeError("All chunks failed during process_chunks")
        elif name == "align_chunks":
            manifest = self._load_manifest()
            self._alignments = run_align_chunks(cfg, paths, manifest)
        elif name == "merge_chunks":
            manifest = self._load_manifest()
            outputs = run_merge_chunks(cfg, paths, manifest, self._alignments)
            result.tiled = TiledResult(
                success=True,
                root=paths.root,
                manifest=manifest,
                chunk_results=self._chunk_status,
                merged_outputs=outputs,
            )
            from instasplat.stages.export import ExportResult

            ply = outputs.get("ply", paths.export / "scene.ply")
            result.export = ExportResult(outputs=outputs, source_ply=ply)
        else:
            raise ValueError(f"Unknown stage: {name}")

    def _load_manifest(self) -> ChunkManifest | _ChunkManifest:
        if self._manifest is not None:
            return self._manifest
        path = chunks_root(self.paths) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Chunk manifest missing: {path}. Run plan_chunks first.")
        self._manifest = _ChunkManifest.load(path)
        return self._manifest
