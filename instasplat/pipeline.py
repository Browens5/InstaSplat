"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from instasplat.config import PipelineConfig
from instasplat.stages.export import ExportResult, run_export
from instasplat.stages.extract import ExtractResult, run_extract
from instasplat.stages.ingest import IngestResult, run_ingest
from instasplat.stages.mask import MaskResult, run_mask
from instasplat.stages.scale import ScaleResult, run_scale
from instasplat.stages.sfm import SfMResult, run_sfm
from instasplat.stages.train import TrainResult, run_train
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger

ProgressCb = Callable[[str, float, str], None]


@dataclass
class PipelineResult:
    success: bool
    paths: JobPaths
    stage_timings: dict[str, float] = field(default_factory=dict)
    ingest: IngestResult | None = None
    extract: ExtractResult | None = None
    mask: MaskResult | None = None
    sfm: SfMResult | None = None
    scale: ScaleResult | None = None
    train: TrainResult | None = None
    export: ExportResult | None = None
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
            "scale": _safe(self.scale),
            "train": _safe(self.train),
            "export": {
                "outputs": {k: str(v) for k, v in self.export.outputs.items()},
                "source_ply": str(self.export.source_ply),
            }
            if self.export
            else None,
        }


class Pipeline:
    STAGE_ORDER: ClassVar[list[str]] = [
        "ingest",
        "extract",
        "mask",
        "sfm",
        "scale",
        "train",
        "export",
    ]

    def __init__(self, cfg: PipelineConfig, on_progress: ProgressCb | None = None):
        self.cfg = cfg
        self.paths = JobPaths(cfg.work_dir())
        self.on_progress = on_progress or (lambda *_: None)
        self.log = get_logger("instasplat.pipeline")

    def run(self, stages: list[str] | None = None) -> PipelineResult:
        self.paths.ensure()
        self.cfg.save(self.paths.config)
        selected = stages or self.cfg.stages or self.STAGE_ORDER
        selected = [s for s in self.STAGE_ORDER if s in selected]

        result = PipelineResult(success=False, paths=self.paths)
        total = len(selected)
        try:
            for i, name in enumerate(selected):
                self.on_progress(name, i / max(total, 1), f"Starting {name}")
                t0 = time.time()
                self._run_stage(name, result)
                result.stage_timings[name] = time.time() - t0
                self.on_progress(name, (i + 1) / max(total, 1), f"Finished {name}")
            result.success = True
        except Exception as exc:
            self.log.exception("Pipeline failed")
            result.error = str(exc)
            result.success = False

        summary = self.paths.root / "result.json"
        summary.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    def _run_stage(self, name: str, result: PipelineResult) -> None:
        cfg, paths = self.cfg, self.paths
        if name == "ingest":
            result.ingest = run_ingest(cfg, paths)
        elif name == "extract":
            result.extract = run_extract(cfg, paths)
        elif name == "mask":
            result.mask = run_mask(cfg, paths)
        elif name == "sfm":
            result.sfm = run_sfm(cfg, paths)
        elif name == "scale":
            model = (
                result.sfm.model_dir
                if result.sfm is not None
                else paths.colmap_model
            )
            if not model.exists() and not cfg.dry_run:
                # Try any sparse model
                candidates = list(paths.colmap_sparse.glob("*"))
                dirs = [c for c in candidates if c.is_dir() and not c.name.endswith("_txt")]
                if not dirs:
                    raise FileNotFoundError("No COLMAP model found; run sfm first")
                model = dirs[0]
            result.scale = run_scale(cfg, paths, model)
        elif name == "train":
            model = (
                result.scale.model_dir
                if result.scale is not None and result.scale.model_dir.exists()
                else paths.scaled_model
            )
            if not model.exists():
                model = result.sfm.model_dir if result.sfm else paths.colmap_model
            result.train = run_train(cfg, paths, model)
        elif name == "export":
            ply = result.train.ply_path if result.train else None
            result.export = run_export(cfg, paths, ply)
        else:
            raise ValueError(f"Unknown stage: {name}")
