"""Post-run quality report for capture, align, and export health."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.utils.chunking import ChunkManifest
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger
from instasplat.utils.telemetry import load_gps, load_gyro, path_length_m


@dataclass
class QualityIssue:
    level: str  # info | warn | error
    code: str
    message: str


@dataclass
class QualityReport:
    grade: str  # excellent | good | fair | poor
    score: float
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "score": self.score,
            "issues": [asdict(i) for i in self.issues],
            "metrics": self.metrics,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def validate_capture(
    *,
    duration_sec: float,
    overlap_sec: float,
    chunk_duration_sec: float,
    min_overlap_ratio: float,
    base_fps: float,
    gyro_csv: Path | None = None,
    gps_csv: Path | None = None,
) -> list[QualityIssue]:
    """Heuristic capture checks inspired by LongSplat / on-the-fly guidelines."""
    issues: list[QualityIssue] = []
    if duration_sec <= 0:
        issues.append(
            QualityIssue("warn", "duration_unknown", "Video duration unknown; chunk plan may be wrong")
        )
        return issues

    ratio = overlap_sec / max(chunk_duration_sec, 1e-6)
    if ratio + 1e-9 < min_overlap_ratio:
        issues.append(
            QualityIssue(
                "warn",
                "low_overlap",
                f"Chunk overlap ratio {ratio:.2f} < min {min_overlap_ratio:.2f}; "
                "tile align may be weak",
            )
        )

    if base_fps < 4.0 and duration_sec > 30:
        issues.append(
            QualityIssue(
                "info",
                "sparse_fps",
                f"base_fps={base_fps} is sparse for long walks; consider 6–10 for turns",
            )
        )

    gps = load_gps(gps_csv) if gps_csv and gps_csv.exists() else None
    if gps is not None and len(gps.t_sec) >= 2:
        plen = path_length_m(gps, float(gps.t_sec[0]), float(gps.t_sec[-1]))
        dt = float(gps.t_sec[-1] - gps.t_sec[0])
        speed = plen / max(dt, 1e-6)
        if speed > 2.5:  # ~jogging / bike
            issues.append(
                QualityIssue(
                    "warn",
                    "fast_motion",
                    f"Mean GPS speed ~{speed:.1f} m/s; increase fps or slow the walk",
                )
            )
        elif speed < 0.05 and duration_sec > 20:
            issues.append(
                QualityIssue(
                    "info",
                    "mostly_static",
                    "GPS path is nearly static; expect limited parallax / thin geometry",
                )
            )

    gyro = load_gyro(gyro_csv) if gyro_csv and gyro_csv.exists() else None
    if gyro is not None and len(gyro.omega) > 10:
        mag = np.linalg.norm(gyro.omega, axis=1)
        p95 = float(np.percentile(mag, 95))
        if p95 > 2.5:  # rad/s
            issues.append(
                QualityIssue(
                    "warn",
                    "harsh_turns",
                    f"95th-percentile gyro rate {p95:.2f} rad/s; expect blur / weak matches",
                )
            )

    return issues


def build_quality_report(cfg: PipelineConfig, paths: JobPaths) -> QualityReport:
    """Aggregate capture + tile + export health into one JSON report."""
    log = get_logger("instasplat.quality", paths.logs / "quality.log")
    issues: list[QualityIssue] = []
    metrics: dict[str, Any] = {}

    duration_sec = 0.0
    manifest_path = paths.chunks / "manifest.json"
    if manifest_path.exists():
        manifest = ChunkManifest.load(manifest_path)
        duration_sec = manifest.duration_sec
        metrics["n_chunks"] = len(manifest.chunks)
        metrics["duration_sec"] = manifest.duration_sec
        metrics["chunk_strategy"] = manifest.strategy
        frame_counts = [len(c.frame_times) for c in manifest.chunks]
        metrics["frames_per_chunk_mean"] = float(np.mean(frame_counts)) if frame_counts else 0
        metrics["frames_per_chunk_min"] = int(min(frame_counts)) if frame_counts else 0
        if frame_counts and min(frame_counts) < 12:
            issues.append(
                QualityIssue(
                    "warn",
                    "thin_chunk",
                    f"Some chunks have only {min(frame_counts)} frames; SfM may fail",
                )
            )
        for note in manifest.notes:
            if "fast" in note or "SfM risk" in note or "low" in note:
                issues.append(QualityIssue("info", "chunk_note", note))
    else:
        metrics["n_chunks"] = 0

    issues.extend(
        validate_capture(
            duration_sec=duration_sec,
            overlap_sec=cfg.chunk.overlap_sec,
            chunk_duration_sec=cfg.chunk.duration_sec,
            min_overlap_ratio=cfg.chunk.min_overlap_ratio,
            base_fps=cfg.chunk.base_fps if cfg.chunk.enabled else cfg.extract.fps,
            gyro_csv=paths.gyro_csv if paths.gyro_csv.exists() else None,
            gps_csv=paths.gps_csv if paths.gps_csv.exists() else None,
        )
    )

    aligns_path = paths.chunks / "alignments.json"
    if aligns_path.exists():
        aligns = json.loads(aligns_path.read_text(encoding="utf-8"))
        rmses = [float(a.get("rmse_m") or 0.0) for a in aligns if a.get("rmse_m") is not None]
        methods = [a.get("method") for a in aligns]
        metrics["align_n"] = len(aligns)
        metrics["align_rmse_mean_m"] = float(np.mean(rmses)) if rmses else None
        metrics["align_rmse_max_m"] = float(np.max(rmses)) if rmses else None
        metrics["align_methods"] = {m: methods.count(m) for m in set(methods)}
        max_rmse = cfg.refine.max_align_rmse_m
        if rmses and max(rmses) > max_rmse:
            issues.append(
                QualityIssue(
                    "warn",
                    "high_align_rmse",
                    f"Max tile align RMSE {max(rmses):.2f}m > gate {max_rmse:.1f}m",
                )
            )
        if any(m == "identity" for m in methods):
            issues.append(
                QualityIssue(
                    "info",
                    "identity_align",
                    "Some tiles used identity Sim3 (anchor / no GPS)",
                )
            )

    export_dir = paths.export if paths.export.exists() else paths.merged
    sizes: dict[str, int] = {}
    if export_dir.exists():
        for p in export_dir.glob("scene.*"):
            if p.is_file():
                sizes[p.name] = p.stat().st_size
        lod = export_dir / "lod-meta.json"
        if lod.exists():
            sizes["lod-meta.json"] = lod.stat().st_size
    metrics["export_bytes"] = sizes
    if not sizes and not cfg.dry_run:
        issues.append(QualityIssue("warn", "no_exports", "No scene.* exports found yet"))

    # Score: start 100, deduct for warnings/errors
    score = 100.0
    for issue in issues:
        if issue.level == "error":
            score -= 25
        elif issue.level == "warn":
            score -= 10
        else:
            score -= 2
    score = max(0.0, min(100.0, score))
    if score >= 90:
        grade = "excellent"
    elif score >= 75:
        grade = "good"
    elif score >= 55:
        grade = "fair"
    else:
        grade = "poor"

    report = QualityReport(grade=grade, score=score, issues=issues, metrics=metrics)
    log.info("Quality grade=%s score=%.1f issues=%d", grade, score, len(issues))
    return report


def run_quality(cfg: PipelineConfig, paths: JobPaths) -> QualityReport:
    report = build_quality_report(cfg, paths)
    out = paths.root / "quality.json"
    if not cfg.dry_run:
        report.save(out)
    return report
