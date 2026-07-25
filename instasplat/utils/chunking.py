"""Temporal / spatial chunk planning for large 8K captures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from instasplat.utils.telemetry import (
    GpsSeries,
    GyroSeries,
    adaptive_frame_times,
    load_gps,
    load_gyro,
    path_length_m,
)


@dataclass
class ChunkPlan:
    chunk_id: str
    index: int
    start_sec: float
    end_sec: float
    overlap_prev_sec: float
    frame_times: list[float]
    gps_start_xyz: list[float] | None = None
    gps_end_xyz: list[float] | None = None
    path_length_m: float | None = None


@dataclass
class ChunkManifest:
    duration_sec: float
    source_fps_hint: float
    chunks: list[ChunkPlan]
    strategy: str
    notes: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "duration_sec": self.duration_sec,
            "source_fps_hint": self.source_fps_hint,
            "strategy": self.strategy,
            "notes": self.notes,
            "chunks": [asdict(c) for c in self.chunks],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ChunkManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = [ChunkPlan(**c) for c in data["chunks"]]
        return cls(
            duration_sec=float(data["duration_sec"]),
            source_fps_hint=float(data.get("source_fps_hint", 30.0)),
            chunks=chunks,
            strategy=data.get("strategy", "temporal"),
            notes=list(data.get("notes") or []),
        )


def probe_duration_sec(video: Path) -> float:
    """Best-effort duration via ffprobe."""
    import shutil
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not video.exists():
        return 0.0
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return max(0.0, float(proc.stdout.strip()))
    except ValueError:
        return 0.0


def plan_chunks(
    *,
    duration_sec: float,
    chunk_duration_sec: float = 25.0,
    overlap_sec: float = 5.0,
    base_fps: float = 6.0,
    max_fps: float = 15.0,
    max_frames_per_chunk: int = 180,
    target_path_length_m: float | None = 40.0,
    gyro: GyroSeries | None = None,
    gps: GpsSeries | None = None,
    source_fps_hint: float = 30.0,
) -> ChunkManifest:
    """
    Auto-chunk a long capture.

    Primary split is temporal with overlap. When GPS is present and
    ``target_path_length_m`` is set, chunk boundaries also respect traveled
    distance so dense walking and sparse transit get sensible tile sizes.
    """
    notes: list[str] = []
    if duration_sec <= 0:
        duration_sec = 60.0
        notes.append("duration unknown; defaulting to 60s placeholder")

    overlap_sec = max(0.0, min(overlap_sec, chunk_duration_sec * 0.8))
    boundaries: list[tuple[float, float]] = []

    if gps is not None and target_path_length_m and target_path_length_m > 0:
        notes.append("spatial-aware temporal chunking via GPS path length")
        t = 0.0
        while t < duration_sec - 1e-6:
            # Grow until path length or max duration hit
            end = min(duration_sec, t + chunk_duration_sec)
            # Binary-ish expand using path length
            lo, hi = t + max(5.0, overlap_sec), end
            best = end
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                plen = path_length_m(gps, t, mid)
                if plen < target_path_length_m:
                    lo = mid
                    best = mid
                else:
                    hi = mid
                    best = mid
            end = min(duration_sec, max(best, t + max(5.0, overlap_sec)))
            boundaries.append((t, end))
            if end >= duration_sec - 1e-6:
                break
            t = max(t + 0.1, end - overlap_sec)
        strategy = "gps_path_temporal"
    else:
        notes.append("pure temporal chunking")
        step = max(1.0, chunk_duration_sec - overlap_sec)
        t = 0.0
        while t < duration_sec - 1e-6:
            end = min(duration_sec, t + chunk_duration_sec)
            boundaries.append((t, end))
            if end >= duration_sec - 1e-6:
                break
            t += step
        strategy = "temporal"

    chunks: list[ChunkPlan] = []
    for i, (start, end) in enumerate(boundaries):
        local_dur = max(0.0, end - start)
        times = adaptive_frame_times(
            local_dur,
            base_fps=base_fps,
            max_fps=max_fps,
            gyro=_slice_gyro(gyro, start, end),
        )
        # Shift to absolute timeline
        times = times + start
        if len(times) > max_frames_per_chunk:
            idx = np.linspace(0, len(times) - 1, max_frames_per_chunk).astype(int)
            times = times[idx]
            notes.append(f"chunk {i:03d} capped to {max_frames_per_chunk} frames")
        gps_start = gps_end = plen = None
        if gps is not None:
            from instasplat.utils.telemetry import interpolate_xyz

            gps_start = interpolate_xyz(gps, start).tolist()
            gps_end = interpolate_xyz(gps, end).tolist()
            plen = path_length_m(gps, start, end)
        chunks.append(
            ChunkPlan(
                chunk_id=f"chunk_{i:03d}",
                index=i,
                start_sec=float(start),
                end_sec=float(end),
                overlap_prev_sec=float(overlap_sec if i else 0.0),
                frame_times=[float(x) for x in times],
                gps_start_xyz=gps_start,
                gps_end_xyz=gps_end,
                path_length_m=plen,
            )
        )

    return ChunkManifest(
        duration_sec=duration_sec,
        source_fps_hint=source_fps_hint,
        chunks=chunks,
        strategy=strategy,
        notes=notes,
    )


def _slice_gyro(gyro: GyroSeries | None, start: float, end: float) -> GyroSeries | None:
    if gyro is None:
        return None
    mask = (gyro.t_sec >= start) & (gyro.t_sec <= end)
    if not np.any(mask):
        return None
    return GyroSeries(t_sec=gyro.t_sec[mask] - start, omega=gyro.omega[mask])


def load_telemetry_pair(
    gyro_csv: Path | None,
    gps_csv: Path | None,
) -> tuple[GyroSeries | None, GpsSeries | None]:
    gyro = load_gyro(gyro_csv) if gyro_csv and gyro_csv.exists() else None
    gps = load_gps(gps_csv) if gps_csv and gps_csv.exists() else None
    return gyro, gps
