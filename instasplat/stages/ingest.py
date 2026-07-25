"""INSV ingest: discover files, extract trailer telemetry, resolve video source."""

from __future__ import annotations

import json
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger, run_cmd


@dataclass
class IngestResult:
    source_video: Path
    source_kind: str  # insv | mp4 | equirect_mp4
    pair_path: Path | None
    gyro_csv: Path | None
    accel_csv: Path | None
    gps_csv: Path | None
    metadata: dict[str, Any]


def find_insv_pair(path: Path) -> Path | None:
    """Locate the companion _10_ file for dual-file 5.7K captures."""
    name = path.name
    if "_00_" in name:
        candidate = path.with_name(name.replace("_00_", "_10_", 1))
        if candidate.exists():
            return candidate
    if "_10_" in name:
        candidate = path.with_name(name.replace("_10_", "_00_", 1))
        if candidate.exists():
            return candidate
    return None


def _parse_insv_trailer(path: Path) -> dict[str, Any]:
    """
    Parse Insta360 INSV trailer records (gyro/accel/exposure/gps).

    Record layout (from end): ... [payload][uint16 id][uint32 size] ...
    Common IDs: 0x300 accel/gyro, 0x400 exposure, 0x700 GPS.
    """
    data: dict[str, Any] = {
        "gyro": [],
        "accel": [],
        "exposure": [],
        "gps": [],
        "maker_notes": {},
    }
    with path.open("rb") as fin:
        fin.seek(0, 2)
        file_size = fin.tell()
        # Trailer length is near the end; try a few known layouts.
        trailer_len = None
        for seek_back in (78, 74, 70, 64):
            if file_size < seek_back + 4:
                continue
            fin.seek(-seek_back, 2)
            buf = fin.read(seek_back)
            # Last 4 bytes of many layouts encode trailer length
            candidate = struct.unpack("<I", buf[-4:])[0]
            if 64 < candidate < file_size:
                trailer_len = candidate
                break
        if trailer_len is None:
            # Fallback: scan last 8 MB for record headers
            trailer_len = min(file_size, 8 * 1024 * 1024)

        offset = -8  # start before trailing magic/size
        gyro: list[list[float]] = []
        accel: list[list[float]] = []
        exposure: list[list[float]] = []
        gps: list[list[float]] = []

        while abs(offset) < trailer_len:
            try:
                fin.seek(offset, 2)
                hdr = fin.read(6)
                if len(hdr) < 6:
                    break
                rec_id, size = struct.unpack("<HI", hdr)
                if size <= 0 or size > trailer_len:
                    offset -= 2
                    continue
                fin.seek(offset - size, 2)
                payload = fin.read(size)
                hid = rec_id
                if hid == 0x300:
                    # Typically interleaved timestamp + gyro + accel samples
                    # Sample size commonly 56 bytes (double ts + 6 floats) or similar.
                    sample = 56
                    n = len(payload) // sample
                    for i in range(n):
                        chunk = payload[i * sample : (i + 1) * sample]
                        if len(chunk) < 32:
                            continue
                        try:
                            # timestamp (double ms), gx,gy,gz, ax,ay,az as floats
                            ts = struct.unpack("<d", chunk[0:8])[0]
                            gxa = struct.unpack("<6f", chunk[8:32])
                            gyro.append([ts, gxa[0], gxa[1], gxa[2]])
                            accel.append([ts, gxa[3], gxa[4], gxa[5]])
                        except struct.error:
                            continue
                elif hid == 0x400:
                    # exposure: pairs of doubles
                    n = len(payload) // 16
                    for i in range(n):
                        chunk = payload[i * 16 : (i + 1) * 16]
                        if len(chunk) < 16:
                            continue
                        a, b = struct.unpack("<dd", chunk)
                        exposure.append([a, b])
                elif hid == 0x700:
                    # GPS: variable; try lat/lon/alt doubles after timestamp
                    sample = 40
                    n = len(payload) // sample
                    for i in range(max(n, 0)):
                        chunk = payload[i * sample : (i + 1) * sample]
                        if len(chunk) < 32:
                            continue
                        try:
                            ts = struct.unpack("<d", chunk[0:8])[0]
                            lat, lon, alt = struct.unpack("<ddd", chunk[8:32])
                            if abs(lat) <= 90 and abs(lon) <= 180:
                                gps.append([ts, lat, lon, alt])
                        except struct.error:
                            continue
                offset = offset - size - 6
            except (OSError, struct.error):
                break

        data["gyro"] = gyro
        data["accel"] = accel
        data["exposure"] = exposure
        data["gps"] = gps
    return data


def _write_csv(path: Path, header: str, rows: list[list[float]]) -> None:
    lines = [header]
    for row in rows:
        lines.append(",".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _exiftool_metadata(path: Path, dry_run: bool) -> dict[str, Any]:
    exe = shutil.which("exiftool")
    if not exe:
        return {}
    proc = run_cmd(
        [exe, "-ee", "-G", "-s", "-j", "-a", str(path)],
        dry_run=dry_run,
        check=False,
    )
    if dry_run or not proc.stdout.strip():
        return {}
    try:
        parsed = json.loads(proc.stdout)
        return parsed[0] if isinstance(parsed, list) and parsed else {}
    except json.JSONDecodeError:
        return {}


def resolve_video_source(cfg: PipelineConfig, paths: JobPaths) -> tuple[Path, str]:
    """
    Resolve an equirectangular (or best-effort) video for frame extraction.

    Mac reality:
    - Official MediaSDK is Windows/Ubuntu — not available natively on macOS.
    - Recommended: export equirect MP4 from Insta360 Studio, or pass a .mp4.
    - INSV can be treated as MP4 container for dual-fisheye fallback.
    """
    src = cfg.input_path
    mode = cfg.extract.stitch_mode

    if src.suffix.lower() in {".mp4", ".mov"}:
        dest = paths.video
        if not dest.exists() or not cfg.skip_existing:
            shutil.copy2(src, dest)
        return dest, "prestitched_mp4"

    if src.suffix.lower() != ".insv":
        raise ValueError(f"Unsupported input type: {src.suffix}")

    if mode == "mediasdk":
        mediasdk = shutil.which("MediaSDKTest")
        if mediasdk:
            out = paths.video
            pair = find_insv_pair(src)
            inputs = [str(src)] + ([str(pair)] if pair else [])
            run_cmd(
                [mediasdk, "-inputs", *inputs, "-output", str(out), "-enable_flowstate"],
                log_file=paths.logs / "mediasdk.log",
                dry_run=cfg.dry_run,
            )
            return out, "mediasdk"
        get_logger("instasplat.ingest").warning(
            "MediaSDKTest not found; falling back to studio_mp4 / ffmpeg path"
        )
        mode = "studio_mp4"

    if mode == "studio_mp4":
        # Look for a sibling pre-exported equirectangular MP4
        candidates = [
            src.with_suffix(".mp4"),
            src.with_name(src.stem + "_equirect.mp4"),
            src.with_name(re.sub(r"_00_", "_", src.stem) + "_360.mp4"),
            paths.ingest / "studio_export.mp4",
        ]
        for c in candidates:
            if c.exists():
                dest = paths.video
                if c.resolve() != dest.resolve():
                    shutil.copy2(c, dest)
                return dest, "studio_mp4"
        if not cfg.extract.allow_prestitched_mp4:
            raise FileNotFoundError(
                "No stitched MP4 found next to INSV. On macOS, export equirectangular "
                "MP4 from Insta360 Studio (or place studio_export.mp4 in the ingest folder)."
            )
        # Fall through to ffmpeg dual-fisheye copy for pipeline testing
        mode = "ffmpeg_fallback"

    if mode == "ffmpeg_fallback":
        dest = paths.video
        # Copy/remux first video stream — dual fisheye, NOT true equirect.
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                str(dest),
            ],
            log_file=paths.logs / "ffmpeg_insv_remux.log",
            dry_run=cfg.dry_run,
            check=False,
        )
        if not cfg.dry_run and not dest.exists():
            # Last resort: literal copy with .mp4 suffix
            shutil.copy2(src, dest)
        meta_note = paths.ingest / "WARNING_UNSTITCHED.txt"
        meta_note.write_text(
            "This video may be dual-fisheye / unstitched. For production metric splats on "
            "macOS, export a stitched equirectangular MP4 from Insta360 Studio first.\n",
            encoding="utf-8",
        )
        return dest, "ffmpeg_fallback"

    if mode == "prestitched":
        raise FileNotFoundError("stitch_mode=prestitched requires an .mp4/.mov input")

    raise RuntimeError(f"Unhandled stitch mode: {mode}")


def run_ingest(cfg: PipelineConfig, paths: JobPaths) -> IngestResult:
    paths.ensure()
    log = get_logger("instasplat.ingest", paths.logs / "ingest.log")
    src = cfg.input_path
    if not src.exists() and not cfg.dry_run:
        raise FileNotFoundError(src)

    pair = find_insv_pair(src) if src.suffix.lower() == ".insv" else None
    log.info("Input: %s (pair=%s)", src, pair)

    video, kind = resolve_video_source(cfg, paths)
    metadata = {
        "input": str(src),
        "pair": str(pair) if pair else None,
        "source_kind": kind,
        "video": str(video),
    }

    gyro_csv = accel_csv = gps_csv = None
    if src.suffix.lower() == ".insv" and not cfg.dry_run:
        try:
            trailer = _parse_insv_trailer(src)
            if trailer["gyro"]:
                gyro_csv = paths.gyro_csv
                _write_csv(gyro_csv, "timestamp_ms,gx,gy,gz", trailer["gyro"])
            if trailer["accel"]:
                accel_csv = paths.accel_csv
                _write_csv(accel_csv, "timestamp_ms,ax,ay,az", trailer["accel"])
            if trailer["gps"]:
                gps_csv = paths.gps_csv
                _write_csv(gps_csv, "timestamp_ms,lat,lon,alt", trailer["gps"])
            metadata["trailer_counts"] = {
                "gyro": len(trailer["gyro"]),
                "accel": len(trailer["accel"]),
                "exposure": len(trailer["exposure"]),
                "gps": len(trailer["gps"]),
            }
        except Exception as exc:  # noqa: BLE001 — trailer formats vary by camera gen
            log.warning("INSV trailer parse failed: %s", exc)
            metadata["trailer_error"] = str(exc)

    metadata["exif"] = _exiftool_metadata(src, cfg.dry_run)
    paths.metadata_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    # Keep a copy of the original input pointer
    (paths.ingest / "input_path.txt").write_text(str(src.resolve()), encoding="utf-8")
    return IngestResult(
        source_video=video,
        source_kind=kind,
        pair_path=pair,
        gyro_csv=gyro_csv,
        accel_csv=accel_csv,
        gps_csv=gps_csv,
        metadata=metadata,
    )
