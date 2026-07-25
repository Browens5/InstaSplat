# Best local Mac pipeline: long 360 → tiled splat

This is InstaSplat’s recommended **local Apple Silicon** path for long
Insta360 captures. It does **not** require CUDA, LingBot-Map, or MediaSDK.

```text
Studio equirect MP4 (+ sibling INSV / gyro+gps CSV)
  → ingest + telemetry
  → plan overlapping tiles (gyro-dense fps, GPS path-aware)
  → preflight (tools, stitch, disk, GPS soft-fallback)
  → per tile (Metal-serialized):
        YOLO MPS masks → cubemap COLMAP → metric scale
        → pose refine → Brush/OpenSplat Metal → PLY
  → GPS/gyro Sim3 align (RANSAC + ICP + RMSE gate)
  → splat-transform merge + prune → ply / sog / spz
  → quality.json + hierarchy LOD + cloud_job.json
```

## One command

```bash
# 1) Export stitched equirect MP4 from Insta360 Studio (8K@30 ok)
# 2) Keep the original .insv next to it (telemetry), or add gyro.csv / gps.csv

instasplat doctor          # confirm mac_long_360=yes
instasplat install-brush   # once — clones + cargo build --release
instasplat mac-360 -i ./capture_equirect_8k.mp4 -o ./runs -n walk_360
# GUI: Pause freezes stages + SIGSTOPs Brush; Resume / Stop also available

# Resume-safe: re-run skips tiles that already have scene.ply
instasplat mac-360 -i ./capture_equirect_8k.mp4 -o ./runs -n walk_360
```

Aliases: `instasplat run --large-8k ...` and `instasplat run --tiled ...`.

## Inputs that work best

| Input | Result |
|-------|--------|
| Studio **equirect MP4** + sibling `.insv` | Best — video + gyro/GPS |
| Equirect MP4 + `gyro.csv` / `gps.csv` sidecars | Good |
| Equirect MP4 alone | Works; no metric GPS, weaker turn densify |
| Raw `.insv` without Studio MP4 | Blocked unless `--allow-unstitched` (test only) |

Sidecar names next to the MP4: `gyro.csv`, `gps.csv`, or `<stem>.gyro.csv`.

## What runs on Metal vs CPU

| Stage | Device |
|-------|--------|
| YOLO people masks | PyTorch **MPS** (auto CPU fallback on known MPS crashes) |
| Cubemap remap | CPU OpenCV |
| COLMAP | CPU (typical Homebrew) |
| Brush train | **Metal / WebGPU** (serialized per tile) |
| OpenSplat train | **Metal MPS** (`--trainer opensplat`) |
| splat-transform merge | CPU Node |

## Outputs

```
runs/walk_360/
  preflight.json
  quality.json
  cloud_job.json
  10_chunks/manifest.json
  10_chunks/chunk_XXX/06_export/scene.ply
  11_merged/scene_merged.ply
  11_merged/hierarchy_manifest.json
  11_merged/lod/lod_levels.json
  06_export/scene.ply
  06_export/scene.sog
  06_export/scene.spz
```

## Safety rails

- **Preflight** fails fast on missing ffmpeg/colmap/trainer/splat-transform, unstitched remux, or critically low disk.
- **GPS soft-fallback** — if `gps.csv` is missing, scale becomes non-metric instead of crashing.
- **Partial tiles** — failed chunks block merge unless `--allow-partial-merge`.
- **Resume** — successful tiles with an export PLY are skipped on re-run.
- **Chunk window extract** — one ffmpeg cut per tile, then local seeks (critical for 8K).

## Tuning (M-series laptops)

| Knob | Safer / faster | Higher quality |
|------|----------------|----------------|
| `chunk.base_fps` | 4–6 | 8–10 |
| `chunk.max_fps` | 10–12 | 15 |
| `chunk.duration_sec` | 20–25 | 15 (more tiles) |
| `sfm.face_resolution` | 1024 | 1280–1536 |
| `train.total_steps` | 12k–15k | 20k–30k |
| `train.max_resolution` | 1280 | 1600 |

Start with defaults from `enable_mac_long_360_defaults()`.

## What this path deliberately skips

- Official MediaSDK stitch (Windows/Linux only) — use Studio on Mac.
- LingBot-Map / 3DGUT / LichtFeld (CUDA) — see `cloud_job.json` + [CLOUD.md](CLOUD.md).
- Kerbl CUDA hierarchy merger — CPU LOD previews are packaged instead.

## See also

- [LARGE_8K.md](LARGE_8K.md) — tile/align details
- [CAPTURE_GUIDELINES.md](CAPTURE_GUIDELINES.md) — walk slow, overlap, avoid crowds
- [SETUP_MACOS.md](SETUP_MACOS.md) — brew / cargo / npm installs
