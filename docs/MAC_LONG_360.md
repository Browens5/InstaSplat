# Best local Mac pipeline: long 360 → tiled splat

This is the **tiled product path** inside InstaSplat’s full Mac
360° video → splat pipeline. For why that pipeline exists and how it is
built end to end, see **[MAC_360_PIPELINE.md](MAC_360_PIPELINE.md)**.

It does **not** require CUDA, LingBot-Map, or MediaSDK.

```text
Studio equirect MP4 (+ sibling INSV / gyro+gps CSV)
  → ingest + telemetry
  → plan overlapping tiles (gyro-dense fps, GPS path-aware)
  → preflight (tools, stitch, disk, GPS soft-fallback)
  → per tile (Metal-serialized):
        YOLO MPS masks → EQUIRECTANGULAR COLMAP → metric scale
        → pose refine → metal_equirect (MPS) → PLY
  → GPS/gyro Sim3 align (RANSAC + ICP + RMSE gate)
  → splat-transform merge + prune → ply / sog / spz
  → quality.json + hierarchy LOD + cloud_job.json
```

## One command

```bash
# 1) Export stitched equirect MP4 from Insta360 Studio (8K@30 ok)
# 2) Keep the original .insv next to it (telemetry), or add gyro.csv / gps.csv

instasplat doctor          # confirm mac_long_360=yes (needs PyTorch)
instasplat mac-360 -i ./capture_equirect_8k.mp4 -o ./runs -n walk_360
# GUI: Pause freezes stages at checkpoints; Unpause / Stop also available

# Resume-safe: re-run skips tiles that already have scene.ply
instasplat mac-360 -i ./capture_equirect_8k.mp4 -o ./runs -n walk_360
```

Aliases: `instasplat run --large-8k ...` and `instasplat run --tiled ...`.

## Continue a previous project

```bash
# CLI — load job folder and run only remaining work
instasplat run --job ./runs/walk_360 --only process_chunks
instasplat run --job ./runs/walk_360 --from align_chunks --to package
```

**GUI:** **Open previous run…** → select `runs/walk_360` (the job folder with
`config.yaml` or `00_ingest/`). Settings reload, remaining stages are
pre-checked, then click **Continue run**. Finished tiles/artifacts are skipped
(`skip_existing`). Use **New project** to clear and start fresh.

The right-hand **Live viewer** polls the job folder during a run: COLMAP sparse
points after the mapper writes `points3D.*`, then splat centers from the newest
training/export PLY. Use the **Artifacts** tab to browse frames, masks, sparse
models, and exports (double-click to open; **Show in 3D** for `points3D` / `.ply`).
Training uses **metal_equirect** (full equirect frames + COLMAP poses) on PyTorch MPS.
See [METAL_EQUIRECT_TRAINER.md](METAL_EQUIRECT_TRAINER.md). Preview JPEGs appear under
`05_train/exports/previews/` during a run.

## Run sections individually

```bash
instasplat stages --mode tiled
instasplat stage process_chunks -j ./runs/walk_360
instasplat run --job ./runs/walk_360 --only align_chunks,merge_chunks
instasplat run --job ./runs/walk_360 --from merge_chunks --to package
instasplat validate --job ./runs/walk_360
```

Do **not** run the single-mode `sfm` stage on a tiled job — top-level
`01_frames/` is empty; tiles live under `10_chunks/`. Use `process_chunks`
(SfM mode: **equirectangular**, COLMAP ≥ 4.1). Full panoramas are used for
both COLMAP and metal_equirect training. Legacy `perspective_cubemap` is an
explicit opt-in only.

GUI: check **Stages** before Run (use “All for mode” to reset).

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
| COLMAP EQUIRECTANGULAR SfM | CPU (Homebrew COLMAP ≥ 4.1); `sfm.mapper: incremental` default, or `global` (GLOMAP) |
| metal_equirect train | **PyTorch MPS / Metal** (serialized per tile) |
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

- [MAC_360_PIPELINE.md](MAC_360_PIPELINE.md) — need + full pipeline architecture
- [METAL_SPLAT_WORKFLOW.md](METAL_SPLAT_WORKFLOW.md) — runbook
- [LARGE_8K.md](LARGE_8K.md) — tile/align details
- [CAPTURE_GUIDELINES.md](CAPTURE_GUIDELINES.md) — walk slow, overlap, avoid crowds
- [SETUP_MACOS.md](SETUP_MACOS.md) — one-shot install
