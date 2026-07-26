# Large 8K tiled Gaussian splat pipeline

Tuning guide for the **long-walk** half of InstaSplat’s full Mac 360 → splat
pipeline ([MAC_360_PIPELINE.md](MAC_360_PIPELINE.md)).

InstaSplat reconstructs **long 8K@30fps** captures by auto-chunking the
timeline, sampling denser frames on turns (gyro), reconstructing each tile,
aligning tiles with **GPS + gyro**, and merging splats — with **Metal-first**
defaults on Apple Silicon.

## Why chunk?

A 10-minute 8K@30 equirect clip is ~18,000 frames. Feeding all of them into
COLMAP + full-res training on a laptop is impractical. Tiled mode:

1. Splits time into overlapping windows (~25s, 5s overlap; GPS path-length aware)
2. Keeps **much more than 1–2 fps** via adaptive sampling (base ~6 fps, up to ~15 on turns)
3. Reconstructs each chunk independently (YOLO MPS → COLMAP → metal_equirect)
4. Estimates a Sim3 per chunk from COLMAP cameras → GPS/gyro world
5. Transforms + merges PLYs with splat-transform → final `.ply` / `.sog`

## Run

```bash
# Recommended one-liner
instasplat run --large-8k -i ./capture_equirect_8k.mp4 -o ./runs -n walk_8k

# Or write a starter config
instasplat init-config --large-8k -o large8k.yaml
instasplat run -c large8k.yaml
```

GUI: enable **Large 8K@30 tiled mode**.

## Metal utilization

| Stage | Metal path |
|-------|------------|
| YOLO people masks | PyTorch **MPS** (`mask.device=mps`) |
| metal_equirect training | **PyTorch MPS** (serialized per chunk by default) |
| COLMAP | CPU on macOS (parallelism limited to avoid RAM thrash) |
| Cubemap remap | CPU OpenCV (per-chunk) |

`metal.serialize_train: true` (default) processes one train job at a time to
avoid Metal memory pressure. Raise `chunk.max_parallel_chunks` only if you have
headroom and set `serialize_train: false`.

## Telemetry alignment

- **GPS** → metric world translation/scale (Umeyama Sim3 on camera centers)
- **Gyro** → turn detection for denser frames + orientation prior / overlap chaining
- **Overlap windows** → chain-align when GPS is sparse

Outputs:

```
runs/<job>/
  10_chunks/manifest.json
  10_chunks/chunk_XXX/...
  10_chunks/alignments.json
  11_merged/scene_merged.ply
  11_merged/hierarchy_manifest.json
  11_merged/lod/lod_levels.json      # CPU preview LODs
  07_nerfstudio/transforms.json
  06_export/scene.ply
  06_export/scene.sog
  06_export/scene.spz
  quality.json
  cloud_job.json                     # 3DGUT / LichtFeld worker handoff
```

```bash
instasplat validate --job ./runs/walk_8k   # capture / align health grade
```

## Tuning for “use as much data as possible”

| Knob | Effect |
|------|--------|
| `chunk.base_fps` ↑ | denser sampling on straight motion |
| `chunk.max_fps` ↑ | denser on turns (still << 30) |
| `chunk.max_frames_per_chunk` ↑ | more views/tile (heavier SfM/train) |
| `chunk.duration_sec` ↓ | more tiles, better locality, more merge seams |
| `chunk.overlap_sec` ↑ | stronger alignment constraints |
| `sfm.face_resolution` ↑ | sharper cubemap faces from 8K source |
| `train.max_resolution` ↑ | higher-fidelity splat (Metal VRAM bound) |

Practical starting point for M-series: `base_fps=6`, `max_fps=15`,
`duration_sec=25`, `face_resolution=1280`, `total_steps=20000` per chunk.
