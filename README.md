# InstaSplat

Local macOS pipeline: **Insta360 `.insv` → frames + gyro → YOLO people masks → COLMAP sparse cloud/cameras → Brush 3D Gaussian splat → `.ply` / `.sog` (via splat-transform)**.

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gui,dev]"
./scripts/setup_macos.sh   # brew/npm/cargo helpers

# Check what can run on this Mac
instasplat doctor

# Run (prefer a Studio-exported equirect MP4 on macOS)
instasplat run -i /path/to/capture_equirect.mp4 -o ./runs -n beach_walk --formats ply,sog

# Or launch the desktop UI
instasplat gui
```

## Pipeline

| Stage | What it does | Primary tools |
|-------|----------------|---------------|
| `ingest` | Resolve video source, parse INSV trailer (gyro/accel/GPS) | exiftool, custom trailer parser |
| `extract` | Sample frames at N fps | ffmpeg |
| `mask` | Segment people → Brush/COLMAP ignore masks | Ultralytics YOLO (**MPS / Metal**) |
| `sfm` | Cubemap (or equirect) views → sparse cloud + poses | COLMAP |
| `scale` | Optional metric scale | known distance / GPS |
| `train` | 3D Gaussian splat training | [Brush](https://github.com/ArthurBrussee/brush) (**Metal/WebGPU**) |
| `export` | Format conversion + transforms | [splat-transform](https://github.com/playcanvas/splat-transform) |
| `plan_chunks` / `process_chunks` / `align_chunks` / `merge_chunks` | **Large 8K tiled mode** | gyro/GPS + Metal |

Outputs land under `runs/<project>/06_export/` (`scene.ply`, `scene.sog`, …).

### Large 8K@30fps (tiled)

```bash
instasplat run --large-8k -i ./capture_equirect_8k.mp4 -o ./runs -n walk_8k
```

Auto-chunks the capture, densifies frames on turns using gyro, reconstructs each
tile on Metal, aligns tiles with GPS/gyro Sim3, and merges into one large splat.
See **[docs/LARGE_8K.md](docs/LARGE_8K.md)**.

## macOS stitching reality check

Official **Insta360 MediaSDK** targets **Windows / Ubuntu**, not macOS. For production quality on a Mac:

1. Stitch + export **equirectangular MP4** in **Insta360 Studio**, then point InstaSplat at that `.mp4`, or
2. Keep the `.insv` beside a Studio export named `*.mp4` / `*_equirect.mp4` / `studio_export.mp4`, or
3. Offload stitching to a **Linux Docker/cloud** worker with MediaSDK (see [docs/CLOUD.md](docs/CLOUD.md)).

`ffmpeg_fallback` remuxes the INSV container for dry testing; it is **not** a proper optical-flow stitch.

## True-to-scale

COLMAP units are arbitrary unless you set `scale.mode`:

- `known_distance` — measure a real-world distance between two reconstructed point IDs
- `gps` — use GPS from the INSV trailer (when present)
- `stereo_baseline` — reserved for dual-lens metric cues (documented; prefer known distance/GPS today)
- `none` — visual splat only (default)

## Feasibility & cloud

See **[docs/FEASIBILITY.md](docs/FEASIBILITY.md)** and **[docs/CLOUD.md](docs/CLOUD.md)**. Short version: the full stack is workable on Apple Silicon for short clips; long 8K captures and large COLMAP jobs are better hybrid (local prep + cloud SfM/train).

## Config

```bash
instasplat init-config -o instasplat.yaml
instasplat run -c instasplat.yaml
```

## License

MIT. Upstream tools keep their own licenses (COLMAP BSD, Brush Apache-2.0, YOLO AGPL/Ultralytics terms, splat-transform MIT, Insta360 SDK proprietary).
