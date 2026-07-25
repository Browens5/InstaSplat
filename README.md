# InstaSplat

Local macOS pipeline: **Insta360 `.insv` → frames + gyro → YOLO people masks → COLMAP sparse cloud/cameras → Brush 3D Gaussian splat → `.ply` / `.sog` (via splat-transform)**.

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gui,dev]"
./scripts/setup_macos.sh   # brew/npm + auto-build Brush
instasplat install-brush   # if Brush not on PATH yet

# Check what can run on this Mac
instasplat doctor

# Best local Mac path: long 360 → tiled Metal splat
instasplat mac-360 -i /path/to/capture_equirect_8k.mp4 -o ./runs -n beach_walk

# Short clip (single scene)
instasplat run -i /path/to/capture_equirect.mp4 -o ./runs -n beach_walk --formats ply,sog

# Desktop UI
instasplat gui
```

## Pipeline

| Stage | What it does | Primary tools |
|-------|----------------|---------------|
| `ingest` | Resolve video source, parse INSV trailer (gyro/accel/GPS) | exiftool, custom trailer parser |
| `extract` | Sample frames at N fps | ffmpeg |
| `mask` | Segment people → Brush/COLMAP ignore masks | Ultralytics YOLO (**MPS / Metal**) |
| `sfm` | Cubemap (or equirect) views → sparse cloud + poses | COLMAP (+ gyro/GPS fallback) |
| `refine` | Bundle adjust + GPS/gyro pose blend | COLMAP BA |
| `scale` | Optional metric scale | known distance / GPS |
| `train` | 3D Gaussian splat training | Brush or OpenSplat (**Metal**) |
| `export` | Format conversion + transforms | [splat-transform](https://github.com/playcanvas/splat-transform) |
| `package` | Nerfstudio / hierarchy / cloud_job / quality | local packaging |
| `plan_chunks` / `process_chunks` / `align_chunks` / `merge_chunks` | **Large 8K tiled mode** | gyro/GPS + Metal |

Outputs land under `runs/<project>/06_export/` (`scene.ply`, `scene.sog`, …).
Large jobs also write `quality.json`, `cloud_job.json`, and optional CPU LOD previews.

### Large / long 360 (tiled, Mac-best)

```bash
instasplat mac-360 -i ./capture_equirect_8k.mp4 -o ./runs -n walk_360
# aliases: --large-8k / --tiled
```

Auto-chunks the capture, densifies frames on turns using gyro, reconstructs each
tile on Metal, aligns tiles with GPS/gyro Sim3, and merges into one large splat.
See **[docs/MAC_LONG_360.md](docs/MAC_LONG_360.md)** and **[docs/LARGE_8K.md](docs/LARGE_8K.md)**.

Trainer backends: `--trainer brush` (default) or `--trainer opensplat` (C++ Metal MPS).
Exports default to `.ply`, `.sog`, and `.spz`. Keep the sibling `.insv` (or `gyro.csv` /
`gps.csv`) next to the Studio MP4 for telemetry.

```bash
instasplat validate --job ./runs/walk_360
```

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

See **[docs/MAC_LONG_360.md](docs/MAC_LONG_360.md)**, **[docs/FEASIBILITY.md](docs/FEASIBILITY.md)**,
**[docs/CLOUD.md](docs/CLOUD.md)**, **[docs/LARGE_8K.md](docs/LARGE_8K.md)**,
**[docs/CAPTURE_GUIDELINES.md](docs/CAPTURE_GUIDELINES.md)**, and
**[docs/RESEARCH_STRATEGIES.md](docs/RESEARCH_STRATEGIES.md)**.

## Config

```bash
instasplat init-config -o instasplat.yaml
instasplat run -c instasplat.yaml
```

## License

MIT. Upstream tools keep their own licenses (COLMAP BSD, Brush Apache-2.0, YOLO AGPL/Ultralytics terms, splat-transform MIT, Insta360 SDK proprietary).
