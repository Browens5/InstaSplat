# Running the Mac 360 → splat pipeline

Operational runbook: install, capture, run, monitor, tune, and troubleshoot.

For **why this pipeline exists** and **how it is architected**, read
**[MAC_360_PIPELINE.md](MAC_360_PIPELINE.md)** first.

Training backend is **metal_equirect** only (full equirect + COLMAP on MPS/Metal).

---

## 1. Pipeline at a glance

```mermaid
flowchart TB
  A["Studio equirect MP4\n(+ .insv / gyro+gps)"] --> B["ingest + extract"]
  B --> C["YOLO masks MPS"]
  C --> D["Stage equirect\nimages_equirect"]
  D --> E["COLMAP EQUIRECTANGULAR"]
  E --> F["Scale / refine"]
  F --> G["metal_equirect train\nsame panoramas + poses"]
  G --> H["export PLY / SOG / SPZ"]

  style G fill:#dceee4,stroke:#1f6f4a
```

| Term | Meaning |
|------|---------|
| **Equirect frames** | Full 360×180 panoramas (`01_frames/equirect/`) |
| **EQUIRECTANGULAR SfM** | COLMAP ≥ 4.1 on full panoramas (`03_sfm/images_equirect`) |
| **SfM mapper** | `incremental` (default) or `global` (GLOMAP / `colmap global_mapper`) |
| **metal_equirect** | PyTorch MPS trainer on the same panoramas + COLMAP poses |
| **Exports** | `scene.ply` (+ sog/spz via splat-transform) |

Trainer internals: [METAL_EQUIRECT_TRAINER.md](METAL_EQUIRECT_TRAINER.md).  
Tiling: [MAC_LONG_360.md](MAC_LONG_360.md).

---

## 2. Install (once)

### Recommended

```bash
git clone https://github.com/Browens5/InstaSplat.git
cd InstaSplat
./scripts/setup_macos.sh
source .venv/bin/activate
instasplat doctor                 # mac_long_360 should be yes
```

| Env | Effect |
|-----|--------|
| `WITH_GUI=0` | Skip PySide6 |
| `SKIP_BREW=1` | Do not run Homebrew |
| `SKIP_PIP=1` | Do not create venv / pip install |

### Manual equivalent

```bash
brew install ffmpeg exiftool colmap git
npm i -g @playcanvas/splat-transform
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gui,dev]"
instasplat setup --verify
instasplat doctor
```

```bash
instasplat setup                  # may install missing brew/npm tools
instasplat setup --verify         # check only
instasplat doctor                 # stage readiness + MPS
```

If Apple Silicon shows PyTorch without MPS, install the wheel from
[pytorch.org](https://pytorch.org), then re-run `doctor`.

More: [SETUP_MACOS.md](SETUP_MACOS.md).

---

## 3. Capture checklist

1. Record with an Insta360 (walk slowly; good overlap; fewer crowds).
2. In **Insta360 Studio**, export **equirectangular** MP4 (not a flat reframe).
3. Keep the sibling `.insv` next to the MP4 (gyro/GPS), **or** add `gyro.csv` / `gps.csv`.

MediaSDK does **not** run on macOS — Studio stitch is the supported local path.
See [CAPTURE_GUIDELINES.md](CAPTURE_GUIDELINES.md).

---

## 4. Choose a run mode

| Mode | When | Command |
|------|------|---------|
| **Tiled (recommended)** | Long walks, 8K, multi-minute clips | `instasplat mac-360 -i …` |
| **Single** | Short clips / experiments | `instasplat run -i …` (no `--large-8k`) |
| **GUI** | Same pipelines, visual controls | `instasplat gui` |

Tiled mode splits time into overlapping chunks, trains each with metal_equirect
(serialized on MPS), then GPS/gyro-aligns and merges.

---

## 5. Stage-by-stage

### A. Tiled long-360 (default product path)

```bash
instasplat mac-360 -i ./capture_equirect.mp4 -o ./runs -n walk_360
```

| Order | Stage | What happens | Key outputs |
|------:|-------|--------------|-------------|
| 1 | `ingest` | Copy/link video; pull gyro/GPS | `00_ingest/equirect.mp4`, `gyro.csv`, `gps.csv` |
| 2 | `plan_chunks` | Overlapping time windows; denser fps on turns | `10_chunks/manifest.json` |
| 3 | `preflight` | ffmpeg/colmap/torch/disk/stitch guards | `preflight.json` |
| 4 | `process_chunks` | Per tile: mask → EQUIRECTANGULAR COLMAP → scale → refine → **train** → export | `10_chunks/chunk_*/…` |
| 5 | `align_chunks` | Sim3 align with GPS/gyro | alignment reports |
| 6 | `merge_chunks` | splat-transform merge + prune | `11_merged/scene_merged.ply` |
| 7 | `package` | quality + LOD + cloud handoff | `quality.json`, `cloud_job.json`, `06_export/` |

Inside each tile’s **train** step:

1. Load equirect frames + masks  
2. Use COLMAP EQUIRECTANGULAR poses (same panorama names)  
3. Init Gaussians from sparse points  
4. Optimize with UT equirect rasterizer (MPS)  
5. Densify / prune; write previews + PLY  

```text
10_chunks/chunk_000/
  01_frames/equirect/
  02_masks/equirect/
  03_sfm/images_equirect/   ← full panoramas for COLMAP
  03_sfm/sparse/0/          ← EQUIRECTANGULAR cameras
  05_train/exports/
    scene.ply
    previews/step_*.jpg
    train_heartbeat.json
```

### B. Single-job (short capture)

```bash
instasplat run -i ./short_equirect.mp4 -o ./runs -n short \
  --stages ingest,extract,mask,sfm,scale,refine,train,export,package
```

### C. Resume / train only

```bash
instasplat run --job ./runs/walk_360 --only process_chunks
instasplat train-equirect -j ./runs/walk_360 --steps 8000
instasplat run --job ./runs/walk_360 --only train
```

GUI: **Open previous run…** → check stages → **Continue run**.

---

## 6. Watching progress

| Signal | Where |
|--------|--------|
| CLI / GUI log | Stage messages + loss / Gaussian count |
| Live 3D viewer | Sparse COLMAP during SfM; splat centers when PLYs appear |
| Artifacts tab | Frames, masks, models, PLYs, preview JPEGs |
| Heartbeat | `05_train/exports/train_heartbeat.json` |
| Previews | `05_train/exports/previews/step_XXXXXX.jpg` |

Pause in the GUI freezes the pipeline at stage checkpoints.

---

## 7. Outputs

```text
runs/walk_360/
  06_export/scene.ply
  06_export/scene.sog
  06_export/scene.spz
  11_merged/…
  quality.json
  cloud_job.json
```

Viewers: PlayCanvas SuperSplat, MetalSplatter, and other Gaussian viewers.

---

## 8. Config knobs (train)

```yaml
sfm:
  mode: equirectangular
  mapper: incremental       # or global (GLOMAP) for speed on large tiles

train:
  backend: metal_equirect
  total_steps: 15000
  max_resolution: 1024
  export_every: 2000
  sh_degree: 1
  lr: 0.01
  with_eval3d: true
  composite: tile           # tile | oit
  sh_warmup_steps: 500
  densify_every: 200

metal:
  prefer_metal: true
  serialize_train: true     # one tile at a time on MPS
```

| Goal | Suggestion |
|------|------------|
| Faster iterate | `total_steps: 8000`, `max_resolution: 768`, `composite: oit` |
| Higher quality | `total_steps: 20000`, `max_resolution: 1280`, `sfm.face_resolution: 1280` |
| Less memory | keep `serialize_train: true`, lower `max_resolution` |

Starter YAML: `examples/large8k.example.yaml`, `instasplat init-config --large-8k`.

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `doctor` train = no | `pip install -e .`; MPS wheel on Apple Silicon |
| Preflight: no trainer | Same — PyTorch required |
| Empty COLMAP / SfM fail | Need COLMAP ≥ 4.1 (`brew upgrade colmap`); tiled jobs use `process_chunks`, not top-level `sfm` |
| No EQUIRECTANGULAR | Upgrade COLMAP to ≥ 4.1 — cubemap fallback is no longer automatic |
| Train: no equirect views | Need `01_frames/equirect` and matching EQUIRECTANGULAR COLMAP image names |
| MPS OOM mid-train | Lower `max_resolution` / `total_steps`; keep `serialize_train` |
| YOLO MPS crash | Pipeline falls back to CPU for masks |
| Merge blocked | Fix failed tiles or `--allow-partial-merge` |

---

## 10. Cheat sheet

```bash
./scripts/setup_macos.sh && source .venv/bin/activate
instasplat setup
instasplat doctor
instasplat mac-360 -i ./capture_equirect.mp4 -o ./runs -n walk
instasplat gui
instasplat stages --mode tiled
instasplat run --job ./runs/walk --only train
instasplat train-equirect -j ./runs/walk --steps 8000
instasplat validate --job ./runs/walk
```

---

## See also

- [MAC_360_PIPELINE.md](MAC_360_PIPELINE.md) — need + how the pipeline is built  
- [SETUP_MACOS.md](SETUP_MACOS.md) — install details  
- [MAC_LONG_360.md](MAC_LONG_360.md) — tiled defaults  
- [METAL_EQUIRECT_TRAINER.md](METAL_EQUIRECT_TRAINER.md) — rasterizer / UT  
- [LARGE_8K.md](LARGE_8K.md) — chunk tuning  
- [CLOUD.md](CLOUD.md) — optional CUDA path  
