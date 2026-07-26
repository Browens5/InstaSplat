# metal_equirect trainer (architecture)

Technical reference for the **full-360 training stage** inside InstaSplat’s Mac
pipeline. Pipeline need & architecture: **[MAC_360_PIPELINE.md](MAC_360_PIPELINE.md)**.
Runbook: **[METAL_SPLAT_WORKFLOW.md](METAL_SPLAT_WORKFLOW.md)**.

## Role in the full pipeline

Without a trainer that can supervise **entire panoramas**, a Mac 360 → splat
path collapses to pinhole crops and loses most of each frame. `metal_equirect`
is that stage: it optimizes 3D Gaussians against full equirectangular views
using COLMAP poses. Projection uses a **3DGUT-style Unscented Transform**
(nonlinear camera), inspired by NVIDIA 3DGUT / [gsplat](https://github.com/nerfstudio-project/gsplat).

On Mac, COLMAP usually runs on **cubemap** faces; training lifts `{stem}_front`
poses back to panoramas so every pixel of the 360 frame can supervise the model.

## Data flow

```text
equirect frames (01_frames/equirect)
        +
COLMAP sparse (cubemap-lifted or EQUIRECTANGULAR)
        │
        ▼
 Dataset assembly  →  GaussianModel (torch)
        │
        ▼
 Equirect rasterizer
   1. world → camera
   2. UT sigma-points          ← 3DGUT idea
   3. equirect project         lon/lat → uv
   4. tile / OIT α-blend
   5. optional eval3d opacity
        │
        ▼
 Latitude-weighted L1 (+ structure) → Adam
        │
        ▼
 MCMC densify/prune → PLY + preview JPEGs
```

| Module | Path |
|--------|------|
| Cameras / UT | `instasplat/metal_equirect/cameras.py` |
| Dataset / pose lift | `instasplat/metal_equirect/dataset.py` |
| Rasterizer | `instasplat/metal_equirect/rasterize.py` |
| Densify | `instasplat/metal_equirect/densify.py` |
| Train loop | `instasplat/metal_equirect/train_loop.py` |
| Pipeline entry | `instasplat/metal_equirect/backend.py` |
| Metal shaders | `instasplat/metal_equirect/metal/EquirectProject.metal` |

## CLI

```bash
instasplat run --job ./runs/walk --only train
instasplat train-equirect -j ./runs/walk --steps 8000 --composite oit
```

## Config

```yaml
train:
  backend: metal_equirect
  total_steps: 15000
  max_resolution: 1024
  export_every: 2000
  sh_degree: 1
  lr: 0.01
  with_eval3d: true
  composite: tile          # tile | oit
  sh_warmup_steps: 500
  densify_every: 200
```

## Device policy

| Device | Behavior |
|--------|----------|
| Apple Silicon + MPS | Default training device |
| CPU | Fallback / CI |
| Metal metallib | Compiled on macOS when `xcrun metal` exists; torch UT path always works |

Cloud CUDA **gsplat 3DGUT** remains optional for scale (`cloud_job.json`).

## Roadmap

1. ~~UT equirect rasterizer, train loop, PLY~~
2. ~~Tile / OIT composite, eval3d, MCMC densify, SH warmup~~
3. ~~Previews, heartbeat, `train-equirect`, images.bin~~
4. Native Metal UT dispatch (metallib present; torch path active)
5. Higher-throughput tile sort (gsplat parity)
