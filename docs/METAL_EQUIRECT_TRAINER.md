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

On Mac, COLMAP runs on **EQUIRECTANGULAR** panoramas (COLMAP ≥ 4.1) so every
pixel of the 360 frame can supervise the model. Legacy cubemap `{stem}_front`
poses are still lifted if present.

## Dataset pairing

COLMAP `NAME` is matched to panoramas under `01_frames/equirect` or
`03_sfm/images_equirect` in this order:

1. Exact basename (and basename of any directory prefix)
2. Case-insensitive stem
3. **Unique trailing frame index** — so `e_000001.jpg` pairs with
   `frame_000001.jpg` (and the reverse)

`images.bin` is read with COLMAP’s official `idddddddi` layout (int32
`image_id`). Quaternions are unit-normalized before `qvec_to_rotmat`.

## Data flow

```text
equirect frames (01_frames/equirect)
        +
COLMAP sparse (EQUIRECTANGULAR; legacy cubemap-lifted OK)
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
  export_every: 500        # incremental equirect_XXXXXX.ply (GUI: 50–1000)
  sh_degree: 1             # 0–3 spherical harmonics
  max_gaussians: 40000     # per tile/job (GUI: up to 30M, shown in thousands)
  viewer_every: 100        # async subsampled live.ply for GUI viewer
  lr: 0.01
  with_eval3d: true
  composite: metal         # metal (fused forward) | oit (torch) | tile
  sh_warmup_steps: 500
  densify_every: 200
```

GUI **Training** group exposes steps and max Gaussians in **thousands**
(`1 k = 1,000`; max Gaussians selectable up to `30,000 k` = 30M per tile/job),
plus SH degree and PLY export interval. The live viewer reloads a **subsampled**
`live.ply` every `viewer_every` steps (default 100, written asynchronously).

### Speed path (Mac)

- **Fused Metal soft-OIT** (`composite: metal`) — Metal forward + torch backward;
  pooled shared MTLBuffers (one D2H / one H2D per step); CPU reference mirrors
  the kernel when PyObjC/metallib is unavailable
- **Vectorized torch OIT** (`composite: oit`) — batched soft splat via `index_add`
- **View cache** — decode panoramas once into RAM/MPS
- **Resolution schedule** — half-res / large tiles early → full-res / fine tiles late
- **Async live preview** — subsampled PLY so training is not blocked by export

## Device policy

| Device | Behavior |
|--------|----------|
| Apple Silicon + MPS | Default training device (all Gaussian params on MPS) |
| CPU | Auto-fallback if an MPS probe forward fails; also CI |
| Metal metallib | Soft-OIT kernels via PyObjC; else CPU-ref fused forward |

Covariance transforms use batched matmul (`R Σ Rᵀ`), not `einsum`, to avoid
PyTorch MPS “Placeholder storage has not been allocated” crashes.

Cloud CUDA **gsplat 3DGUT** remains optional for scale (`cloud_job.json`).

## Roadmap

1. ~~UT equirect rasterizer, train loop, PLY~~
2. ~~Tile / OIT composite, eval3d, MCMC densify, SH warmup~~
3. ~~Previews, heartbeat, `train-equirect`, images.bin~~
4. ~~Vectorized OIT, view cache, schedule, async live PLY~~
5. ~~Fused Metal soft-OIT metallib (+ optional PyObjC)~~
6. ~~Metal-first composite (`composite: metal`) with torch backward~~
7. ~~Pooled MTL shared buffers (one D2H/H2D per step, no numpy round-trip)~~
8. True zero-copy MPS↔MTLBuffer (PyTorch MPS storage interop) when exposed
