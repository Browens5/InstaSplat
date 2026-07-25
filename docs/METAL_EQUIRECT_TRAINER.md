# Mac-native Metal equirect Gaussian trainer

InstaSplat’s **`metal_equirect`** training backend optimizes 3D Gaussians
directly against **equirectangular** frames, using COLMAP poses. It is the
local Mac counterpart to cloud `gsplat_3dgut` (see [CLOUD.md](CLOUD.md)).

## Why not Brush?

Brush / OpenSplat expect **pinhole** COLMAP datasets. The cubemap path works
but expands views ~6× and never trains on the full 360×180 pixel domain.
Native equirect training needs a **nonlinear camera projection** in the
rasterizer — the idea behind NVIDIA **3DGUT** (Unscented Transform through
arbitrary projections) as integrated in [gsplat](https://github.com/nerfstudio-project/gsplat).

## Architecture (gsplat / 3DGUT-inspired)

```text
equirect frames (01_frames/equirect)
        +
COLMAP sparse (cubemap-lifted or EQUIRECTANGULAR)
        │
        ▼
┌─────────────────────────────┐
│  Dataset assembly           │  images + masks + viewmats
│  (lift *_front poses → 360) │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  GaussianModel (torch)      │  means, quats, scales, opacity, SH
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Equirect rasterizer        │
│  1. world → camera          │
│  2. UT sigma-points         │  ← 3DGUT idea
│  3. equirect project        │  lon/lat → uv
│  4. 2D cov + soft α-blend   │
│  5. optional Metal kernels  │  projection / tile prep (Mac)
└─────────────┬───────────────┘
              ▼
  L1 + SSIM (latitude-weighted) → Adam → PLY export
```

| Piece | Role | Reference |
|-------|------|-----------|
| Equirect project | \(u,v\) from camera rays | spherical mapping |
| Unscented Transform | project mean+σ points for 2D mean/cov | 3DGUT / gsplat `with_ut` |
| Eval in 3D (planned) | particle response along ray | gsplat `with_eval3d` |
| MCMC densify (v1: clone/split) | grow/prune Gaussians | gsplat MCMC for 3DGUT |
| Metal `.metal` shaders | accelerate project/sort on Apple GPU | Mac-native path |
| PyTorch MPS/CPU | autodiff training loop | portable Mac + CI |

## Usage

```bash
# After SfM (cubemap or equirect) has a sparse model:
instasplat run --job ./runs/walk_360 --only train --trainer metal_equirect

# Or GUI: Trainer → metal_equirect
```

Config (`train` section):

```yaml
backend: metal_equirect
total_steps: 15000
max_resolution: 1024          # equirect width; height = width/2
export_every: 2000
sh_degree: 1
lr: 0.01
```

## Pose sources

1. **Native EQUIRECTANGULAR COLMAP** — image names match equirect stems; cameras marked equirect.
2. **Cubemap SfM (default on Mac)** — take `{stem}_front` pose as the panorama pose (front face = yaw 0 / pitch 0), load `{stem}.jpg` from equirect frames.

## Device policy

| Device | Behavior |
|--------|----------|
| Apple Silicon + PyTorch MPS | Primary training device |
| CPU | Fallback / CI smoke |
| Metal shaders | Compiled on Mac via `xcrun metal` when present; Python UT path always available |

CUDA **gsplat 3DGUT** remains the cloud upgrade for large jobs (`cloud_job.json`).

## Roadmap

1. ~~Core UT equirect rasterizer + train loop + PLY export~~ (this package)
2. Tile-based Metal raster kernels + sorting (parity with gsplat speed)
3. Full MCMC densification + SH degree schedule
4. GUI live preview of equirect renders during training
5. Optional eval3d particle response (closer 3DGUT parity)
