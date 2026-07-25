# Feasibility — Insta360 → metric Gaussian splat on Mac

## Verdict

**Feasible as a local Mac application**, with one hard caveat: **high-quality INSV stitching is not natively available via Insta360 MediaSDK on macOS**. Everything after a stitched equirectangular MP4 (frames, YOLO masks, COLMAP, metal_equirect, splat-transform) runs well on Apple Silicon. For a polished product, treat stitching as either **Studio-assisted** or **cloud/Linux MediaSDK**.

## Stage-by-stage

### 1. INSV ingest (frames + gyro)

| Aspect | Assessment |
|--------|------------|
| Video container | `.insv` is MP4-like; ffmpeg can remux streams |
| Proper 360 stitch | Needs MediaSDK (Win/Ubuntu) or Insta360 Studio (Mac OK) |
| Gyro / accel / exposure | Trailer records (e.g. `0x300`, `0x400`) are parseable; formats vary by camera generation |
| GPS | Present on some captures; useful for metric scale |
| Dual-file 5.7K | Older models use `_00_` / `_10_` pairs; X4+ often single file dual-track |

**Risk:** Relying on raw dual-fisheye without Studio/MediaSDK yields poor SfM and splat quality.

### 2. YOLO people masking

| Aspect | Assessment |
|--------|------------|
| Runtime | Ultralytics YOLO-seg on **MPS** is practical for 2–5 fps extractions |
| Integration | metal_equirect supports equirect masks; COLMAP accepts mask paths during feature extraction |
| Quality | Works for tourists/operators; fails on heavy occlusion, mirrors, tiny distant figures |
| License | Ultralytics YOLO licensing must be reviewed for commercial redistribution |

**Risk:** Over-masking removes useful structure; under-masking leaves “ghost people” in the splat.

### 3. COLMAP sparse reconstruction

| Aspect | Assessment |
|--------|------------|
| Cubemap / perspective faces from equirect | **Reliable** with stock Homebrew COLMAP on Mac |
| Native `EQUIRECTANGULAR` model | Available in newer COLMAP; may need building from source |
| Apple Silicon | CPU mapper is fine for hundreds of faces; thousands get slow |
| Texture-poor scenes | Water, sky, snow → weak matches; gyro priors help but need custom integration |

**Risk:** Long walking videos need frame decimation + sequential matching; exhaustive matching does not scale.

### 4. metal_equirect training (Mac-native)

| Aspect | Assessment |
|--------|------------|
| Platform | Designed for **macOS / Metal / WebGPU** — best open trainer fit for this app |
| Input | COLMAP or Nerfstudio layouts |
| Masks | First-class |
| Output | PLY / compressed PLY (CLI flags vary by version) |

**Risk:** CLI flags evolve; InstaSplat keeps a fallback invocation and documents manual export from the viewer.

### 5. splat-transform exports

| Aspect | Assessment |
|--------|------------|
| `.ply` ↔ `.sog` | First-class |
| Also | `spz`, `glb`, `html`, `csv`, compressed PLY, filters, rotate/translate/scale |
| Install | `npm i -g @playcanvas/splat-transform` |

**Risk:** Low. This is the most stable “export options” layer.

### 6. True-to-scale

Gaussian splats inherit COLMAP’s **unknown global scale** unless constrained:

1. **Known distance** between two 3D points (most reliable for indoor/set pieces)
2. **GPS path length** vs camera path (outdoor, needs decent GPS)
3. **Stereo baseline** of dual fisheyes (~6–7 cm class) — theoretically metric, operationally fiddly
4. Survey markers / AprilTags (recommended enhancement)

Without one of these, the splat looks right but is **not** metrically true.

## Hardware expectations (Apple Silicon)

| Capture | Extract @ 2 fps | YOLO | COLMAP | metal_equirect 15k steps |
|---------|-----------------|------|--------|-----------------|
| 30 s, 5.7K | Light | Light | Moderate | Moderate (M-series GPU) |
| 5 min walk, 8K | Heavy disk | Heavy | Heavy CPU | Heavy; may prefer cloud |
| 20 min tour | Usually too much locally | Batch/cloud | Cloud recommended | Cloud or chunked scenes |

Rough local sweet spot: **short clips**, **2 fps or less**, **≤ ~200–400 training views** after cubemap expansion (remember: 1 equirect × 6 faces).

For **long 8K@30** captures, use **tiled mode** (`--large-8k`): overlapping temporal chunks, gyro-adaptive sampling (≈6–15 fps), per-tile metal_equirect on MPS, GPS/gyro Sim3 alignment, splat-transform merge. See `docs/LARGE_8K.md`.

## Product architecture recommendation

```
┌─────────────────────────────────────────────────────────┐
│ Mac app (InstaSplat)                                    │
│  UI + orchestration + YOLO + metal_equirect + splat-transform │
└───────────────┬───────────────────────────┬─────────────┘
                │                           │
        Studio / local MP4           Optional cloud worker
                │                     (MediaSDK + COLMAP)
                └────────────┬──────────────┘
                             ▼
                      metal_equirect (local) or gsplat (cloud)
                             ▼
                      splat-transform exports
```

**Ship local-first** with clear Studio stitching instructions; add cloud workers for MediaSDK + large SfM without blocking the Mac UX.

## Open-source stack (chosen)

| Role | Tool | Why |
|------|------|-----|
| Stitch (Linux/cloud) | Insta360 MediaSDK | Best optical-flow / FlowState |
| Stitch (Mac practical) | Insta360 Studio export | Only robust Mac path today |
| Frames | ffmpeg | Ubiquitous |
| Telemetry | Custom trailer parser + exiftool | No official Mac SDK needed |
| People masks | YOLO-seg | Fast, MPS-friendly |
| SfM | COLMAP | COLMAP cameras/points for training |
| Splats | metal_equirect | Native Mac equirect training |
| Export | splat-transform | `.sog` / `.ply` / more |

Alternatives worth knowing: nerfstudio/gsplat (CUDA-centric), SphereSfM, OpenMVG spherical, Postshot (not open), Lichtfeld Studio.

## Legal / packaging notes

- Bundling **Insta360 MediaSDK** requires their developer agreement; do not redistribute proprietary binaries casually.
- YOLO / PyTorch wheels inflate app size; consider shipping masks as an optional downloadable component.
- A notarized Mac `.app` can wrap the PySide6 GUI + CLI; heavy tools (COLMAP, PyTorch) are better detected on `PATH` or installed via the setup script.
