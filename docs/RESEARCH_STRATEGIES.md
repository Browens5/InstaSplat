# Research strategies → InstaSplat improvements

Synthesis of open-source 3DGS / radiance-field projects and how they should shape
InstaSplat’s Mac-local, tiled 8K pipeline.

## Priority matrix

| Priority | Strategy | Source | InstaSplat action |
|----------|----------|--------|-------------------|
| P0 | Native equirect trainer (Metal/MPS) | metal_equirect (3DGUT-inspired) | `train.backend = metal_equirect` |
| P0 | Keep sequential + overlapping capture | on-the-fly-nvs, LongSplat | Capture guidelines + overlap checks |
| P0 | Compressed delivery (SPZ / streamed SOG) | spz, splat-transform, LichtFeld | Default large-8k exports include `spz` |
| P1 | Hierarchical / LOD after tile merge | hierarchical-3d-gaussians, threedtiles | Optional LOD / streamed-SOG export |
| P1 | Equirect / distorted-camera training | Self-Cali-GS, 3DGUT/LichtFeld | Prefer native equirect when trainer supports GUT |
| P1 | Anchor-style coarsen distant Gaussians | on-the-fly-nvs, LongSplat | Post-merge prune + future anchor LOD |
| P2 | Incremental / unposed pose fallback | LongSplat (MASt3R), on-the-fly-nvs | Optional unposed path when COLMAP fails |
| P2 | Joint pose–appearance refine | Self-Cali-GS, SplaTAM, RTG-SLAM | Optional BA / pose refine stage |
| P3 | CUDA-only power tools | LichtFeld, 3dgrut, instant-ngp | Cloud worker backends only |
| P3 | RGB-D SLAM densification | RTG-SLAM, SplaTAM | Future depth/LiDAR path |

---

## Project-by-project takeaways

### LongSplat (NVlabs) — long casual video
- **Unposed** incremental 3DGS with MASt3R poses + adaptive **octree anchors**.
- Practical capture advice: ~10 fps subsample, resize long runs (~512px width for speed), avoid dynamics, keep overlap.
- Convert specialized representation back to standard 3DGS PLY with pre-pruning.
- **Use:** inspire denser-than-2fps sampling (we already do 6–15), add prune-before-merge, consider unposed fallback.

### LichtFeld Studio (MrNeRF) — production workstation
- Single native app: train, inspect, edit, export (`PLY`/`SOG`/`SPZ`/HTML).
- Research features: MCMC, bilateral grid, **3DGUT** for distorted cameras.
- **NVIDIA/CUDA-first** — not a Mac Metal trainer; good cloud/secondary backend + UX reference.
- **Use:** export parity (SPZ/HTML), plugin/MCP automation ideas, GUT/equirect training on cloud GPU.

### on-the-fly-nvs (graphdeco) — scalable sequential reconstruction
- Joint pose + Gaussian optimization for **ordered** sequences (not COLMAP drop-in).
- **Sliding-window anchors**: offload sub-pixel Gaussians to CPU, cluster, merge neighbors (Kerbl hierarchy-style).
- Trigger when >40% of active Gaussians are <1px from previous camera.
- **Use:** model our tile merge as “anchors along the path”; add post-merge coarsening; enforce sequential capture guidelines.

### 3dgrut (nv-tlabs) — 3DGRT / 3DGUT
- Ray-traced Gaussians + **3DGUT** for distorted / rolling-shutter cameras inside rasterization.
- Production tip: use **gsplat** for modular training; 3DGUT is the path for true equirect/fisheye without cubemap.
- **Use:** cloud CUDA scale path; local Mac uses metal_equirect (cubemap SfM + equirect train).

### threedtiles (ebeaufay) — streaming large scenes
- 3D Tiles viewer for three.js (LOD, streaming).
- **Use:** long-term delivery format for city-scale walks; near-term use splat-transform streamed SOG / hierarchical LOD instead of inventing tiles.

### Self-Cali-GS — large-FOV calibration
- Joint optimize extrinsics + intrinsics + **distortion** (iResNet); cubemap sampling for wide FOV.
- COLMAP with `OPENCV_FISHEYE` then refine.
- **Use:** critical for Insta360 optical quality — add optional distortion-refine stage; prefer fisheye-aware SfM before splat train.

### awesome-3D-gaussian-splatting (MrNeRF)
- Living index of trainers, viewers, converters.
- Notable utilities: **splatreg** (Sim3 align/merge), SuperSplat, gsplat, OpenSplat, Metal viewers.
- **Use:** adopt splatreg-quality merge when our Umeyama is weak; keep watching hierarchical / compression papers.

### spz (Niantic)
- ~10× smaller than PLY; v4 ZSTD streams; explicit coordinate systems (RUB default).
- **Use:** first-class export; document RUB vs COLMAP/RDF when packing.

### hierarchical-3d-gaussians (graphdeco)
- Chunk → optimize → **hierarchy generator/merger** → real-time LOD viewer.
- Monocular depth priors (Depth Anything V2) help large outdoor scenes.
- **Use:** after our GPS/gyro tile merge, build hierarchical LOD for viewing; optional depth prior stage.

### RTG-SLAM — real-time RGB-D GS-SLAM
- Opaque vs near-transparent Gaussians; optimize only **unstable** Gaussians; selective pixel render.
- **Use:** densification heuristics if we add depth; not primary for monocular Insta360.

### MetalSplatter (scier) — Apple Metal viewer
- Swift/Metal renderer for PLY/SPZ/.splat on iOS/macOS/visionOS.
- **Use:** recommended Mac viewer for InstaSplat exports; not a trainer.

### OpenSplat (pierotofy) — portable C++ trainer
- COLMAP/OpenSfM/ODM/nerfstudio in → PLY/splat out.
- **Metal (`-DGPU_RUNTIME=MPS`)**, CUDA, HIP, or CPU.
- Resume training; AGPL license.
- **Use (historical):** evaluated as a Mac trainer; InstaSplat’s product path is
  **metal_equirect** (full equirect) instead.

### instant-ngp (NVlabs)
- Hash-grid NeRF speed culture; CUDA.
- **Use:** cloud optional NeRF path only; not Mac-primary.

### nerfstudio
- Dataset conventions, splatfacto, viewers, gsplat backend.
- **Use:** optional export of Nerfstudio transforms.json; cloud splatfacto/gsplat+3DGUT.

### SplaTAM — RGB-D Gaussian SLAM
- Splat, track, map; post-opt with classic 3DGS; PLY export.
- **Use:** if we later support iPhone LiDAR / RGB-D companions alongside Insta360.

---

## Concrete InstaSplat roadmap

### Now (implemented or wiring)
1. Trainer: `metal_equirect` (sole backend; equirect + COLMAP).
2. Large-8k default exports: `ply`, `sog`, `spz` (+ optional streamed LOD).
3. Capture guidelines doc (sequential, overlap, speed, dynamics).
4. Post-merge opacity/NaN prune (LongSplat-style size control).
5. Pose refine: COLMAP BA + GPS/gyro blend (Self-Cali-inspired).
6. Robust tile align: RANSAC Umeyama + ICP + RMSE quality gate.
7. Telemetry pose fallback when COLMAP fails.
8. Nerfstudio `transforms.json` + hierarchy anchor manifest packaging.
9. Capture validation + `quality.json` grade (overlap / speed / align RMSE).
10. `cloud_job.json` for 3DGUT / LichtFeld / splatfacto workers.
11. CPU LOD previews from hierarchy anchors (XYZ+opacity; Kerbl CUDA still cloud).

### Local Mac long-360 (shipped)
See `instasplat mac-360` / [MAC_LONG_360.md](MAC_LONG_360.md): windowed ffmpeg extract,
preflight gates, INSV/sidecar telemetry, scale-before-refine, resume tiles, merge checks.

### Next (cloud / research)
1. **LingBot-Map** as CUDA cloud SfM backend (streaming poses+depth) → COLMAP/Nerfstudio export.
2. **Native equirect training** — local `metal_equirect` backend (UT equirect rasterizer;
   see [METAL_EQUIRECT_TRAINER.md](METAL_EQUIRECT_TRAINER.md)); cloud 3DGUT/gsplat remains
   the CUDA scale path (`cloud_job.json` worker runner TBD).
3. **Kerbl hierarchy merger** binary integration for true LOD trees.
4. Full Self-Cali distortion network (iResNet) for raw fisheye.

### Later / cloud-only
1. LichtFeld or 3dgrut workers for CUDA MCMC / GUT.
2. 3D Tiles packaging for web GIS viewers.
3. RGB-D SLAM densify path (RTG-SLAM / SplaTAM) with depth sensors.

---

## Mac Metal reality check

| Component | Best open option on Mac |
|-----------|-------------------------|
| Train | **metal_equirect** (native 360, MPS/Metal) |
| View | MetalSplatter, PlayCanvas, InstaSplat GUI |
| Compress | splat-transform → SOG/SPZ |
| Distorted 360 train | **metal_equirect** locally; 3DGUT/LichtFeld in cloud for scale |
| Large LOD | Hierarchical-3DGS merger (CUDA) or streamed SOG |

---

## References

- https://github.com/NVlabs/LongSplat
- https://github.com/MrNeRF/LichtFeld-Studio
- https://github.com/graphdeco-inria/on-the-fly-nvs
- https://github.com/nv-tlabs/3dgrut
- https://github.com/ebeaufay/threedtiles
- https://github.com/denghilbert/Self-Cali-GS
- https://github.com/MrNeRF/awesome-3D-gaussian-splatting
- https://github.com/nianticlabs/spz
- https://github.com/graphdeco-inria/hierarchical-3d-gaussians
- https://github.com/MisEty/RTG-SLAM
- https://github.com/scier/MetalSplatter
- https://github.com/pierotofy/OpenSplat
- https://github.com/NVlabs/instant-ngp
- https://github.com/nerfstudio-project/nerfstudio
- https://github.com/spla-tam/SplaTAM
