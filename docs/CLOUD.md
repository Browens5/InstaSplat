# Cloud & hybrid processing options

InstaSplat is designed **local-first** on macOS. Use cloud when stitching, SfM, or training exceeds laptop resources, or when you need official MediaSDK.

## What to run where

| Stage | Local Mac | Cloud GPU/CPU | Notes |
|-------|-----------|---------------|-------|
| Studio stitch | ✅ Best UX | ❌ | Manual or Automator-assisted |
| MediaSDK stitch | ❌ Native | ✅ Linux VM/Docker | Official SDK platforms |
| Frame extract | ✅ | ✅ | Cheap either way |
| Gyro parse | ✅ | ✅ | Tiny |
| YOLO masks | ✅ MPS | ✅ CUDA | Cloud wins on long clips |
| COLMAP | ✅ CPU (small) | ✅ Strongly recommended for large sets | |
| Brush train | ✅ Metal | ✅ If you need speed/headless scale | Brush also runs locally well |
| splat-transform | ✅ | ✅ | Lightweight Node CLI |

## Option A — Hybrid (recommended)

1. On Mac: Studio → equirect MP4, or upload raw INSV.
2. Cloud worker: MediaSDK stitch (if needed) → ffmpeg frames → YOLO → COLMAP.
3. Download `sparse/0` + images/masks to Mac.
4. Local Brush training + splat-transform export (interactive viewer).

This keeps the **interactive splat loop** on the Mac (Brush’s strength) while outsourcing the **least Mac-friendly** pieces.

## Option B — Full cloud batch

Package stages as containers:

```text
instasplat-stitch   # Linux + MediaSDK (licensed)
instasplat-prep     # ffmpeg + YOLO
instasplat-sfm      # COLMAP/GLOMAP
instasplat-train    # Brush headless or gsplat CUDA
instasplat-export   # splat-transform
```

Orchestrate with the same `instasplat.yaml` (stages subset per service). Object storage for frames; return `scene.ply` / `scene.sog`.

### Example sketch (Docker Compose)

```yaml
services:
  prep:
    image: instasplat-prep:latest
    volumes: ["./data:/data"]
    command: ["instasplat", "run", "-c", "/data/job.yaml", "--stages", "ingest,extract,mask"]
  sfm:
    image: instasplat-sfm:latest
    depends_on: [prep]
    command: ["instasplat", "run", "-c", "/data/job.yaml", "--stages", "sfm,scale"]
  train:
    image: instasplat-train:latest
    gpus: all
    depends_on: [sfm]
    command: ["instasplat", "run", "-c", "/data/job.yaml", "--stages", "train,export"]
```

MediaSDK images must be built from **your** licensed SDK drop — not redistributed here.

## Option C — Managed 3DGS services

If open-source ops overhead is too high:

- Upload sampled frames + masks to a managed Gaussian training API
- Keep InstaSplat for INSV prep + YOLO + export normalization via splat-transform

Tradeoff: less control over metric scale and masking, recurring cost.

## Provider cheat sheet

| Provider | Good for | Watch-outs |
|----------|----------|------------|
| RunPod / Vast / Lambda | COLMAP + CUDA training | Ephemeral disks; pin versions |
| AWS `g5`/`g6` + CPU for COLMAP | Production pipelines | Egress cost for 8K frames |
| GCP Spot GPUs | Batch jobs | Preemption mid-train |
| Mac Stadium / cloud Mac | If you insist on Studio automation | Still no MediaSDK |

## Bandwidth tips

- Upload **stitched equirect MP4** or **JPEG frames @ 1–2 fps**, not raw dual 8K INSV, when possible.
- Cubemap faces multiply storage (~6×); consider generating cubemaps in the cloud next to COLMAP.
- Return **PLY + SOG** only; discard intermediate features/db unless debugging.

## Security / privacy

People masking exists partly for privacy. If you cloud-process:

- Prefer running YOLO **before upload**, or
- Use a VPC/private worker and delete frames after job completion
- Document retention in any product ToS

## Cloud job manifest (implemented)

After `package`, large jobs write `cloud_job.json` at the job root:

```bash
instasplat run --large-8k -i ./capture.mp4 -o ./runs -n walk_8k
# → runs/walk_8k/cloud_job.json
# → runs/walk_8k/07_nerfstudio/transforms.json
# → runs/walk_8k/11_merged/hierarchy_manifest.json
# → runs/walk_8k/quality.json
```

Recommended backends in the manifest:

| Backend | When to use |
|---------|-------------|
| `gsplat_3dgut` | Native equirect / fisheye training (skip cubemap). Local Mac counterpart: `train.backend: metal_equirect` — see [METAL_EQUIRECT_TRAINER.md](METAL_EQUIRECT_TRAINER.md) |
| `lichtfeld` | CUDA MCMC / workstation-grade train + export |
| `nerfstudio_splatfacto` | Use packaged `transforms.json` |
| `hierarchical_merge_cuda` | Kerbl hierarchy merger on tile anchors |

Disable with `--no-cloud-manifest` / `--no-quality`.

## Still to wire

1. `PipelineConfig.execution_backend: local | ssh | http`
2. `stages/remote.py` that rsyncs `JobPaths` folders and runs `instasplat run --stages ...`
3. Presigned S3 URLs for `06_export/scene.sog`

The local CLI already supports **per-stage runs**, which is the main requirement for splitting across machines.
