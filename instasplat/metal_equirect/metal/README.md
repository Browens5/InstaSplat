# Metal kernels for equirect soft-OIT

These shaders implement equirect projection helpers and a fused soft-OIT
accumulate/normalize path used by `instasplat.metal_equirect` on Apple Silicon.

On a Mac with Xcode CLT:

```bash
cd instasplat/metal_equirect/metal
xcrun -sdk macosx metal -c EquirectProject.metal -o EquirectProject.air
xcrun -sdk macosx metallib EquirectProject.air -o EquirectProject.metallib
```

Kernels:

| Kernel | Role |
|--------|------|
| `project_means_equirect` | means_cam → UV |
| `unscented_project_equirect` | 7 sigma-point UT projection |
| `soft_oit_accumulate` | Per-Gaussian footprint soft splat (atomic RGB/weight) |
| `soft_oit_normalize` | `rgb = color / (weight + ε)` |

## Composite path (`composite: metal`)

`metal_runtime.fused_soft_oit` is the training composite entry:

1. **Forward** — Metal soft-OIT when PyObjC + metallib load; else CPU reference
   (`soft_oit_ref.py`) that mirrors the kernel.
2. **Backward** — recompute vectorized torch OIT so grads flow to means / SH / etc.

Inference / `torch.no_grad()` still uses the fused forward only (no torch composite).

Set `train.composite: metal` (Mac defaults) or keep `oit` / `tile` for pure torch.
