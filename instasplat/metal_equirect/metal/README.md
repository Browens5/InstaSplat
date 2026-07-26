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

Dispatch (optional): `metal_runtime.metal_fused_oit_ste` loads the metallib via
PyObjC when available and uses a straight-through estimator so PyTorch grads
still flow through the vectorized torch OIT path. Without PyObjC / metallib,
training uses the vectorized PyTorch OIT (MPS/CPU) only — CI and Linux stay green.
