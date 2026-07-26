# Metal kernels for equirect projection

These shaders implement the GPU side of the 3DGUT-style Unscented Transform
projection used by `instasplat.metal_equirect`.

On a Mac with Xcode CLT:

```bash
cd instasplat/metal_equirect/metal
xcrun -sdk macosx metal -c EquirectProject.metal -o EquirectProject.air
xcrun -sdk macosx metallib EquirectProject.air -o EquirectProject.metallib
```

The training loop currently uses the PyTorch UT path (MPS/CPU) so CI and
Linux agents can run without Metal. A follow-up will load `EquirectProject.metallib`
via Metal Performance Shaders / PyObjC when `sys.platform == "darwin"`.
