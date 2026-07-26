"""PyObjC Metal dispatch for EquirectProject.metallib (optional, macOS only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MetalPipeline:
    """Minimal MTLComputePipeline for soft_oit_accumulate + soft_oit_normalize."""

    def __init__(self, metallib: Path) -> None:
        self.ok = False
        self._device = None
        self._queue = None
        self._acc_pso = None
        self._norm_pso = None
        try:
            import Metal  # type: ignore  # noqa: F401
            import objc  # type: ignore  # noqa: F401
            from Metal import (  # type: ignore
                MTLCreateSystemDefaultDevice,
                MTLResourceStorageModeShared,
                MTLSize,
            )

            device = MTLCreateSystemDefaultDevice()
            if device is None:
                return
            lib_url = metallib.resolve().as_uri()
            # MTLDevice newLibraryWithURL
            err = objc.nil
            library, err = device.newLibraryWithURL_error_(lib_url, None)
            if library is None:
                # Fallback: file path via NSURL
                from Foundation import NSURL  # type: ignore

                url = NSURL.fileURLWithPath_(str(metallib.resolve()))
                library, err = device.newLibraryWithURL_error_(url, None)
            if library is None:
                return
            acc_fn = library.newFunctionWithName_("soft_oit_accumulate")
            norm_fn = library.newFunctionWithName_("soft_oit_normalize")
            if acc_fn is None or norm_fn is None:
                return
            acc_pso, _ = device.newComputePipelineStateWithFunction_error_(acc_fn, None)
            norm_pso, _ = device.newComputePipelineStateWithFunction_error_(norm_fn, None)
            if acc_pso is None or norm_pso is None:
                return
            self._device = device
            self._queue = device.newCommandQueue()
            self._acc_pso = acc_pso
            self._norm_pso = norm_pso
            self._MTLSize = MTLSize
            self._storage = MTLResourceStorageModeShared
            self.ok = self._queue is not None
        except Exception:
            self.ok = False

    def soft_oit(
        self,
        mean_2d: np.ndarray,
        cov_2d: np.ndarray,
        radius: np.ndarray,
        opacities: np.ndarray,
        colors: np.ndarray,
        depth: np.ndarray,
        valid: np.ndarray,
        height: int,
        width: int,
        *,
        footprint: int = 24,
    ) -> np.ndarray:
        if not self.ok:
            raise RuntimeError("Metal pipeline not ready")

        n = int(mean_2d.shape[0])
        n_pix = int(height * width)
        mean_2d = np.ascontiguousarray(mean_2d, dtype=np.float32)
        # pack cov as (a, b, c, 0)
        cov_pack = np.zeros((n, 4), dtype=np.float32)
        cov_pack[:, 0] = cov_2d[:, 0, 0]
        cov_pack[:, 1] = cov_2d[:, 0, 1]
        cov_pack[:, 2] = cov_2d[:, 1, 1]
        radius = np.ascontiguousarray(radius, dtype=np.float32)
        opacities = np.ascontiguousarray(opacities, dtype=np.float32)
        colors = np.ascontiguousarray(colors, dtype=np.float32)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        valid_u8 = np.ascontiguousarray(valid.astype(np.uint8))

        color_acc = np.zeros(n_pix * 3, dtype=np.float32)
        weight_acc = np.zeros(n_pix, dtype=np.float32)
        rgb_out = np.zeros((n_pix, 3), dtype=np.float32)

        # SoftOITUniforms: n, width, height, footprint, depth_tau
        u_bytes = np.zeros(1, dtype=np.dtype(
            [
                ("n", "<u4"),
                ("width", "<u4"),
                ("height", "<u4"),
                ("footprint", "<i4"),
                ("depth_tau", "<f4"),
            ]
        ))
        u_bytes["n"] = n
        u_bytes["width"] = width
        u_bytes["height"] = height
        u_bytes["footprint"] = int(footprint)
        u_bytes["depth_tau"] = 0.15

        def _buf(arr: np.ndarray):
            nbytes = arr.nbytes
            buf = self._device.newBufferWithLength_options_(nbytes, self._storage)
            mv = buf.contents().as_buffer(nbytes)
            mv[:] = arr.tobytes()
            return buf

        b_mean = _buf(mean_2d)
        b_cov = _buf(cov_pack)
        b_rad = _buf(radius)
        b_op = _buf(opacities)
        b_col = _buf(colors)
        b_dep = _buf(depth)
        b_val = _buf(valid_u8)
        b_cacc = _buf(color_acc)
        b_wacc = _buf(weight_acc)
        b_uni = _buf(np.frombuffer(u_bytes.tobytes(), dtype=np.uint8))
        b_out = _buf(rgb_out)
        b_npix = _buf(np.array([n_pix], dtype=np.uint32))

        cmd = self._queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self._acc_pso)
        enc.setBuffer_offset_atIndex_(b_mean, 0, 0)
        enc.setBuffer_offset_atIndex_(b_cov, 0, 1)
        enc.setBuffer_offset_atIndex_(b_rad, 0, 2)
        enc.setBuffer_offset_atIndex_(b_op, 0, 3)
        enc.setBuffer_offset_atIndex_(b_col, 0, 4)
        enc.setBuffer_offset_atIndex_(b_dep, 0, 5)
        enc.setBuffer_offset_atIndex_(b_val, 0, 6)
        enc.setBuffer_offset_atIndex_(b_cacc, 0, 7)
        enc.setBuffer_offset_atIndex_(b_wacc, 0, 8)
        enc.setBuffer_offset_atIndex_(b_uni, 0, 9)
        tg = max(1, int(self._acc_pso.maxTotalThreadsPerThreadgroup()))
        grid = self._MTLSize(n, 1, 1)
        group = self._MTLSize(min(tg, max(n, 1)), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, group)

        enc.setComputePipelineState_(self._norm_pso)
        enc.setBuffer_offset_atIndex_(b_cacc, 0, 0)
        enc.setBuffer_offset_atIndex_(b_wacc, 0, 1)
        enc.setBuffer_offset_atIndex_(b_out, 0, 2)
        enc.setBuffer_offset_atIndex_(b_npix, 0, 3)
        grid2 = self._MTLSize(n_pix, 1, 1)
        group2 = self._MTLSize(min(tg, max(n_pix, 1)), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid2, group2)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        out_mv = b_out.contents().as_buffer(rgb_out.nbytes)
        result = np.frombuffer(bytes(out_mv), dtype=np.float32).reshape(height, width, 3).copy()
        return result
