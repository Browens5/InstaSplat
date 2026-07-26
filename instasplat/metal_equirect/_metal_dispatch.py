"""PyObjC Metal dispatch for EquirectProject.metallib (optional, macOS only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from instasplat.metal_equirect._metal_buffers import SharedBufferPool, pack_cov2d


class MetalPipeline:
    """MTLComputePipeline for soft_oit with pooled shared buffers."""

    def __init__(self, metallib: Path) -> None:
        self.ok = False
        self.shared_buffers = False
        self._device = None
        self._queue = None
        self._acc_pso = None
        self._norm_pso = None
        self._pool: SharedBufferPool | None = None
        self._MTLSize = None
        self._storage = None
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
            library, _err = device.newLibraryWithURL_error_(
                metallib.resolve().as_uri(), None
            )
            if library is None:
                from Foundation import NSURL  # type: ignore

                url = NSURL.fileURLWithPath_(str(metallib.resolve()))
                library, _err = device.newLibraryWithURL_error_(url, None)
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
            self._pool = SharedBufferPool(device, MTLResourceStorageModeShared)
            self.ok = self._queue is not None
            self.shared_buffers = self.ok
        except Exception:
            self.ok = False
            self.shared_buffers = False

    def soft_oit_from_tensors(
        self,
        mean_2d: torch.Tensor,
        cov_2d: torch.Tensor,
        radius: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        depth: torch.Tensor,
        valid: torch.Tensor,
        height: int,
        width: int,
        *,
        footprint: int = 24,
        out_device: torch.device | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Soft-OIT using pooled shared MTLBuffers.

        Copies inputs once into shared host memory (``tensor.copy_``), runs the
        kernels, then copies the RGB result once to ``out_device`` (default: same
        as ``mean_2d``).
        """
        if not self.ok or self._pool is None:
            raise RuntimeError("Metal pipeline not ready")

        pool = self._pool
        n = int(mean_2d.shape[0])
        n_pix = int(height * width)
        out_device = out_device or mean_2d.device
        out_dtype = out_dtype or mean_2d.dtype

        cov_pack = pack_cov2d(cov_2d)
        b_mean = pool.copy_in("mean_2d", mean_2d.float(), (n, 2))
        b_cov = pool.copy_in("cov_pack", cov_pack, (n, 4))
        b_rad = pool.copy_in("radius", radius.float(), (n,))
        b_op = pool.copy_in("opacity", opacities.float(), (n,))
        b_col = pool.copy_in("color", colors.float(), (n, 3))
        b_dep = pool.copy_in("depth", depth.float(), (n,))
        valid_u8 = valid.detach().to(torch.uint8).reshape(n)
        b_val = pool.copy_in("valid", valid_u8, (n,))

        # Accumulators + output live in shared memory; zero each step
        b_cacc = pool.ensure("color_acc", n_pix * 3 * 4)
        pool.torch_view("color_acc", (n_pix * 3,), torch.float32).zero_()
        b_wacc = pool.ensure("weight_acc", n_pix * 4)
        pool.torch_view("weight_acc", (n_pix,), torch.float32).zero_()
        b_out = pool.ensure("rgb_out", n_pix * 3 * 4)
        out_view = pool.torch_view("rgb_out", (height, width, 3), torch.float32)
        out_view.zero_()

        u = np.zeros(
            1,
            dtype=np.dtype(
                [
                    ("n", "<u4"),
                    ("width", "<u4"),
                    ("height", "<u4"),
                    ("footprint", "<i4"),
                    ("depth_tau", "<f4"),
                ]
            ),
        )
        u["n"] = n
        u["width"] = width
        u["height"] = height
        u["footprint"] = int(footprint)
        u["depth_tau"] = 0.15
        u_bytes = np.frombuffer(u.tobytes(), dtype=np.uint8).copy()
        b_uni = pool.copy_in(
            "uniforms", torch.from_numpy(u_bytes), (u_bytes.size,)
        )
        npix_t = torch.tensor([n_pix], dtype=torch.int32)
        b_npix = pool.copy_in("n_pix", npix_t, (1,))

        self._encode_and_run(
            n=n,
            n_pix=n_pix,
            buffers=(
                b_mean,
                b_cov,
                b_rad,
                b_op,
                b_col,
                b_dep,
                b_val,
                b_cacc,
                b_wacc,
                b_uni,
                b_out,
                b_npix,
            ),
        )

        # One H2D (or memcpy) into the caller's device
        result = torch.empty(height, width, 3, device=out_device, dtype=out_dtype)
        result.copy_(out_view.to(dtype=out_dtype))
        return result

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
        """NumPy entry (tests / legacy) — wraps ``soft_oit_from_tensors``."""
        out = self.soft_oit_from_tensors(
            torch.from_numpy(np.ascontiguousarray(mean_2d, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(cov_2d, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(radius, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(opacities, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(colors, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(valid.astype(np.uint8))),
            height,
            width,
            footprint=footprint,
            out_device=torch.device("cpu"),
            out_dtype=torch.float32,
        )
        return out.numpy()

    def _encode_and_run(self, *, n: int, n_pix: int, buffers: tuple[Any, ...]) -> None:
        (
            b_mean,
            b_cov,
            b_rad,
            b_op,
            b_col,
            b_dep,
            b_val,
            b_cacc,
            b_wacc,
            b_uni,
            b_out,
            b_npix,
        ) = buffers
        cmd = self._queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self._acc_pso)
        for i, buf in enumerate(
            (b_mean, b_cov, b_rad, b_op, b_col, b_dep, b_val, b_cacc, b_wacc, b_uni)
        ):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
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
