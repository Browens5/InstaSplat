"""Reusable Metal shared-storage buffers + torch staging views.

Goal: avoid the per-step path
  MPS → numpy → tobytes → new MTLBuffer → kernel → bytes → numpy → MPS

Instead:
  1. Keep pooled ``MTLResourceStorageModeShared`` buffers across steps
  2. Host ``contents()`` is wrapped as a torch CPU tensor view
  3. ``staging.copy_(src)`` does one D2H (MPS→shared) or memcpy (CPU→shared)
  4. Kernel reads/writes the same MTLBuffer (no extra upload)
  5. ``dst.copy_(out_staging)`` does one H2D back to MPS
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def pack_cov2d(cov_2d: torch.Tensor) -> torch.Tensor:
    """Pack 2×2 cov as float4 ``(a, b_sym, c, 0)`` — matches Metal kernel."""
    cov = cov_2d.detach().float().contiguous()
    n = cov.shape[0]
    out = cov.new_zeros(n, 4)
    out[:, 0] = cov[:, 0, 0]
    out[:, 1] = 0.5 * (cov[:, 0, 1] + cov[:, 1, 0])
    out[:, 2] = cov[:, 1, 1]
    return out


def _memoryview_from_mtl_buffer(buf: Any, nbytes: int):
    """Return a writable memoryview over an MTLBuffer's host contents."""
    contents = buf.contents()
    if isinstance(contents, memoryview):
        return contents[:nbytes] if len(contents) >= nbytes else contents
    if hasattr(contents, "as_buffer"):
        return contents.as_buffer(nbytes)
    import ctypes

    ptr = int(contents)
    return memoryview((ctypes.c_char * nbytes).from_address(ptr))


class SharedBufferPool:
    """
    Grow-only pool of shared MTLBuffers with torch CPU views onto host memory.

    Safe to call from the training thread only (Metal command queue is serial).
    """

    def __init__(self, device: Any, storage_mode: Any) -> None:
        self._device = device
        self._storage = storage_mode
        self._bufs: dict[str, Any] = {}
        self._caps: dict[str, int] = {}
        self._views: dict[str, torch.Tensor] = {}

    def clear_views(self) -> None:
        """Drop cached torch views (call if Metal buffers were reallocated)."""
        self._views.clear()

    def ensure(self, name: str, nbytes: int) -> Any:
        nbytes = max(int(nbytes), 64)
        cap = self._caps.get(name, 0)
        if name in self._bufs and cap >= nbytes:
            return self._bufs[name]
        # Grow ~1.5× to reduce realloc churn
        grow = max(nbytes, int(cap * 1.5) if cap else nbytes)
        buf = self._device.newBufferWithLength_options_(grow, self._storage)
        if buf is None:
            raise RuntimeError(f"MTLBuffer alloc failed for {name} ({grow} bytes)")
        self._bufs[name] = buf
        self._caps[name] = grow
        self._views.pop(name, None)
        return buf

    def torch_view(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Host torch tensor viewing the shared buffer (CPU, contiguous)."""
        n_elem = 1
        for s in shape:
            n_elem *= int(s)
        if dtype == torch.float32:
            item = 4
        elif dtype == torch.uint8:
            item = 1
        elif dtype == torch.int32:
            item = 4
        else:
            raise TypeError(f"unsupported dtype {dtype}")
        nbytes = n_elem * item
        buf = self.ensure(name, nbytes)
        key = name
        cached = self._views.get(key)
        if cached is not None and cached.shape == shape and cached.dtype == dtype:
            return cached
        mv = _memoryview_from_mtl_buffer(buf, self._caps[name])
        # Narrow to exact element count
        if dtype == torch.float32:
            arr = np.frombuffer(mv, dtype=np.float32, count=n_elem)
        elif dtype == torch.uint8:
            arr = np.frombuffer(mv, dtype=np.uint8, count=n_elem)
        else:
            arr = np.frombuffer(mv, dtype=np.int32, count=n_elem)
        # Writable copy-on-write avoidance: frombuffer is writable if mv is
        arr = np.reshape(arr, shape)
        tensor = torch.from_numpy(arr)
        self._views[key] = tensor
        return tensor

    def copy_in(self, name: str, src: torch.Tensor, shape: tuple[int, ...] | None = None) -> Any:
        """
        Copy ``src`` into shared buffer ``name``.

        One device transfer when ``src`` is MPS/CUDA; memcpy when already CPU.
        Returns the MTLBuffer.
        """
        src = src.detach().contiguous()
        if shape is None:
            shape = tuple(src.shape)
        dtype = src.dtype
        if dtype not in (torch.float32, torch.uint8, torch.int32):
            if dtype == torch.bool:
                src = src.to(torch.uint8)
                dtype = torch.uint8
            else:
                src = src.float()
                dtype = torch.float32
        view = self.torch_view(name, shape, dtype)
        # copy_ handles MPS→CPU into the shared host memory in one sync
        view.copy_(src.reshape(shape).to(dtype=dtype))
        return self._bufs[name]

    def zero(self, name: str, nbytes: int) -> Any:
        view_f = self.torch_view(name, (nbytes // 4,), torch.float32)
        view_f.zero_()
        return self._bufs[name]

    def buffer(self, name: str) -> Any:
        return self._bufs[name]
