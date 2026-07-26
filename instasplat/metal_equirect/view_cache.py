"""Decode equirect training views once; serve from RAM/MPS."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from instasplat.metal_equirect.dataset import TrainView


@dataclass
class CachedView:
    rgb_full: torch.Tensor  # (H, W, 3) on device
    mask_full: torch.Tensor | None
    R: torch.Tensor
    t: torch.Tensor
    full_width: int
    full_height: int


class ViewCache:
    """LRU-ish cache of decoded panoramas on the training device."""

    def __init__(self, device: torch.device, *, max_views: int = 512) -> None:
        self.device = device
        self.max_views = max(8, int(max_views))
        self._store: dict[str, CachedView] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._store)

    def preload(self, views: list[TrainView]) -> int:
        """Decode up to ``max_views`` panoramas up front."""
        n = 0
        for v in views[: self.max_views]:
            self.get(v)
            n += 1
        return n

    def get(self, view: TrainView) -> CachedView:
        key = f"{view.image_path}|{view.width}x{view.height}"
        hit = self._store.get(key)
        if hit is not None:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            return hit

        img = cv2.imread(str(view.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(view.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != view.width or img.shape[0] != view.height:
            img = cv2.resize(img, (view.width, view.height), interpolation=cv2.INTER_AREA)
        rgb = torch.from_numpy(img.astype(np.float32) / 255.0).to(self.device)

        mask_t = None
        if view.mask_path is not None and view.mask_path.exists():
            m = cv2.imread(str(view.mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[1] != view.width or m.shape[0] != view.height:
                    m = cv2.resize(
                        m, (view.width, view.height), interpolation=cv2.INTER_NEAREST
                    )
                mask_t = torch.from_numpy((m.astype(np.float32) / 255.0)).to(self.device)

        R = torch.from_numpy(view.R_w2c).to(self.device)
        t = torch.from_numpy(view.t_w2c).to(self.device)
        cached = CachedView(
            rgb_full=rgb,
            mask_full=mask_t,
            R=R,
            t=t,
            full_width=view.width,
            full_height=view.height,
        )
        self._store[key] = cached
        self._order.append(key)
        while len(self._order) > self.max_views:
            old = self._order.pop(0)
            self._store.pop(old, None)
        return cached

    def get_scaled(
        self, view: TrainView, *, width_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, int, int]:
        """
        Return RGB/mask/R/t and effective (W, H) at ``width_scale`` of the view size.
        """
        cached = self.get(view)
        scale = float(max(0.25, min(1.0, width_scale)))
        w = max(16, int(round(cached.full_width * scale)))
        # Keep equirect aspect
        h = max(8, int(round(w * cached.full_height / max(cached.full_width, 1))))
        if w == cached.full_width and h == cached.full_height:
            return cached.rgb_full, cached.mask_full, cached.R, cached.t, w, h

        rgb = (
            F.interpolate(
                cached.rgb_full.permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
        mask = None
        if cached.mask_full is not None:
            mask = F.interpolate(
                cached.mask_full.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="nearest",
            ).squeeze(0).squeeze(0)
        return rgb, mask, cached.R, cached.t, w, h
