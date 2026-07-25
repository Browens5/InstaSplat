"""YOLO segmentation masks to remove people (and optionally other classes)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from instasplat.config import PipelineConfig
from instasplat.utils.paths import JobPaths
from instasplat.utils.process import get_logger


@dataclass
class MaskResult:
    mask_dir: Path
    count: int
    model: str


def _resolve_device(requested: str) -> str:
    try:
        import torch

        if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if requested.startswith("cuda") and torch.cuda.is_available():
            return requested
    except ImportError:
        pass
    return "cpu"


def run_mask(cfg: PipelineConfig, paths: JobPaths) -> MaskResult:
    paths.ensure()
    log = get_logger("instasplat.mask", paths.logs / "mask.log")
    src_dir = paths.equirect_frames
    dst_dir = paths.equirect_masks
    dst_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(
        list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpeg"))
    )
    if not frames:
        raise FileNotFoundError(f"No frames in {src_dir}")

    if not cfg.mask.enabled:
        log.info("Masking disabled; writing full-white keep masks")
        for fr in frames:
            img = cv2.imread(str(fr), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            mask = np.full((h, w), 255, dtype=np.uint8)
            cv2.imwrite(str(dst_dir / (fr.stem + ".png")), mask)
        return MaskResult(dst_dir, len(frames), "disabled")

    existing = list(dst_dir.glob("*.png"))
    if existing and cfg.skip_existing and len(existing) >= len(frames):
        log.info("Skipping mask; found %d masks", len(existing))
        return MaskResult(dst_dir, len(existing), cfg.mask.model)

    if cfg.dry_run:
        log.info("Dry-run: would run YOLO %s on %d frames", cfg.mask.model, len(frames))
        return MaskResult(dst_dir, 0, cfg.mask.model)

    from ultralytics import YOLO

    device = _resolve_device(cfg.mask.device)
    log.info("Loading %s on %s", cfg.mask.model, device)
    model = YOLO(cfg.mask.model)

    for fr in tqdm(frames, desc="YOLO masks"):
        img = cv2.imread(str(fr), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        # Start with keep-everything; paint people as ignore (0)
        keep = np.full((h, w), 255, dtype=np.uint8)
        results = model.predict(
            source=img,
            conf=cfg.mask.conf,
            iou=cfg.mask.iou,
            classes=cfg.mask.classes,
            device=device,
            verbose=False,
            retina_masks=True,
        )
        for r in results:
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy()
            for m in masks:
                m_resized = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                bin_m = (m_resized > 0.5).astype(np.uint8) * 255
                if cfg.mask.dilate_px > 0:
                    k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (cfg.mask.dilate_px * 2 + 1, cfg.mask.dilate_px * 2 + 1),
                    )
                    bin_m = cv2.dilate(bin_m, k)
                keep[bin_m > 0] = 0
        if cfg.mask.invert:
            keep = 255 - keep
        out = dst_dir / f"{fr.stem}.png"
        cv2.imwrite(str(out), keep)

    count = len(list(dst_dir.glob("*.png")))
    log.info("Wrote %d people-exclusion masks", count)
    return MaskResult(dst_dir, count, cfg.mask.model)
