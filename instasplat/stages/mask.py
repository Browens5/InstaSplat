"""YOLO segmentation masks to remove people (and optionally other classes)."""

from __future__ import annotations

import os
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
    device_used: str = ""
    mps_fallbacks: int = 0
    skipped_errors: int = 0


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


def _is_mps_indexing_error(exc: BaseException) -> bool:
    """Detect known PyTorch MPS / Ultralytics seg indexing crashes."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if "acceleratorerror" in name.lower():
        return True
    if "index" in msg and "out of bounds" in msg:
        return True
    if "mps" in msg and ("index" in msg or "ndarray" in msg or "subrange" in msg):
        return True
    return False


def _write_keep_mask(path: Path, h: int, w: int, invert: bool = False) -> None:
    keep = np.full((h, w), 0 if invert else 255, dtype=np.uint8)
    cv2.imwrite(str(path), keep)


def _apply_masks_to_keep(
    keep: np.ndarray,
    results,
    *,
    dilate_px: int,
    invert: bool,
) -> np.ndarray:
    h, w = keep.shape[:2]
    for r in results:
        if r.masks is None:
            continue
        masks = r.masks.data
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        else:
            masks = np.asarray(masks)
        for m in masks:
            m_resized = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            bin_m = (m_resized > 0.5).astype(np.uint8) * 255
            if dilate_px > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (dilate_px * 2 + 1, dilate_px * 2 + 1),
                )
                bin_m = cv2.dilate(bin_m, k)
            keep[bin_m > 0] = 0
    if invert:
        keep = 255 - keep
    return keep


def _predict_kwargs(cfg: PipelineConfig, device: str) -> dict:
    """Build YOLO predict kwargs (FP32; avoid deprecated ``half``)."""
    kwargs: dict = {
        "conf": cfg.mask.conf,
        "iou": cfg.mask.iou,
        "classes": cfg.mask.classes,
        "device": device,
        "verbose": False,
        # retina_masks=True triggers extra MPS native ops that crash intermittently
        "retina_masks": bool(cfg.mask.retina_masks),
        # Ultralytics replaced half=False with quantize=32 / "fp32"
        "quantize": 32,
    }
    return kwargs


def _predict_frame(model, img: np.ndarray, cfg: PipelineConfig, device: str):
    """Run YOLO predict with MPS-safe defaults."""
    kwargs = _predict_kwargs(cfg, device)
    try:
        return model.predict(source=img, **kwargs)
    except TypeError as exc:
        # Older Ultralytics: no quantize arg — drop it (default is FP32)
        if "quantize" in str(exc).lower() or "unexpected keyword" in str(exc).lower():
            kwargs.pop("quantize", None)
            return model.predict(source=img, **kwargs)
        raise


def run_mask(cfg: PipelineConfig, paths: JobPaths) -> MaskResult:
    paths.ensure()
    log = get_logger("instasplat.mask", paths.logs / "mask.log")
    # Helps some missing MPS ops fall back instead of hard-failing
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
            if img is None:
                continue
            h, w = img.shape[:2]
            _write_keep_mask(dst_dir / f"{fr.stem}.png", h, w, invert=cfg.mask.invert)
        return MaskResult(dst_dir, len(frames), "disabled", device_used="none")

    # Resume: only process frames that do not yet have a mask
    pending = [fr for fr in frames if not (dst_dir / f"{fr.stem}.png").exists()]
    if cfg.skip_existing and not pending:
        log.info("Skipping mask; found masks for all %d frames", len(frames))
        return MaskResult(dst_dir, len(frames), cfg.mask.model, device_used="skipped")
    if cfg.skip_existing and pending:
        log.info("Resuming masks: %d done, %d remaining", len(frames) - len(pending), len(pending))
    else:
        pending = frames

    if cfg.dry_run:
        log.info("Dry-run: would run YOLO %s on %d frames", cfg.mask.model, len(pending))
        return MaskResult(dst_dir, 0, cfg.mask.model, device_used="dry_run")

    from ultralytics import YOLO

    device = _resolve_device(cfg.mask.device)
    log.info(
        "Loading %s on %s (retina_masks=%s)",
        cfg.mask.model,
        device,
        cfg.mask.retina_masks,
    )
    model = YOLO(cfg.mask.model)

    mps_fallbacks = 0
    skipped_errors = 0
    consecutive_mps_fails = 0
    # After this many MPS crashes, abandon MPS for the rest of the job
    mps_fail_limit = max(1, int(cfg.mask.mps_fail_limit))

    for fr in tqdm(pending, desc="YOLO masks"):
        out = dst_dir / f"{fr.stem}.png"
        if out.exists() and cfg.skip_existing:
            continue
        img = cv2.imread(str(fr), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("Could not read %s — skipping", fr)
            continue
        h, w = img.shape[:2]
        keep = np.full((h, w), 255, dtype=np.uint8)

        try:
            results = _predict_frame(model, img, cfg, device)
            keep = _apply_masks_to_keep(
                keep, results, dilate_px=cfg.mask.dilate_px, invert=cfg.mask.invert
            )
            consecutive_mps_fails = 0
        except Exception as exc:  # noqa: BLE001 — MPS crashes are intermittent & fatal otherwise
            if device == "mps" and _is_mps_indexing_error(exc):
                consecutive_mps_fails += 1
                mps_fallbacks += 1
                log.warning(
                    "MPS YOLO crash on %s (%s) — retrying on CPU",
                    fr.name,
                    type(exc).__name__,
                )
                try:
                    results = _predict_frame(model, img, cfg, "cpu")
                    keep = _apply_masks_to_keep(
                        keep, results, dilate_px=cfg.mask.dilate_px, invert=cfg.mask.invert
                    )
                except Exception as cpu_exc:  # noqa: BLE001
                    skipped_errors += 1
                    log.error(
                        "CPU retry also failed for %s: %s — writing keep-all mask",
                        fr.name,
                        cpu_exc,
                    )
                    _write_keep_mask(out, h, w, invert=cfg.mask.invert)
                    if consecutive_mps_fails >= mps_fail_limit:
                        log.warning(
                            "MPS failed %d times — switching device to cpu for remaining frames",
                            consecutive_mps_fails,
                        )
                        device = "cpu"
                    continue
                if consecutive_mps_fails >= mps_fail_limit:
                    log.warning(
                        "MPS failed %d times — switching device to cpu for remaining frames",
                        consecutive_mps_fails,
                    )
                    device = "cpu"
            else:
                skipped_errors += 1
                log.error("YOLO failed on %s: %s — writing keep-all mask", fr.name, exc)
                _write_keep_mask(out, h, w, invert=cfg.mask.invert)
                continue

        cv2.imwrite(str(out), keep)

    count = len(list(dst_dir.glob("*.png")))
    log.info(
        "Wrote %d people-exclusion masks (device=%s, mps_fallbacks=%d, skipped_errors=%d)",
        count,
        device,
        mps_fallbacks,
        skipped_errors,
    )
    return MaskResult(
        dst_dir,
        count,
        cfg.mask.model,
        device_used=device,
        mps_fallbacks=mps_fallbacks,
        skipped_errors=skipped_errors,
    )
