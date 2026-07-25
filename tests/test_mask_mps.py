"""Tests for MPS-safe YOLO mask helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from instasplat.config import PipelineConfig
from instasplat.stages.mask import (
    _is_mps_indexing_error,
    _predict_frame,
    _predict_kwargs,
    _write_keep_mask,
)


def test_detect_mps_indexing_error() -> None:
    class AcceleratorError(Exception):
        pass

    assert _is_mps_indexing_error(
        AcceleratorError("index 72057594037927936 is out of bounds for dimension 0 with size 2")
    )
    assert _is_mps_indexing_error(RuntimeError("index 5 is out of bounds for dimension 0 with size 2"))
    assert not _is_mps_indexing_error(ValueError("unrelated"))


def test_mask_config_mps_safe_defaults() -> None:
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=Path("r"), project_name="t")
    assert cfg.mask.retina_masks is False
    assert cfg.mask.mps_fail_limit == 3


def test_predict_kwargs_uses_quantize_not_half() -> None:
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=Path("r"), project_name="t")
    kwargs = _predict_kwargs(cfg, "cpu")
    assert "half" not in kwargs
    assert kwargs["quantize"] == 32
    assert kwargs["retina_masks"] is False


def test_predict_frame_drops_quantize_on_old_ultralytics() -> None:
    cfg = PipelineConfig(input_path=Path("a.mp4"), output_dir=Path("r"), project_name="t")
    calls: list[dict] = []

    class FakeModel:
        def predict(self, source=None, **kwargs):
            calls.append(kwargs)
            if "quantize" in kwargs:
                raise TypeError("predict() got an unexpected keyword argument 'quantize'")
            return ["ok"]

    out = _predict_frame(FakeModel(), np.zeros((8, 8, 3), dtype=np.uint8), cfg, "cpu")
    assert out == ["ok"]
    assert len(calls) == 2
    assert "quantize" in calls[0]
    assert "quantize" not in calls[1]
    assert "half" not in calls[0] and "half" not in calls[1]


def test_write_keep_mask(tmp_path: Path) -> None:
    out = tmp_path / "m.png"
    _write_keep_mask(out, 8, 12)
    assert out.exists()
    import cv2

    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    assert img.shape == (8, 12)
    assert int(img.max()) == 255
