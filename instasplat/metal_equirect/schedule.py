"""Resolution / tile schedule for progressive training quality."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainSchedule:
    """Per-step render settings (coarse early → fine later)."""

    width_scale: float
    tile_size: int
    max_per_tile: int
    max_gaussians_render: int
    phase: str  # coarse | mid | fine


def schedule_at_step(
    step: int,
    total_steps: int,
    *,
    base_max_gaussians_render: int = 8_000,
) -> TrainSchedule:
    """
    Progressive schedule:

    - first 30%: half-res, large tiles, fewer splats
    - 30–70%: ¾ res
    - final 30%: full res, finer tiles
    """
    total_steps = max(1, int(total_steps))
    t = float(step) / float(total_steps)
    if t <= 0.30:
        return TrainSchedule(
            width_scale=0.5,
            tile_size=64,
            max_per_tile=128,
            max_gaussians_render=max(2_000, base_max_gaussians_render // 2),
            phase="coarse",
        )
    if t <= 0.70:
        return TrainSchedule(
            width_scale=0.75,
            tile_size=32,
            max_per_tile=96,
            max_gaussians_render=max(3_000, (base_max_gaussians_render * 3) // 4),
            phase="mid",
        )
    return TrainSchedule(
        width_scale=1.0,
        tile_size=16,
        max_per_tile=64,
        max_gaussians_render=base_max_gaussians_render,
        phase="fine",
    )
