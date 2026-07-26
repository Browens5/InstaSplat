"""Mac-native equirectangular Gaussian splat trainer (3DGUT / gsplat-inspired)."""

from instasplat.metal_equirect.backend import run_metal_equirect_train
from instasplat.metal_equirect.metal_runtime import metal_status

__all__ = ["run_metal_equirect_train", "metal_status"]
