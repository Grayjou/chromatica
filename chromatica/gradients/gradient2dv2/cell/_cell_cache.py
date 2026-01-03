# chromatica/gradients/gradient2dv2/cell/_cell_cache.py
"""Cache reuse logic in one place."""

from __future__ import annotations
from typing import Optional
import numpy as np
from ....types.color_types import ColorModes, is_hue_space, HueDirection


def get_reusable_slice(
    cached_value: Optional[np.ndarray],
    start_idx: int,
    end_idx: int,
    current_space: ColorModes,
    target_space: ColorModes,
    current_hue_dir: Optional[HueDirection],
    target_hue_dir: Optional[HueDirection]
) -> Optional[np.ndarray]:
    """Return sliced cached value if it's safe to reuse, else None."""
    if cached_value is None:
        return None
    
    if current_space != target_space:
        return None
    
    if is_hue_space(current_space):
        if current_hue_dir != target_hue_dir:
            return None
    
    return cached_value[:, start_idx:end_idx, :].copy()