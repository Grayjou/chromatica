"""
High-level typed wrappers for hue interpolation with array/constant borders.

Mirrors the style of interp_2d_/wrappers.py, providing type hints and light
input normalization before calling the Cython kernels.
"""
from __future__ import annotations

from typing import Optional, Union
import numpy as np
from enum import IntEnum
from functools import partial

from .interp_hue import hue_lerp_between_lines_dispatch as _hue_dispatch
from .interp_hue2d_array_border import (
    hue_lerp_between_lines_array_border,
    hue_lerp_between_lines_array_border_flat,
    hue_lerp_between_lines_array_border_x_discrete,
    hue_lerp_between_lines_array_border_flat_x_discrete,
)
from .interp_hue_corners import (
    hue_lerp_from_corners_full_feathered as _hue_corners_dispatch,
)
from .interp_hue_corners_array_border import (
    hue_lerp_from_corners_array_border,
    hue_lerp_from_corners_flat_array_border,
    hue_lerp_from_corners_multichannel_array_border,
    hue_lerp_from_corners_multichannel_flat_array_border,
)
from ..border_handler import BorderMode
from enum import IntEnum


class DistanceMode(IntEnum):
    """Distance metrics for 2D border computation."""
    MAX_NORM = 1
    MANHATTAN = 2
    SCALED_MANHATTAN = 3
    ALPHA_MAX = 4
    ALPHA_MAX_SIMPLE = 5
    TAYLOR = 6
    EUCLIDEAN = 7
    WEIGHTED_MINMAX = 8

class HueMode(IntEnum):
    CW = 0
    CCW = 1
    SHORTEST = 2
    LONGEST = 3


BorderModeInput = Union[BorderMode, str, int]
DistanceModeInput = Union[str, int]


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float64 and arr.flags['C_CONTIGUOUS']:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _normalize_distance_mode(distance_mode: DistanceModeInput) -> int:
    """Map distance mode to the integer expected by Cython kernels."""
    if isinstance(distance_mode, str):
        name = distance_mode.lower()
        if name in {"euclidean", "euclid"}:
            return 7  # matches EUCLIDEAN in interp_utils
        if name in {"manhattan"}:
            return 2  # MANHATTAN in interp_utils (per codebase constants)
        if name in {"max_norm", "max"}:
            return 1  # MAX_NORM
        # Fallback: try int conversion; will raise if invalid
        return int(distance_mode)
    return int(distance_mode)


def _normalize_border_mode(border_mode: BorderModeInput) -> int:
    """Normalize border_mode to int for kernel compatibility."""
    if isinstance(border_mode, BorderMode):
        return int(border_mode.value if hasattr(border_mode, "value") else border_mode)
    if isinstance(border_mode, int):
        return border_mode
    raise TypeError(f"border_mode must be BorderMode or int, got {type(border_mode)}")


def hue_lerp_between_lines_typed(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray,
    *,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: float = 0.0,
    border_array: Optional[np.ndarray] = None,
    border_feathering: float = 0.0,
    distance_mode: DistanceModeInput = DistanceMode.EUCLIDEAN,
    num_threads: int = 1,
    x_discrete: bool = False,
) -> np.ndarray:
    """Typed wrapper around hue line interpolation with optional array border."""
    line0 = _ensure_float64(np.asarray(line0))
    line1 = _ensure_float64(np.asarray(line1))
    coords = _ensure_float64(np.asarray(coords))
    border_arr = None if border_array is None else _ensure_float64(np.asarray(border_array))

    return _hue_dispatch(
        line0,
        line1,
        coords,
        mode_x=int(mode_x),
        mode_y=int(mode_y),
        border_mode=_normalize_border_mode(border_mode),
        border_constant=border_constant,
        border_array=border_arr,
        border_feathering=border_feathering,
        distance_mode=_normalize_distance_mode(distance_mode),
        num_threads=num_threads,
        x_discrete=x_discrete,
    )


def hue_lerp_between_lines_array_border_typed(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray,
    border_array: np.ndarray,
    *,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CONSTANT,
    border_feathering: float = 0.0,
    distance_mode: DistanceModeInput = DistanceMode.EUCLIDEAN,
    num_threads: int = 1,
    x_discrete: bool = False,
) -> np.ndarray:
    """Directly call hue array-border kernels with type hints."""
    line0 = _ensure_float64(np.asarray(line0))
    line1 = _ensure_float64(np.asarray(line1))
    coords = _ensure_float64(np.asarray(coords))
    border_array = _ensure_float64(np.asarray(border_array))

    if coords.ndim == 3:
        if x_discrete:
            return hue_lerp_between_lines_array_border_x_discrete(
                line0,
                line1,
                coords,
                border_array,
                border_feathering=border_feathering,
                border_mode=_normalize_border_mode(border_mode),
                mode_y=int(mode_y),
                distance_mode=_normalize_distance_mode(distance_mode),
                num_threads=num_threads,
            )
        return hue_lerp_between_lines_array_border(
            line0,
            line1,
            coords,
            border_array,
            border_feathering=border_feathering,
            border_mode=_normalize_border_mode(border_mode),
            mode_x=int(mode_x),
            mode_y=int(mode_y),
            distance_mode=_normalize_distance_mode(distance_mode),
            num_threads=num_threads,
        )

    if coords.ndim == 2:
        if x_discrete:
            return hue_lerp_between_lines_array_border_flat_x_discrete(
                line0,
                line1,
                coords,
                border_array,
                border_feathering=border_feathering,
                border_mode=_normalize_border_mode(border_mode),
                mode_y=int(mode_y),
                distance_mode=_normalize_distance_mode(distance_mode),
                num_threads=num_threads,
            )
        return hue_lerp_between_lines_array_border_flat(
            line0,
            line1,
            coords,
            border_array,
            border_feathering=border_feathering,
            border_mode=_normalize_border_mode(border_mode),
            mode_x=int(mode_x),
            mode_y=int(mode_y),
            distance_mode=_normalize_distance_mode(distance_mode),
            num_threads=num_threads,
        )

    raise ValueError("coords must be (H, W, 2) or (N, 2)")


# =============================================================================
# Hue Corner Interpolation
# =============================================================================
def hue_lerp_from_corners_typed(
    corners: np.ndarray,
    coords: np.ndarray,
    *,
    mode: HueMode = HueMode.SHORTEST,
    border_mode: BorderMode = BorderMode.CLAMP,
    border_constant: float = 0.0,
    border_feathering: float = 0.0,
    distance_mode: DistanceMode = DistanceMode.EUCLIDEAN,
    num_threads: int = 1,
) -> np.ndarray:
    """Typed wrapper around hue corner interpolation with feathering."""
    corners = _ensure_float64(np.asarray(corners))
    coords = _ensure_float64(np.asarray(coords))
    
    return _hue_corners_dispatch(
        corners,
        coords,
        border_constant=border_constant,
        border_mode=_normalize_border_mode(border_mode),
        border_feathering=border_feathering,
        distance_mode=_normalize_distance_mode(distance_mode),
        num_threads=num_threads,
    )


def hue_lerp_from_corners_array_border_typed(
    corners: np.ndarray,
    coords: np.ndarray,
    border_array: np.ndarray,
    *,
    mode: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CONSTANT,
    border_feathering: float = 0.0,
    distance_mode: DistanceModeInput = DistanceMode.EUCLIDEAN,
    num_threads: int = 1,
) -> np.ndarray:
    """Typed wrapper for hue corner with array border."""
    corners = _ensure_float64(np.asarray(corners))
    coords = _ensure_float64(np.asarray(coords))
    border_array = _ensure_float64(np.asarray(border_array))
    
    # Single-channel case
    if corners.ndim == 1:
        if coords.ndim == 3:
            return hue_lerp_from_corners_array_border(
                corners, coords, border_array,
                border_feathering=border_feathering,
                border_mode=_normalize_border_mode(border_mode),
                distance_mode=_normalize_distance_mode(distance_mode),
                num_threads=num_threads,
            )
        elif coords.ndim == 2:
            return hue_lerp_from_corners_flat_array_border(
                corners, coords, border_array,
                border_feathering=border_feathering,
                border_mode=_normalize_border_mode(border_mode),
                distance_mode=_normalize_distance_mode(distance_mode),
                num_threads=num_threads,
            )
    
    # Multi-channel case
    elif corners.ndim == 2:
        if coords.ndim == 3:
            return hue_lerp_from_corners_multichannel_array_border(
                corners, coords, border_array,
                border_feathering=border_feathering,
                border_mode=_normalize_border_mode(border_mode),
                distance_mode=_normalize_distance_mode(distance_mode),
                num_threads=num_threads,
            )
        elif coords.ndim == 2:
            return hue_lerp_from_corners_multichannel_flat_array_border(
                corners, coords, border_array,
                border_feathering=border_feathering,
                border_mode=_normalize_border_mode(border_mode),
                distance_mode=_normalize_distance_mode(distance_mode),
                num_threads=num_threads,
            )
    
    raise ValueError("Invalid corners/coords shapes")

hue_lerp_between_lines_typed_x_discrete = partial(hue_lerp_between_lines_typed, x_discrete=True)
hue_lerp_between_lines_array_border_typed_x_discrete = partial(hue_lerp_between_lines_array_border_typed, x_discrete=True)




__all__ = [
    "HueMode",
    # Lines
    "hue_lerp_between_lines_typed",
    "hue_lerp_between_lines_array_border_typed",
    "hue_lerp_between_lines_typed_x_discrete",
    "hue_lerp_between_lines_array_border_typed_x_discrete",
    # Corners
    "hue_lerp_from_corners_typed",
    "hue_lerp_from_corners_array_border_typed",

]
