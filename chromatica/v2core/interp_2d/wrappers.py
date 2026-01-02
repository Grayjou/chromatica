"""
Interpolation dispatcher module.

Provides unified interfaces for 2D line and corner interpolation with
flexible border handling, feathering, and multi-threading support.
"""
#chromatica\v2core\interp_2d_\wrappers.py
from .interp_2d_fast_ import (
    lerp_between_lines_full_feathered as _lerp_between_lines,
)
from .corner_interp_2d_fast_ import (
    lerp_from_corners_full_feathered as _lerp_from_corners,
)
from .interp_2d_array_border import (
    lerp_between_lines_inplace as _lerp_between_lines_inplace,
    lerp_between_lines_onto_array as _lerp_between_lines_onto_array,
)

from .corner_interp_2d_border_ import (
    lerp_from_corners_array_border_full as _lerp_from_corners_array_border,
)
from ..border_handler import BorderMode, BorderModeInput, DistanceMode
from typing import List, Optional, Union
import numpy as np
from ...types.array_types import ndarray_1d, ndarray_2d, ndarray_3d
from functools import partial
from enum import IntEnum




# =============================================================================
# Type Aliases
# =============================================================================


BorderConstantInput = Optional[Union[float, List[float], ndarray_1d]]
CoordsInput = Union[ndarray_2d, List[ndarray_2d]]


# =============================================================================
# Internal Helpers
# =============================================================================

def _ensure_coords_array(coords: CoordsInput) -> np.ndarray:
    """Convert coords input to a float64 numpy array."""
    if isinstance(coords, list):
        coords = coords[0] if len(coords) == 1 else np.array(coords)
    return np.asarray(coords, dtype=np.float64)


def _normalize_border_mode(
    border_mode: BorderModeInput,
) -> Union[BorderMode, np.ndarray]:
    """Normalize border mode(s) to scalar or int32 array."""
    if isinstance(border_mode, list):
        if len(border_mode) == 1:
            return border_mode[0]
        return np.asarray(border_mode, dtype=np.int32)
    if isinstance(border_mode, np.ndarray):
        return border_mode.astype(np.int32)
    return border_mode


def _normalize_border_constant(
    border_constant: BorderConstantInput,
) -> Optional[Union[float, np.ndarray]]:
    """Normalize border constant(s) to scalar or float64 array."""
    if border_constant is None:
        return None
    if isinstance(border_constant, list):
        if len(border_constant) == 1:
            return float(border_constant[0])
        return np.asarray(border_constant, dtype=np.float64)
    if isinstance(border_constant, np.ndarray):
        return border_constant.astype(np.float64)
    return float(border_constant)


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Ensure array is float64, avoiding copy if already correct dtype."""
    if arr.dtype == np.float64:
        return arr
    return arr.astype(np.float64)


def _ensure_float64_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is float64 and C-contiguous, avoiding copy if possible."""
    if arr.dtype == np.float64 and arr.flags['C_CONTIGUOUS']:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _pack_corners(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
) -> np.ndarray:
    """
    Pack corner arrays into a single corners array.
    
    Args:
        top_left: Corner value(s), shape () or (C,)
        top_right: Corner value(s), shape () or (C,)
        bottom_left: Corner value(s), shape () or (C,)
        bottom_right: Corner value(s), shape () or (C,)
    
    Returns:
        Packed corners array, shape (4,) or (4, C)
    """
    top_left = np.atleast_1d(np.asarray(top_left))
    top_right = np.atleast_1d(np.asarray(top_right))
    bottom_left = np.atleast_1d(np.asarray(bottom_left))
    bottom_right = np.atleast_1d(np.asarray(bottom_right))
    
    if top_left.ndim > 2:
        raise ValueError(f"Invalid corner shape: {top_left.shape}. Expected 1D or 2D.")
    
    return np.stack(
        [top_left, top_right, bottom_left, bottom_right], axis=0
    ).astype(np.float64)


# =============================================================================
# Core Interpolation Functions
# =============================================================================

def lerp_between_lines(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordsInput,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstantInput = None,
    border_feathering: float = 0.0,
    num_threads: int = -1,
    x_discrete: bool = False,
    distance_mode: DistanceMode = DistanceMode.ALPHA_MAX,
) -> np.ndarray:
    """
    Interpolate between two lines at given coordinates.
    
    Performs bilinear interpolation between line0 (y=0) and line1 (y=1),
    with configurable border handling and feathering.
    
    Args:
        line0: First line values at y=0, shape (L,) or (L, C)
        line1: Second line values at y=1, shape (L,) or (L, C)
        coords: Coordinate array:
            - (H, W, 2): Grid coordinates
            - (N, 2): Flat coordinates
            - (C, H, W, 2): Per-channel grid coordinates
            - (C, N, 2): Per-channel flat coordinates
        border_mode: Border handling mode(s):
            - BorderMode: Same for all channels
            - List/array of BorderMode: Per-channel, shape (C,)
        border_constant: Value(s) for CONSTANT border mode:
            - float: Same for all channels
            - List/array: Per-channel constants, shape (C,)
        border_feathering: Smooth transition width at borders (0.0 = hard edge)
        num_threads: Thread count (-1=auto, 0=serial, >0=specific count)
        x_discrete: If True, use nearest-neighbor sampling in x direction
        distance_mode: Distance metric for 2D border distance calculation
    
    Returns:
        Interpolated values with shape derived from coords
    """
    return _lerp_between_lines(
        _ensure_float64(line0),
        _ensure_float64(line1),
        _ensure_coords_array(coords),
        border_mode=_normalize_border_mode(border_mode),
        border_constant=_normalize_border_constant(border_constant),
        border_feathering=border_feathering,
        distance_mode=distance_mode,
        num_threads=num_threads,
        x_discrete=x_discrete,
    )


def lerp_from_corners(
    corners: Union[List[float], np.ndarray],
    coords: CoordsInput,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstantInput = None,
    border_feathering: float = 0.0,
    num_threads: int = -1,
) -> np.ndarray:
    """
    Interpolate from corner values at given coordinates.
    
    Performs bilinear interpolation from four corner values over a unit square.
    
    Args:
        corners: Corner values:
            - List/tuple of 4 scalars: [top_left, top_right, bottom_left, bottom_right]
            - Array shape (4,): Single channel
            - Array shape (4, C): Multi-channel
        coords: Coordinate array (see lerp_between_lines for shapes)
        border_mode: Border handling mode(s)
        border_constant: Value(s) for CONSTANT border mode
        border_feathering: Smooth transition width at borders
        num_threads: Thread count (-1=auto, 0=serial, >0=specific)
    
    Returns:
        Interpolated values with shape derived from coords
    """
    if isinstance(corners, (list, tuple)):
        corners = np.array(corners)
    
    return _lerp_from_corners(
        _ensure_float64(corners),
        _ensure_coords_array(coords),
        _normalize_border_mode(border_mode),
        _normalize_border_constant(border_constant),
        border_feathering,
        num_threads,
    )


def lerp_from_unpacked_corners(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
    coords: CoordsInput,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstantInput = None,
    border_feathering: float = 0.0,
    num_threads: int = -1,
) -> np.ndarray:
    """
    Interpolate from unpacked corner values.
    
    Convenience wrapper around lerp_from_corners that accepts individual
    corner arrays instead of a packed array.
    
    Args:
        top_left: Top-left corner value(s)
        top_right: Top-right corner value(s)
        bottom_left: Bottom-left corner value(s)
        bottom_right: Bottom-right corner value(s)
        coords: Coordinate array
        border_mode: Border handling mode(s)
        border_constant: Value(s) for CONSTANT border mode
        border_feathering: Smooth transition width at borders
        num_threads: Thread count
    
    Returns:
        Interpolated values
    """
    return lerp_from_corners(
        _pack_corners(top_left, top_right, bottom_left, bottom_right),
        coords,
        border_mode=border_mode,
        border_constant=border_constant,
        border_feathering=border_feathering,
        num_threads=num_threads,
    )


# =============================================================================
# In-Place and Composite Operations
# =============================================================================

def lerp_between_lines_inplace(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordsInput,
    target_array: np.ndarray,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_feathering: float = 0.0,
    distance_mode: DistanceMode = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    x_discrete: bool = False,
) -> None:
    """
    Interpolate between lines and write results in-place to target array.
    
    Modifies target_array directly. In-bounds interpolated values overwrite
    existing values; out-of-bounds regions are left unchanged (or blended
    with feathering).
    
    Args:
        line0: First line values at y=0, shape (L,) or (L, C)
        line1: Second line values at y=1, shape (L,) or (L, C)
        coords: Coordinate array, shape (H, W, 2) or (N, 2)
        target_array: Array to modify in-place, shape (H, W) or (H, W, C)
        border_mode: Border handling mode(s)
        border_feathering: Smooth blending width at borders
        distance_mode: Distance metric for 2D border computation
        num_threads: Thread count
        x_discrete: If True, use nearest-neighbor sampling in x direction
    
    Returns:
        None (modifies target_array in-place)
    """
    coords = _ensure_coords_array(coords)
    line0 = _ensure_float64(line0)
    line1 = _ensure_float64(line1)
    border_mode = _normalize_border_mode(border_mode)
    
    # Check if we can operate truly in-place
    can_inplace = (
        target_array.dtype == np.float64 and 
        target_array.flags['C_CONTIGUOUS']
    )
    
    if can_inplace:
        _lerp_between_lines_inplace(
            line0,
            line1,
            coords,
            target_array,
            border_mode=border_mode,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
            x_discrete=x_discrete,
        )
    else:
        # Fallback: work on a copy, then write back
        working_array = np.ascontiguousarray(target_array, dtype=np.float64)
        _lerp_between_lines_inplace(
            line0,
            line1,
            coords,
            working_array,
            border_mode=border_mode,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
            x_discrete=x_discrete,
        )
        np.copyto(target_array, working_array.astype(target_array.dtype))


def lerp_between_lines_onto_array(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordsInput,
    background_array: np.ndarray,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_feathering: float = 0.0,
    distance_mode: DistanceMode = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    x_discrete: bool = False,
) -> np.ndarray:
    """
    Interpolate between lines and composite onto a background array.
    
    Creates a new array where in-bounds regions contain interpolated values
    and out-of-bounds regions contain values from background_array (with
    optional feathered blending at boundaries).
    
    Args:
        line0: First line values at y=0, shape (L,) or (L, C)
        line1: Second line values at y=1, shape (L,) or (L, C)
        coords: Coordinate array:
            - (H, W, 2): Grid coordinates
            - (N, 2): Flat coordinates
            - (C, H, W, 2): Per-channel grid coordinates
            - (C, N, 2): Per-channel flat coordinates
        background_array: Background to composite onto:
            - (H, W): Single channel grid
            - (H, W, C): Multi-channel grid
            - (N,): Single channel flat
            - (N, C): Multi-channel flat
        border_mode: Border handling mode(s)
        border_feathering: Smooth blending width at borders
        distance_mode: Distance metric for 2D border computation
        num_threads: Thread count
        x_discrete: If True, use nearest-neighbor sampling in x direction
    
    Returns:
        New array with interpolated values composited onto background
    """
    return _lerp_between_lines_onto_array(
        _ensure_float64(line0),
        _ensure_float64(line1),
        _ensure_coords_array(coords),
        _ensure_float64(background_array),
        border_mode=_normalize_border_mode(border_mode),
        border_feathering=border_feathering,
        distance_mode=distance_mode,
        num_threads=num_threads,
        x_discrete=x_discrete,
    )


# =============================================================================
# Unified Dispatcher
# =============================================================================

def lerp_between_lines_full(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordsInput,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_value: Optional[Union[float, ndarray_1d, ndarray_2d, ndarray_3d]] = None,
    border_feathering: float = 0.0,
    distance_mode: DistanceMode = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    modify_inplace: bool = False,
    x_discrete: bool = False,
) -> Optional[np.ndarray]:
    """
    Full-featured line interpolation with flexible border handling.
    
    Unified dispatcher that routes to the appropriate kernel based on inputs:
    - If border_value is a 2D/3D array and modify_inplace=True: in-place modification
    - If border_value is a 2D/3D array and modify_inplace=False: composite onto copy
    - Otherwise: standard interpolation with scalar/per-channel border constants
    
    Args:
        line0: First line values at y=0, shape (L,) or (L, C)
        line1: Second line values at y=1, shape (L,) or (L, C)
        coords: Coordinate array (see lerp_between_lines for shapes)
        border_mode: Border handling mode(s)
        border_value: Border handling values:
            - None: Use default (0.0 for CONSTANT mode)
            - float: Scalar constant for all channels
            - 1D array: Per-channel constants, shape (C,)
            - 2D/3D array: Background array to composite onto
        border_feathering: Smooth transition width at borders
        distance_mode: Distance metric for 2D border computation
        num_threads: Thread count
        modify_inplace: If True and border_value is array, modify it in-place
        x_discrete: If True, use nearest-neighbor sampling in x direction
    
    Returns:
        - If modify_inplace=True with array border_value: None (modifies in-place)
        - Otherwise: Interpolated values array
    
    Examples:
        >>> # Standard interpolation with constant border
        >>> result = lerp_between_lines_full(line0, line1, coords, border_value=0.5)
        
        >>> # Composite onto background image
        >>> result = lerp_between_lines_full(
        ...     line0, line1, coords,
        ...     border_value=background_image,
        ...     border_feathering=0.1
        ... )
        
        >>> # Modify image in-place
        >>> lerp_between_lines_full(
        ...     line0, line1, coords,
        ...     border_value=image,
        ...     modify_inplace=True,
        ...     border_feathering=0.1
        ... )
    """
    border_is_array = isinstance(border_value, np.ndarray) and border_value.ndim >= 2
    
    if border_is_array:
        if modify_inplace:
            lerp_between_lines_inplace(
                line0,
                line1,
                coords,
                target_array=border_value,
                border_mode=border_mode,
                border_feathering=border_feathering,
                distance_mode=distance_mode,
                num_threads=num_threads,
                x_discrete=x_discrete,
            )
            return None
        else:
            return lerp_between_lines_onto_array(
                line0,
                line1,
                coords,
                background_array=border_value,
                border_mode=border_mode,
                border_feathering=border_feathering,
                distance_mode=distance_mode,
                num_threads=num_threads,
                x_discrete=x_discrete,
            )
    else:
        return lerp_between_lines(
            line0,
            line1,
            coords,
            border_mode=border_mode,
            border_constant=border_value,
            border_feathering=border_feathering,
            num_threads=num_threads,
            x_discrete=x_discrete,
            distance_mode=distance_mode,
        )


# =============================================================================
# Corner Interpolation with Array Border
# =============================================================================
def lerp_from_corners_array_border(
    corners: Union[List[float], np.ndarray],
    coords: CoordsInput,
    background_array: np.ndarray,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_feathering: float = 0.0,
    distance_mode: DistanceMode = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
) -> np.ndarray:
    """
    Interpolate from corner values and composite onto a background array.
    
    Creates a new array where in-bounds regions contain interpolated corner
    values and out-of-bounds regions contain values from background_array
    (with optional feathered blending at boundaries).
    
    Args:
        corners: Corner values:
            - List/tuple of 4 scalars: [top_left, top_right, bottom_left, bottom_right]
            - Array shape (4,): Single channel
            - Array shape (4, C): Multi-channel
        coords: Coordinate array:
            - (H, W, 2): Grid coordinates
            - (N, 2): Flat coordinates
            - (C, H, W, 2): Per-channel grid coordinates
            - (C, N, 2): Per-channel flat coordinates
        background_array: Background to composite onto:
            - (H, W): Single channel grid
            - (H, W, C): Multi-channel grid
            - (N,): Single channel flat
            - (N, C): Multi-channel flat
        border_mode: Border handling mode(s):
            - BorderMode: Same for all channels
            - List/array of BorderMode: Per-channel, shape (C,)
        border_feathering: Smooth blending width at borders (0.0 = hard edge)
        distance_mode: Distance metric for 2D border computation
        num_threads: Thread count (-1=auto, 0=serial, >0=specific count)
    
    Returns:
        New array with corner values composited onto background
    """
    if isinstance(corners, (list, tuple)):
        corners = np.array(corners)
    
    return _lerp_from_corners_array_border(
        _ensure_float64(corners),
        _ensure_coords_array(coords),
        _ensure_float64(background_array),
        border_mode=_normalize_border_mode(border_mode),
        border_feathering=border_feathering,
        distance_mode=distance_mode,
        num_threads=num_threads,
    )


# =============================================================================
# Convenience Partials
# =============================================================================

lerp_between_lines_x_discrete = partial(lerp_between_lines, x_discrete=True)
lerp_between_lines_onto_array_x_discrete = partial(
    lerp_between_lines_onto_array, x_discrete=True
)
lerp_between_lines_full_x_discrete = partial(lerp_between_lines_full, x_discrete=True)
lerp_between_lines_inplace_x_discrete = partial(lerp_between_lines_inplace, x_discrete=True)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "DistanceMode",
    # Core line interpolation
    "lerp_between_lines",
    "lerp_between_lines_inplace",
    "lerp_between_lines_onto_array",
    # Core corner interpolation
    "lerp_from_corners",
    "lerp_from_unpacked_corners",
    "lerp_from_corners_array_border",
    # Unified dispatchers
    "lerp_between_lines_full",
    # Convenience variants
    "lerp_between_lines_x_discrete",
    "lerp_between_lines_onto_array_x_discrete",
    "lerp_between_lines_full_x_discrete",
    "lerp_between_lines_inplace_x_discrete",
]