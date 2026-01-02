"""
High-level typed wrappers for hue interpolation with array/constant borders.

Mirrors the style of interp_2d_/wrappers.py, providing type hints and light
input normalization before calling the Cython kernels.
"""
#chromatica\v2core\interp_hue_\wrappers.py
from __future__ import annotations

from typing import Optional, Union, List, Tuple
import numpy as np
from enum import IntEnum
from functools import partial

from ...types.color_types import HueMode
from ...v2core.border_handler import BorderMode
from ...types.array_types import ndarray_1d, ndarray_2d, ndarray_3d
from ..border_handler import BorderModeInput, DistanceMode, BorderConstant
# Direct imports - no lazy loading, no getters
from .interp_hue import (
    hue_lerp_between_lines_dispatch as _hue_lerp_between_lines,
    hue_lerp_from_corners_dispatch as _hue_lerp_from_corners,
    hue_multidim_lerp as _hue_multidim_lerp,
    hue_lerp_simple as _hue_lerp_simple,
)

from .interp_hue2d_array_border import (
    hue_lerp_between_lines_array_border as _hue_lerp_between_lines_array_border,
    hue_lerp_between_lines_array_border_flat as _hue_lerp_between_lines_array_border_flat,
    hue_lerp_between_lines_array_border_x_discrete as _hue_lerp_between_lines_array_border_x_discrete,
    hue_lerp_between_lines_array_border_flat_x_discrete as _hue_lerp_between_lines_array_border_flat_x_discrete,
)

from .interp_hue_corners_array_border import (
    hue_lerp_from_corners_array_border as _hue_lerp_from_corners_array_border,
    hue_lerp_from_corners_flat_array_border as _hue_lerp_from_corners_flat_array_border,
    hue_lerp_from_corners_multichannel_array_border as _hue_lerp_from_corners_multichannel_array_border,
    hue_lerp_from_corners_multichannel_flat_array_border as _hue_lerp_from_corners_multichannel_flat_array_border,
)





# =============================================================================
# Type Aliases
# =============================================================================



DistanceModeInput = Union[DistanceMode, str, int]
CoordsInput = Union[ndarray_2d, List[ndarray_2d], np.ndarray]


CornersInput = Union[List[float], Tuple[float, float, float, float], np.ndarray]


# =============================================================================
# Normalization Helpers
# =============================================================================

def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Ensure array is float64 and C-contiguous."""
    if arr.dtype == np.float64 and arr.flags['C_CONTIGUOUS']:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _normalize_hue_mode(mode: HueMode) -> int:
    """Convert hue mode to int."""
    if mode is None:
        return HueMode.SHORTEST
    return HueMode(mode)


def _normalize_border_mode(mode: BorderModeInput) -> int:
    """Convert border mode to int."""
    if mode is None:
        return BorderMode.CLAMP
    elif isinstance(mode, BorderMode):
        return int(mode)
    elif isinstance(mode, int):
        return mode
    elif isinstance(mode, str):
        name = mode.lower()
        mapping = {
            'repeat': BorderMode.REPEAT,
            'periodic': BorderMode.REPEAT,
            'mirror': BorderMode.MIRROR,
            'reflect': BorderMode.MIRROR,
            'constant': BorderMode.CONSTANT,
            'fill': BorderMode.CONSTANT,
            'clamp': BorderMode.CLAMP,
            'edge': BorderMode.CLAMP,
            'clip': BorderMode.CLAMP,
            'overflow': BorderMode.OVERFLOW,
            'ignore': BorderMode.OVERFLOW,
        }
        if name in mapping:
            return int(mapping[name])
        raise ValueError(f"Unknown border mode: {mode}")

    raise TypeError(f"border mode must be BorderMode, int, or str, got {type(mode)}")


def _normalize_distance_mode(mode: DistanceModeInput) -> int:
    """Convert distance mode to int."""
    if isinstance(mode, DistanceMode):
        return int(mode)
    if isinstance(mode, int):
        return mode
    if isinstance(mode, str):
        name = mode.lower()
        mapping = {
            'max_norm': DistanceMode.MAX_NORM,
            'max': DistanceMode.MAX_NORM,
            'manhattan': DistanceMode.MANHATTAN,
            'l1': DistanceMode.MANHATTAN,
            'scaled_manhattan': DistanceMode.SCALED_MANHATTAN,
            'alpha_max': DistanceMode.ALPHA_MAX,
            'alpha_max_simple': DistanceMode.ALPHA_MAX_SIMPLE,
            'taylor': DistanceMode.TAYLOR,
            'euclidean': DistanceMode.EUCLIDEAN,  
            'l2': DistanceMode.EUCLIDEAN,
            'true_distance': DistanceMode.EUCLIDEAN,
            'weighted_minmax': DistanceMode.WEIGHTED_MINMAX,
        }
        if name in mapping:
            return int(mapping[name])
        raise ValueError(f"Unknown distance mode: {mode}")
    raise TypeError(f"distance mode must be DistanceMode, int, or str, got {type(mode)}")


def _normalize_coords(coords: CoordsInput) -> np.ndarray:
    """Normalize coordinates to float64 array."""
    if isinstance(coords, list):
        # Handle list of arrays or list of lists
        if len(coords) == 0:
            raise ValueError("coords list cannot be empty")
        if isinstance(coords[0], np.ndarray):
            coords = np.stack(coords, axis=0)
        else:
            coords = np.array(coords)
    return _ensure_float64(coords)


def _normalize_border_constant(constant: BorderConstant) -> Optional[Union[float, np.ndarray]]:
    """Normalize border constant."""
    if constant is None:
        return None
    if isinstance(constant, (int, float)):
        return float(constant)
    if isinstance(constant, list):
        constant = np.array(constant)
    if isinstance(constant, np.ndarray):
        return _ensure_float64(constant)
    raise TypeError(f"border_constant must be None, number, list, or ndarray, got {type(constant)}")


def _pack_corners(
    top_left: Union[float, np.ndarray],
    top_right: Union[float, np.ndarray],
    bottom_left: Union[float, np.ndarray],
    bottom_right: Union[float, np.ndarray],
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
    
    return _ensure_float64(np.stack(
        [top_left, top_right, bottom_left, bottom_right], axis=0
    ))


# =============================================================================
# Line-based Hue Interpolation
# =============================================================================

def hue_lerp_between_lines(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordsInput,
    *,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstant = 0.0,
    border_array: ndarray_2d = None,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    x_discrete: bool = False,
) -> np.ndarray:
    """
    Hue interpolation between two lines at given coordinates.
    
    Performs bilinear hue interpolation between line0 (y=0) and line1 (y=1),
    with configurable border handling and feathering.
    
    Args:
        line0: First hue line at y=0, shape (L,) or (L, C)
        line1: Second hue line at y=1, shape (L,) or (L, C)
        coords: Coordinate array:
            - (H, W, 2): Grid coordinates
            - (N, 2): Flat coordinates
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
        border_array: Optional per-pixel border values
        border_feathering: Smooth transition width at borders (0.0 = hard edge)
        distance_mode: Distance metric for feathering
        num_threads: Thread count (-1=auto, 0=serial, >0=specific)
        x_discrete: If True, use nearest-neighbor sampling in x direction
    
    Returns:
        Interpolated hue values with shape derived from coords
    """
    # Normalize inputs
    line0_norm = _ensure_float64(np.asarray(line0))
    line1_norm = _ensure_float64(np.asarray(line1))
    coords_norm = _normalize_coords(coords)
    
    mode_x_norm = _normalize_hue_mode(mode_x)
    mode_y_norm = _normalize_hue_mode(mode_y)
    border_mode_norm = _normalize_border_mode(border_mode)
    distance_mode_norm = _normalize_distance_mode(distance_mode)
    
    # Handle border array if provided
    border_array_norm = None
    if border_array is not None:
        border_array_norm = _ensure_float64(np.asarray(border_array))
    
    border_constant_norm = _normalize_border_constant(border_constant)
    
    # Use the dispatcher for consistent border handling
    return _hue_lerp_between_lines(
        line0_norm,
        line1_norm,
        coords_norm,
        mode_x=mode_x_norm,
        mode_y=mode_y_norm,
        border_mode=border_mode_norm,
        border_constant=border_constant_norm,
        border_array=border_array_norm,
        border_feathering=border_feathering,
        feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
        distance_mode=distance_mode_norm,
        num_threads=num_threads,
        x_discrete=x_discrete,
    )


def hue_lerp_between_lines_array_border(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordsInput,
    border_array: np.ndarray,
    *,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CONSTANT,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    x_discrete: bool = False,
) -> np.ndarray:
    """
    Hue interpolation with array-based border values.
    
    Creates a new array where in-bounds regions contain interpolated hue values
    and out-of-bounds regions contain values from border_array (with optional
    feathered blending at boundaries).
    
    Args:
        line0: First hue line at y=0, shape (L,) or (L, C)
        line1: Second hue line at y=1, shape (L,) or (L, C)
        coords: Coordinate array:
            - (H, W, 2): Grid coordinates
            - (N, 2): Flat coordinates
        border_array: Border values array:
            - (H, W): Single channel grid
            - (H, W, C): Multi-channel grid
            - (N,): Single channel flat
            - (N, C): Multi-channel flat
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode
        border_feathering: Smooth blending width at borders
        distance_mode: Distance metric for 2D border computation
        num_threads: Thread count
        x_discrete: If True, use nearest-neighbor sampling in x direction
    
    Returns:
        New array with interpolated hue values composited onto border_array
    """
    # Normalize inputs
    line0_norm = _ensure_float64(np.asarray(line0))
    line1_norm = _ensure_float64(np.asarray(line1))
    coords_norm = _normalize_coords(coords)
    border_array_norm = _ensure_float64(np.asarray(border_array))
    
    mode_x_norm = _normalize_hue_mode(mode_x)
    mode_y_norm = _normalize_hue_mode(mode_y)
    border_mode_norm = _normalize_border_mode(border_mode)
    distance_mode_norm = _normalize_distance_mode(distance_mode)
    
    # Determine which function to call based on shape
    if coords_norm.ndim == 3:  # (H, W, 2)
        if x_discrete:
            return _hue_lerp_between_lines_array_border_x_discrete(
                line0_norm,
                line1_norm,
                coords_norm,
                border_array_norm,
                mode_y=mode_y_norm,
                border_mode=border_mode_norm,
                border_feathering=border_feathering,
                feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
                distance_mode=distance_mode_norm,
                num_threads=num_threads,
            )
        else:
            return _hue_lerp_between_lines_array_border(
                line0_norm,
                line1_norm,
                coords_norm,
                border_array_norm,
                mode_x=mode_x_norm,
                mode_y=mode_y_norm,
                border_mode=border_mode_norm,
                border_feathering=border_feathering,
                feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
                distance_mode=distance_mode_norm,
                num_threads=num_threads,
            )
    elif coords_norm.ndim == 2:  # (N, 2)
        if x_discrete:
            return _hue_lerp_between_lines_array_border_flat_x_discrete(
                line0_norm,
                line1_norm,
                coords_norm,
                border_array_norm,
                mode_y=mode_y_norm,
                border_mode=border_mode_norm,
                border_feathering=border_feathering,
                feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
                distance_mode=distance_mode_norm,
                num_threads=num_threads,
            )
        else:
            return _hue_lerp_between_lines_array_border_flat(
                line0_norm,
                line1_norm,
                coords_norm,
                border_array_norm,
                mode_x=mode_x_norm,
                mode_y=mode_y_norm,
                border_mode=border_mode_norm,
                border_feathering=border_feathering,
                feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
                distance_mode=distance_mode_norm,
                num_threads=num_threads,
            )
    else:
        raise ValueError(f"coords must be (H, W, 2) or (N, 2), got shape {coords_norm.shape}")


def hue_lerp_between_lines_inplace(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordsInput,
    target_array: np.ndarray,
    *,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    x_discrete: bool = False,
) -> None:
    """
    Hue interpolation that modifies target_array in-place.
    
    Modifies target_array directly. In-bounds interpolated hue values overwrite
    existing values; out-of-bounds regions are left unchanged (or blended with
    feathering).
    
    Args:
        line0: First hue line at y=0, shape (L,) or (L, C)
        line1: Second hue line at y=1, shape (L,) or (L, C)
        coords: Coordinate array, shape (H, W, 2) or (N, 2)
        target_array: Array to modify in-place, shape (H, W) or (H, W, C) or (N,) or (N, C)
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode
        border_feathering: Smooth blending width at borders
        distance_mode: Distance metric for 2D border computation
        num_threads: Thread count
        x_discrete: If True, use nearest-neighbor sampling in x direction
    
    Returns:
        None (modifies target_array in-place)
    """
    # For in-place, we use array border function and copy result back
    result = hue_lerp_between_lines_array_border(
        line0=line0,
        line1=line1,
        coords=coords,
        border_array=target_array,
        mode_x=mode_x,
        mode_y=mode_y,
        border_mode=border_mode,
        border_feathering=border_feathering,
        feather_hue_mode=feather_hue_mode,
        distance_mode=distance_mode,
        num_threads=num_threads,
        x_discrete=x_discrete,
    )
    
    # Copy result back to target array
    np.copyto(target_array, result)


# =============================================================================
# Corner-based Hue Interpolation
# =============================================================================

def hue_lerp_from_corners(
    corners: CornersInput,
    coords: CoordsInput,
    *,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_array: ndarray_2d = None,
    border_constant: BorderConstant = 0.0,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
) -> np.ndarray:
    """
    Smart dispatcher for hue corner interpolation.
    
    Automatically routes to the appropriate implementation based on:
    - corners shape: (4,) for single-channel, (4, C) for multi-channel
    - coords shape: (H, W, 2), (N, 2), or (C, H, W, 2) for per-channel coords
    - border_array: if provided, uses array-border variant
    - modes_x/modes_y arrays: if provided, uses per-channel modes
    
    Args:
        corners: Corner hue values, shape (4,) or (4, C)
        coords: Coordinate array, shape (H, W, 2), (N, 2), or (C, H, W, 2)
        mode_x: Hue interpolation mode for X axis (int enum)
        mode_y: Hue interpolation mode for Y axis (int enum)
        modes_x: Optional per-channel X modes, shape (C,)
        modes_y: Optional per-channel Y modes, shape (C,)
        border_mode: Border handling mode (int enum)
        border_constant: Constant value for BORDER_CONSTANT mode
        border_array: Optional per-pixel border values
        border_feathering: Feathering distance
        feather_hue_mode: Hue interpolation mode for feathering blend (int enum)
        distance_mode: Distance metric for feathering (int enum)
        num_threads: Number of threads (-1 for auto)
    
    Returns:
        Interpolated hue values with shape matching coords grid
    """
    # Normalize inputs
    if isinstance(corners, (list, tuple)):
        corners = np.array(corners)
    corners_norm = _ensure_float64(np.asarray(corners))
    coords_norm = _normalize_coords(coords)
    
    mode_x_norm = _normalize_hue_mode(mode_x)
    mode_y_norm = _normalize_hue_mode(mode_y)
    border_mode_norm = _normalize_border_mode(border_mode)
    border_constant_norm = _normalize_border_constant(border_constant)
    distance_mode_norm = _normalize_distance_mode(distance_mode)
    
    return _hue_lerp_from_corners(
        corners_norm,
        coords_norm,
        mode_x=mode_x_norm,
        mode_y=mode_y_norm,
        border_mode=border_mode_norm,
        border_array=border_array,
        border_constant=border_constant_norm,
        border_feathering=border_feathering,
        feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
        distance_mode=distance_mode_norm,
        num_threads=num_threads,
    )


def hue_lerp_from_corners_array_border(
    corners: CornersInput,
    coords: CoordsInput,
    border_array: np.ndarray,
    *,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CONSTANT,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
) -> np.ndarray:
    """
    Hue corner interpolation composited onto a border array.
    
    Creates a new array where in-bounds regions contain interpolated corner
    hue values and out-of-bounds regions contain values from border_array
    (with optional feathered blending at boundaries).
    
    Args:
        corners: Corner values:
            - List/tuple of 4 scalars: [top_left, top_right, bottom_left, bottom_right]
            - Array shape (4,): Single channel
            - Array shape (4, C): Multi-channel
        coords: Coordinate array:
            - (H, W, 2): Grid coordinates
            - (N, 2): Flat coordinates
        border_array: Border values array:
            - (H, W): Single channel grid
            - (H, W, C): Multi-channel grid
            - (N,): Single channel flat
            - (N, C): Multi-channel flat
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode
        border_feathering: Smooth blending width at borders
        distance_mode: Distance metric for 2D border computation
        num_threads: Thread count
    
    Returns:
        New array with corner hue values composited onto border_array
    """
    # Normalize inputs
    if isinstance(corners, (list, tuple)):
        corners = np.array(corners)
    corners_norm = _ensure_float64(np.asarray(corners))
    coords_norm = _normalize_coords(coords)
    border_array_norm = _ensure_float64(np.asarray(border_array))
    
    mode_x_norm = _normalize_hue_mode(mode_x)
    mode_y_norm = _normalize_hue_mode(mode_y)
    border_mode_norm = _normalize_border_mode(border_mode)
    distance_mode_norm = _normalize_distance_mode(distance_mode)
    
    # Single-channel case
    if corners_norm.ndim == 1:
        if coords_norm.ndim == 3:  # (H, W, 2)
            return _hue_lerp_from_corners_array_border(
                corners_norm,
                coords_norm,
                border_array_norm,
                mode_x=mode_x_norm,
                mode_y=mode_y_norm,
                border_mode=border_mode_norm,
                border_feathering=border_feathering,
                feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
                distance_mode=distance_mode_norm,
                num_threads=num_threads,
            )
        elif coords_norm.ndim == 2:  # (N, 2)
            return _hue_lerp_from_corners_flat_array_border(
                corners_norm,
                coords_norm,
                border_array_norm,
                mode_x=mode_x_norm,
                mode_y=mode_y_norm,
                border_mode=border_mode_norm,
                border_feathering=border_feathering,
                feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
                distance_mode=distance_mode_norm,
                num_threads=num_threads,
            )
    
    # Multi-channel case
    elif corners_norm.ndim == 2:
        if coords_norm.ndim == 3:  # (H, W, 2)
            return _hue_lerp_from_corners_multichannel_array_border(
                corners_norm,
                coords_norm,
                border_array_norm,
                mode_x=mode_x_norm,
                mode_y=mode_y_norm,
                border_mode=border_mode_norm,
                border_feathering=border_feathering,
                feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
                distance_mode=distance_mode_norm,
                num_threads=num_threads,
            )
        elif coords_norm.ndim == 2:  # (N, 2)
            return _hue_lerp_from_corners_multichannel_flat_array_border(
                corners_norm,
                coords_norm,
                border_array_norm,
                mode_x=mode_x_norm,
                mode_y=mode_y_norm,
                border_mode=border_mode_norm,
                border_feathering=border_feathering,
                feather_hue_mode=_normalize_hue_mode(feather_hue_mode),
                distance_mode=distance_mode_norm,
                num_threads=num_threads,
            )
    
    raise ValueError(f"Invalid corners shape: {corners_norm.shape}")


def hue_lerp_from_unpacked_corners(
    top_left: Union[float, np.ndarray],
    top_right: Union[float, np.ndarray],
    bottom_left: Union[float, np.ndarray],
    bottom_right: Union[float, np.ndarray],
    coords: CoordsInput,
    *,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_array: ndarray_2d = None,
    border_constant: BorderConstant = 0.0,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
) -> np.ndarray:
    """
    Hue interpolation from unpacked corner values.
    
    Convenience wrapper around hue_lerp_from_corners that accepts individual
    corner arrays instead of a packed array.
    
    Args:
        top_left: Top-left corner hue value(s)
        top_right: Top-right corner hue value(s)
        bottom_left: Bottom-left corner hue value(s)
        bottom_right: Bottom-right corner hue value(s)
        coords: Coordinate array
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
        border_feathering: Smooth transition width at borders
        distance_mode: Distance metric for 2D border computation
        num_threads: Thread count
    
    Returns:
        Interpolated hue values
    """
    corners = _pack_corners(top_left, top_right, bottom_left, bottom_right)
    return hue_lerp_from_corners(
        corners,
        coords,
        mode_x=mode_x,
        mode_y=mode_y,
        border_mode=border_mode,
        border_array=border_array,
        border_constant=border_constant,
        border_feathering=border_feathering,
        feather_hue_mode=feather_hue_mode,
        distance_mode=distance_mode,
        num_threads=num_threads,
    )


# =============================================================================
# Multi-dimensional Hue Interpolation
# =============================================================================

def hue_multidim_lerp(
    starts: np.ndarray,
    ends: np.ndarray,
    coeffs: np.ndarray,
    modes: Union[List[HueMode], np.ndarray],
) -> np.ndarray:
    """
    Multi-dimensional hue interpolation with recursive bilinear-like scheme.
    
    Performs hue-aware interpolation across N dimensions using recursive
    bilinear-like interpolation with hue modes for each dimension.
    
    Args:
        starts: Start hue values for corner pairs, shape (2^{N-1},)
        ends: End hue values for corner pairs, shape (2^{N-1},)
        coeffs: Interpolation coefficients for L points across N dimensions,
                shape (L, N) for 1D spatial grid or (H, W, N) for 2D spatial grid
        modes: Interpolation mode for each dimension:
               - List of N modes (each can be HueMode, str, or int)
               - Array shape (N,) of ints
    
    Returns:
        Interpolated hue values with shape derived from coeffs
    """
    # Normalize inputs
    starts_norm = _ensure_float64(np.asarray(starts))
    ends_norm = _ensure_float64(np.asarray(ends))
    coeffs_norm = _ensure_float64(np.asarray(coeffs))
    
    # Normalize modes
    if isinstance(modes, list):
        modes_norm = np.array([_normalize_hue_mode(m) for m in modes], dtype=np.int32)
    else:
        modes_norm = np.asarray(modes, dtype=np.int32)
    
    return _hue_multidim_lerp(starts_norm, ends_norm, coeffs_norm, modes_norm)


# =============================================================================
# Unified Dispatcher
# =============================================================================

def hue_lerp_between_lines_full(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordsInput,
    *,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_value: Optional[Union[float, np.ndarray]] = None,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    modify_inplace: bool = False,
    x_discrete: bool = False,
) -> Optional[np.ndarray]:
    """
    Full-featured hue line interpolation with flexible border handling.
    
    Unified dispatcher that routes to appropriate implementation:
    - If border_value is 2D/3D array and modify_inplace=True: in-place modification
    - If border_value is 2D/3D array and modify_inplace=False: composite onto copy
    - Otherwise: standard interpolation with scalar border constant
    
    Args:
        line0: First hue line at y=0, shape (L,) or (L, C)
        line1: Second hue line at y=1, shape (L,) or (L, C)
        coords: Coordinate array
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode
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
        - Otherwise: Interpolated hue values array
    """
    border_is_array = isinstance(border_value, np.ndarray) and border_value.ndim >= 2
    
    if border_is_array:
        if modify_inplace:
            hue_lerp_between_lines_inplace(
                line0,
                line1,
                coords,
                target_array=border_value,
                mode_x=mode_x,
                mode_y=mode_y,
                border_mode=border_mode,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,
                distance_mode=distance_mode,
                num_threads=num_threads,
                x_discrete=x_discrete,
            )
            return None
        else:
            return hue_lerp_between_lines_array_border(
                line0,
                line1,
                coords,
                border_array=border_value,
                mode_x=mode_x,
                mode_y=mode_y,
                border_mode=border_mode,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,
                distance_mode=distance_mode,
                num_threads=num_threads,
                x_discrete=x_discrete,
            )
    else:
        return hue_lerp_between_lines(
            line0,
            line1,
            coords,
            mode_x=mode_x,
            mode_y=mode_y,
            border_mode=border_mode,
            border_constant=border_value,
            border_feathering=border_feathering,
            feather_hue_mode=feather_hue_mode,
            distance_mode=distance_mode,
            num_threads=num_threads,
            x_discrete=x_discrete,
        )


def hue_lerp_simple(
    hue0: float,
    hue1: float,
    t: np.ndarray,
    mode: HueMode = HueMode.SHORTEST,
) -> float:
    """
    Simple hue interpolation between two scalar hue values.
    
    Args:
        hue0: Starting hue value
        hue1: Ending hue value
        t: Interpolation factor in [0.0, 1.0]
        mode: Hue interpolation mode
    
    Returns:
        Interpolated hue value
    """
    hue0_norm = float(hue0)
    hue1_norm = float(hue1)

    mode_norm = _normalize_hue_mode(mode)
    
    return _hue_lerp_simple(hue0_norm, hue1_norm, t, mode_norm)


def _resolve_hue_modes(modes: Optional[np.ndarray | List[HueMode]], ndims=2):
    """Helper to resolve modes input."""
    if modes is None:
        return np.full((ndims,), HueMode.SHORTEST, dtype=np.int32)
    if isinstance(modes, (list, tuple)):
        return np.array(modes, dtype=np.int32)
    return modes.astype(np.int32)

#For backwards compatibility until I remove all references to hue_lerp_2d_spatial
def hue_lerp_2d_spatial(
    start_hues: np.ndarray,
    end_hues: np.ndarray,
    coeffs: np.ndarray,
    modes: np.ndarray | List[HueMode] | None = None,  # SHORTER
) -> np.ndarray:
    """
    2D spatial hue interpolation.
    
    Args:
        start_hues: Starting hue values, shape (H, W)
        end_hues: Ending hue values, shape (H, W)
        coefficients: Interpolation coefficients in [0, 1], shape (H, W)
        mode: Hue interpolation mode (0=SHORTER, 1=LONGER, 2=INCREASING, 3=DECREASING)
        
    Returns:
        Interpolated hue values, shape (H, W)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    modes = _resolve_hue_modes(modes, ndims=start_hues.ndim)

    corners = np.array([
        start_hues[0], end_hues[0],
        start_hues[-1], end_hues[-1]
    ], dtype=np.float64)
    return hue_lerp_from_corners(
        corners=corners,
        coords=coeffs,
        mode_x=modes[0],
        mode_y=modes[1],
    )

# =============================================================================
# Convenience Partials
# =============================================================================

hue_lerp_between_lines_x_discrete = partial(hue_lerp_between_lines, x_discrete=True, mode_x=None)
hue_lerp_between_lines_array_border_x_discrete = partial(
    hue_lerp_between_lines_array_border, x_discrete=True, mode_x=None
)
hue_lerp_between_lines_full_x_discrete = partial(
    hue_lerp_between_lines_full, x_discrete=True, mode_x=None
)
hue_lerp_between_lines_inplace_x_discrete = partial(
    hue_lerp_between_lines_inplace, x_discrete=True, mode_x=None
)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "HueMode",
    "BorderMode",
    "DistanceMode",
    
    # Line interpolation
    "hue_lerp_between_lines",
    "hue_lerp_between_lines_array_border",
    "hue_lerp_between_lines_inplace",
    "hue_lerp_between_lines_full",
    
    # Corner interpolation
    "hue_lerp_from_corners",
    "hue_lerp_from_corners_array_border",
    "hue_lerp_from_unpacked_corners",
    
    # Multi-dimensional interpolation
    "hue_multidim_lerp",
    
    # Convenience variants
    "hue_lerp_between_lines_x_discrete",
    "hue_lerp_between_lines_array_border_x_discrete",
    "hue_lerp_between_lines_full_x_discrete",
    "hue_lerp_between_lines_inplace_x_discrete",
]