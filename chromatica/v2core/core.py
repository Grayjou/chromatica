# ===================== core.py =====================
"""
Core interpolation functions with Cython-accelerated backends.
"""

import numpy as np
from enum import IntEnum
from typing import List, Tuple, Union, Sequence

from boundednumbers import BoundType, bound_type_to_np_function
from ..types.array_types import ndarray_1d
from .interp import (  # type: ignore
    lerp_bounded_1d_spatial_fast,
    lerp_bounded_2d_spatial_fast,
    single_channel_multidim_lerp_bounded_cython_fast,
)
from .interp_hue import (  # type: ignore
    hue_lerp_1d_spatial,
    hue_lerp_2d_spatial,
    hue_lerp_simple,
    hue_lerp_arrays,
    hue_lerp_2d_with_modes,
    hue_lerp_between_lines,
    hue_multidim_lerp,
)
from .interp_2d import (  # type: ignore
    lerp_between_lines,
    lerp_between_planes,
)

# =============================================================================
# Type Aliases
# =============================================================================
BoundTypeSequence = Union[List[BoundType], Tuple[BoundType, ...]]


# =============================================================================
# Hue Interpolation Mode
# =============================================================================
class HueMode(IntEnum):
    """
    Hue interpolation modes for cyclical color space.
    
    CW:       Clockwise (increasing hue direction)
    CCW:      Counterclockwise (decreasing hue direction)
    SHORTEST: Shortest path (≤180° arc) - most common
    LONGEST:  Longest path (≥180° arc)
    """
    CW = 0
    CCW = 1
    SHORTEST = 2
    LONGEST = 3


HueModeSequence = Union[List[HueMode], Tuple[HueMode, ...], HueMode]


def _prepare_hue_modes(
    modes: HueModeSequence,
    n_dims: int,
) -> np.ndarray:
    """Convert hue modes to int32 array for Cython."""
    if isinstance(modes, HueMode):
        return np.full(n_dims, int(modes), dtype=np.int32)
    
    modes_list = list(modes)
    if len(modes_list) < n_dims:
        # Pad with SHORTEST
        modes_list += [HueMode.SHORTEST] * (n_dims - len(modes_list))
    
    return np.array([int(m) for m in modes_list[:n_dims]], dtype=np.int32)


# =============================================================================
# Bounding Utilities
# =============================================================================
def _prepare_bound_types(
    bound_types: Union[BoundType, BoundTypeSequence],
) -> BoundTypeSequence:
    """Normalize bound_types to a sequence."""
    if isinstance(bound_types, BoundType):
        return [bound_types]
    if isinstance(bound_types, (list, tuple)):
        return bound_types
    raise ValueError("Invalid bound_types argument")


def _bound_stacked(U: np.ndarray, bound_type: BoundType) -> np.ndarray:
    """Apply a single bound type to an array."""
    fn = bound_type_to_np_function[bound_type]
    return fn(U, 0.0, 1.0)


def _apply_bound(arr: np.ndarray, bound_type: BoundType) -> np.ndarray:
    """Apply bounding to coefficients if needed."""
    if bound_type is BoundType.IGNORE:
        return arr
    return _bound_stacked(arr, bound_type)


def bound_coeffs(
    coeffs: List[ndarray_1d],
    bound_types: BoundTypeSequence,
) -> List[ndarray_1d]:
    """Apply per-dimension bounding to coefficient arrays."""
    coeffs = [np.asarray(c) for c in coeffs]
    D = len(coeffs)

    if len(bound_types) < D:
        bound_types = list(bound_types) + [BoundType.CLAMP] * (D - len(bound_types))

    if all(bt is BoundType.IGNORE for bt in bound_types):
        return coeffs

    U = np.stack(coeffs, axis=-1)
    U_out = np.empty_like(U)

    for bt in set(bound_types):
        idx = [i for i, t in enumerate(bound_types) if t is bt]
        if bt is BoundType.IGNORE:
            U_out[..., idx] = U[..., idx]
        else:
            U_out[..., idx] = _bound_stacked(U[..., idx], bt)

    return [U_out[..., i] for i in range(D)]


def bound_coeffs_fused(
    coeffs: List[np.ndarray],
    bound_types: BoundTypeSequence,
) -> List[np.ndarray]:
    """Apply per-dimension bounding with in-place operations where possible."""
    coeffs = [np.asarray(c) for c in coeffs]
    D = len(coeffs)

    if len(bound_types) < D:
        bound_types = list(bound_types) + [BoundType.CLAMP] * (D - len(bound_types))

    if all(bt is BoundType.IGNORE for bt in bound_types):
        return coeffs

    U = np.stack(coeffs, axis=-1)

    for i, bt in enumerate(bound_types):
        if bt is BoundType.IGNORE:
            continue
        fn = bound_type_to_np_function[bt]
        try:
            fn(U[..., i], 0.0, 1.0, out=U[..., i])
        except TypeError:
            U[..., i] = fn(U[..., i], 0.0, 1.0)

    return [U[..., i] for i in range(D)]


# =============================================================================
# Multi-dimensional Interpolation (Generic)
# =============================================================================
def single_channel_multidim_lerp(
    starts: np.ndarray,
    ends: np.ndarray,
    coeffs: np.ndarray,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """Vectorized multi-dimensional linear interpolation."""
    bounded = _bound_stacked(coeffs, bound_type)
    return single_channel_multidim_lerp_bounded_cython_fast(starts, ends, bounded)


# =============================================================================
# 1D/2D Multi-value Interpolation
# =============================================================================
def multival1d_lerp(
    starts: np.ndarray,
    ends: np.ndarray,
    coeffs: List[np.ndarray],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
    prefer_float64: bool = True,
) -> np.ndarray:
    """Multi-channel 1D linear interpolation."""
    starts = np.asarray(starts, dtype=np.float64).ravel()
    ends = np.asarray(ends, dtype=np.float64).ravel()
    coeffs = [np.asarray(c, dtype=np.float64) for c in coeffs]

    num_channels = len(starts)
    num_steps = coeffs[0].shape[0]

    if len(ends) != num_channels:
        raise ValueError("starts and ends must have same length")
    if len(coeffs) != num_channels:
        raise ValueError("coeffs must have one array per channel")

    if bound_types is BoundType.IGNORE:
        bounded = coeffs
    else:
        bounded = bound_coeffs(coeffs, _prepare_bound_types(bound_types))

    out = np.empty((num_steps, num_channels), dtype=np.float64)

    for ch in range(num_channels):
        U = bounded[ch][:, np.newaxis]
        out[:, ch] = lerp_bounded_1d_spatial_fast(
            np.array([starts[ch]], dtype=np.float64),
            np.array([ends[ch]], dtype=np.float64),
            U,
        )

    return out


def multival2d_lerp(
    starts: np.ndarray,
    ends: np.ndarray,
    coeffs: List[np.ndarray],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
) -> np.ndarray:
    """Multi-channel 2D linear interpolation over a spatial grid."""
    starts = np.asarray(starts, dtype=np.float64).ravel()
    ends = np.asarray(ends, dtype=np.float64).ravel()
    coeffs = [np.asarray(c, dtype=np.float64) for c in coeffs]

    num_channels = len(starts)
    H, W = coeffs[0].shape[:2]

    if len(ends) != num_channels:
        raise ValueError("starts and ends must have same length")
    if len(coeffs) != num_channels:
        raise ValueError("coeffs must have one array per channel")
    if any(c.shape[:2] != (H, W) for c in coeffs):
        raise ValueError("All coeff arrays must have same spatial shape")

    if bound_types is BoundType.IGNORE:
        bounded = coeffs
    else:
        bounded = bound_coeffs(coeffs, _prepare_bound_types(bound_types))

    out = np.empty((H, W, num_channels), dtype=np.float64)

    for ch in range(num_channels):
        U = bounded[ch][:, :, np.newaxis]
        out[:, :, ch] = lerp_bounded_2d_spatial_fast(
            np.array([starts[ch]], dtype=np.float64),
            np.array([ends[ch]], dtype=np.float64),
            U,
        )

    return out


def multival1d_lerp_uniform(
    starts: np.ndarray,
    ends: np.ndarray,
    coeff: np.ndarray,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """1D interpolation with same coefficient for all channels."""
    num_channels = len(np.atleast_1d(starts))
    coeffs = [coeff] * num_channels
    return multival1d_lerp(starts, ends, coeffs, bound_types=bound_type)


def multival2d_lerp_uniform(
    starts: np.ndarray,
    ends: np.ndarray,
    coeff: np.ndarray,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """2D interpolation with same coefficient for all channels."""
    num_channels = len(np.atleast_1d(starts))
    coeffs = [coeff] * num_channels
    return multival2d_lerp(starts, ends, coeffs, bound_types=bound_type)


# =============================================================================
# Hue Interpolation
# =============================================================================
def hue_lerp(
    h0: float,
    h1: float,
    coeffs: np.ndarray,
    mode: HueMode = HueMode.SHORTEST,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Simple 1D hue interpolation between two values.
    
    Args:
        h0: Start hue in degrees [0, 360)
        h1: End hue in degrees [0, 360)
        coeffs: Interpolation coefficients, shape (N,)
        mode: Interpolation mode (CW, CCW, SHORTEST, LONGEST)
        bound_type: How to bound coefficients
        
    Returns:
        Interpolated hues, shape (N,), values in [0, 360)
        
    Example:
        >>> # Interpolate from red (0°) to blue (240°) via shortest path
        >>> t = np.linspace(0, 1, 10)
        >>> hues = hue_lerp(0, 240, t, HueMode.SHORTEST)
        
        >>> # Same but via longest path (through green)
        >>> hues_long = hue_lerp(0, 240, t, HueMode.LONGEST)
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    coeffs = _apply_bound(coeffs, bound_type)
    return hue_lerp_simple(float(h0), float(h1), coeffs, int(mode))


def hue_lerp_multi(
    h0_arr: np.ndarray,
    h1_arr: np.ndarray,
    coeffs: np.ndarray,
    mode: HueMode = HueMode.SHORTEST,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Vectorized hue interpolation for multiple hue pairs.
    
    Args:
        h0_arr: Start hues, shape (M,)
        h1_arr: End hues, shape (M,)
        coeffs: Interpolation coefficients, shape (N,)
        mode: Interpolation mode
        bound_type: How to bound coefficients
        
    Returns:
        Interpolated hues, shape (N, M)
    """
    h0_arr = np.asarray(h0_arr, dtype=np.float64)
    h1_arr = np.asarray(h1_arr, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    coeffs = _apply_bound(coeffs, bound_type)
    return hue_lerp_arrays(h0_arr, h1_arr, coeffs, int(mode))


def hue_multidim_lerp_bounded(
    starts: np.ndarray,
    ends: np.ndarray,
    coeffs: np.ndarray,
    modes: HueModeSequence = HueMode.SHORTEST,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Multi-dimensional hue interpolation with per-dimension modes.
    
    This is the hue-aware version of single_channel_multidim_lerp.
    Each dimension can have a different interpolation mode.
    
    Args:
        starts: Corner start values, shape (2^{N-1},)
        ends: Corner end values, shape (2^{N-1},)
        coeffs: Coefficient grid, shape (D_1, ..., D_k, N)
        modes: Interpolation mode per dimension (single or sequence)
        bound_type: How to bound coefficients
        
    Returns:
        Interpolated hues, shape (D_1, ..., D_k), values in [0, 360)
        
    Example:
        >>> # 2D hue gradient with different modes per axis
        >>> starts = np.array([0.0])    # Red
        >>> ends = np.array([240.0])    # Blue
        >>> 
        >>> H, W = 100, 200
        >>> ux = np.linspace(0, 1, W)
        >>> uy = np.linspace(0, 1, H)
        >>> coeffs = np.stack(np.meshgrid(ux, uy, indexing='xy'), axis=-1)
        >>> 
        >>> # X-axis: shortest path, Y-axis: clockwise
        >>> modes = [HueMode.SHORTEST, HueMode.CW]
        >>> result = hue_multidim_lerp_bounded(starts, ends, coeffs, modes)
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    
    n_dims = coeffs.shape[-1]
    modes_arr = _prepare_hue_modes(modes, n_dims)
    
    # Bound coefficients
    coeffs = _apply_bound(coeffs, bound_type)
    
    return hue_multidim_lerp(starts, ends, coeffs, modes_arr)


def hue_gradient_1d(
    h_start: float,
    h_end: float,
    n_steps: int,
    mode: HueMode = HueMode.SHORTEST,
) -> np.ndarray:
    """
    Create a 1D hue gradient.
    
    Args:
        h_start: Start hue in degrees
        h_end: End hue in degrees
        n_steps: Number of steps
        mode: Interpolation mode
        
    Returns:
        Hue values, shape (n_steps,)
        
    Example:
        >>> # Rainbow gradient (full circle clockwise)
        >>> hues = hue_gradient_1d(0, 360, 100, HueMode.CW)
    """
    t = np.linspace(0, 1, n_steps)
    return hue_lerp(h_start, h_end, t, mode)


def hue_gradient_2d(
    corners: Tuple[float, float, float, float],
    shape: Tuple[int, int],
    modes: Tuple[HueMode, HueMode] = (HueMode.SHORTEST, HueMode.SHORTEST),
) -> np.ndarray:
    """
    Create a 2D hue gradient from 4 corner values.
    
    Args:
        corners: (top_left, top_right, bottom_left, bottom_right) hues
        shape: (H, W) output shape
        modes: (x_mode, y_mode) interpolation modes
        
    Returns:
        Hue values, shape (H, W)
        
    Example:
        >>> # Hue wheel quadrant
        >>> corners = (0, 90, 270, 180)  # Red, Yellow, Purple, Cyan
        >>> hues = hue_gradient_2d(corners, (100, 100), (HueMode.CW, HueMode.CCW))
    """
    tl, tr, bl, br = corners
    H, W = shape
    
    # For bilinear: 2 corners for 2D (starts=first half, ends=second half)
    starts = np.array([tl, tr], dtype=np.float64)
    ends = np.array([bl, br], dtype=np.float64)
    
    ux = np.linspace(0, 1, W)
    uy = np.linspace(0, 1, H)
    coeffs = np.stack(np.meshgrid(ux, uy, indexing='xy'), axis=-1)
    
    modes_arr = np.array([int(modes[0]), int(modes[1])], dtype=np.int32)
    
    return hue_lerp_2d_spatial(starts, ends, coeffs, modes_arr)


# =============================================================================
# Section 6: Hue interpolation between lines
# =============================================================================
def sample_hue_between_lines(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Sample hue between two pre-computed hue gradient lines.
    
    This is the hue-aware version of sample_between_lines for Section 6.
    Both the within-line sampling and between-line blending respect
    the cyclical nature of hue values.
    
    Args:
        line0: First hue gradient line, shape (L,)
        line1: Second hue gradient line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[..., 0] = u_x (position along lines)
                coords[..., 1] = u_y (blend between lines)
        mode_x: Interpolation mode for sampling within lines
        mode_y: Interpolation mode for blending between lines
        bound_type: How to bound coordinates
        
    Returns:
        Interpolated hues, shape (H, W), values in [0, 360)
        
    Example:
        >>> # Two hue gradient lines
        >>> line0 = hue_gradient_1d(0, 120, 100, HueMode.CW)    # Red→Green
        >>> line1 = hue_gradient_1d(240, 360, 100, HueMode.CW)  # Blue→Red
        >>> 
        >>> # Create coordinate grid
        >>> H, W = 50, 100
        >>> ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        >>> coords = np.stack([ux, uy], axis=-1)
        >>> 
        >>> # Sample with shortest path for both axes
        >>> result = sample_hue_between_lines(line0, line1, coords)
    """
    line0 = np.asarray(line0, dtype=np.float64)
    line1 = np.asarray(line1, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    
    # Bound coordinates
    coords = _apply_bound(coords, bound_type)
    
    return hue_lerp_between_lines(line0, line1, coords, int(mode_x), int(mode_y))


def make_hue_line_sampler(
    line0: np.ndarray,
    line1: np.ndarray,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    bound_type: BoundType = BoundType.CLAMP,
):
    """
    Create a reusable sampler for a pair of hue lines.
    
    Returns:
        A callable: sampler(coords) → interpolated hues
    """
    line0 = np.ascontiguousarray(line0, dtype=np.float64)
    line1 = np.ascontiguousarray(line1, dtype=np.float64)
    mx, my = int(mode_x), int(mode_y)
    
    def sampler(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        coords = _apply_bound(coords, bound_type)
        return hue_lerp_between_lines(line0, line1, coords, mx, my)
    
    return sampler


# =============================================================================
# Section 6: Line-to-plane for regular values (re-export with bounds)
# =============================================================================
def sample_between_lines(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Sample between two pre-computed gradient lines at arbitrary coordinates.
    
    Args:
        line0: First gradient line, shape (L,) or (L, C)
        line1: Second gradient line, shape (L,) or (L, C)
        coords: Coordinate grid, shape (H, W, 2) or (N, 2)
        bound_type: How to bound coordinates
        
    Returns:
        Sampled values, shape matching coords spatial dims
    """
    coords = np.asarray(coords, dtype=np.float64)
    coords = _apply_bound(coords, bound_type)
    return lerp_between_lines(line0, line1, coords)