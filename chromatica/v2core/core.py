#chromatica\v2core\core.py
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


)
from ..types.color_types import HueDirection

from .interp_hue import (

    hue_lerp_2d_spatial,
    hue_lerp_simple,


    hue_lerp_between_lines,

)

from .interp_2d import (  # type: ignore
    lerp_between_lines,

)
from ..utils.list_mismatch import handle_list_size_mismatch

# =============================================================================
# Type Aliases
# =============================================================================
BoundTypeSequence = Union[List[BoundType], Tuple[BoundType, ...]]



HueDirectionSequence = Union[List[HueDirection], Tuple[HueDirection, ...], HueDirection]


def _prepare_hue_modes(
    modes: HueDirectionSequence,
    n_dims: int,
) -> np.ndarray:
    """Convert hue modes to int32 array for Cython."""
    if isinstance(modes, HueDirection):
        return np.full(n_dims, int(modes), dtype=np.int32)
    
    modes_list = list(modes)
    if len(modes_list) < n_dims:
        # Pad with SHORTEST
        modes_list += [HueDirection.SHORTEST] * (n_dims - len(modes_list))
    
    return np.array([int(m) for m in modes_list[:n_dims]], dtype=np.int32)


# =============================================================================
# Bounding Utilities
# =============================================================================
def _prepare_bound_types(
    bound_types: Union[BoundType, BoundTypeSequence],
    num_channels: int = 1
) -> BoundTypeSequence:
    """Normalize bound_types to a sequence."""
    if isinstance(bound_types, BoundType):
        bt =  [bound_types]*num_channels
    elif isinstance(bound_types, (list, tuple)):
        bt = list(bound_types)
    else:
        raise ValueError(f"Invalid bound_types argument: {bound_types}", type(bound_types))
    if len(bt) < num_channels:
        bt = handle_list_size_mismatch(input_list=bt, target_size=num_channels, 
                                       fill_value=BoundType.CLAMP)
    return bt

def _bound_stacked(U: np.ndarray, bound_type: BoundType) -> np.ndarray:
    """Apply a single bound type to an array."""

    fn = bound_type_to_np_function[bound_type]
    return fn(U, 0.0, 1.0)


def _apply_bound(arr: np.ndarray, bound_type: BoundType) -> np.ndarray:
    """Apply bounding to coefficients if needed."""
    if bound_type is BoundType.IGNORE:
        return arr
    return _bound_stacked(arr, bound_type)

def apply_bounds(arr: np.ndarray | List[np.ndarray], bound_types: BoundTypeSequence) -> np.ndarray | List[np.ndarray]:
    """Apply bounding to array(s) based on bound_types."""
    if isinstance(arr, list):
        return [apply_bounds(a, [bound_types[i] if i < len(bound_types) else BoundType.CLAMP])
                        for i, a in enumerate(arr)]
    else:
        if len(bound_types) == 1:
            bt = bound_types[0]
        else:
            bt = bound_types[0]  # For single array, use first bound type
        return _apply_bound(arr, bt)

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





# =============================================================================
# Hue Interpolation
# =============================================================================
def hue_lerp(
    h0: float,
    h1: float,
    coeffs: np.ndarray,
    mode: HueDirection = HueDirection.SHORTEST,
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
        >>> hues = hue_lerp(0, 240, t, HueDirection.SHORTEST)
        
        >>> # Same but via longest path (through green)
        >>> hues_long = hue_lerp(0, 240, t, HueDirection.LONGEST)
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    coeffs = _apply_bound(coeffs, bound_type)
    return hue_lerp_simple(float(h0), float(h1), coeffs, int(mode))





def hue_gradient_1d(
    h_start: float,
    h_end: float,
    n_steps: int,
    mode: HueDirection = HueDirection.SHORTEST,
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
        >>> hues = hue_gradient_1d(0, 360, 100, HueDirection.CW)
    """
    t = np.linspace(0, 1, n_steps)
    return hue_lerp(h_start, h_end, t, mode)


def hue_gradient_2d(
    corners: Tuple[float, float, float, float],
    shape: Tuple[int, int],
    modes: Tuple[HueDirection, HueDirection] = (HueDirection.SHORTEST, HueDirection.SHORTEST),
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
        >>> hues = hue_gradient_2d(corners, (100, 100), (HueDirection.CW, HueDirection.CCW))
    """
    tl, tr, bl, br = corners

    H, W = shape
    
    # For bilinear: 2 corners for 2D (starts=first half, ends=second half)
    starts = np.array([tl, bl], dtype=np.float64)
    ends = np.array([tr, br], dtype=np.float64)
    
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
    mode_x: HueDirection = HueDirection.SHORTEST,
    mode_y: HueDirection = HueDirection.SHORTEST,
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
        >>> line0 = hue_gradient_1d(0, 120, 100, HueDirection.CW)    # Red→Green
        >>> line1 = hue_gradient_1d(240, 360, 100, HueDirection.CW)  # Blue→Red
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
    mode_x: HueDirection = HueDirection.SHORTEST,
    mode_y: HueDirection = HueDirection.SHORTEST,
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