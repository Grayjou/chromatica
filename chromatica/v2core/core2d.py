# ===================== core2d.py =====================
"""
2D interpolation wrappers with proper type hints and BoundType support.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from enum import IntEnum

from boundednumbers import BoundType, bound_type_to_np_function
from ..types.array_types import ndarray_1d

from .interp_2d import (
    lerp_from_corners,
    lerp_between_lines,
    
    lerp_between_lines_x_discrete as lerp_between_lines_x_discrete_multichannel,

)
from .interp_hue import (  # type: ignore
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
)

from .core import (
    HueDirection,
    HueDirectionSequence,
    BoundTypeSequence,
    _prepare_bound_types,
    _apply_bound,
    apply_bounds
)
from .border_handler import (

    BorderMode
)

def _optimize_border_mode(bound_type: BoundType, border_mode: BorderMode) -> BorderMode:
    if bound_type != BoundType.IGNORE:
        #if we are bounding, no need for complex border handling
        return BorderMode.OVERFLOW
    return border_mode

# core2d.py - Add this helper function

def _prepare_border_constant(
    border_constant: Optional[Union[float, np.ndarray, List[float]]],
    num_channels: int,
) -> np.ndarray:
    """
    Pre-resolve border_constant to a contiguous float64 array.
    
    This allows Cython to receive typed data and release the GIL.
    
    Args:
        border_constant: Scalar, array-like, or None
        num_channels: Number of channels (C)
        
    Returns:
        Contiguous float64 array of shape (C,)
    """
    if border_constant is None:
        return np.zeros(num_channels, dtype=np.float64)
    elif isinstance(border_constant, (int, float)):
        return np.full(num_channels, float(border_constant), dtype=np.float64)
    else:
        arr = np.asarray(border_constant, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != num_channels:
            raise ValueError(
                f"border_constant must be scalar or array of length {num_channels}, "
                f"got shape {arr.shape}"
            )
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        return arr


# =============================================================================
# Type Aliases
# =============================================================================
CoordArray2D = np.ndarray  # Shape (H, W, 2)
CoordArrayFlat = np.ndarray  # Shape (N, 2)
LineArray = np.ndarray  # Shape (L,) or (L, C)


def _ensure_list_ndarray(arr: Union[List[np.ndarray], np.ndarray]) -> List[np.ndarray]:
    """Ensure the input is a list of ndarrays."""
    if isinstance(arr, list):
        return arr
    else:
        return list((arr[i] for i in range(arr.shape[0])))

# =============================================================================
# Regular (Non-Hue) Line Interpolation
# =============================================================================
def sample_between_lines_continuous(
    line0: LineArray,
    line1: LineArray,
    coords: Union[CoordArray2D, CoordArrayFlat],
    bound_type: BoundType = BoundType.CLAMP,
    border_mode: BorderMode = BorderMode.OVERFLOW,
    border_constant: Optional[float] = None,
) -> np.ndarray:
    """
    Sample between two pre-computed gradient lines with continuous x-interpolation.
    
    Args:
        line0: First gradient line, shape (L,) or (L, C)
        line1: Second gradient line, shape (L,) or (L, C)
        coords: Coordinate grid, shape (H, W, 2) or (N, 2)
                coords[..., 0] = u_x (position along lines, continuous)
                coords[..., 1] = u_y (blend between lines)
        bound_type: How to bound coordinates
        
    Returns:
        Sampled values, shape matching coords spatial dims + channels if any
        
    Example:
        >>> # Two RGB gradient lines
        >>> line0 = np.linspace([255, 0, 0], [255, 255, 0], 100)  # Red → Yellow
        >>> line1 = np.linspace([0, 0, 255], [0, 255, 0], 100)    # Blue → Green
        >>> 
        >>> # Create coordinate grid
        >>> H, W = 50, 100
        >>> ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        >>> coords = np.stack([ux, uy], axis=-1)
        >>> 
        >>> result = sample_between_lines_continuous(line0, line1, coords)
    """
    border_mode = _optimize_border_mode(bound_type, border_mode)
    coords = np.asarray(coords, dtype=np.float64)
    coords = _apply_bound(coords, bound_type)
    # Detect if single or multi-channel
    # lerp_bertween_lines now detects this internally
    return lerp_between_lines(line0, line1, coords, border_mode=border_mode, border_constant=border_constant)


# =============================================================================
# Hue Line Interpolation
# =============================================================================


def sample_hue_between_lines_continuous(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordArray2D,
    mode_x: HueDirection = HueDirection.SHORTEST,
    mode_y: HueDirection = HueDirection.SHORTEST,
    bound_type: BoundType = BoundType.CLAMP,
    border_mode: BorderMode = BorderMode.OVERFLOW,
    border_constant: Optional[float] = None,
) -> np.ndarray:
    """
    Sample hue between two pre-computed hue gradient lines with continuous x-interpolation.
    
    Args:
        line0: First hue gradient line, shape (L,)
        line1: Second hue gradient line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[..., 0] = u_x (position along lines, continuous)
                coords[..., 1] = u_y (blend between lines)
        mode_x: Interpolation mode for sampling within lines
        mode_y: Interpolation mode for blending between lines
        bound_type: How to bound coordinates
        
    Returns:
        Interpolated hues, shape (H, W), values in [0, 360)
    """

    line0 = np.asarray(line0, dtype=np.float64)
    line1 = np.asarray(line1, dtype=np.float64)
    if line0.ndim == 2:
        # (1, L)
        line0 = line0.flatten()
    if line1.ndim == 2:
        # (1, L)
        line1 = line1.flatten()
    coords = np.asarray(coords, dtype=np.float64)
    coords = _apply_bound(coords, bound_type)
    border_mode = _optimize_border_mode(bound_type, border_mode)
    # Detect if single or multi-channel


    result = hue_lerp_between_lines(
        line0=line0, line1=line1, coords=coords, mode_x=int(mode_x), mode_y=int(mode_y), border_mode=border_mode, border_constant=border_constant if border_constant is not None else 0.0)

    return result


def sample_hue_between_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordArray2D,
    mode_y: HueDirection = HueDirection.SHORTEST,
    bound_type: BoundType = BoundType.CLAMP,
    border_mode: BorderMode = BorderMode.OVERFLOW,
    border_constant: Optional[float] = None,
) -> np.ndarray:
    """
    Sample hue between two pre-computed hue gradient lines with discrete x-sampling.
    
    Args:
        line0: First hue gradient line, shape (L,)
        line1: Second hue gradient line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[..., 0] = u_x (maps to nearest index in lines)
                coords[..., 1] = u_y (blend between lines)
        mode_y: Interpolation mode for blending between lines
        bound_type: How to bound coordinates
        
    Returns:
        Interpolated hues, shape (H, W), values in [0, 360)
    """
    border_mode = _optimize_border_mode(bound_type, border_mode)
    line0 = np.asarray(line0, dtype=np.float64)
    line1 = np.asarray(line1, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    coords = _apply_bound(coords, bound_type)

    return hue_lerp_between_lines_x_discrete(
        line0=line0, 
        line1=line1, 
        coords=coords,
        mode_y=int(mode_y),
        border_mode=border_mode,
        border_constant=border_constant if border_constant is not None else 0.0,
    )


# =============================================================================
# Multi-channel 2D Interpolation
# =============================================================================
def multival2d_lerp_between_lines_continuous(
    starts_lines: List[np.ndarray],
    ends_lines: List[np.ndarray],
    coords: List[CoordArray2D],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
    border_mode: BorderMode = BorderMode.OVERFLOW,
    border_constant: Optional[float] = None,
) -> np.ndarray:
    """
    Multi-channel 2D interpolation via line sampling with per-channel coordinates.
    
    Args:
        starts_lines: List of start gradient lines, one per channel, each shape (L,)
        ends_lines: List of end gradient lines, one per channel, each shape (L,)
        coords: List of coordinate grids, one per channel, each shape (H, W, 2)
        bound_types: Bound types per channel
        
    Returns:
        Interpolated values, shape (H, W, num_channels)
    """
    num_channels = len(starts_lines)
    if len(ends_lines) != num_channels:
        raise ValueError("All lists must have same length (num_channels)")
    
    H, W = coords[0].shape[:2]

    bound_types = _prepare_bound_types(bound_types, num_channels=num_channels)

    coords = apply_bounds(coords, bound_types)
    border_mode = _optimize_border_mode(bound_types if isinstance(bound_types, BoundType) else bound_types[0], border_mode)
    #Make lines shape (L, C)
    start_line = np.stack(starts_lines, axis=-1).astype(np.float64)
    end_line = np.stack(ends_lines, axis=-1).astype(np.float64)

    return lerp_between_lines(
        line0=start_line,
        line1=end_line,
        coords=coords,
        border_mode=border_mode,
        border_constant=border_constant,
    )



def multival2d_lerp_between_lines_discrete(
    starts_lines: List[np.ndarray] | np.ndarray,
    ends_lines: List[np.ndarray] | np.ndarray,
    coords: List[CoordArray2D],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
    border_mode: BorderMode = BorderMode.OVERFLOW,
    border_constant: Optional[float] = None,
) -> np.ndarray:
    """
    Multi-channel 2D interpolation via discrete line sampling with per-channel coordinates.
    
    Args:
        starts_lines: List of start gradient lines, one per channel, each shape (L,)
        ends_lines: List of end gradient lines, one per channel, each shape (L,)
        coords: List of coordinate grids, one per channel, each shape (H, W, 2)
        bound_types: Bound types per channel
        
    Returns:
        Interpolated values, shape (H, W, num_channels)
    """
    num_channels = len(starts_lines)

    #starts_lines = _ensure_list_ndarray(starts_lines)
    #ends_lines = _ensure_list_ndarray(ends_lines)


    if len(ends_lines) != num_channels:
        raise ValueError("All lists must have same length (num_channels)")
    
    H, W = coords[0].shape[:2]
    print("multival2d_lerp_between_lines_discrete:", num_channels, H, W)
    bound_types = _prepare_bound_types(bound_types, num_channels=num_channels) 
    coords = apply_bounds(coords, bound_types)
    border_mode = _optimize_border_mode(bound_types if isinstance(bound_types, BoundType) else bound_types[0], border_mode)
    start_line = np.stack(starts_lines, axis=-1).astype(np.float64)
    end_line = np.stack(ends_lines, axis=-1).astype(np.float64)
    return lerp_between_lines_x_discrete_multichannel(
        line0=start_line,
        line1=end_line,
        coords=coords,
        border_mode=border_mode,
        border_constant=border_constant,
    )

# =============================================================================
# Corner-based 2D Interpolation
# =============================================================================

def multival2d_lerp_from_corners(
    corners: np.ndarray,
    coords: List[CoordArray2D],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
    border_mode: BorderMode = BorderMode.OVERFLOW,
    border_constant: Optional[float] = None,
) -> np.ndarray:
    """
    Multi-channel 2D interpolation from corner values with per-channel coordinates.
    
    Args:
        corners: Corner values, shape (4, num_channels)
                 Order: [top_left, top_right, bottom_left, bottom_right]
        coords: List of coordinate grids, one per channel, each shape (H, W, 2)
                coords[..., 0] = u_x, coords[..., 1] = u_y
        bound_types: Bound types per channel
        
    Returns:
        Interpolated values, shape (H, W, num_channels)
    """

    border_mode = _optimize_border_mode(bound_types if isinstance(bound_types, BoundType) else bound_types[0], border_mode)

    num_channels = corners.shape[1]
    bound_types = _prepare_bound_types(bound_types, num_channels=num_channels) 
    coords = apply_bounds(coords, bound_types)
    H, W = coords[0].shape[:2]

    return lerp_from_corners(
        corners,
        coords,
        border_mode=border_mode,
        border_constant=border_constant,
    )





# =============================================================================
# Re-export from core.py for convenience
# =============================================================================
from .core import (
    hue_gradient_2d,
)

__all__ = [
    # Regular line interpolation
    'sample_between_lines_continuous',

    
    # Hue line interpolation
    'sample_hue_between_lines_continuous',
    'sample_hue_between_lines_discrete',
    
    # Multi-channel 2D
    'multival2d_lerp_between_lines_continuous',
    'multival2d_lerp_between_lines_discrete',
    'multival2d_lerp_from_corners',
    

    
    # Re-exports
    'multival2d_lerp',

    'hue_gradient_2d',
    
    # Types and enums
    'HueDirection',
    'BoundType',
]
