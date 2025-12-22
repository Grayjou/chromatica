# ===================== core2d.py =====================
"""
2D interpolation wrappers with proper type hints and BoundType support.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from enum import IntEnum

from boundednumbers import BoundType, bound_type_to_np_function
from ...types.array_types import ndarray_1d
from .interp_2d import (  # type: ignore
    lerp_between_lines,
    lerp_between_lines_x_discrete_1ch,
    lerp_between_lines_x_discrete_multichannel,
    lerp_between_planes,
)
from .interp_hue import (  # type: ignore
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
)
from .core import (
    HueMode,
    HueModeSequence,
    BoundTypeSequence,
    _prepare_bound_types,
    _apply_bound,
)

# =============================================================================
# Type Aliases
# =============================================================================
CoordArray2D = np.ndarray  # Shape (H, W, 2)
CoordArrayFlat = np.ndarray  # Shape (N, 2)
LineArray = np.ndarray  # Shape (L,) or (L, C)


# =============================================================================
# Regular (Non-Hue) Line Interpolation
# =============================================================================
def sample_between_lines_continuous(
    line0: LineArray,
    line1: LineArray,
    coords: Union[CoordArray2D, CoordArrayFlat],
    bound_type: BoundType = BoundType.CLAMP,
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
    coords = np.asarray(coords, dtype=np.float64)
    coords = _apply_bound(coords, bound_type)
    return lerp_between_lines(line0, line1, coords)


def sample_between_lines_discrete(
    line0: LineArray,
    line1: LineArray,
    coords: Union[CoordArray2D, CoordArrayFlat],
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Sample between two pre-computed gradient lines with discrete x-sampling.
    
    More efficient when the x-coordinate maps directly to line indices
    (e.g., when L == W).
    
    Args:
        line0: First gradient line, shape (L,) or (L, C)
        line1: Second gradient line, shape (L,) or (L, C)
        coords: Coordinate grid, shape (H, W, 2) or (N, 2)
                coords[..., 0] = u_x (maps to nearest index in lines)
                coords[..., 1] = u_y (blend between lines)
        bound_type: How to bound coordinates
        
    Returns:
        Sampled values, shape matching coords spatial dims + channels if any
    """
    line0 = np.asarray(line0, dtype=np.float64)
    line1 = np.asarray(line1, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    coords = _apply_bound(coords, bound_type)
    
    # Detect if single or multi-channel
    if line0.ndim == 1:
        # Coordinates must be 3D for the Cython function
        if coords.ndim == 2:
            # Flat coords not supported for discrete version yet
            raise NotImplementedError("Discrete line sampling only supports grid coords (H, W, 2)")
        return lerp_between_lines_x_discrete_1ch(line0, line1, coords)
    elif line0.ndim == 2:
        if coords.ndim == 2:
            raise NotImplementedError("Discrete line sampling only supports grid coords (H, W, 2)")
        return lerp_between_lines_x_discrete_multichannel(line0, line1, coords)
    else:
        raise ValueError(f"Unsupported line dimensions: {line0.ndim}")


# =============================================================================
# Hue Line Interpolation
# =============================================================================
def sample_hue_between_lines_continuous(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordArray2D,
    mode_x: HueMode = HueMode.SHORTEST,
    mode_y: HueMode = HueMode.SHORTEST,
    bound_type: BoundType = BoundType.CLAMP,
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
    coords = np.asarray(coords, dtype=np.float64)
    coords = _apply_bound(coords, bound_type)
    
    return hue_lerp_between_lines(line0, line1, coords, int(mode_x), int(mode_y))


def sample_hue_between_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: CoordArray2D,
    mode_y: HueMode = HueMode.SHORTEST,
    bound_type: BoundType = BoundType.CLAMP,
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
    line0 = np.asarray(line0, dtype=np.float64)
    line1 = np.asarray(line1, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    coords = _apply_bound(coords, bound_type)
    
    return hue_lerp_between_lines_x_discrete(line0, line1, coords, int(mode_y))


# =============================================================================
# Multi-channel 2D Interpolation
# =============================================================================
def multival2d_lerp_between_lines_continuous(
    starts_lines: List[np.ndarray],
    ends_lines: List[np.ndarray],
    coords: List[CoordArray2D],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
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
    if len(ends_lines) != num_channels or len(coords) != num_channels:
        raise ValueError("All lists must have same length (num_channels)")
    
    H, W = coords[0].shape[:2]
    
    # Prepare bound types
    if isinstance(bound_types, BoundType):
        bound_types = [bound_types] * num_channels
    else:
        bound_types = list(bound_types)
        if len(bound_types) < num_channels:
            bound_types += [BoundType.CLAMP] * (num_channels - len(bound_types))
    
    out = np.empty((H, W, num_channels), dtype=np.float64)
    
    for ch in range(num_channels):
        out[:, :, ch] = sample_between_lines_continuous(
            starts_lines[ch],
            ends_lines[ch],
            coords[ch],
            bound_types[ch],
        )
    
    return out


def multival2d_lerp_between_lines_discrete(
    starts_lines: List[np.ndarray],
    ends_lines: List[np.ndarray],
    coords: List[CoordArray2D],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
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
    if len(ends_lines) != num_channels or len(coords) != num_channels:
        raise ValueError("All lists must have same length (num_channels)")
    
    H, W = coords[0].shape[:2]
    
    # Prepare bound types
    if isinstance(bound_types, BoundType):
        bound_types = [bound_types] * num_channels
    else:
        bound_types = list(bound_types)
        if len(bound_types) < num_channels:
            bound_types += [BoundType.CLAMP] * (num_channels - len(bound_types))
    
    out = np.empty((H, W, num_channels), dtype=np.float64)
    
    for ch in range(num_channels):
        out[:, :, ch] = sample_between_lines_discrete(
            starts_lines[ch],
            ends_lines[ch],
            coords[ch],
            bound_types[ch],
        )
    
    return out


# =============================================================================
# Corner-based 2D Interpolation
# =============================================================================
def multival2d_lerp_from_corners(
    corners: np.ndarray,
    coords: List[CoordArray2D],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
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
    from .core import multival2d_lerp
    
    num_channels = corners.shape[1]
    if len(coords) != num_channels:
        raise ValueError("coords must have one array per channel")
    
    H, W = coords[0].shape[:2]
    
    # Prepare bound types
    if isinstance(bound_types, BoundType):
        bound_types = [bound_types] * num_channels
    else:
        bound_types = list(bound_types)
        if len(bound_types) < num_channels:
            bound_types += [BoundType.CLAMP] * (num_channels - len(bound_types))
    
    out = np.empty((H, W, num_channels), dtype=np.float64)
    
    # For each channel, use bilinear interpolation from corners
    for ch in range(num_channels):
        tl, tr, bl, br = corners[:, ch]
        starts = np.array([tl, tr], dtype=np.float64)
        ends = np.array([bl, br], dtype=np.float64)
        
        # Stack coordinates for multival2d_lerp
        # It expects shape (H, W, 2) and treats as (u_x, u_y)
        # We need to provide as list of 2D arrays
        coord = coords[ch]  # Shape (H, W, 2)
        coeffs = [coord[:, :, 0], coord[:, :, 1]]  # Split into u_x and u_y
        
        # Use the existing multival2d_lerp which handles bilinear interpolation
        result = multival2d_lerp(
            starts=starts,
            ends=ends,
            coeffs=coeffs,
            bound_types=bound_types[ch],
        )
        out[:, :, ch] = result[:, :, ch] if result.ndim == 3 else result
    
    return out


# =============================================================================
# Plane Interpolation
# =============================================================================
def sample_between_planes(
    plane0: np.ndarray,
    plane1: np.ndarray,
    coords: np.ndarray,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Sample between two pre-computed gradient planes.
    
    Args:
        plane0: First gradient plane, shape (H, W) or (H, W, C)
        plane1: Second gradient plane, shape (H, W) or (H, W, C)
        coords: Coordinate grid, shape (D, H, W, 3)
                coords[..., 0] = u_x, coords[..., 1] = u_y, coords[..., 2] = u_z
        bound_type: How to bound coordinates
        
    Returns:
        Sampled values, shape (D, H, W) or (D, H, W, C)
    """
    coords = np.asarray(coords, dtype=np.float64)
    coords = _apply_bound(coords, bound_type)
    return lerp_between_planes(plane0, plane1, coords)


# =============================================================================
# Re-export from core.py for convenience
# =============================================================================
from .core import (
    multival2d_lerp,
    multival2d_lerp_uniform,
    hue_gradient_2d,
)

__all__ = [
    # Regular line interpolation
    'sample_between_lines_continuous',
    'sample_between_lines_discrete',
    
    # Hue line interpolation
    'sample_hue_between_lines_continuous',
    'sample_hue_between_lines_discrete',
    
    # Multi-channel 2D
    'multival2d_lerp_between_lines_continuous',
    'multival2d_lerp_between_lines_discrete',
    'multival2d_lerp_from_corners',
    
    # Plane interpolation
    'sample_between_planes',
    
    # Re-exports
    'multival2d_lerp',
    'multival2d_lerp_uniform',
    'hue_gradient_2d',
    
    # Types and enums
    'HueMode',
    'BoundType',
]
