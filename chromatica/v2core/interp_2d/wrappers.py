"""
Type-hinted wrappers for 2D interpolation functions.

This module provides Python wrappers with type hints for the Cython-optimized
interpolation functions. The wrappers add type information while maintaining
the same behavior as the underlying Cython functions.
"""

from typing import Optional
import numpy as np

# Import Cython functions as internal
try:

    from .interp_2d_fast import (
        lerp_between_lines_x_discrete_1ch_fast as _lerp_between_lines_x_discrete_1ch,#_fast,
        lerp_between_lines_multichannel_fast as _lerp_between_lines_multichannel,#_fast,
        lerp_between_lines_full_fast as _lerp_between_lines,#_fast,
        lerp_between_lines_x_discrete_full_fast as _lerp_between_lines_x_discrete_multichannel,#_fast,

    )

    from .corner_interp_2d_fast import (
        lerp_from_corners_1ch_flat_fast as _lerp_from_corners_1ch_flat,#_fast,
        lerp_from_corners_multichannel_fast as _lerp_from_corners_multichannel,#_fast,
        lerp_from_corners_multichannel_same_coords_fast as _lerp_from_corners_multichannel_same_coords,#_fast,
        lerp_from_corners_multichannel_flat_fast as _lerp_from_corners_multichannel_flat,#_fast,
        lerp_from_corners_multichannel_flat_same_coords_fast as _lerp_from_corners_multichannel_flat_same_coords,#_fast,
        lerp_from_corners_fast as _lerp_from_corners,#_fast,
    )
    from .interp_planes import (
        lerp_between_planes as _lerp_between_planes,
    )
    CYTHON_AVAILABLE = True
except ImportError as e:
    CYTHON_AVAILABLE = False
    print("Cython interp_2d extensions not available:", e)

# =============================================================================
# Line Interpolation Functions
# =============================================================================

def _ensure_coords_array(
    coords: np.ndarray | list[np.ndarray]
) -> np.ndarray:
    """Ensure coords is a single numpy array."""
    if isinstance(coords, list):
        return np.array(coords)
    return coords

def lerp_between_lines(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray | list[np.ndarray],
    border_mode: int = 3,  # BORDER_CLAMP
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate between two 1D lines using 2D coordinates with continuous x-interpolation.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x (position along lines, 0-1, continuous)
                coords[h, w, 1] = u_y (blend factor between lines, 0-1)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
                    0=REPEAT, 1=MIRROR, 2=CONSTANT, 3=CLAMP, 4=OVERFLOW
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (H, W)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    coords = _ensure_coords_array(coords)
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_between_lines(line0, line1, coords, border_mode, border_constant)


def lerp_between_lines_x_discrete_1ch(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray | list[np.ndarray],
    border_mode: int = 3,  # BORDER_CLAMP
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate between two 1D lines using 2D coordinates with discrete x-sampling.
    
    More efficient when u_x maps directly to line indices (e.g., when L == W).
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x (maps to nearest index in lines, 0-1)
                coords[h, w, 1] = u_y (blend factor between lines, 0-1)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (H, W)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    coords = _ensure_coords_array(coords)
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_between_lines_x_discrete_multichannel(line0, line1, coords, border_mode, border_constant)


def lerp_between_lines_multichannel(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray | list[np.ndarray],
    border_mode: int = 3,  # BORDER_CLAMP
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate between two multichannel lines using 2D coordinates with continuous x-interpolation.
    
    Args:
        line0: First line, shape (L, C) where C is number of channels
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x (position along lines, 0-1, continuous)
                coords[h, w, 1] = u_y (blend factor between lines, 0-1)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (H, W, C)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    coords = _ensure_coords_array(coords)
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return lerp_between_lines(line0, line1, coords, border_mode, border_constant)


def lerp_between_lines_x_discrete_multichannel(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray | list[np.ndarray],
    border_mode: int = 3,  # BORDER_CLAMP
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate between two multichannel lines using 2D coordinates with discrete x-sampling.
    
    Args:
        line0: First line, shape (L, C) where C is number of channels
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x (maps to nearest index in lines, 0-1)
                coords[h, w, 1] = u_y (blend factor between lines, 0-1)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (H, W, C)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    coords = _ensure_coords_array(coords)
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_between_lines_x_discrete_multichannel(line0, line1, coords, border_mode, border_constant)


# =============================================================================
# Corner Interpolation Functions
# =============================================================================

def lerp_from_corners(
    corners: np.ndarray,
    coords_list: list,
    border_mode: int = 3,  # BORDER_CLAMP
    border_constant: Optional[float] = None,
) -> np.ndarray:
    """
    Interpolate from corner values using coordinate grids.
    
    Args:
        corners: Corner values, shape (4, C) where C is number of channels
                Order: [top_left, top_right, bottom_left, bottom_right]
        coords_list: List of coordinate grids, one per channel
                    Each grid has shape (H, W, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (H, W, C)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_from_corners(corners, coords_list, border_mode, border_constant)


def lerp_from_corners_1ch_flat(
    c_tl: float,
    c_tr: float,
    c_bl: float,
    c_br: float,
    coords: np.ndarray,
    border_mode: int = 3,
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate from 4 corner scalars using flat coordinates.
    
    Args:
        c_tl: Top-left corner value
        c_tr: Top-right corner value
        c_bl: Bottom-left corner value
        c_br: Bottom-right corner value
        coords: Flat coordinate array, shape (N, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (N,)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_from_corners_1ch_flat(c_tl, c_tr, c_bl, c_br, coords, border_mode, border_constant)


def lerp_from_corners_multichannel(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    coords: np.ndarray,
    border_mode: int = 3,
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate from 4 corner value arrays using 2D coordinates.
    
    Args:
        c_tl: Top-left corner values, shape (C,)
        c_tr: Top-right corner values, shape (C,)
        c_bl: Bottom-left corner values, shape (C,)
        c_br: Bottom-right corner values, shape (C,)
        coords: Coordinate grid, shape (H, W, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (H, W, C)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_from_corners_multichannel(c_tl, c_tr, c_bl, c_br, coords, border_mode, border_constant)


def lerp_from_corners_multichannel_same_coords(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    coords: np.ndarray,
    border_mode: int = 3,
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate from corner values using same coordinates for all channels.
    
    Args:
        c_tl: Top-left corner values, shape (C,)
        c_tr: Top-right corner values, shape (C,)
        c_bl: Bottom-left corner values, shape (C,)
        c_br: Bottom-right corner values, shape (C,)
        coords: Single coordinate grid for all channels, shape (H, W, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (H, W, C)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_from_corners_multichannel_same_coords(c_tl, c_tr, c_bl, c_br, coords, border_mode, border_constant)


def lerp_from_corners_multichannel_flat(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    coords: np.ndarray,
    border_mode: int = 3,
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate from corner values using flat coordinates.
    
    Args:
        c_tl: Top-left corner values, shape (C,)
        c_tr: Top-right corner values, shape (C,)
        c_bl: Bottom-left corner values, shape (C,)
        c_br: Bottom-right corner values, shape (C,)
        coords: Flat coordinate array, shape (N, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (N, C)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_from_corners_multichannel_flat(c_tl, c_tr, c_bl, c_br, coords, border_mode, border_constant)


def lerp_from_corners_multichannel_flat_same_coords(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    coords: np.ndarray,
    border_mode: int = 3,
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate from corner values using flat coordinates (same for all channels).
    
    Args:
        c_tl: Top-left corner values, shape (C,)
        c_tr: Top-right corner values, shape (C,)
        c_bl: Bottom-left corner values, shape (C,)
        c_br: Bottom-right corner values, shape (C,)
        coords: Flat coordinate array, shape (N, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (N, C)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_from_corners_multichannel_flat_same_coords(c_tl, c_tr, c_bl, c_br, coords, border_mode, border_constant)


# =============================================================================
# Plane Interpolation Functions
# =============================================================================

def lerp_between_planes(
    plane0: np.ndarray,
    plane1: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """
    Interpolate between two pre-computed planes using 3D coordinates.
    
    Args:
        plane0: First plane, shape (H, W) or (H, W, C)
        plane1: Second plane, shape (H, W) or (H, W, C)
        coords: Coordinate grid, shape (D, H, W, 3)
                coords[..., 0] = u_x
                coords[..., 1] = u_y
                coords[..., 2] = u_z (blend between planes, 0-1)
        
    Returns:
        Interpolated values, shape (D, H, W) or (D, H, W, C)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_2d extensions not available. Please build extensions.")
    return _lerp_between_planes(plane0, plane1, coords)


__all__ = [
    'lerp_between_lines',
    'lerp_between_lines_x_discrete_1ch',
    'lerp_between_lines_multichannel',
    'lerp_between_lines_x_discrete_multichannel',
    'lerp_from_corners',
    'lerp_from_corners_1ch_flat',
    'lerp_from_corners_multichannel',
    'lerp_from_corners_multichannel_same_coords',
    'lerp_from_corners_multichannel_flat',
    'lerp_from_corners_multichannel_flat_same_coords',
    'lerp_between_planes',
]
