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

    """        
    from .interp_2d_fast import (
        lerp_between_lines_full_fast as _lerp_between_lines,#_fast,
        lerp_between_lines_x_discrete_full_fast as _lerp_between_lines_x_discrete_multichannel,#_fast,
        lerp_between_lines_2d_multichannel_same_coords as _lerp_between_lines_multichannel_same_coords,#_fast,
        lerp_between_lines_2d_x_discrete_multichannel_same_coords as _lerp_between_lines_x_discrete_multichannel_same_coords,#_fast,
    )"""
    #Temporary overrides for testing

    from ..interp_2d_.wrappers import (lerp_between_lines as _lerp_between_lines,
                                       lerp_between_lines_x_discrete as _lerp_between_lines_x_discrete_multichannel)

    from ..interp_2d_.wrappers import (
        lerp_from_corners as _lerp_from_corners,
        lerp_from_unpacked_corners
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
    num_channels: Optional[int] = None,
) -> np.ndarray:
    """
    Interpolate between two 1D lines using 2D coordinates with continuous x-interpolation.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Handles all coordinate configurations:
            - Single channel: (H, W, 2) or (N, 2)
            - Multi-channel same coords: (H, W, 2) or (N, 2)  
            - Multi-channel per-channel coords: (C, H, W, 2) or (C, N, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
                    0=REPEAT, 1=MIRROR, 2=CONSTANT, 3=CLAMP, 4=OVERFLOW
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated values, shape (H, W)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    return _lerp_between_lines(line0, line1, coords, border_mode, border_constant)







def lerp_between_lines_x_discrete_multichannel(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray | list[np.ndarray],
    border_mode: int = 3,  # BORDER_CLAMP
    border_constant: float = 0.0,
    num_channels: Optional[int] = None,
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









__all__ = [
    'lerp_between_lines',
    'lerp_between_lines_x_discrete_1ch',
    'lerp_between_lines_multichannel',
    'lerp_between_lines_x_discrete_multichannel',
    'lerp_from_corners',
    'lerp_from_corners_1ch_flat',
    'lerp_from_corners_multichannel',




]
