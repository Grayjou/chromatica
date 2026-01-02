"""
Type-hinted wrappers for hue interpolation functions.

This module provides Python wrappers with type hints for the Cython-optimized
hue interpolation functions. The wrappers add type information while maintaining
the same behavior as the underlying Cython functions.
"""

from typing import List, Optional, Tuple
import numpy as np

# Import Cython functions as internal
try:

    from ..interp_hue_.wrappers import (hue_lerp_between_lines as _hue_lerp_between_lines,
                                        hue_lerp_between_lines_x_discrete as _hue_lerp_between_lines_x_discrete,
                                        hue_lerp_simple as _hue_lerp_simple,
                                        hue_lerp_from_corners

                                        )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
from ...types.color_types import HueMode

# =============================================================================
# 1D Hue Interpolation Functions
# =============================================================================

def _resolve_hue_modes(modes: Optional[np.ndarray | List[HueMode]], ndims=2):
    """Helper to resolve modes input."""
    if modes is None:
        return np.full((ndims,), HueMode.SHORTEST, dtype=np.int32)
    if isinstance(modes, (list, tuple)):
        return np.array(modes, dtype=np.int32)
    return modes.astype(np.int32)




def hue_lerp_simple(
    start_hue: float,
    end_hue: float,
    coef: float,
    mode: int = 0,  # SHORTER
) -> float:
    """
    Simple hue interpolation between two scalar hue values.
    
    Args:
        start_hue: Starting hue value (0-1)
        end_hue: Ending hue value (0-1)
        coef: Interpolation coefficient in [0, 1]
        mode: Hue interpolation mode (0=SHORTER, 1=LONGER, 2=INCREASING, 3=DECREASING)
        
    Returns:
        Interpolated hue value (0-1)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_hue extensions not available. Please build extensions.")
    return _hue_lerp_simple(start_hue, end_hue, coef, mode)



# =============================================================================
# 2D Hue Interpolation Functions
# =============================================================================

def hue_lerp_between_lines(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray,
    mode_x: HueMode = HueMode.SHORTEST,  # SHORTER
    mode_y: HueMode = HueMode.SHORTEST,  # SHORTER
    border_mode: int = 3,  # BORDER_CLAMP
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate hue values between two lines with continuous x-interpolation.
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x (position along lines, 0-1, continuous)
                coords[h, w, 1] = u_y (blend factor between lines, 0-1)
        mode_x: Hue mode for horizontal interpolation (along lines)
        mode_y: Hue mode for vertical interpolation (between lines)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
                    0=REPEAT, 1=MIRROR, 2=CONSTANT, 3=CLAMP, 4=OVERFLOW
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated hue values, shape (H, W)
        
    Raises:
        ImportError: If Cython extensions are not built
    """

    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_hue extensions not available. Please build extensions.")
    if isinstance(border_constant, np.ndarray):
        border_constant = border_constant[0]
    return _hue_lerp_between_lines(
        line0=line0,
        line1=line1,
        coords=coords,
        mode_x=mode_x,
        mode_y=mode_y,
        border_mode=border_mode,
        border_constant=border_constant
    )
    #return _hue_lerp_between_lines(line0, line1, coords, mode_x, mode_y, border_mode, border_constant)


def hue_lerp_between_lines_x_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray,
    mode_y: HueMode = HueMode.SHORTEST,  # SHORTER
    border_mode: int = 3,  # BORDER_CLAMP
    border_constant: float = 0.0,
) -> np.ndarray:
    """
    Interpolate hue values between two lines with discrete x-sampling.
    
    More efficient when u_x maps directly to line indices (e.g., when L == W).
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x (maps to nearest index in lines, 0-1)
                coords[h, w, 1] = u_y (blend factor between lines, 0-1)
        mode_y: Hue mode for vertical interpolation (between lines)
        border_mode: Border handling mode (default: BORDER_CLAMP=3)
        border_constant: Value for out-of-bounds coordinates when border_mode=CONSTANT
        
    Returns:
        Interpolated hue values, shape (H, W)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_hue extensions not available. Please build extensions.")
    return _hue_lerp_between_lines_x_discrete(
        line0=line0,
        line1=line1,
        coords=coords,
        mode_y=mode_y,
        border_mode=border_mode,
        border_constant=border_constant
    )
    #return _hue_lerp_between_lines_x_discrete(line0, line1, coords, mode_y, border_mode, border_constant)


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
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_hue extensions not available. Please build extensions.")
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





__all__ = [

    'hue_lerp_simple',
    'hue_lerp_between_lines',
    'hue_lerp_between_lines_x_discrete',
    'hue_lerp_2d_spatial',

]
