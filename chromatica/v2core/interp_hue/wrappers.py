"""
Type-hinted wrappers for hue interpolation functions.

This module provides Python wrappers with type hints for the Cython-optimized
hue interpolation functions. The wrappers add type information while maintaining
the same behavior as the underlying Cython functions.
"""

from typing import Optional, Tuple
import numpy as np

# Import Cython functions as internal
try:
    from .interp_hue import (
        hue_lerp_1d_spatial as _hue_lerp_1d_spatial,
        hue_lerp_simple as _hue_lerp_simple,
        hue_lerp_arrays as _hue_lerp_arrays,
        hue_multidim_lerp as _hue_multidim_lerp,
    )
    from .interp_hue2d import (
        hue_lerp_between_lines as _hue_lerp_between_lines,
        hue_lerp_between_lines_x_discrete as _hue_lerp_between_lines_x_discrete,
        hue_lerp_2d_spatial as _hue_lerp_2d_spatial,
        hue_lerp_2d_with_modes as _hue_lerp_2d_with_modes,
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


# =============================================================================
# 1D Hue Interpolation Functions
# =============================================================================

def hue_lerp_1d_spatial(
    start_hues: np.ndarray,
    end_hues: np.ndarray,
    coefficients: np.ndarray,
    mode: int = 0,  # SHORTER
) -> np.ndarray:
    """
    Spatially interpolate hues along a 1D array with a specific hue mode.
    
    Args:
        start_hues: Starting hue values, shape (N,)
        end_hues: Ending hue values, shape (N,)
        coefficients: Interpolation coefficients in [0, 1], shape (N,)
        mode: Hue interpolation mode (0=SHORTER, 1=LONGER, 2=INCREASING, 3=DECREASING)
        
    Returns:
        Interpolated hue values, shape (N,)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_hue extensions not available. Please build extensions.")
    return _hue_lerp_1d_spatial(start_hues, end_hues, coefficients, mode)


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


def hue_lerp_arrays(
    start_hues: np.ndarray,
    end_hues: np.ndarray,
    coefficients: np.ndarray,
    mode: int = 0,  # SHORTER
) -> np.ndarray:
    """
    Interpolate between arrays of hue values.
    
    Args:
        start_hues: Starting hue values, shape (N,)
        end_hues: Ending hue values, shape (N,)
        coefficients: Interpolation coefficients in [0, 1], shape (N,)
        mode: Hue interpolation mode (0=SHORTER, 1=LONGER, 2=INCREASING, 3=DECREASING)
        
    Returns:
        Interpolated hue values, shape (N,)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_hue extensions not available. Please build extensions.")
    return _hue_lerp_arrays(start_hues, end_hues, coefficients, mode)


def hue_multidim_lerp(
    start_hues: np.ndarray,
    end_hues: np.ndarray,
    coefficients: np.ndarray,
    mode: int = 0,  # SHORTER
) -> np.ndarray:
    """
    Multidimensional hue interpolation.
    
    Handles arbitrary-shaped arrays for hue interpolation.
    
    Args:
        start_hues: Starting hue values, shape (...,)
        end_hues: Ending hue values, shape (...,)
        coefficients: Interpolation coefficients in [0, 1], shape (...,)
        mode: Hue interpolation mode (0=SHORTER, 1=LONGER, 2=INCREASING, 3=DECREASING)
        
    Returns:
        Interpolated hue values, shape (...,)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_hue extensions not available. Please build extensions.")
    return _hue_multidim_lerp(start_hues, end_hues, coefficients, mode)


# =============================================================================
# 2D Hue Interpolation Functions
# =============================================================================

def hue_lerp_between_lines(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray,
    mode_x: int = 0,  # SHORTER
    mode_y: int = 0,  # SHORTER
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
    return _hue_lerp_between_lines(line0, line1, coords, mode_x, mode_y, border_mode, border_constant)


def hue_lerp_between_lines_x_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    coords: np.ndarray,
    mode_y: int = 0,  # SHORTER
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
    return _hue_lerp_between_lines_x_discrete(line0, line1, coords, mode_y, border_mode, border_constant)


def hue_lerp_2d_spatial(
    start_hues: np.ndarray,
    end_hues: np.ndarray,
    coefficients: np.ndarray,
    mode: int = 0,  # SHORTER
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
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_hue extensions not available. Please build extensions.")
    return _hue_lerp_2d_spatial(start_hues, end_hues, coefficients, mode)


def hue_lerp_2d_with_modes(
    start_hues: np.ndarray,
    end_hues: np.ndarray,
    coefficients: np.ndarray,
    modes: np.ndarray,
) -> np.ndarray:
    """
    2D hue interpolation with per-pixel mode selection.
    
    Args:
        start_hues: Starting hue values, shape (H, W)
        end_hues: Ending hue values, shape (H, W)
        coefficients: Interpolation coefficients in [0, 1], shape (H, W)
        modes: Hue mode per pixel, shape (H, W), values in {0, 1, 2, 3}
              (0=SHORTER, 1=LONGER, 2=INCREASING, 3=DECREASING)
        
    Returns:
        Interpolated hue values, shape (H, W)
        
    Raises:
        ImportError: If Cython extensions are not built
    """
    if not CYTHON_AVAILABLE:
        raise ImportError("Cython interp_hue extensions not available. Please build extensions.")
    return _hue_lerp_2d_with_modes(start_hues, end_hues, coefficients, modes)


__all__ = [
    'hue_lerp_1d_spatial',
    'hue_lerp_simple',
    'hue_lerp_arrays',
    'hue_multidim_lerp',
    'hue_lerp_between_lines',
    'hue_lerp_between_lines_x_discrete',
    'hue_lerp_2d_spatial',
    'hue_lerp_2d_with_modes',
]
