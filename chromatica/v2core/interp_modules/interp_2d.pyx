# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np

ctypedef np.float64_t f64

# Import the underlying functions from the split modules
from .interp_2d_1ch import (
    lerp_between_lines_1ch,
    lerp_between_lines_flat_1ch,
    lerp_between_planes_1ch,
    lerp_between_lines_x_discrete_1ch,
)

from .interp_2d_multichannel import (
    lerp_between_lines_multichannel,
    lerp_between_lines_flat_multichannel,
    lerp_between_lines_x_discrete_multichannel,
)


# =============================================================================
# Dispatcher: lerp_between_planes
# =============================================================================
def lerp_between_planes(
    np.ndarray plane0,
    np.ndarray plane1,
    np.ndarray coords,
):
    """
    Interpolate between two planes at arbitrary (u_x, u_y, u_z) coordinates.
    
    Extends Section 6 to higher dimensions.
    
    Args:
        plane0: First plane, shape (H, W) or (H, W, C)
        plane1: Second plane, shape (H, W) or (H, W, C)
        coords: Coordinates, shape (..., 3) where last dim is (u_x, u_y, u_z)
    
    Returns:
        Interpolated values
    """
    if plane0.dtype != np.float64 or not plane0.flags['C_CONTIGUOUS']:
        plane0 = np.ascontiguousarray(plane0, dtype=np.float64)
    if plane1.dtype != np.float64 or not plane1.flags['C_CONTIGUOUS']:
        plane1 = np.ascontiguousarray(plane1, dtype=np.float64)
    if coords.dtype != np.float64 or not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    if plane0.ndim == 2 and coords.ndim == 4:
        return lerp_between_planes_1ch(plane0, plane1, coords)
    
    raise ValueError("Unsupported shapes for lerp_between_planes")


# =============================================================================
# Dispatcher: lerp_between_lines
# =============================================================================
def lerp_between_lines(
    np.ndarray line0,
    np.ndarray line1,
    np.ndarray coords,
):
    """
    Interpolate between two lines at arbitrary (u_x, u_y) coordinates.
    
    This implements Section 6 of the interpolation article:
    creating gradient planes by sampling between pre-existing gradient lines.
    
    Args:
        line0: First line, shape (L,) or (L, C) for multi-channel
        line1: Second line, shape (L,) or (L, C) for multi-channel
        coords: Coordinates where to sample
                - Shape (H, W, 2): grid of (u_x, u_y) pairs → output (H, W) or (H, W, C)
                - Shape (N, 2): flat list of pairs → output (N,) or (N, C)
                
                u_x ∈ [0, 1]: position along the lines
                u_y ∈ [0, 1]: blend factor (0 = line0, 1 = line1)
    
    Returns:
        Interpolated values at each coordinate
        
    Example:
        >>> # Create two RGB gradient lines
        >>> line0 = np.linspace([255, 0, 0], [255, 255, 0], 100)  # Red → Yellow
        >>> line1 = np.linspace([0, 0, 255], [0, 255, 0], 100)    # Blue → Green
        >>> 
        >>> # Create coordinate grid
        >>> H, W = 50, 100
        >>> u_x = np.linspace(0, 1, W)  # Sample along lines
        >>> u_y = np.linspace(0, 1, H)  # Blend between lines
        >>> coords = np.stack(np.meshgrid(u_x, u_y, indexing='xy'), axis=-1)
        >>> 
        >>> result = lerp_between_lines(line0, line1, coords)
        >>> result.shape
        (50, 100, 3)
    """
    # Ensure float64 contiguous
    if line0.dtype != np.float64 or not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0, dtype=np.float64)
    if line1.dtype != np.float64 or not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1, dtype=np.float64)
    if coords.dtype != np.float64 or not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    # Determine shapes
    line_ndim = line0.ndim
    coords_ndim = coords.ndim
    
    # Grid coords (H, W, 2)
    if coords_ndim == 3 and coords.shape[2] == 2:
        if line_ndim == 1:
            return lerp_between_lines_1ch(line0, line1, coords)
        elif line_ndim == 2:
            return lerp_between_lines_multichannel(line0, line1, coords)
    
    # Flat coords (N, 2)
    elif coords_ndim == 2 and coords.shape[1] == 2:
        if line_ndim == 1:
            return lerp_between_lines_flat_1ch(line0, line1, coords)
        elif line_ndim == 2:
            return lerp_between_lines_flat_multichannel(line0, line1, coords)
    
    raise ValueError(
        "Unsupported shapes for lerp_between_lines"
    )
