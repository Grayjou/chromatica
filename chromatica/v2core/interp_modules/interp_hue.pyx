# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Main dispatcher for hue interpolation functions.

Imports all hue interpolation functionality from split modules.
"""

import numpy as np
cimport numpy as np

# Import from split modules
from .interp_hue_simple import (
    hue_lerp_simple,
    hue_lerp_arrays,
)

from .interp_hue_spatial import (
    hue_lerp_1d_spatial,
    hue_lerp_2d_spatial,
)

from .interp_hue_between_lines import (
    hue_lerp_2d_with_modes,
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
)

# =============================================================================
# Dispatcher for multi-dim hue lerp
# =============================================================================
def hue_multidim_lerp(
    np.ndarray starts,
    np.ndarray ends,
    np.ndarray coeffs,
    np.ndarray modes,
):
    """
    Dispatcher for multi-dimensional hue interpolation.
    
    Args:
        starts: Corner values, shape (2^{N-1},)
        ends: Corner values, shape (2^{N-1},)
        coeffs: Coefficient grid, shape (..., N) where N is num dimensions
        modes: Mode per dimension, shape (N,)
    
    Returns:
        Interpolated hues with shape matching coeffs[..., 0]
    """
    if starts.dtype != np.float64 or not starts.flags['C_CONTIGUOUS']:
        starts = np.ascontiguousarray(starts, dtype=np.float64)
    if ends.dtype != np.float64 or not ends.flags['C_CONTIGUOUS']:
        ends = np.ascontiguousarray(ends, dtype=np.float64)
    if coeffs.dtype != np.float64 or not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
    if modes.dtype != np.int32 or not modes.flags['C_CONTIGUOUS']:
        modes = np.ascontiguousarray(modes, dtype=np.int32)
    
    cdef int spatial_ndims = coeffs.ndim - 1
    
    if spatial_ndims == 1:
        return hue_lerp_1d_spatial(starts, ends, coeffs, modes)
    elif spatial_ndims == 2:
        return hue_lerp_2d_spatial(starts, ends, coeffs, modes)
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_ndims}")
