#interp_hue.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Hue interpolation with cyclical color space support.

Supports four interpolation modes per dimension:
- CW (0): Clockwise interpolation (always positive direction)
- CCW (1): Counterclockwise interpolation (always negative direction)
- SHORTEST (2): Shortest angular distance (≤180°)
- LONGEST (3): Longest angular distance (≥180°)

Functions in this module handle hue interpolation in multi-dimensional spaces
with proper handling of the cyclical nature of hue values (0-360° range).
"""
 
import numpy as np
cimport numpy as np
from libc.math cimport fmod, floor
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free

# Import border handling
from ..border_handling cimport (
    handle_border_lines_2d,
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)
from .interp_hue_utils cimport (
    wrap_hue,
    adjust_end_for_mode,
    lerp_hue_single,
)




from .interp_hue2d cimport (
    hue_lerp_2d_spatial,
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
    hue_lerp_2d_with_modes,

)
ctypedef np.float64_t f64
ctypedef np.int32_t i32





# =============================================================================
# Hue Mode Constants
# =============================================================================
DEF HUE_CW = 0
DEF HUE_CCW = 1
DEF HUE_SHORTEST = 2
DEF HUE_LONGEST = 3


# =============================================================================
# 1D Hue Interpolation: coeffs (L, N), modes (N,) -> output (L,)
# =============================================================================
def hue_lerp_1d_spatial(
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=2] coeffs,
    np.ndarray[i32, ndim=1] modes,
):
    """
    Multi-dimensional hue interpolation along a 1D spatial grid.
    
    Performs recursive bilinear-like interpolation across N dimensions using
    hue-aware interpolation modes for each dimension.
    
    Args:
        starts: Start hue values for corner pairs, shape (2^{N-1},)
                Each element corresponds to a start hue for a pair of corners.
        ends: End hue values for corner pairs, shape (2^{N-1},)
              Each element corresponds to an end hue for a pair of corners.
        coeffs: Interpolation coefficients for L points across N dimensions,
                shape (L, N). Each row contains coefficients for one point.
        modes: Interpolation mode for each dimension, shape (N,)
               Each element must be one of:
               0=CW, 1=CCW, 2=SHORTEST, 3=LONGEST
    
    Returns:
        np.ndarray: Interpolated hue values for L points, shape (L,)
                    Values are wrapped to [0, 360) range.
    
    Notes:
        - Total number of corners in the N-dimensional hypercube is 2^N
        - starts and ends arrays each contain half of these corners (2^{N-1})
        - Interpolation proceeds recursively: first interpolate along dimension 0
          between each start-end pair, then along dimension 1, etc.
    """
    cdef Py_ssize_t num_points = coeffs.shape[0]
    cdef Py_ssize_t num_dims = coeffs.shape[1]
    cdef Py_ssize_t num_corners = starts.shape[0]
    
    if ends.shape[0] != num_corners:
        raise ValueError("starts and ends must have same length")
    if modes.shape[0] != num_dims:
        raise ValueError("modes must have length equal to num dimensions")
    if num_corners != (1 << (num_dims - 1)):
        raise ValueError(
            f"starts/ends length {num_corners} doesn't match "
            f"2^(num_dims-1)={1 << (num_dims - 1)} for num_dims={num_dims}"
        )
    
    # Ensure contiguous
    if not starts.flags['C_CONTIGUOUS']:
        starts = np.ascontiguousarray(starts)
    if not ends.flags['C_CONTIGUOUS']:
        ends = np.ascontiguousarray(ends)
    if not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs)
    if not modes.flags['C_CONTIGUOUS']:
        modes = np.ascontiguousarray(modes)
    
    cdef f64[::1] starts_mv = starts
    cdef f64[::1] ends_mv = ends
    cdef f64[:, ::1] coeffs_mv = coeffs
    cdef i32[::1] modes_mv = modes
    
    cdef np.ndarray[f64, ndim=1] result = np.empty(num_points, dtype=np.float64)
    cdef f64[::1] result_mv = result
    
    cdef Py_ssize_t p, i, j, half, curr_size
    cdef f64 u, h0, h1, h1_adj
    cdef int mode
    
    # Working buffers
    cdef Py_ssize_t MAX_CORNERS = 256
    cdef f64 a_stack[256]
    cdef f64 b_stack[256]
    cdef f64* a = NULL
    cdef f64* b = NULL
    
    if num_corners <= MAX_CORNERS:
        for p in range(num_points):
            # Initialize working buffers
            memcpy(&a_stack[0], &starts_mv[0], num_corners * sizeof(f64))
            memcpy(&b_stack[0], &ends_mv[0], num_corners * sizeof(f64))
            
            curr_size = num_corners
            
            for i in range(num_dims):
                u = coeffs_mv[p, i]
                mode = modes_mv[i]
                
                # Lerp each pair with hue mode
                for j in range(curr_size):
                    h0 = a_stack[j]
                    h1 = b_stack[j]
                    h1_adj = adjust_end_for_mode(h0, h1, mode)
                    a_stack[j] = wrap_hue(h0 + u * (h1_adj - h0))
                
                # Split for next iteration
                if curr_size > 1:
                    half = curr_size >> 1
                    memcpy(&b_stack[0], &a_stack[half], half * sizeof(f64))
                    curr_size = half
            
            result_mv[p] = a_stack[0]
        
        return result
    
    # Heap fallback for large corner counts
    a = <f64*>malloc(num_corners * sizeof(f64))
    b = <f64*>malloc(num_corners * sizeof(f64))
    if a == NULL or b == NULL:
        if a != NULL: free(a)
        if b != NULL: free(b)
        raise MemoryError("Failed to allocate working buffers")
    
    try:
        for p in range(num_points):
            memcpy(a, &starts_mv[0], num_corners * sizeof(f64))
            memcpy(b, &ends_mv[0], num_corners * sizeof(f64))
            
            curr_size = num_corners
            
            for i in range(num_dims):
                u = coeffs_mv[p, i]
                mode = modes_mv[i]
                
                for j in range(curr_size):
                    h0 = a[j]
                    h1 = b[j]
                    h1_adj = adjust_end_for_mode(h0, h1, mode)
                    a[j] = wrap_hue(h0 + u * (h1_adj - h0))
                
                if curr_size > 1:
                    half = curr_size >> 1
                    memcpy(b, a + half, half * sizeof(f64))
                    curr_size = half
            
            result_mv[p] = a[0]
    finally:
        free(a)
        free(b)
    
    return result



# =============================================================================
# Simple 1D Hue Lerp: Single dimension interpolation
# =============================================================================
def hue_lerp_simple(
    f64 h0,
    f64 h1,
    np.ndarray[f64, ndim=1] coeffs,
    int mode = HUE_SHORTEST,
):
    """
    Simple 1D hue interpolation between two values for multiple coefficients.
    
    Args:
        h0: Start hue value in degrees
        h1: End hue value in degrees
        coeffs: Interpolation coefficients, shape (N,)
                Each coefficient u should typically be in [0, 1]
        mode: Interpolation mode (0=CW, 1=CCW, 2=SHORTEST, 3=LONGEST)
    
    Returns:
        np.ndarray: Interpolated hue values for each coefficient, shape (N,)
                    Values are wrapped to [0, 360) range.
    
    Examples:
        >>> hue_lerp_simple(10, 350, [0.0, 0.5, 1.0], HUE_SHORTEST)
        array([10., 0., 350.])  # Shortest path goes through 0°
    """
    cdef Py_ssize_t N = coeffs.shape[0]
    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out
    cdef f64[::1] c = coeffs
    
    cdef f64 h1_adj = adjust_end_for_mode(h0, h1, mode)
    cdef Py_ssize_t i
    cdef f64 u
    
    for i in range(N):
        u = c[i]
        out_mv[i] = wrap_hue(h0 + u * (h1_adj - h0))
    
    return out


# =============================================================================
# Vectorized Simple Lerp: Arrays of start/end pairs
# =============================================================================
def hue_lerp_arrays(
    np.ndarray[f64, ndim=1] h0_arr,
    np.ndarray[f64, ndim=1] h1_arr,
    np.ndarray[f64, ndim=1] coeffs,
    int mode = HUE_SHORTEST,
):
    """
    Vectorized 1D hue interpolation for multiple hue pairs and coefficients.
    
    Interpolates between corresponding pairs in h0_arr and h1_arr for each
    coefficient in coeffs, producing a 2D grid of results.
    
    Args:
        h0_arr: Array of start hue values, shape (M,)
        h1_arr: Array of end hue values, shape (M,)
                Must match length of h0_arr
        coeffs: Interpolation coefficients, shape (N,)
                Each coefficient u should typically be in [0, 1]
        mode: Interpolation mode (0=CW, 1=CCW, 2=SHORTEST, 3=LONGEST)
    
    Returns:
        np.ndarray: Interpolated hue values, shape (N, M)
                    out[i, j] = interpolation between h0_arr[j] and h1_arr[j]
                    at coefficient coeffs[i]
                    Values are wrapped to [0, 360) range.
    
    Notes:
        - Broadcasts N coefficients across M hue pairs
        - Equivalent to N calls to hue_lerp_simple for different start/end pairs
    """
    cdef Py_ssize_t M = h0_arr.shape[0]
    cdef Py_ssize_t N = coeffs.shape[0]
    
    if h1_arr.shape[0] != M:
        raise ValueError("h0_arr and h1_arr must have same length")
    
    if not h0_arr.flags['C_CONTIGUOUS']:
        h0_arr = np.ascontiguousarray(h0_arr)
    if not h1_arr.flags['C_CONTIGUOUS']:
        h1_arr = np.ascontiguousarray(h1_arr)
    if not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs)
    
    cdef f64[::1] h0_mv = h0_arr
    cdef f64[::1] h1_mv = h1_arr
    cdef f64[::1] c = coeffs
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((N, M), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t i, j
    cdef f64 u, h0, h1, h1_adj
    
    for j in range(M):
        h0 = h0_mv[j]
        h1 = h1_mv[j]
        h1_adj = adjust_end_for_mode(h0, h1, mode)
        
        for i in range(N):
            u = c[i]
            out_mv[i, j] = wrap_hue(h0 + u * (h1_adj - h0))
    
    return out



# =============================================================================
# Hue interpolation between lines (Section 6 style)
# =============================================================================


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
    
    Automatically routes to appropriate implementation based on spatial
    dimensions of the coefficient grid.
    
    Args:
        starts: Start hue values for corner pairs, shape (2^{N-1},)
        ends: End hue values for corner pairs, shape (2^{N-1},)
        coeffs: Coefficient grid, shape (..., N) where N is number of dimensions
                Last dimension must match length of modes
        modes: Mode per dimension, shape (N,)
               Each element must be one of:
               0=CW, 1=CCW, 2=SHORTEST, 3=LONGEST
    
    Returns:
        np.ndarray: Interpolated hue values with shape matching coeffs[..., 0]
                    (i.e., spatial dimensions of coeffs without the last dim)
    
    Raises:
        ValueError: If shapes are incompatible
        NotImplementedError: If spatial dimensions > 2
    
    Notes:
        - For 1D spatial grid (coeffs shape: (L, N)) -> returns shape (L,)
        - For 2D spatial grid (coeffs shape: (H, W, N)) -> returns shape (H, W)
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
        raise NotImplementedError(
            f"Hue lerp for {spatial_ndims}D spatial grid not implemented. "
            f"Only 1D (L, N) and 2D (H, W, N) coefficient grids are supported."
        )