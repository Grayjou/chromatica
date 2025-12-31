"""
Hue interpolation with cyclical color space support.

Modernized with feathering, distance modes, and array-border handling to
mirror the corner interpolators. Includes prange support for parallelism.
"""

import numpy as np
cimport numpy as np
from libc.math cimport fmod, floor
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange

# Import border handling and modern features
from ..border_handling cimport (
    handle_border_lines_2d,
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,

)
from ..interp_utils cimport (
    BorderResult,
    process_border_2d,
    MAX_NORM,
    MANHATTAN,
    SCALED_MANHATTAN,
    ALPHA_MAX,
    ALPHA_MAX_SIMPLE,
    TAYLOR,
    EUCLIDEAN,
    WEIGHTED_MINMAX,
)
from .interp_hue_utils cimport (
    f64,                   
    i32,                   
    HueMode,
    HUE_CW,
    HUE_CCW,
    HUE_SHORTEST,
    HUE_LONGEST,
    wrap_hue,               
    adjust_end_for_mode,    
    lerp_hue_single,
    _interp_line_1ch_hue,
    _interp_line_discrete_hue,
    process_hue_border_2d,
)

cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_spatial(
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=3] coeffs,
    np.ndarray[i32, ndim=1] modes,
):
    """
    Multi-dimensional hue interpolation over a 2D spatial grid.
    ... (rest of docstring)
    """
    cdef Py_ssize_t H = coeffs.shape[0]
    cdef Py_ssize_t W = coeffs.shape[1]
    cdef Py_ssize_t num_dims = coeffs.shape[2]
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
    cdef f64[:, :, ::1] coeffs_mv = coeffs
    cdef i32[::1] modes_mv = modes
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t h, w, i, j, half, curr_size
    cdef f64 u, h0, h1, h1_adj
    cdef int mode
    
    cdef Py_ssize_t MAX_CORNERS = 256
    cdef f64 a_stack[256]
    cdef f64 b_stack[256]
    cdef f64* a = NULL
    cdef f64* b = NULL
    
    if num_corners <= MAX_CORNERS:
        for h in range(H):
            for w in range(W):
                memcpy(&a_stack[0], &starts_mv[0], num_corners * sizeof(f64))
                memcpy(&b_stack[0], &ends_mv[0], num_corners * sizeof(f64))
                
                curr_size = num_corners
                
                for i in range(num_dims):
                    u = coeffs_mv[h, w, i]
                    mode = modes_mv[i]
                    
                    for j in range(curr_size):
                        h0 = a_stack[j]
                        h1 = b_stack[j]
                        h1_adj = adjust_end_for_mode(h0, h1, mode)
                        a_stack[j] = wrap_hue(h0 + u * (h1_adj - h0))
                    
                    if curr_size > 1:
                        half = curr_size >> 1
                        memcpy(&b_stack[0], &a_stack[half], half * sizeof(f64))
                        curr_size = half
                
                out_mv[h, w] = a_stack[0]
        
        return out
    
    # Heap fallback
    a = <f64*>malloc(num_corners * sizeof(f64))
    b = <f64*>malloc(num_corners * sizeof(f64))
    if a == NULL or b == NULL:
        if a != NULL: free(a)
        if b != NULL: free(b)
        raise MemoryError("Failed to allocate working buffers")
    
    try:
        for h in range(H):
            for w in range(W):
                memcpy(a, &starts_mv[0], num_corners * sizeof(f64))
                memcpy(b, &ends_mv[0], num_corners * sizeof(f64))
                
                curr_size = num_corners
                
                for i in range(num_dims):
                    u = coeffs_mv[h, w, i]
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
                
                out_mv[h, w] = a[0]
    finally:
        free(a)
        free(b)
    
    return out


cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_x = HUE_SHORTEST,
    int mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CLAMP,
    f64 border_constant = 0.0,
    f64 border_feathering = 0.0,
    int num_threads = -1,
    int distance_mode = ALPHA_MAX,
):
    """
    Interpolate hue between two 1D hue lines with 2D coordinate mapping.
    Adds feathering, distance mode, and threading support.
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    
    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, ::1] out_mv = out
    
    if num_threads < 0:
        import os
        num_threads = os.cpu_count() or 4
    elif num_threads == 0:
        num_threads = 1
    
    _lerp_1ch_hue_feathered_kernel(
        l0, l1, c, out_mv,
        border_constant, border_feathering,
        H, W, L, border_mode, num_threads,
        mode_x, mode_y, distance_mode,
    )
    return out


cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_x_discrete(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CLAMP,
    f64 border_constant = 0.0,
    f64 border_feathering = 0.0,
    int num_threads = -1,
    int distance_mode = ALPHA_MAX,
):
    """
    Interpolate hue between two 1D hue lines with nearest-neighbor x-sampling.
    Adds feathering, distance mode, and threading support.
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    
    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, ::1] out_mv = out
    
    if num_threads < 0:
        import os
        num_threads = os.cpu_count() or 4
    elif num_threads == 0:
        num_threads = 1
    
    _lerp_1ch_hue_discrete_feathered_kernel(
        l0, l1, c, out_mv,
        border_constant, border_feathering,
        H, W, L, border_mode, num_threads,
        mode_y, distance_mode,
    )
    return out


# =============================================================================
# 2D Grid Hue Lerp with per-pixel mode
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_with_modes(
    np.ndarray[f64, ndim=2] h0_grid,
    np.ndarray[f64, ndim=2] h1_grid,
    np.ndarray[f64, ndim=2] coeffs,
    np.ndarray[i32, ndim=2] modes,
):
    """
    2D grid interpolation with per-pixel start, end, coefficient, and mode.
    
    Each pixel in the output is interpolated independently using its own
    start hue, end hue, interpolation coefficient, and interpolation mode.
    
    Args:
        h0_grid: Grid of start hue values, shape (H, W)
        h1_grid: Grid of end hue values, shape (H, W)
                 Must match shape of h0_grid
        coeffs: Grid of interpolation coefficients, shape (H, W)
                Each coefficient should typically be in [0, 1]
        modes: Grid of interpolation modes, shape (H, W)
               Each element must be one of:
               0=CW, 1=CCW, 2=SHORTEST, 3=LONGEST
    
    Returns:
        np.ndarray: Interpolated hue grid, shape (H, W)
                    Values are wrapped to [0, 360) range.
    
    Notes:
        - Useful for spatial varying interpolation parameters
        - Each pixel's interpolation is independent
    """
    cdef Py_ssize_t H = h0_grid.shape[0]
    cdef Py_ssize_t W = h0_grid.shape[1]
    
    if h1_grid.shape[0] != H or h1_grid.shape[1] != W:
        raise ValueError("h1_grid shape mismatch")
    if coeffs.shape[0] != H or coeffs.shape[1] != W:
        raise ValueError("coeffs shape mismatch")
    if modes.shape[0] != H or modes.shape[1] != W:
        raise ValueError("modes shape mismatch")
    
    if not h0_grid.flags['C_CONTIGUOUS']:
        h0_grid = np.ascontiguousarray(h0_grid)
    if not h1_grid.flags['C_CONTIGUOUS']:
        h1_grid = np.ascontiguousarray(h1_grid)
    if not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs)
    if not modes.flags['C_CONTIGUOUS']:
        modes = np.ascontiguousarray(modes)
    
    cdef f64[:, ::1] h0_mv = h0_grid
    cdef f64[:, ::1] h1_mv = h1_grid
    cdef f64[:, ::1] c_mv = coeffs
    cdef i32[:, ::1] m_mv = modes
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t h, w
    cdef f64 h0, h1, u, h1_adj
    cdef int mode
    
    for h in range(H):
        for w in range(W):
            h0 = h0_mv[h, w]
            h1 = h1_mv[h, w]
            u = c_mv[h, w]
            mode = m_mv[h, w]
            
            h1_adj = adjust_end_for_mode(h0, h1, mode)
            out_mv[h, w] = wrap_hue(h0 + u * (h1_adj - h0))
    
    return out