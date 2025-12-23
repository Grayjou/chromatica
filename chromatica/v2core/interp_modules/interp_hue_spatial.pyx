# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from .hue_common cimport (
    f64, i32, HUE_CW, HUE_CCW, HUE_SHORTEST, HUE_LONGEST,
    wrap_hue, adjust_end_for_mode, lerp_hue_single
)

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
    Multi-dimensional hue interpolation over 1D spatial grid.
    
    Args:
        starts: Corner start values, shape (2^{N-1},)
        ends: Corner end values, shape (2^{N-1},)
        coeffs: Interpolation coefficients, shape (L, N)
        modes: Interpolation mode per dimension, shape (N,)
               0=CW, 1=CCW, 2=SHORTEST, 3=LONGEST
    
    Returns:
        Interpolated hues, shape (L,), values in [0, 360)
    """
    cdef Py_ssize_t num_points = coeffs.shape[0]
    cdef Py_ssize_t num_dims = coeffs.shape[1]
    cdef Py_ssize_t num_corners = starts.shape[0]
    
    if ends.shape[0] != num_corners:
        raise ValueError("starts and ends must have same length")
    if modes.shape[0] != num_dims:
        raise ValueError("modes must have length equal to num dimensions")
    
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
# 2D Hue Interpolation: coeffs (H, W, N), modes (N,) -> output (H, W)
# =============================================================================
def hue_lerp_2d_spatial(
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=3] coeffs,
    np.ndarray[i32, ndim=1] modes,
):
    """
    Multi-dimensional hue interpolation over 2D spatial grid.
    
    Args:
        starts: Corner start values, shape (2^{N-1},)
        ends: Corner end values, shape (2^{N-1},)
        coeffs: Interpolation coefficients, shape (H, W, N)
        modes: Interpolation mode per dimension, shape (N,)
    
    Returns:
        Interpolated hues, shape (H, W), values in [0, 360)
    """
    cdef Py_ssize_t H = coeffs.shape[0]
    cdef Py_ssize_t W = coeffs.shape[1]
    cdef Py_ssize_t num_dims = coeffs.shape[2]
    cdef Py_ssize_t num_corners = starts.shape[0]
    
    if ends.shape[0] != num_corners:
        raise ValueError("starts and ends must have same length")
    if modes.shape[0] != num_dims:
        raise ValueError("modes must have length equal to num dimensions")
    
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
        raise MemoryError()
    
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
