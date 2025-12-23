# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from .hue_common cimport (
    f64, i32, HUE_CW, HUE_CCW, HUE_SHORTEST, HUE_LONGEST,
    wrap_hue, adjust_end_for_mode, lerp_hue_single
)

# =============================================================================
# Simple 1D Hue Interpolation
# =============================================================================
def hue_lerp_simple(
    f64 h0,
    f64 h1,
    np.ndarray[f64, ndim=1] coeffs,
    int mode = HUE_SHORTEST,
):
    """
    Simple 1D hue interpolation between two values.
    
    Args:
        h0: Start hue (degrees)
        h1: End hue (degrees)
        coeffs: Interpolation coefficients, shape (N,)
        mode: Interpolation mode (0=CW, 1=CCW, 2=SHORTEST, 3=LONGEST)
    
    Returns:
        Interpolated hues, shape (N,)
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
    Vectorized 1D hue interpolation for arrays of hue pairs.
    
    Args:
        h0_arr: Start hues, shape (M,)
        h1_arr: End hues, shape (M,)
        coeffs: Interpolation coefficients, shape (N,)
        mode: Interpolation mode
    
    Returns:
        Interpolated hues, shape (N, M)
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
