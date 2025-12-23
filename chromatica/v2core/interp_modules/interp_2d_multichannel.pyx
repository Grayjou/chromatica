# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport floor

ctypedef np.float64_t f64

# =============================================================================
# Multi-channel interpolation between two 1D lines
# =============================================================================
def lerp_between_lines_multichannel(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
):
    """
    Interpolate between two multi-channel 1D lines.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    
    # Ensure contiguous
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    cdef Py_ssize_t h, w, ch
    cdef Py_ssize_t idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac
    cdef f64 v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            # Map u_x to line index
            idx_f = u_x * L_minus_1
            idx_lo = <Py_ssize_t>floor(idx_f)
            
            if idx_lo < 0:
                idx_lo = 0
            if idx_lo >= L - 1:
                idx_lo = L - 2
            
            idx_hi = idx_lo + 1
            frac = idx_f - <f64>idx_lo
            
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            
            # Process each channel
            for ch in range(C):
                v0 = l0[idx_lo, ch] + frac * (l0[idx_hi, ch] - l0[idx_lo, ch])
                v1 = l1[idx_lo, ch] + frac * (l1[idx_hi, ch] - l1[idx_lo, ch])
                out_mv[h, w, ch] = v0 + u_y * (v1 - v0)
    
    return out


# =============================================================================
# Multi-channel interpolation with flat coordinates
# =============================================================================
def lerp_between_lines_flat_multichannel(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=2] coords,
):
    """
    Interpolate at arbitrary (u_x, u_y) points (flat list), multi-channel.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate pairs, shape (N, 2)
    
    Returns:
        Interpolated values, shape (N, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t N = coords.shape[0]
    
    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, ::1] c = coords
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t n, ch, idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac, v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    for n in range(N):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        idx_f = u_x * L_minus_1
        idx_lo = <Py_ssize_t>floor(idx_f)
        
        if idx_lo < 0:
            idx_lo = 0
        if idx_lo >= L - 1:
            idx_lo = L - 2
        
        idx_hi = idx_lo + 1
        frac = idx_f - <f64>idx_lo
        
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0
        
        for ch in range(C):
            v0 = l0[idx_lo, ch] + frac * (l0[idx_hi, ch] - l0[idx_lo, ch])
            v1 = l1[idx_lo, ch] + frac * (l1[idx_hi, ch] - l1[idx_lo, ch])
            out_mv[n, ch] = v0 + u_y * (v1 - v0)
    
    return out


# =============================================================================
# Multi-channel discrete x interpolation
# =============================================================================
def lerp_between_lines_x_discrete_multichannel(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
):
    """
    Interpolate between two multi-channel 1D lines with discrete x-sampling.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
                coords[..., 0] = u_x (position along lines, maps to nearest index)
                coords[..., 1] = u_y (blend between lines)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    cdef Py_ssize_t h, w, ch
    cdef Py_ssize_t idx
    cdef f64 u_x, u_y, idx_f
    
    # Handle edge case when L == 1
    if L == 1:
        for h in range(H):
            for w in range(W):
                u_y = c[h, w, 1]
                if u_y < 0.0:
                    u_y = 0.0
                elif u_y > 1.0:
                    u_y = 1.0
                for ch in range(C):
                    out_mv[h, w, ch] = l0[0, ch] + u_y * (l1[0, ch] - l0[0, ch])
        return out
    
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            # Map u_x to nearest index by rounding
            idx_f = u_x * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)  # Round to nearest
            
            # Clamp index to valid range [0, L-1]
            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1
            
            # Clamp u_y to [0, 1] for safety
            if u_y < 0.0:
                u_y = 0.0
            elif u_y > 1.0:
                u_y = 1.0
            
            # Blend between lines at discrete index for each channel
            for ch in range(C):
                out_mv[h, w, ch] = l0[idx, ch] + u_y * (l1[idx, ch] - l0[idx, ch])
    
    return out
