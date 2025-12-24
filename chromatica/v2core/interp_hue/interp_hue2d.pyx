# interp_hue2d.pyx
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
 
# ADD f64, i32 to this import!
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
)

# =============================================================================
# 2D Hue Interpolation: coeffs (H, W, N), modes (N,) -> output (H, W)
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_spatial(   # <-- ADD RETURN TYPE
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=3] coeffs,
    np.ndarray[i32, ndim=1] modes,
):
    """
    Multi-dimensional hue interpolation over a 2D spatial grid.
    ... (rest of docstring)
    """
    # ... rest of function body unchanged ...
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


# =============================================================================
# Hue interpolation between lines (Section 6 style)
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines(   # <-- ADD RETURN TYPE
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_x = HUE_SHORTEST,
    int mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CLAMP,
    f64 border_constant = 0.0,
):
    """
    Interpolate hue between two 1D hue lines with 2D coordinate mapping.
    ... (rest of docstring)
    """
    # ... rest of function body unchanged ...
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
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
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t h, w
    cdef Py_ssize_t idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac
    cdef f64 h0_lo, h0_hi, h1_lo, h1_hi
    cdef f64 v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    cdef tuple border_result
    cdef f64 new_u_x, new_u_y
    cdef f64 wrapped_border = wrap_hue(border_constant)
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            if border_mode == BORDER_CONSTANT:
                if u_x < 0.0 or u_x > 1.0 or u_y < 0.0 or u_y > 1.0:
                    out_mv[h, w] = wrapped_border
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                border_result = handle_border_lines_2d(u_x, u_y, border_mode)
                if border_result is None:
                    out_mv[h, w] = wrapped_border
                    continue
                new_u_x, new_u_y = border_result
            
            idx_f = new_u_x * L_minus_1
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
            
            h0_lo = l0[idx_lo]
            h0_hi = l0[idx_hi]
            v0 = lerp_hue_single(h0_lo, h0_hi, frac, mode_x)
            
            h1_lo = l1[idx_lo]
            h1_hi = l1[idx_hi]
            v1 = lerp_hue_single(h1_lo, h1_hi, frac, mode_x)
            
            out_mv[h, w] = lerp_hue_single(v0, v1, new_u_y, mode_y)
    
    return out


# =============================================================================
# Hue interpolation between lines with discrete x-sampling (Section 6 style)
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_x_discrete(   # <-- ADD RETURN TYPE
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CLAMP,
    f64 border_constant = 0.0,
):
    """
    Interpolate hue between two 1D hue lines with nearest-neighbor x-sampling.
    ... (rest of docstring)
    """
    # ... rest of function body unchanged ...
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
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
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t h, w
    cdef Py_ssize_t idx
    cdef f64 u_x, u_y, idx_f
    cdef f64 v0, v1
    
    cdef tuple border_result
    cdef f64 new_u_x, new_u_y
    cdef f64 wrapped_border = wrap_hue(border_constant)
    
    if L == 1:
        v0 = l0[0]
        v1 = l1[0]
        for h in range(H):
            for w in range(W):
                u_x = c[h, w, 0]
                u_y = c[h, w, 1]
                
                if border_mode == BORDER_CONSTANT:
                    if u_x < 0.0 or u_x > 1.0 or u_y < 0.0 or u_y > 1.0:
                        out_mv[h, w] = wrapped_border
                        continue
                    new_u_y = u_y
                elif border_mode == BORDER_OVERFLOW:
                    new_u_y = u_y
                else:
                    border_result = handle_border_lines_2d(u_x, u_y, border_mode)
                    if border_result is None:
                        out_mv[h, w] = wrapped_border
                        continue
                    new_u_y = border_result[1]
                
                if border_mode != BORDER_OVERFLOW:
                    if new_u_y < 0.0:
                        new_u_y = 0.0
                    elif new_u_y > 1.0:
                        new_u_y = 1.0
                
                out_mv[h, w] = lerp_hue_single(v0, v1, new_u_y, mode_y)
        return out
    
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            if border_mode == BORDER_CONSTANT:
                if u_x < 0.0 or u_x > 1.0 or u_y < 0.0 or u_y > 1.0:
                    out_mv[h, w] = wrapped_border
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                border_result = handle_border_lines_2d(u_x, u_y, border_mode)
                if border_result is None:
                    out_mv[h, w] = wrapped_border
                    continue
                new_u_x, new_u_y = border_result
            
            idx_f = new_u_x * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)
            
            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1
            
            v0 = l0[idx]
            v1 = l1[idx]
            
            out_mv[h, w] = lerp_hue_single(v0, v1, new_u_y, mode_y)
    
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