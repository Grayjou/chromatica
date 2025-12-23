# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Hue interpolation with cyclical color space support.

Supports four interpolation modes per dimension:
- CW (0): Clockwise
- CCW (1): Counterclockwise  
- SHORTEST (2): Shortest path (≤180°)
- LONGEST (3): Longest path (≥180°)
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
# Inline Helpers
# =============================================================================
cdef inline f64 wrap_hue(f64 h) noexcept nogil:
    """Wrap hue to [0, 360)."""
    h = fmod(h, 360.0)
    if h < 0.0:
        h += 360.0
    return h


cdef inline f64 adjust_end_for_mode(f64 h0, f64 h1, int mode) noexcept nogil:
    """
    Adjust h1 relative to h0 based on interpolation mode.
    Returns adjusted h1 (may be outside [0, 360)).
    """
    cdef f64 d = h1 - h0
    
    if mode == HUE_CW:
        # Clockwise: ensure h1 > h0
        if h0 >= h1:
            return h1 + 360.0
        return h1
        
    elif mode == HUE_CCW:
        # Counterclockwise: ensure h1 < h0
        if h0 <= h1:
            return h1 - 360.0
        return h1
        
    elif mode == HUE_SHORTEST:
        # Shortest path: |d| <= 180
        if d > 180.0:
            return h1 - 360.0
        elif d < -180.0:
            return h1 + 360.0
        return h1
        
    elif mode == HUE_LONGEST:
        # Longest path: |d| >= 180
        if d >= 0.0 and d < 180.0:
            return h1 - 360.0
        elif d < 0.0 and d > -180.0:
            return h1 + 360.0
        return h1
    
    # Default: no adjustment
    return h1


cdef inline f64 lerp_hue_single(f64 h0, f64 h1, f64 u, int mode) noexcept nogil:
    """Lerp between two hues with mode, returning wrapped result."""
    cdef f64 h1_adj = adjust_end_for_mode(h0, h1, mode)
    cdef f64 result = h0 + u * (h1_adj - h0)
    return wrap_hue(result)


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
    if num_corners != (1 << (num_dims - 1)) * 2:
        # Actually num_corners should be 2^{n-1} for each of starts and ends
        # but they're passed as the two halves... let me reconsider
        pass
    
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


# =============================================================================
# 2D Grid Hue Lerp with per-pixel mode
# =============================================================================
def hue_lerp_2d_with_modes(
    np.ndarray[f64, ndim=2] h0_grid,
    np.ndarray[f64, ndim=2] h1_grid,
    np.ndarray[f64, ndim=2] coeffs,
    np.ndarray[i32, ndim=2] modes,
):
    """
    2D hue interpolation with per-pixel start, end, coeff, and mode.
    
    Args:
        h0_grid: Start hues, shape (H, W)
        h1_grid: End hues, shape (H, W)
        coeffs: Interpolation coefficients, shape (H, W)
        modes: Per-pixel modes, shape (H, W)
    
    Returns:
        Interpolated hues, shape (H, W)
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


# =============================================================================
# Hue interpolation between lines (Section 6 style)
# =============================================================================

def hue_lerp_between_lines(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_x = HUE_SHORTEST,
    int mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CLAMP,
    f64 border_constant = 0.0,
):
    """
    Interpolate hue between two 1D lines with modes for each axis.
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[..., 0] = u_x (position along lines)
                coords[..., 1] = u_y (blend between lines)
        mode_x: Interpolation mode for sampling within lines
        mode_y: Interpolation mode for blending between lines
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Hue value for BORDER_CONSTANT mode (default: 0.0)
    
    Returns:
        Interpolated hues, shape (H, W), values in [0, 360)
    """
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
    
    # Border handling variables
    cdef tuple border_result
    cdef f64 new_u_x, new_u_y
    
    # Wrap border_constant to valid hue range
    cdef f64 wrapped_border = wrap_hue(border_constant)
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            # Handle border conditions
            if border_mode == BORDER_CONSTANT:
                # Check if coordinates are out of bounds
                if u_x < 0.0 or u_x > 1.0 or u_y < 0.0 or u_y > 1.0:
                    out_mv[h, w] = wrapped_border
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                # No border handling - just use the coordinates
                new_u_x = u_x
                new_u_y = u_y
            else:
                # Use border handling function for other modes (CLAMP, REPEAT, MIRROR)
                border_result = handle_border_lines_2d(u_x, u_y, border_mode)
                if border_result is None:
                    out_mv[h, w] = wrapped_border
                    continue
                new_u_x, new_u_y = border_result
            
            # Map new_u_x to line index
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
            
            # Sample line0 at new_u_x with hue mode
            h0_lo = l0[idx_lo]
            h0_hi = l0[idx_hi]
            v0 = lerp_hue_single(h0_lo, h0_hi, frac, mode_x)
            
            # Sample line1 at new_u_x with hue mode
            h1_lo = l1[idx_lo]
            h1_hi = l1[idx_hi]
            v1 = lerp_hue_single(h1_lo, h1_hi, frac, mode_x)
            
            # Blend between lines with hue mode
            out_mv[h, w] = lerp_hue_single(v0, v1, new_u_y, mode_y)
    
    return out

# =============================================================================
# Hue interpolation between lines with discrete x-sampling (Section 6 style)
# =============================================================================
def hue_lerp_between_lines_x_discrete(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CLAMP,
    f64 border_constant = 0.0,
):
    """
    Interpolate hue between two 1D lines with discrete x-sampling (nearest index).
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[..., 0] = u_x (position along lines, maps to nearest index)
                coords[..., 1] = u_y (blend between lines)
        mode_y: Interpolation mode for blending between lines
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Hue value for BORDER_CONSTANT mode (default: 0.0)
    
    Returns:
        Interpolated hues, shape (H, W), values in [0, 360)
    """
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
    
    # Border handling variables
    cdef tuple border_result
    cdef f64 new_u_x, new_u_y
    
    # Wrap border_constant to valid hue range
    cdef f64 wrapped_border = wrap_hue(border_constant)
    
    # Handle edge case when L == 1
    if L == 1:
        v0 = l0[0]
        v1 = l1[0]
        for h in range(H):
            for w in range(W):
                u_x = c[h, w, 0]
                u_y = c[h, w, 1]
                
                # Handle border conditions
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
                
                # Clamp u_y for safety (only if not using OVERFLOW)
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
            
            # Handle border conditions
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
            
            # Map new_u_x to nearest index by rounding
            idx_f = new_u_x * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)
            
            # Clamp index to valid range [0, L-1]
            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1
            
            # Sample lines at discrete index
            v0 = l0[idx]
            v1 = l1[idx]
            
            # Blend between lines with hue mode
            out_mv[h, w] = lerp_hue_single(v0, v1, new_u_y, mode_y)
    
    return out
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
        raise NotImplementedError(f"Hue lerp for {spatial_ndims}D not implemented")