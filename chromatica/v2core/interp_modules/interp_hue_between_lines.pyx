# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport floor
from .hue_common cimport (
    HUE_CW, HUE_CCW, HUE_SHORTEST, HUE_LONGEST,
    wrap_hue, adjust_end_for_mode, lerp_hue_single
)

ctypedef np.float64_t f64
ctypedef np.int32_t i32

# =============================================================================
# 2D Hue interpolation with per-pixel modes
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
    cdef f64 v0, v1, h_adj
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
            
            # Sample line0 at u_x with hue mode
            h0_lo = l0[idx_lo]
            h0_hi = l0[idx_hi]
            v0 = lerp_hue_single(h0_lo, h0_hi, frac, mode_x)
            
            # Sample line1 at u_x with hue mode
            h1_lo = l1[idx_lo]
            h1_hi = l1[idx_hi]
            v1 = lerp_hue_single(h1_lo, h1_hi, frac, mode_x)
            
            # Blend between lines with hue mode
            out_mv[h, w] = lerp_hue_single(v0, v1, u_y, mode_y)
    
    return out


# =============================================================================
# Hue interpolation between lines with discrete x-sampling
# =============================================================================
def hue_lerp_between_lines_x_discrete(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_y = HUE_SHORTEST,
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
    
    Returns:
        Interpolated hues, shape (H, W), values in [0, 360)
    
    Note:
        For efficiency when L = W, use this function instead of hue_lerp_between_lines.
        The x-coordinate is mapped to the nearest index by rounding.
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
    
    # Handle edge case when L == 1
    if L == 1:
        # Only one index available, so v0 and v1 are simply the first elements
        v0 = l0[0]
        v1 = l1[0]
        for h in range(H):
            for w in range(W):
                u_y = c[h, w, 1]
                # Clamp u_y to [0, 1] for safety
                if u_y < 0.0:
                    u_y = 0.0
                elif u_y > 1.0:
                    u_y = 1.0
                # Blend between the single values from each line
                out_mv[h, w] = lerp_hue_single(v0, v1, u_y, mode_y)
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
            
            # Sample lines at discrete index
            v0 = l0[idx]
            v1 = l1[idx]
            
            # Blend between lines with hue mode in y-direction
            out_mv[h, w] = lerp_hue_single(v0, v1, u_y, mode_y)
    
    return out
