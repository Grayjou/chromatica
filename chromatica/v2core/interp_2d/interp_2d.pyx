# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport floor, fmod, fabs
from libc.stdlib cimport malloc, free
from ..border_handling cimport (
    handle_border_edges_2d,
    handle_border_lines_2d,
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)
ctypedef np.float64_t f64

from .helpers cimport (
    handle_border_1d,
    is_out_of_bounds_1d,
    is_out_of_bounds_2d,
    prepare_border_constant_array,
    is_out_of_bounds_3d,

)
# =============================================================================
# Core: Interpolate between two 1D lines at arbitrary (u_x, u_y) coordinates
# Single channel version
# =============================================================================

def lerp_between_lines_1ch(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
):
    """
    Interpolate between two 1D lines using a 2D grid of (u_x, u_y) coordinates.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x (position along lines, 0-1)
                coords[h, w, 1] = u_y (blend factor between lines, 0-1)
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value to use for BORDER_CONSTANT mode (default: 0.0)
    
    Returns:
        Interpolated values, shape (H, W)
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
    cdef f64 v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    out_mv[h, w] = border_constant
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)
            
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
            
            v0 = l0[idx_lo] + frac * (l0[idx_hi] - l0[idx_lo])
            v1 = l1[idx_lo] + frac * (l1[idx_hi] - l1[idx_lo])
            out_mv[h, w] = v0 + new_u_y * (v1 - v0)
    
    return out


# =============================================================================
# Multi-channel version: lines have shape (L, C)
# =============================================================================
def lerp_between_lines_multichannel(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Interpolate between two multi-channel 1D lines.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value to use for BORDER_CONSTANT mode.
                        Can be scalar, array of shape (C,), or None (zeros).
                        Examples: 0.0, [1.0, 0.0, 0.0], (255, 0, 255)
    
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
    
    cdef np.ndarray[f64, ndim=1] border_const_arr = prepare_border_constant_array(border_constant, C)
    
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] border_const_mv = border_const_arr
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    cdef Py_ssize_t h, w, ch
    cdef Py_ssize_t idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac
    cdef f64 v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    for ch in range(C):
                        out_mv[h, w, ch] = border_const_mv[ch]
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)
            
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
            
            for ch in range(C):
                v0 = l0[idx_lo, ch] + frac * (l0[idx_hi, ch] - l0[idx_lo, ch])
                v1 = l1[idx_lo, ch] + frac * (l1[idx_hi, ch] - l1[idx_lo, ch])
                out_mv[h, w, ch] = v0 + new_u_y * (v1 - v0)
    
    return out


# =============================================================================
# Flat coordinates version: coords shape (N, 2) -> output (N,) or (N, C)
# =============================================================================
def lerp_between_lines_flat_1ch(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
):
    """
    Interpolate at arbitrary (u_x, u_y) points (flat list).
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, ::1] c = coords

    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out

    cdef Py_ssize_t n, idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac, v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y

    for n in range(N):
        u_x = c[n, 0]
        u_y = c[n, 1]

        if border_mode == BORDER_CONSTANT:
            if is_out_of_bounds_2d(u_x, u_y):
                out_mv[n] = border_constant
                continue
            new_u_x = u_x
            new_u_y = u_y
        elif border_mode == BORDER_OVERFLOW:
            new_u_x = u_x
            new_u_y = u_y
        else:
            new_u_x = handle_border_1d(u_x, border_mode)
            new_u_y = handle_border_1d(u_y, border_mode)

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

        v0 = l0[idx_lo] + frac * (l0[idx_hi] - l0[idx_lo])
        v1 = l1[idx_lo] + frac * (l1[idx_hi] - l1[idx_lo])
        out_mv[n] = v0 + new_u_y * (v1 - v0)

    return out


def lerp_between_lines_flat_multichannel(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Interpolate at arbitrary (u_x, u_y) points (flat list), multi-channel.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate pairs, shape (N, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value for BORDER_CONSTANT mode.
                        Can be scalar, array of shape (C,), or None.
                        Examples: 0.0, [1.0, 0.0, 0.0], (255, 0, 255)

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

    cdef np.ndarray[f64, ndim=1] border_const_arr = prepare_border_constant_array(border_constant, C)

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[::1] border_const_mv = border_const_arr

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    cdef Py_ssize_t n, ch, idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac, v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y

    for n in range(N):
        u_x = c[n, 0]
        u_y = c[n, 1]

        if border_mode == BORDER_CONSTANT:
            if is_out_of_bounds_2d(u_x, u_y):
                for ch in range(C):
                    out_mv[n, ch] = border_const_mv[ch]
                continue
            new_u_x = u_x
            new_u_y = u_y
        elif border_mode == BORDER_OVERFLOW:
            new_u_x = u_x
            new_u_y = u_y
        else:
            new_u_x = handle_border_1d(u_x, border_mode)
            new_u_y = handle_border_1d(u_y, border_mode)

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

        for ch in range(C):
            v0 = l0[idx_lo, ch] + frac * (l0[idx_hi, ch] - l0[idx_lo, ch])
            v1 = l1[idx_lo, ch] + frac * (l1[idx_hi, ch] - l1[idx_lo, ch])
            out_mv[n, ch] = v0 + new_u_y * (v1 - v0)

    return out


# =============================================================================
# Discrete x-sampling versions (nearest neighbor in x-direction)
# =============================================================================
def lerp_between_lines_x_discrete_1ch(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
):
    """
    Interpolate between two 1D lines with discrete x-sampling (nearest index).
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
    cdef f64 new_u_x, new_u_y
    cdef f64 v0, v1
    
    # Handle edge case when L == 1
    if L == 1:
        v0 = l0[0]
        v1 = l1[0]
        for h in range(H):
            for w in range(W):
                u_y = c[h, w, 1]
                
                if border_mode == BORDER_CONSTANT:
                    if is_out_of_bounds_1d(u_y):
                        out_mv[h, w] = border_constant
                        continue
                    new_u_y = u_y
                elif border_mode == BORDER_OVERFLOW:
                    new_u_y = u_y
                else:
                    new_u_y = handle_border_1d(u_y, border_mode)
                
                out_mv[h, w] = v0 + new_u_y * (v1 - v0)
        return out
    
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    out_mv[h, w] = border_constant
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)
            
            idx_f = new_u_x * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)
            
            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1
            
            v0 = l0[idx]
            v1 = l1[idx]
            out_mv[h, w] = v0 + new_u_y * (v1 - v0)
    
    return out

def lerp_between_lines_x_discrete_multichannel(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Interpolate between two multi-channel 1D lines with discrete x-sampling.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value for BORDER_CONSTANT mode.
                        Can be scalar, array of shape (C,), or None.
    
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
    
    cdef np.ndarray[f64, ndim=1] border_const_arr = prepare_border_constant_array(border_constant, C)
    
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] border_const_mv = border_const_arr
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    cdef Py_ssize_t h, w, ch
    cdef Py_ssize_t idx
    cdef f64 u_x, u_y, idx_f
    cdef f64 new_u_x, new_u_y
    
    # Handle edge case when L == 1
    if L == 1:
        for h in range(H):
            for w in range(W):
                u_y = c[h, w, 1]
                
                if border_mode == BORDER_CONSTANT:
                    if is_out_of_bounds_1d(u_y):
                        for ch in range(C):
                            out_mv[h, w, ch] = border_const_mv[ch]
                        continue
                    new_u_y = u_y
                elif border_mode == BORDER_OVERFLOW:
                    new_u_y = u_y
                else:
                    new_u_y = handle_border_1d(u_y, border_mode)
                
                for ch in range(C):
                    out_mv[h, w, ch] = l0[0, ch] + new_u_y * (l1[0, ch] - l0[0, ch])
        return out
    
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    for ch in range(C):
                        out_mv[h, w, ch] = border_const_mv[ch]
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)
            
            idx_f = new_u_x * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)
            
            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1
            
            for ch in range(C):
                out_mv[h, w, ch] = l0[idx, ch] + new_u_y * (l1[idx, ch] - l0[idx, ch])
    
    return out


# =============================================================================
# Dispatcher: lerp_between_lines
# =============================================================================
def lerp_between_lines(
    np.ndarray line0,
    np.ndarray line1,
    np.ndarray coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Interpolate between two lines at arbitrary (u_x, u_y) coordinates.
    
    Args:
        line0: First line, shape (L,) or (L, C)
        line1: Second line, shape (L,) or (L, C)
        coords: Coordinates, shape (H, W, 2) or (N, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value for BORDER_CONSTANT mode.
                        For single channel: scalar (default: 0.0)
                        For multi-channel: scalar or array of shape (C,)
                        Examples: 0.0, [1.0, 0.0, 0.0], (255, 0, 255)
    
    Returns:
        Interpolated values
    """
    if line0.dtype != np.float64 or not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0, dtype=np.float64)
    if line1.dtype != np.float64 or not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1, dtype=np.float64)
    if coords.dtype != np.float64 or not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    line_ndim = line0.ndim
    coords_ndim = coords.ndim
    
    # Handle default border_constant
    if border_constant is None:
        if line_ndim == 1:
            border_constant = 0.0
        # For multi-channel, leave as None and let prepare_border_constant_array handle it
    
    # Grid coords (H, W, 2)
    if coords_ndim == 3 and coords.shape[2] == 2:
        if line_ndim == 1:
            return lerp_between_lines_1ch(line0, line1, coords, border_mode, <f64>border_constant)
        elif line_ndim == 2:
            return lerp_between_lines_multichannel(line0, line1, coords, border_mode, border_constant)
    
    # Flat coords (N, 2)
    elif coords_ndim == 2 and coords.shape[1] == 2:
        if line_ndim == 1:
            return lerp_between_lines_flat_1ch(line0, line1, coords, border_mode, <f64>border_constant)
        elif line_ndim == 2:
            return lerp_between_lines_flat_multichannel(line0, line1, coords, border_mode, border_constant)
    
    raise ValueError(f"Unsupported shapes for lerp_between_lines: line0.ndim={line_ndim}, coords.ndim={coords_ndim}")


# =============================================================================
# Dispatcher: lerp_between_lines_x_discrete  
# =============================================================================
def lerp_between_lines_x_discrete(
    np.ndarray line0,
    np.ndarray line1,
    np.ndarray coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Interpolate between two lines with discrete x-sampling (nearest neighbor).
    
    Args:
        line0: First line, shape (L,) or (L, C)
        line1: Second line, shape (L,) or (L, C)
        coords: Coordinate grid, shape (H, W, 2)
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value for BORDER_CONSTANT mode.
                        For single channel: scalar (default: 0.0)
                        For multi-channel: scalar or array of shape (C,)
                        Examples: 0.0, [1.0, 0.0, 0.0], (255, 0, 255)
    
    Returns:
        Interpolated values, shape (H, W) or (H, W, C)
    """
    if line0.dtype != np.float64 or not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0, dtype=np.float64)
    if line1.dtype != np.float64 or not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1, dtype=np.float64)
    if coords.dtype != np.float64 or not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    if coords.ndim != 3 or coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    
    line_ndim = line0.ndim
    
    if line_ndim == 1:
        if border_constant is None:
            border_constant = 0.0
        return lerp_between_lines_x_discrete_1ch(line0, line1, coords, border_mode, <f64>border_constant)
    elif line_ndim == 2:
        return lerp_between_lines_x_discrete_multichannel(line0, line1, coords, border_mode, border_constant)
    
    raise ValueError(f"Unsupported line dimensions: {line_ndim}")