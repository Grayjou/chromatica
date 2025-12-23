# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport floor, fmod, fabs
from libc.stdlib cimport malloc, free
from .border_handling cimport (
    handle_border_edges_2d,
    handle_border_lines_2d,
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)
ctypedef np.float64_t f64


# =============================================================================
# Helper: Inline border handling for a single coordinate
# =============================================================================
cdef inline f64 handle_border_1d(f64 u, int border_mode) noexcept nogil:
    """Handle border for a single coordinate in [0, 1] range."""
    cdef f64 result
    
    if border_mode == BORDER_CLAMP:
        if u < 0.0:
            return 0.0
        elif u > 1.0:
            return 1.0
        return u
    elif border_mode == BORDER_REPEAT:
        result = fmod(u, 1.0)
        if result < 0.0:
            result += 1.0
        return result
    elif border_mode == BORDER_MIRROR:
        result = fmod(fabs(u), 2.0)
        if result > 1.0:
            result = 2.0 - result
        return result
    else:
        return u


cdef inline bint is_out_of_bounds_1d(f64 u) noexcept nogil:
    """Check if coordinate is out of [0, 1] bounds."""
    return u < 0.0 or u > 1.0


cdef inline bint is_out_of_bounds_2d(f64 u_x, f64 u_y) noexcept nogil:
    """Check if 2D coordinates are out of [0, 1] bounds."""
    return u_x < 0.0 or u_x > 1.0 or u_y < 0.0 or u_y > 1.0


cdef inline bint is_out_of_bounds_3d(f64 u_x, f64 u_y, f64 u_z) noexcept nogil:
    """Check if 3D coordinates are out of [0, 1] bounds."""
    return u_x < 0.0 or u_x > 1.0 or u_y < 0.0 or u_y > 1.0 or u_z < 0.0 or u_z > 1.0


# =============================================================================
# Helper: Prepare border constants for multi-channel data
# =============================================================================
cdef np.ndarray[f64, ndim=1] prepare_border_constant_array(object border_constant, Py_ssize_t C):
    """
    Prepare border constants array for multi-channel data.
    
    Args:
        border_constant: Can be:
            - None: returns zeros array of shape (C,)
            - scalar (int/float): returns array filled with that value
            - array-like of shape (C,): returns that array
        C: Number of channels
    
    Returns:
        Contiguous float64 array of shape (C,)
    """
    cdef np.ndarray[f64, ndim=1] result
    
    if border_constant is None:
        result = np.zeros(C, dtype=np.float64)
    elif isinstance(border_constant, (int, float)):
        result = np.full(C, <f64>border_constant, dtype=np.float64)
    else:
        # Assume array-like (list, tuple, ndarray)
        result = np.asarray(border_constant, dtype=np.float64)
        if result.ndim != 1:
            raise ValueError(f"border_constant must be 1D, got {result.ndim}D")
        if result.shape[0] != C:
            raise ValueError(f"border_constant must have length {C}, got {result.shape[0]}")
    
    if not result.flags['C_CONTIGUOUS']:
        result = np.ascontiguousarray(result)
    
    return result


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
# Planes: Single channel version
# =============================================================================
def lerp_between_planes_1ch(
    np.ndarray[f64, ndim=2] plane0,
    np.ndarray[f64, ndim=2] plane1,
    np.ndarray[f64, ndim=4] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
):
    """
    Interpolate between two 2D planes using a 3D grid of (u_x, u_y, u_z) coordinates.
    
    Args:
        plane0: First plane, shape (H_src, W_src)
        plane1: Second plane, shape (H_src, W_src)
        coords: Coordinate grid, shape (D, H, W, 3)
                coords[..., 0] = u_x (x position in planes, 0-1)
                coords[..., 1] = u_y (y position in planes, 0-1)
                coords[..., 2] = u_z (blend factor between planes, 0-1)
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value to use for BORDER_CONSTANT mode (default: 0.0)
    
    Returns:
        Interpolated values, shape (D, H, W)
    """
    cdef Py_ssize_t H_src = plane0.shape[0]
    cdef Py_ssize_t W_src = plane0.shape[1]
    cdef Py_ssize_t D_out = coords.shape[0]
    cdef Py_ssize_t H_out = coords.shape[1]
    cdef Py_ssize_t W_out = coords.shape[2]
    
    if plane1.shape[0] != H_src or plane1.shape[1] != W_src:
        raise ValueError("Planes must have same shape")
    if coords.shape[3] != 3:
        raise ValueError("coords last dim must be 3")
    
    if not plane0.flags['C_CONTIGUOUS']:
        plane0 = np.ascontiguousarray(plane0)
    if not plane1.flags['C_CONTIGUOUS']:
        plane1 = np.ascontiguousarray(plane1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] p0 = plane0
    cdef f64[:, ::1] p1 = plane1
    cdef f64[:, :, :, ::1] c = coords
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((D_out, H_out, W_out), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    cdef Py_ssize_t d, h, w
    cdef Py_ssize_t x_lo, x_hi, y_lo, y_hi
    cdef f64 u_x, u_y, u_z, x_f, y_f, x_frac, y_frac
    cdef f64 new_u_x, new_u_y, new_u_z
    cdef f64 v00_0, v01_0, v10_0, v11_0
    cdef f64 v00_1, v01_1, v10_1, v11_1
    cdef f64 v0, v1
    cdef f64 H_minus_1 = <f64>(H_src - 1)
    cdef f64 W_minus_1 = <f64>(W_src - 1)
    
    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):
                u_x = c[d, h, w, 0]
                u_y = c[d, h, w, 1]
                u_z = c[d, h, w, 2]
                
                if border_mode == BORDER_CONSTANT:
                    if is_out_of_bounds_3d(u_x, u_y, u_z):
                        out_mv[d, h, w] = border_constant
                        continue
                    new_u_x = u_x
                    new_u_y = u_y
                    new_u_z = u_z
                elif border_mode == BORDER_OVERFLOW:
                    new_u_x = u_x
                    new_u_y = u_y
                    new_u_z = u_z
                else:
                    new_u_x = handle_border_1d(u_x, border_mode)
                    new_u_y = handle_border_1d(u_y, border_mode)
                    new_u_z = handle_border_1d(u_z, border_mode)
                
                x_f = new_u_x * W_minus_1
                y_f = new_u_y * H_minus_1
                
                x_lo = <Py_ssize_t>floor(x_f)
                y_lo = <Py_ssize_t>floor(y_f)
                
                if x_lo < 0: x_lo = 0
                if x_lo >= W_src - 1: x_lo = W_src - 2
                if y_lo < 0: y_lo = 0
                if y_lo >= H_src - 1: y_lo = H_src - 2
                
                x_hi = x_lo + 1
                y_hi = y_lo + 1
                
                x_frac = x_f - <f64>x_lo
                y_frac = y_f - <f64>y_lo
                
                if x_frac < 0.0: x_frac = 0.0
                if x_frac > 1.0: x_frac = 1.0
                if y_frac < 0.0: y_frac = 0.0
                if y_frac > 1.0: y_frac = 1.0
                
                # Bilinear sample plane0
                v00_0 = p0[y_lo, x_lo]
                v01_0 = p0[y_lo, x_hi]
                v10_0 = p0[y_hi, x_lo]
                v11_0 = p0[y_hi, x_hi]
                v0 = (v00_0 * (1 - x_frac) + v01_0 * x_frac) * (1 - y_frac) + \
                     (v10_0 * (1 - x_frac) + v11_0 * x_frac) * y_frac
                
                # Bilinear sample plane1
                v00_1 = p1[y_lo, x_lo]
                v01_1 = p1[y_lo, x_hi]
                v10_1 = p1[y_hi, x_lo]
                v11_1 = p1[y_hi, x_hi]
                v1 = (v00_1 * (1 - x_frac) + v01_1 * x_frac) * (1 - y_frac) + \
                     (v10_1 * (1 - x_frac) + v11_1 * x_frac) * y_frac
                
                out_mv[d, h, w] = v0 + new_u_z * (v1 - v0)
    
    return out


# =============================================================================
# Planes: Multi-channel version (e.g., RGB images)
# =============================================================================
def lerp_between_planes_multichannel(
    np.ndarray[f64, ndim=3] plane0,
    np.ndarray[f64, ndim=3] plane1,
    np.ndarray[f64, ndim=4] coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Interpolate between two multi-channel 2D planes (e.g., RGB images).
    
    Args:
        plane0: First plane, shape (H_src, W_src, C)
        plane1: Second plane, shape (H_src, W_src, C)
        coords: Coordinate grid, shape (D, H, W, 3)
                coords[..., 0] = u_x (x position in planes, 0-1)
                coords[..., 1] = u_y (y position in planes, 0-1)
                coords[..., 2] = u_z (blend factor between planes, 0-1)
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value to use for BORDER_CONSTANT mode.
                        Can be scalar, array of shape (C,), or None.
                        Examples: 0.0, [1.0, 0.0, 0.0], (255, 0, 255)
    
    Returns:
        Interpolated values, shape (D, H, W, C)
    """
    cdef Py_ssize_t H_src = plane0.shape[0]
    cdef Py_ssize_t W_src = plane0.shape[1]
    cdef Py_ssize_t C = plane0.shape[2]
    cdef Py_ssize_t D_out = coords.shape[0]
    cdef Py_ssize_t H_out = coords.shape[1]
    cdef Py_ssize_t W_out = coords.shape[2]
    
    if plane1.shape[0] != H_src or plane1.shape[1] != W_src or plane1.shape[2] != C:
        raise ValueError("Planes must have same shape")
    if coords.shape[3] != 3:
        raise ValueError("coords last dim must be 3")
    
    cdef np.ndarray[f64, ndim=1] border_const_arr = prepare_border_constant_array(border_constant, C)
    
    if not plane0.flags['C_CONTIGUOUS']:
        plane0 = np.ascontiguousarray(plane0)
    if not plane1.flags['C_CONTIGUOUS']:
        plane1 = np.ascontiguousarray(plane1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, :, ::1] p0 = plane0
    cdef f64[:, :, ::1] p1 = plane1
    cdef f64[:, :, :, ::1] c = coords
    cdef f64[::1] border_const_mv = border_const_arr
    
    cdef np.ndarray[f64, ndim=4] out = np.empty((D_out, H_out, W_out, C), dtype=np.float64)
    cdef f64[:, :, :, ::1] out_mv = out
    
    cdef Py_ssize_t d, h, w, ch
    cdef Py_ssize_t x_lo, x_hi, y_lo, y_hi
    cdef f64 u_x, u_y, u_z, x_f, y_f, x_frac, y_frac
    cdef f64 new_u_x, new_u_y, new_u_z
    cdef f64 v00_0, v01_0, v10_0, v11_0
    cdef f64 v00_1, v01_1, v10_1, v11_1
    cdef f64 v0, v1
    cdef f64 H_minus_1 = <f64>(H_src - 1)
    cdef f64 W_minus_1 = <f64>(W_src - 1)
    
    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):
                u_x = c[d, h, w, 0]
                u_y = c[d, h, w, 1]
                u_z = c[d, h, w, 2]
                
                if border_mode == BORDER_CONSTANT:
                    if is_out_of_bounds_3d(u_x, u_y, u_z):
                        for ch in range(C):
                            out_mv[d, h, w, ch] = border_const_mv[ch]
                        continue
                    new_u_x = u_x
                    new_u_y = u_y
                    new_u_z = u_z
                elif border_mode == BORDER_OVERFLOW:
                    new_u_x = u_x
                    new_u_y = u_y
                    new_u_z = u_z
                else:
                    new_u_x = handle_border_1d(u_x, border_mode)
                    new_u_y = handle_border_1d(u_y, border_mode)
                    new_u_z = handle_border_1d(u_z, border_mode)
                
                x_f = new_u_x * W_minus_1
                y_f = new_u_y * H_minus_1
                
                x_lo = <Py_ssize_t>floor(x_f)
                y_lo = <Py_ssize_t>floor(y_f)
                
                if x_lo < 0: x_lo = 0
                if x_lo >= W_src - 1: x_lo = W_src - 2
                if y_lo < 0: y_lo = 0
                if y_lo >= H_src - 1: y_lo = H_src - 2
                
                x_hi = x_lo + 1
                y_hi = y_lo + 1
                
                x_frac = x_f - <f64>x_lo
                y_frac = y_f - <f64>y_lo
                
                if x_frac < 0.0: x_frac = 0.0
                if x_frac > 1.0: x_frac = 1.0
                if y_frac < 0.0: y_frac = 0.0
                if y_frac > 1.0: y_frac = 1.0
                
                # Process each channel
                for ch in range(C):
                    # Bilinear sample plane0
                    v00_0 = p0[y_lo, x_lo, ch]
                    v01_0 = p0[y_lo, x_hi, ch]
                    v10_0 = p0[y_hi, x_lo, ch]
                    v11_0 = p0[y_hi, x_hi, ch]
                    v0 = (v00_0 * (1 - x_frac) + v01_0 * x_frac) * (1 - y_frac) + \
                         (v10_0 * (1 - x_frac) + v11_0 * x_frac) * y_frac
                    
                    # Bilinear sample plane1
                    v00_1 = p1[y_lo, x_lo, ch]
                    v01_1 = p1[y_lo, x_hi, ch]
                    v10_1 = p1[y_hi, x_lo, ch]
                    v11_1 = p1[y_hi, x_hi, ch]
                    v1 = (v00_1 * (1 - x_frac) + v01_1 * x_frac) * (1 - y_frac) + \
                         (v10_1 * (1 - x_frac) + v11_1 * x_frac) * y_frac
                    
                    out_mv[d, h, w, ch] = v0 + new_u_z * (v1 - v0)
    
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
# Dispatcher: lerp_between_planes
# =============================================================================
def lerp_between_planes(
    np.ndarray plane0,
    np.ndarray plane1,
    np.ndarray coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Interpolate between two planes at arbitrary (u_x, u_y, u_z) coordinates.
    
    Args:
        plane0: First plane, shape (H, W) or (H, W, C)
        plane1: Second plane, shape (H, W) or (H, W, C)
        coords: Coordinates, shape (..., 3) where last dim is (u_x, u_y, u_z)
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value for BORDER_CONSTANT mode.
                        For single channel: scalar (default: 0.0)
                        For multi-channel: scalar or array of shape (C,)
                        Examples: 0.0, [1.0, 0.0, 0.0], (255, 0, 255)
    
    Returns:
        Interpolated values
    """
    if plane0.dtype != np.float64 or not plane0.flags['C_CONTIGUOUS']:
        plane0 = np.ascontiguousarray(plane0, dtype=np.float64)
    if plane1.dtype != np.float64 or not plane1.flags['C_CONTIGUOUS']:
        plane1 = np.ascontiguousarray(plane1, dtype=np.float64)
    if coords.dtype != np.float64 or not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    # Single channel planes
    if plane0.ndim == 2 and coords.ndim == 4:
        if border_constant is None:
            border_constant = 0.0
        return lerp_between_planes_1ch(plane0, plane1, coords, border_mode, <f64>border_constant)
    
    # Multi-channel planes (e.g., RGB)
    if plane0.ndim == 3 and coords.ndim == 4:
        return lerp_between_planes_multichannel(plane0, plane1, coords, border_mode, border_constant)
    
    raise ValueError(f"Unsupported shapes for lerp_between_planes: plane0.ndim={plane0.ndim}, coords.ndim={coords.ndim}")


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