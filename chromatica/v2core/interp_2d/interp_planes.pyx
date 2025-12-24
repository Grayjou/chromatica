

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


