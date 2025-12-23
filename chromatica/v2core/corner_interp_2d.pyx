# corner_interp_2d.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport floor, fmod, fabs
from .border_handling cimport (
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)

ctypedef np.float64_t f64
ctypedef np.uint8_t u8


# =============================================================================
# Helper Functions
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
    else:  # BORDER_OVERFLOW or BORDER_CONSTANT (handled separately)
        return u


cdef inline bint is_out_of_bounds_2d(f64 u_x, f64 u_y) noexcept nogil:
    """Check if 2D coordinates are out of [0, 1] bounds."""
    return u_x < 0.0 or u_x > 1.0 or u_y < 0.0 or u_y > 1.0


cdef np.ndarray[f64, ndim=1] prepare_border_constant_array(object border_constant, Py_ssize_t C):
    """
    Prepare border constants array for multi-channel data.
    """
    cdef np.ndarray[f64, ndim=1] result
    
    if border_constant is None:
        result = np.zeros(C, dtype=np.float64)
    elif isinstance(border_constant, (int, float)):
        result = np.full(C, <f64>border_constant, dtype=np.float64)
    else:
        result = np.asarray(border_constant, dtype=np.float64)
        if result.ndim != 1:
            raise ValueError(f"border_constant must be 1D, got {result.ndim}D")
        if result.shape[0] != C:
            raise ValueError(f"border_constant must have length {C}, got {result.shape[0]}")
    
    if not result.flags['C_CONTIGUOUS']:
        result = np.ascontiguousarray(result)
    
    return result


# =============================================================================
# Single-channel: Corner-based 2D Interpolation
# =============================================================================
def lerp_from_corners_1ch(
    f64 tl,
    f64 tr, 
    f64 bl,
    f64 br,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
):
    """
    Single-channel 2D interpolation from corner values.
    
    Args:
        tl, tr, bl, br: Corner values (top-left, top-right, bottom-left, bottom-right)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x, coords[h, w, 1] = u_y
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value for BORDER_CONSTANT mode (default: 0.0)
    
    Returns:
        Interpolated values, shape (H, W)
    """
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, :, ::1] coords_mv = coords
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 top_edge, bottom_edge
    
    for h in range(H):
        for w in range(W):
            u_x = coords_mv[h, w, 0]
            u_y = coords_mv[h, w, 1]
            
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
            
            # Bilinear interpolation
            top_edge = tl + (tr - tl) * new_u_x
            bottom_edge = bl + (br - bl) * new_u_x
            out_mv[h, w] = top_edge + (bottom_edge - top_edge) * new_u_y
    
    return out


def lerp_from_corners_1ch_flat(
    f64 tl,
    f64 tr,
    f64 bl,
    f64 br,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
):
    """
    Single-channel 2D interpolation from corner values with flat coordinates.
    
    Args:
        tl, tr, bl, br: Corner values
        coords: Coordinate pairs, shape (N, 2)
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
    
    Returns:
        Interpolated values, shape (N,)
    """
    cdef Py_ssize_t N = coords.shape[0]
    
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] coords_mv = coords
    
    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out
    
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 top_edge, bottom_edge
    
    for n in range(N):
        u_x = coords_mv[n, 0]
        u_y = coords_mv[n, 1]
        
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
        
        top_edge = tl + (tr - tl) * new_u_x
        bottom_edge = bl + (br - bl) * new_u_x
        out_mv[n] = top_edge + (bottom_edge - top_edge) * new_u_y
    
    return out


# =============================================================================
# Multi-channel: Per-channel coordinates (4D coords array)
# =============================================================================
def lerp_from_corners_multichannel(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=4] coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Multi-channel 2D interpolation from corner values with per-channel coordinates.
    
    Args:
        corners: Corner values, shape (4, num_channels)
                 Order: [top_left, top_right, bottom_left, bottom_right]
        coords: Coordinate grids, shape (num_channels, H, W, 2)
                coords[ch, h, w, 0] = u_x, coords[ch, h, w, 1] = u_y
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value for BORDER_CONSTANT mode.
                        Can be scalar or array of shape (num_channels,)
    
    Returns:
        Interpolated values, shape (H, W, num_channels)
    """
    cdef Py_ssize_t num_channels = corners.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t H = coords.shape[1]
    cdef Py_ssize_t W = coords.shape[2]
    
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, num_channels)")
    if coords.shape[3] != 2:
        raise ValueError("coords must have shape (num_channels, H, W, 2)")
    if C_coords != num_channels:
        raise ValueError(f"Number of channels in coords ({C_coords}) must match corners ({num_channels})")
    
    cdef np.ndarray[f64, ndim=1] border_const_arr = prepare_border_constant_array(border_constant, num_channels)
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, :, :, ::1] coords_mv = coords
    cdef f64[::1] border_const_mv = border_const_arr
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, num_channels), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    cdef Py_ssize_t ch, h, w
    cdef f64 tl, tr, bl, br
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 top_edge, bottom_edge
    
    for ch in range(num_channels):
        tl = corners_mv[0, ch]
        tr = corners_mv[1, ch]
        bl = corners_mv[2, ch]
        br = corners_mv[3, ch]
        
        for h in range(H):
            for w in range(W):
                u_x = coords_mv[ch, h, w, 0]
                u_y = coords_mv[ch, h, w, 1]
                
                if border_mode == BORDER_CONSTANT:
                    if is_out_of_bounds_2d(u_x, u_y):
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
                
                top_edge = tl + (tr - tl) * new_u_x
                bottom_edge = bl + (br - bl) * new_u_x
                out_mv[h, w, ch] = top_edge + (bottom_edge - top_edge) * new_u_y
    
    return out


# =============================================================================
# Multi-channel: Same coordinates for all channels (3D coords array)
# =============================================================================
def lerp_from_corners_multichannel_same_coords(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Multi-channel 2D interpolation from corner values with same coordinates for all channels.
    
    Args:
        corners: Corner values, shape (4, num_channels)
                 Order: [top_left, top_right, bottom_left, bottom_right]
        coords: Coordinate grid, shape (H, W, 2) - same for all channels
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value for BORDER_CONSTANT mode.
                        Can be scalar or array of shape (num_channels,)
    
    Returns:
        Interpolated values, shape (H, W, num_channels)
    """
    cdef Py_ssize_t num_channels = corners.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, num_channels)")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    
    cdef np.ndarray[f64, ndim=1] border_const_arr = prepare_border_constant_array(border_constant, num_channels)
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, :, ::1] coords_mv = coords
    cdef f64[::1] border_const_mv = border_const_arr
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, num_channels), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    # Pre-compute bounded coordinates and out-of-bounds mask
    cdef np.ndarray[f64, ndim=2] new_u_x_arr = np.empty((H, W), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] new_u_y_arr = np.empty((H, W), dtype=np.float64)
    cdef np.ndarray[u8, ndim=2] oob_mask = np.zeros((H, W), dtype=np.uint8)
    
    cdef f64[:, ::1] new_u_x_mv = new_u_x_arr
    cdef f64[:, ::1] new_u_y_mv = new_u_y_arr
    cdef u8[:, ::1] oob_mask_mv = oob_mask
    
    cdef Py_ssize_t ch, h, w
    cdef f64 tl, tr, bl, br
    cdef f64 u_x, u_y
    cdef f64 top_edge, bottom_edge
    
    # First pass: compute bounded coordinates
    for h in range(H):
        for w in range(W):
            u_x = coords_mv[h, w, 0]
            u_y = coords_mv[h, w, 1]
            
            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    oob_mask_mv[h, w] = 1
                    # Still store coords (won't be used, but avoids uninitialized values)
                    new_u_x_mv[h, w] = u_x
                    new_u_y_mv[h, w] = u_y
                else:
                    new_u_x_mv[h, w] = u_x
                    new_u_y_mv[h, w] = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x_mv[h, w] = u_x
                new_u_y_mv[h, w] = u_y
            else:
                new_u_x_mv[h, w] = handle_border_1d(u_x, border_mode)
                new_u_y_mv[h, w] = handle_border_1d(u_y, border_mode)
    
    # Second pass: interpolate for each channel
    for ch in range(num_channels):
        tl = corners_mv[0, ch]
        tr = corners_mv[1, ch]
        bl = corners_mv[2, ch]
        br = corners_mv[3, ch]
        
        for h in range(H):
            for w in range(W):
                if border_mode == BORDER_CONSTANT and oob_mask_mv[h, w]:
                    out_mv[h, w, ch] = border_const_mv[ch]
                else:
                    u_x = new_u_x_mv[h, w]
                    u_y = new_u_y_mv[h, w]
                    top_edge = tl + (tr - tl) * u_x
                    bottom_edge = bl + (br - bl) * u_x
                    out_mv[h, w, ch] = top_edge + (bottom_edge - top_edge) * u_y
    
    return out


# =============================================================================
# Multi-channel: Flat coordinates with per-channel coords
# =============================================================================
def lerp_from_corners_multichannel_flat(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Multi-channel 2D interpolation from corners with flat per-channel coordinates.
    
    Args:
        corners: Corner values, shape (4, num_channels)
        coords: Coordinate pairs, shape (num_channels, N, 2)
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
    
    Returns:
        Interpolated values, shape (N, num_channels)
    """
    cdef Py_ssize_t num_channels = corners.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t N = coords.shape[1]
    
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, num_channels)")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (num_channels, N, 2)")
    if C_coords != num_channels:
        raise ValueError(f"Number of channels in coords ({C_coords}) must match corners ({num_channels})")
    
    cdef np.ndarray[f64, ndim=1] border_const_arr = prepare_border_constant_array(border_constant, num_channels)
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, :, ::1] coords_mv = coords
    cdef f64[::1] border_const_mv = border_const_arr
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((N, num_channels), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t ch, n
    cdef f64 tl, tr, bl, br
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 top_edge, bottom_edge
    
    for ch in range(num_channels):
        tl = corners_mv[0, ch]
        tr = corners_mv[1, ch]
        bl = corners_mv[2, ch]
        br = corners_mv[3, ch]
        
        for n in range(N):
            u_x = coords_mv[ch, n, 0]
            u_y = coords_mv[ch, n, 1]
            
            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
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
            
            top_edge = tl + (tr - tl) * new_u_x
            bottom_edge = bl + (br - bl) * new_u_x
            out_mv[n, ch] = top_edge + (bottom_edge - top_edge) * new_u_y
    
    return out


def lerp_from_corners_multichannel_flat_same_coords(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Multi-channel 2D interpolation from corners with flat coordinates (same for all channels).
    
    Args:
        corners: Corner values, shape (4, num_channels)
        coords: Coordinate pairs, shape (N, 2) - same for all channels
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
    
    Returns:
        Interpolated values, shape (N, num_channels)
    """
    cdef Py_ssize_t num_channels = corners.shape[1]
    cdef Py_ssize_t N = coords.shape[0]
    
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, num_channels)")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    
    cdef np.ndarray[f64, ndim=1] border_const_arr = prepare_border_constant_array(border_constant, num_channels)
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, ::1] coords_mv = coords
    cdef f64[::1] border_const_mv = border_const_arr
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((N, num_channels), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    # Pre-compute bounded coordinates
    cdef np.ndarray[f64, ndim=1] new_u_x_arr = np.empty(N, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] new_u_y_arr = np.empty(N, dtype=np.float64)
    cdef np.ndarray[u8, ndim=1] oob_mask = np.zeros(N, dtype=np.uint8)
    
    cdef f64[::1] new_u_x_mv = new_u_x_arr
    cdef f64[::1] new_u_y_mv = new_u_y_arr
    cdef u8[::1] oob_mask_mv = oob_mask
    
    cdef Py_ssize_t ch, n
    cdef f64 tl, tr, bl, br
    cdef f64 u_x, u_y
    cdef f64 top_edge, bottom_edge
    
    # First pass: compute bounded coordinates
    for n in range(N):
        u_x = coords_mv[n, 0]
        u_y = coords_mv[n, 1]
        
        if border_mode == BORDER_CONSTANT:
            if is_out_of_bounds_2d(u_x, u_y):
                oob_mask_mv[n] = 1
                new_u_x_mv[n] = u_x
                new_u_y_mv[n] = u_y
            else:
                new_u_x_mv[n] = u_x
                new_u_y_mv[n] = u_y
        elif border_mode == BORDER_OVERFLOW:
            new_u_x_mv[n] = u_x
            new_u_y_mv[n] = u_y
        else:
            new_u_x_mv[n] = handle_border_1d(u_x, border_mode)
            new_u_y_mv[n] = handle_border_1d(u_y, border_mode)
    
    # Second pass: interpolate for each channel
    for ch in range(num_channels):
        tl = corners_mv[0, ch]
        tr = corners_mv[1, ch]
        bl = corners_mv[2, ch]
        br = corners_mv[3, ch]
        
        for n in range(N):
            if border_mode == BORDER_CONSTANT and oob_mask_mv[n]:
                out_mv[n, ch] = border_const_mv[ch]
            else:
                u_x = new_u_x_mv[n]
                u_y = new_u_y_mv[n]
                top_edge = tl + (tr - tl) * u_x
                bottom_edge = bl + (br - bl) * u_x
                out_mv[n, ch] = top_edge + (bottom_edge - top_edge) * u_y
    
    return out

# =============================================================================
# Dispatcher
# =============================================================================
def lerp_from_corners(
    corners,
    coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
):
    """
    Bilinear interpolation from corner values.
    
    Args:
        corners: Corner values. Can be:
            - Tuple/list of 4 scalars (tl, tr, bl, br) for single channel
            - Array of shape (4,) for single channel
            - Array of shape (4, num_channels) for multi-channel
        coords: Coordinates. Shape determines behavior:
            - (H, W, 2): Grid coords, same for all channels
            - (N, 2): Flat coords, same for all channels  
            - (num_channels, H, W, 2): Grid coords, per-channel
            - (num_channels, N, 2): Flat coords, per-channel
        border_mode: Border handling mode (default: BORDER_CLAMP)
        border_constant: Value for BORDER_CONSTANT mode.
                        Scalar or array of shape (num_channels,)
    
    Returns:
        Interpolated values
    """
    # Convert corners to array - use object type for flexibility in dispatcher
    cdef bint single_channel = False
    
    if isinstance(corners, (list, tuple)) and len(corners) == 4:
        # Check if it's 4 scalars or 4 arrays
        if all(isinstance(c, (int, float)) for c in corners):
            # Single channel: 4 scalars
            single_channel = True
            corners_arr = np.array(corners, dtype=np.float64)
        else:
            # Multi-channel: list of 4 arrays -> shape (4, num_channels)
            corners_arr = np.asarray(corners, dtype=np.float64)
    else:
        corners_arr = np.asarray(corners, dtype=np.float64)
    
    if not corners_arr.flags['C_CONTIGUOUS']:
        corners_arr = np.ascontiguousarray(corners_arr)
    
    # Determine if single or multi-channel from corners shape
    if corners_arr.ndim == 1:
        if corners_arr.shape[0] != 4:
            raise ValueError("For single channel, corners must have 4 values")
        single_channel = True
    elif corners_arr.ndim == 2:
        if corners_arr.shape[0] != 4:
            raise ValueError("corners must have shape (4,) or (4, num_channels)")
        single_channel = False
    else:
        raise ValueError("corners must be 1D or 2D array")
    
    # Convert coords - don't use cdef to keep Python object access
    coords_arr = np.asarray(coords, dtype=np.float64)
    if not coords_arr.flags['C_CONTIGUOUS']:
        coords_arr = np.ascontiguousarray(coords_arr)
    
    cdef Py_ssize_t num_channels
    
    # Dispatch based on shapes
    if single_channel:
        # Single channel
        if border_constant is None:
            border_constant = 0.0
        
        if coords_arr.ndim == 3 and coords_arr.shape[2] == 2:
            # Grid coords (H, W, 2)
            return lerp_from_corners_1ch(
                corners_arr[0], corners_arr[1], corners_arr[2], corners_arr[3],
                coords_arr, border_mode, <f64>border_constant
            )
        elif coords_arr.ndim == 2 and coords_arr.shape[1] == 2:
            # Flat coords (N, 2)
            return lerp_from_corners_1ch_flat(
                corners_arr[0], corners_arr[1], corners_arr[2], corners_arr[3],
                coords_arr, border_mode, <f64>border_constant
            )
        else:
            raise ValueError(f"Invalid coords shape for single channel: {coords_arr.shape}")
    
    else:
        # Multi-channel
        num_channels = corners_arr.shape[1]
        
        if coords_arr.ndim == 3 and coords_arr.shape[2] == 2:
            # Grid coords, same for all channels (H, W, 2)
            return lerp_from_corners_multichannel_same_coords(
                corners_arr, coords_arr, border_mode, border_constant
            )
        elif coords_arr.ndim == 2 and coords_arr.shape[1] == 2:
            # Flat coords, same for all channels (N, 2)
            return lerp_from_corners_multichannel_flat_same_coords(
                corners_arr, coords_arr, border_mode, border_constant
            )
        elif coords_arr.ndim == 4 and coords_arr.shape[3] == 2:
            # Grid coords, per-channel (num_channels, H, W, 2)
            if coords_arr.shape[0] != num_channels:
                raise ValueError(f"coords channels ({coords_arr.shape[0]}) must match corners ({num_channels})")
            return lerp_from_corners_multichannel(
                corners_arr, coords_arr, border_mode, border_constant
            )
        elif coords_arr.ndim == 3 and coords_arr.shape[2] == 2 and coords_arr.shape[0] == num_channels:
            # Flat coords, per-channel (num_channels, N, 2)
            return lerp_from_corners_multichannel_flat(
                corners_arr, coords_arr, border_mode, border_constant
            )
        else:
            raise ValueError(f"Invalid coords shape for multi-channel: {coords_arr.shape}")