# corner_interp_2d.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from cython.parallel cimport prange, parallel
from libc.math cimport floor, fmod, fabs

cdef int BORDER_REPEAT = 0    # Modulo repeat
cdef int BORDER_MIRROR = 1    # Mirror repeat
cdef int BORDER_CONSTANT = 2  # Constant color fill
cdef int BORDER_CLAMP = 3     # Clamp to edge
cdef int BORDER_OVERFLOW = 4  # Allow overflow (no border handling)

ctypedef np.float64_t f64
ctypedef np.uint8_t u8

from .helpers cimport (
    handle_border_1d,
    is_out_of_bounds_2d,
)

# =============================================================================
# Parallel Kernels - Single Channel
# =============================================================================
cdef inline void _corner_1ch_kernel_parallel(
    f64 tl,
    f64 tr,
    f64 bl,
    f64 br,
    const f64[:, :, ::1] coords_mv,
    f64[:, ::1] out_mv,
    f64 border_const,
    Py_ssize_t H,
    Py_ssize_t W,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """
    Parallel single-channel bilinear interpolation kernel.
    
    Bilinear formula:
        top = tl + (tr - tl) * u_x
        bottom = bl + (br - bl) * u_x
        result = top + (bottom - top) * u_y
    """
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 top_edge, bottom_edge
    # Pre-compute differences for efficiency
    cdef f64 tr_minus_tl = tr - tl
    cdef f64 br_minus_bl = br - bl

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = coords_mv[h, w, 0]
            u_y = coords_mv[h, w, 1]

            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    out_mv[h, w] = border_const
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)

            top_edge = tl + tr_minus_tl * new_u_x
            bottom_edge = bl + br_minus_bl * new_u_x
            out_mv[h, w] = top_edge + (bottom_edge - top_edge) * new_u_y


cdef inline void _corner_1ch_flat_kernel_parallel(
    f64 tl,
    f64 tr,
    f64 bl,
    f64 br,
    const f64[:, ::1] coords_mv,
    f64[::1] out_mv,
    f64 border_const,
    Py_ssize_t N,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """Parallel single-channel flat coordinates kernel."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 top_edge, bottom_edge
    cdef f64 tr_minus_tl = tr - tl
    cdef f64 br_minus_bl = br - bl

    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = coords_mv[n, 0]
        u_y = coords_mv[n, 1]

        if border_mode == BORDER_CONSTANT:
            if is_out_of_bounds_2d(u_x, u_y):
                out_mv[n] = border_const
                continue
            new_u_x = u_x
            new_u_y = u_y
        elif border_mode == BORDER_OVERFLOW:
            new_u_x = u_x
            new_u_y = u_y
        else:
            new_u_x = handle_border_1d(u_x, border_mode)
            new_u_y = handle_border_1d(u_y, border_mode)

        top_edge = tl + tr_minus_tl * new_u_x
        bottom_edge = bl + br_minus_bl * new_u_x
        out_mv[n] = top_edge + (bottom_edge - top_edge) * new_u_y


# =============================================================================
# Parallel Kernels - Multi-Channel (Same Coords)
# =============================================================================
cdef inline void _corner_multichannel_same_coords_kernel_parallel(
    const f64[:, ::1] corners_mv,
    const f64[:, :, ::1] coords_mv,
    f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv,
    Py_ssize_t H,
    Py_ssize_t W,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """
    Parallel multi-channel bilinear interpolation with same coords for all channels.
    
    Parallelizes over pixels (h, w) and processes all channels per pixel.
    This is more cache-friendly than parallelizing over channels.
    """
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 tl, tr, bl, br
    cdef f64 top_edge, bottom_edge
    cdef bint is_oob

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = coords_mv[h, w, 0]
            u_y = coords_mv[h, w, 1]

            # Compute bounded coords once for all channels
            is_oob = False
            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    is_oob = True
                else:
                    new_u_x = u_x
                    new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)

            # Process all channels for this pixel
            for ch in range(C):
                if is_oob:
                    out_mv[h, w, ch] = border_const_mv[ch]
                else:
                    tl = corners_mv[0, ch]
                    tr = corners_mv[1, ch]
                    bl = corners_mv[2, ch]
                    br = corners_mv[3, ch]
                    
                    top_edge = tl + (tr - tl) * new_u_x
                    bottom_edge = bl + (br - bl) * new_u_x
                    out_mv[h, w, ch] = top_edge + (bottom_edge - top_edge) * new_u_y


cdef inline void _corner_multichannel_flat_same_coords_kernel_parallel(
    const f64[:, ::1] corners_mv,
    const f64[:, ::1] coords_mv,
    f64[:, ::1] out_mv,
    const f64[::1] border_const_mv,
    Py_ssize_t N,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """Parallel multi-channel flat coordinates kernel (same coords)."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 tl, tr, bl, br
    cdef f64 top_edge, bottom_edge
    cdef bint is_oob

    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = coords_mv[n, 0]
        u_y = coords_mv[n, 1]

        is_oob = False
        if border_mode == BORDER_CONSTANT:
            if is_out_of_bounds_2d(u_x, u_y):
                is_oob = True
            else:
                new_u_x = u_x
                new_u_y = u_y
        elif border_mode == BORDER_OVERFLOW:
            new_u_x = u_x
            new_u_y = u_y
        else:
            new_u_x = handle_border_1d(u_x, border_mode)
            new_u_y = handle_border_1d(u_y, border_mode)

        for ch in range(C):
            if is_oob:
                out_mv[n, ch] = border_const_mv[ch]
            else:
                tl = corners_mv[0, ch]
                tr = corners_mv[1, ch]
                bl = corners_mv[2, ch]
                br = corners_mv[3, ch]
                
                top_edge = tl + (tr - tl) * new_u_x
                bottom_edge = bl + (br - bl) * new_u_x
                out_mv[n, ch] = top_edge + (bottom_edge - top_edge) * new_u_y


# =============================================================================
# Parallel Kernels - Multi-Channel (Per-Channel Coords)
# =============================================================================
cdef inline void _corner_multichannel_per_channel_kernel_parallel(
    const f64[:, ::1] corners_mv,
    const f64[:, :, :, ::1] coords_mv,
    f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv,
    Py_ssize_t H,
    Py_ssize_t W,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """
    Parallel multi-channel bilinear interpolation with per-channel coords.
    
    coords shape: (C, H, W, 2)
    Parallelizes over (h, w) since each pixel's channels have different coords.
    """
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 tl, tr, bl, br
    cdef f64 top_edge, bottom_edge

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            for ch in range(C):
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

                tl = corners_mv[0, ch]
                tr = corners_mv[1, ch]
                bl = corners_mv[2, ch]
                br = corners_mv[3, ch]

                top_edge = tl + (tr - tl) * new_u_x
                bottom_edge = bl + (br - bl) * new_u_x
                out_mv[h, w, ch] = top_edge + (bottom_edge - top_edge) * new_u_y


cdef inline void _corner_multichannel_flat_per_channel_kernel_parallel(
    const f64[:, ::1] corners_mv,
    const f64[:, :, ::1] coords_mv,
    f64[:, ::1] out_mv,
    const f64[::1] border_const_mv,
    Py_ssize_t N,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """
    Parallel multi-channel flat per-channel coordinates kernel.
    
    coords shape: (C, N, 2)
    """
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, new_u_x, new_u_y
    cdef f64 tl, tr, bl, br
    cdef f64 top_edge, bottom_edge

    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for ch in range(C):
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

            tl = corners_mv[0, ch]
            tr = corners_mv[1, ch]
            bl = corners_mv[2, ch]
            br = corners_mv[3, ch]

            top_edge = tl + (tr - tl) * new_u_x
            bottom_edge = bl + (br - bl) * new_u_x
            out_mv[n, ch] = top_edge + (bottom_edge - top_edge) * new_u_y



# =============================================================================
# Public API - Single Channel
# =============================================================================
DEF MIN_PARALLEL_SIZE = 10000

def lerp_from_corners_1ch_fast(
    f64 tl,
    f64 tr,
    f64 bl,
    f64 br,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    int num_threads=-1,
):
    """
    Fast parallel single-channel bilinear interpolation from corner values.
    
    Args:
        tl, tr, bl, br: Corner values (top-left, top-right, bottom-left, bottom-right)
        coords: Coordinate grid, shape (H, W, 2)
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
        num_threads: Thread count (-1=auto, 0=serial, >0=specific)
    
    Returns:
        Interpolated values, shape (H, W)
    """
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    # Determine thread count
    cdef int n_threads = num_threads
    cdef Py_ssize_t total_size = H * W
    if n_threads < 0:
        if total_size < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)

    cdef f64[:, :, ::1] coords_mv = coords
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _corner_1ch_kernel_parallel(
            tl, tr, bl, br, coords_mv, out_mv,
            border_constant, H, W, border_mode, n_threads
        )

    return out


def lerp_from_corners_1ch_flat_fast(
    f64 tl,
    f64 tr,
    f64 bl,
    f64 br,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    int num_threads=-1,
):
    """Fast parallel single-channel flat coordinates interpolation."""
    cdef Py_ssize_t N = coords.shape[0]

    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    cdef int n_threads = num_threads
    if n_threads < 0:
        if N < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)

    cdef f64[:, ::1] coords_mv = coords
    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out

    with nogil:
        _corner_1ch_flat_kernel_parallel(
            tl, tr, bl, br, coords_mv, out_mv,
            border_constant, N, border_mode, n_threads
        )

    return out


# =============================================================================
# Public API - Multi-Channel (Same Coords)
# =============================================================================
def lerp_from_corners_multichannel_same_coords_fast(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """
    Fast parallel multi-channel bilinear interpolation with same coords for all channels.
    
    Args:
        corners: Corner values, shape (4, C) - [tl, tr, bl, br]
        coords: Coordinate grid, shape (H, W, 2) - same for all channels
        border_mode: Border handling mode
        border_constant: Pre-resolved border values, shape (C,)
        num_threads: Thread count (-1=auto)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t C = corners.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, num_channels)")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    # Handle border constant
    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    cdef Py_ssize_t total_size = H * W
    if n_threads < 0:
        if total_size < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, :, ::1] coords_mv = coords
    cdef f64[::1] bc_mv = border_constant

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _corner_multichannel_same_coords_kernel_parallel(
            corners_mv, coords_mv, out_mv, bc_mv,
            H, W, C, border_mode, n_threads
        )

    return out


def lerp_from_corners_multichannel_flat_same_coords_fast(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """Fast parallel multi-channel flat coordinates (same for all channels)."""
    cdef Py_ssize_t C = corners.shape[1]
    cdef Py_ssize_t N = coords.shape[0]

    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, num_channels)")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    if n_threads < 0:
        if N < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, ::1] coords_mv = coords
    cdef f64[::1] bc_mv = border_constant

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _corner_multichannel_flat_same_coords_kernel_parallel(
            corners_mv, coords_mv, out_mv, bc_mv,
            N, C, border_mode, n_threads
        )

    return out


# =============================================================================
# Public API - Multi-Channel (Per-Channel Coords)
# =============================================================================
def lerp_from_corners_multichannel_fast(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=4] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """
    Fast parallel multi-channel bilinear interpolation with per-channel coords.
    
    Args:
        corners: Corner values, shape (4, C)
        coords: Per-channel coordinate grids, shape (C, H, W, 2)
        border_mode: Border handling mode
        border_constant: Pre-resolved border values, shape (C,)
        num_threads: Thread count (-1=auto)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t C = corners.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t H = coords.shape[1]
    cdef Py_ssize_t W = coords.shape[2]

    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, num_channels)")
    if coords.shape[3] != 2:
        raise ValueError("coords must have shape (num_channels, H, W, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match corners ({C})")

    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    cdef Py_ssize_t total_size = H * W
    if n_threads < 0:
        if total_size < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, :, :, ::1] coords_mv = coords
    cdef f64[::1] bc_mv = border_constant

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _corner_multichannel_per_channel_kernel_parallel(
            corners_mv, coords_mv, out_mv, bc_mv,
            H, W, C, border_mode, n_threads
        )

    return out


def lerp_from_corners_multichannel_flat_fast(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """Fast parallel multi-channel flat per-channel coordinates."""
    cdef Py_ssize_t C = corners.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t N = coords.shape[1]

    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, num_channels)")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (num_channels, N, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match corners ({C})")

    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    if n_threads < 0:
        if N < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, :, ::1] coords_mv = coords
    cdef f64[::1] bc_mv = border_constant

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _corner_multichannel_flat_per_channel_kernel_parallel(
            corners_mv, coords_mv, out_mv, bc_mv,
            N, C, border_mode, n_threads
        )

    return out


# =============================================================================
# Smart Dispatcher
# =============================================================================
def lerp_from_corners_fast(
    corners,
    coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
    int num_threads=-1,
):
    """
    Smart dispatcher for fast parallel corner interpolation.
    
    Automatically selects the appropriate kernel based on input shapes.
    
    Args:
        corners: Corner values. Can be:
            - Tuple/list of 4 scalars for single channel
            - Array of shape (4,) for single channel
            - Array of shape (4, C) for multi-channel
        coords: Coordinates. Shape determines behavior:
            - (H, W, 2): Grid coords, same for all channels
            - (N, 2): Flat coords, same for all channels
            - (C, H, W, 2): Grid coords, per-channel
            - (C, N, 2): Flat coords, per-channel
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
        num_threads: Thread count (-1=auto, 0=serial, >0=specific)
    
    Returns:
        Interpolated values
    """
    # Convert corners
    cdef bint single_channel = False
    
    if isinstance(corners, (list, tuple)) and len(corners) == 4:
        if all(isinstance(c, (int, float)) for c in corners):
            single_channel = True
            corners_arr = np.array(corners, dtype=np.float64)
        else:
            corners_arr = np.asarray(corners, dtype=np.float64)
    else:
        corners_arr = np.asarray(corners, dtype=np.float64)
    
    if not corners_arr.flags['C_CONTIGUOUS']:
        corners_arr = np.ascontiguousarray(corners_arr)
    
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
    
    # Convert coords
    coords_arr = np.ascontiguousarray(coords, dtype=np.float64)
    
    cdef Py_ssize_t num_channels
    cdef np.ndarray[f64, ndim=1] bc_arr
    
    # Dispatch
    if single_channel:
        bc = 0.0 if border_constant is None else float(border_constant)
        
        if coords_arr.ndim == 3 and coords_arr.shape[2] == 2:
            return lerp_from_corners_1ch_fast(
                corners_arr[0], corners_arr[1], corners_arr[2], corners_arr[3],
                coords_arr, border_mode, bc, num_threads
            )
        elif coords_arr.ndim == 2 and coords_arr.shape[1] == 2:
            return lerp_from_corners_1ch_flat_fast(
                corners_arr[0], corners_arr[1], corners_arr[2], corners_arr[3],
                coords_arr, border_mode, bc, num_threads
            )
        else:
            raise ValueError(f"Invalid coords shape for single channel: {coords_arr.shape}")
    
    else:
        num_channels = corners_arr.shape[1]
        
        # Prepare border constant
        if border_constant is None:
            bc_arr = np.zeros(num_channels, dtype=np.float64)
        elif isinstance(border_constant, (int, float)):
            bc_arr = np.full(num_channels, float(border_constant), dtype=np.float64)
        else:
            bc_arr = np.ascontiguousarray(border_constant, dtype=np.float64)
            if bc_arr.shape[0] != num_channels:
                raise ValueError(f"border_constant must have length {num_channels}")
        
        if coords_arr.ndim == 3 and coords_arr.shape[2] == 2:
            # Same coords for all channels (H, W, 2)
            return lerp_from_corners_multichannel_same_coords_fast(
                corners_arr, coords_arr, border_mode, bc_arr, num_threads
            )
        elif coords_arr.ndim == 2 and coords_arr.shape[1] == 2:
            # Flat same coords (N, 2)
            return lerp_from_corners_multichannel_flat_same_coords_fast(
                corners_arr, coords_arr, border_mode, bc_arr, num_threads
            )
        elif coords_arr.ndim == 4 and coords_arr.shape[3] == 2:
            # Per-channel grid coords (C, H, W, 2)
            if coords_arr.shape[0] != num_channels:
                raise ValueError(f"coords channels ({coords_arr.shape[0]}) must match corners ({num_channels})")
            return lerp_from_corners_multichannel_fast(
                corners_arr, coords_arr, border_mode, bc_arr, num_threads
            )
        elif coords_arr.ndim == 3 and coords_arr.shape[2] == 2 and coords_arr.shape[0] == num_channels:
            # Per-channel flat coords (C, N, 2)
            return lerp_from_corners_multichannel_flat_fast(
                corners_arr, coords_arr, border_mode, bc_arr, num_threads
            )
        else:
            raise ValueError(f"Invalid coords shape for multi-channel: {coords_arr.shape}")

