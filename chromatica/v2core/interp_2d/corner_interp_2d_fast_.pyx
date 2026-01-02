# corner_interp_2d_fast_.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport floor

ctypedef np.float64_t f64
ctypedef np.int32_t i32

cdef int BORDER_REPEAT = 0
cdef int BORDER_MIRROR = 1
cdef int BORDER_CONSTANT = 2
cdef int BORDER_CLAMP = 3
cdef int BORDER_OVERFLOW = 4

from ..border_handling cimport handle_border_1d
from ..interp_utils cimport (
    DistanceMode,
    MAX_NORM,
    MANHATTAN,
    SCALED_MANHATTAN,
    ALPHA_MAX,
    ALPHA_MAX_SIMPLE,
    TAYLOR,
    EUCLIDEAN,
    WEIGHTED_MINMAX,
    compute_interp_idx,
    BorderResult,
    clamp_01,
    process_border_2d
)


# =============================================================================
# Bilinear Interpolation Helper Functions
# =============================================================================
cdef inline f64 _bilinear_interp_1ch(
    f64 tl, f64 tr, f64 bl, f64 br,
    f64 u_x, f64 u_y
) noexcept nogil:
    """Bilinear interpolation for single channel."""
    cdef f64 top = tl + (tr - tl) * u_x
    cdef f64 bottom = bl + (br - bl) * u_x
    return top + (bottom - top) * u_y

cdef inline f64 _bilinear_interp_multichannel(
    const f64[:, ::1] corners_mv,  # shape (4, C)
    f64 u_x, f64 u_y,
    Py_ssize_t ch
) noexcept nogil:
    """Bilinear interpolation for multi-channel."""
    cdef f64 tl = corners_mv[0, ch]
    cdef f64 tr = corners_mv[1, ch]
    cdef f64 bl = corners_mv[2, ch]
    cdef f64 br = corners_mv[3, ch]
    
    cdef f64 top = tl + (tr - tl) * u_x
    cdef f64 bottom = bl + (br - bl) * u_x
    return top + (bottom - top) * u_y

# =============================================================================
# Single-Channel Kernels with Feathering
# =============================================================================
cdef inline void _corner_1ch_feathered_kernel(
    f64 tl, f64 tr, f64 bl, f64 br,
    const f64[:, :, ::1] coords_mv,
    f64[:, ::1] out_mv,
    f64 border_const,
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W,
    int border_mode,
    i32 distance_mode,
    int num_threads,
) noexcept nogil:
    """Parallel single-channel bilinear interpolation with feathering."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = coords_mv[h, w, 0]
            u_y = coords_mv[h, w, 1]
            
            border_res = process_border_2d(u_x, u_y, border_mode, border_feathering, distance_mode)
            
            if border_res.use_border_directly:
                out_mv[h, w] = border_const
            else:
                edge_val = _bilinear_interp_1ch(tl, tr, bl, br, 
                                               border_res.u_x_final, border_res.u_y_final)
                if border_res.blend_factor > 0.0:
                    out_mv[h, w] = edge_val + border_res.blend_factor * (border_const - edge_val)
                else:
                    out_mv[h, w] = edge_val

cdef inline void _corner_1ch_flat_feathered_kernel(
    f64 tl, f64 tr, f64 bl, f64 br,
    const f64[:, ::1] coords_mv,
    f64[::1] out_mv,
    f64 border_const,
    f64 border_feathering,
    Py_ssize_t N,
    int border_mode,
    i32 distance_mode,
    int num_threads,
) noexcept nogil:
    """Parallel single-channel flat coordinates with feathering."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, edge_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = coords_mv[n, 0]
        u_y = coords_mv[n, 1]
        
        border_res = process_border_2d(u_x, u_y, border_mode, border_feathering, distance_mode)
        
        if border_res.use_border_directly:
            out_mv[n] = border_const
        else:
            edge_val = _bilinear_interp_1ch(tl, tr, bl, br, 
                                           border_res.u_x_final, border_res.u_y_final)
            if border_res.blend_factor > 0.0:
                out_mv[n] = edge_val + border_res.blend_factor * (border_const - edge_val)
            else:
                out_mv[n] = edge_val

# =============================================================================
# Multi-Channel Kernels with Per-Channel Border Modes and Feathering
# =============================================================================
cdef inline void _corner_multichannel_per_ch_border_feathered_kernel(
    const f64[:, ::1] corners_mv,
    const f64[:, :, ::1] coords_mv,
    f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv,
    const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Multi-channel with same coords but per-channel border modes and feathering."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = coords_mv[h, w, 0]
            u_y = coords_mv[h, w, 1]
            
            for ch in range(C):
                border_res = process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch])
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_const_mv[ch]
                else:
                    edge_val = _bilinear_interp_multichannel(corners_mv, 
                                                            border_res.u_x_final, 
                                                            border_res.u_y_final, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_const_mv[ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val

cdef inline void _corner_multichannel_flat_per_ch_border_feathered_kernel(
    const f64[:, ::1] corners_mv,
    const f64[:, ::1] coords_mv,
    f64[:, ::1] out_mv,
    const f64[::1] border_const_mv,
    const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,
    f64 border_feathering,
    Py_ssize_t N, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Flat multi-channel with per-channel border modes and feathering."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = coords_mv[n, 0]
        u_y = coords_mv[n, 1]
        
        for ch in range(C):
            border_res = process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch])
            
            if border_res.use_border_directly:
                out_mv[n, ch] = border_const_mv[ch]
            else:
                edge_val = _bilinear_interp_multichannel(corners_mv, 
                                                        border_res.u_x_final, 
                                                        border_res.u_y_final, ch)
                if border_res.blend_factor > 0.0:
                    border_val = border_const_mv[ch]
                    out_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[n, ch] = edge_val

# =============================================================================
# Multi-Channel with Per-Channel Coordinates
# =============================================================================
cdef inline void _corner_multichannel_per_ch_coords_feathered_kernel(
    const f64[:, ::1] corners_mv,
    const f64[:, :, :, ::1] coords_mv,
    f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv,
    const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Multi-channel with per-channel coords and per-channel border modes."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            for ch in range(C):
                u_x = coords_mv[ch, h, w, 0]
                u_y = coords_mv[ch, h, w, 1]
                
                border_res = process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch])
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_const_mv[ch]
                else:
                    edge_val = _bilinear_interp_multichannel(corners_mv, 
                                                            border_res.u_x_final, 
                                                            border_res.u_y_final, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_const_mv[ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val

cdef inline void _corner_multichannel_flat_per_ch_coords_feathered_kernel(
    const f64[:, ::1] corners_mv,
    const f64[:, :, ::1] coords_mv,
    f64[:, ::1] out_mv,
    const f64[::1] border_const_mv,
    const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,
    f64 border_feathering,
    Py_ssize_t N, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Flat multi-channel with per-channel coords and per-channel border modes."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for ch in range(C):
            u_x = coords_mv[ch, n, 0]
            u_y = coords_mv[ch, n, 1]
            
            border_res = process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch])
            
            if border_res.use_border_directly:
                out_mv[n, ch] = border_const_mv[ch]
            else:
                edge_val = _bilinear_interp_multichannel(corners_mv, 
                                                        border_res.u_x_final, 
                                                        border_res.u_y_final, ch)
                if border_res.blend_factor > 0.0:
                    border_val = border_const_mv[ch]
                    out_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[n, ch] = edge_val

# =============================================================================
# Public API - Single Channel
# =============================================================================
def lerp_from_corners_1ch_feathered(
    f64 tl,
    f64 tr,
    f64 bl,
    f64 br,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,
    int num_threads=-1,
):
    """
    Fast parallel single-channel bilinear interpolation from corners with feathering.
    
    Args:
        tl, tr, bl, br: Corner values
        coords: Coordinate grid, shape (H, W, 2)
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
        border_feathering: Feathering width (0.0 = hard edge)
        distance_mode: Distance metric for 2D border computation
        num_threads: Thread count (-1=auto, 0=serial, >0=specific)
    
    Returns:
        Interpolated values, shape (H, W)
    """
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    
    cdef int n_threads = num_threads
    if n_threads < 0:
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
        _corner_1ch_feathered_kernel(
            tl, tr, bl, br, coords_mv, out_mv,
            border_constant, border_feathering,
            H, W, border_mode, distance_mode, n_threads
        )
    
    return out

def lerp_from_corners_1ch_flat_feathered(
    f64 tl,
    f64 tr,
    f64 bl,
    f64 br,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,
    int num_threads=-1,
):
    """Fast parallel single-channel flat coordinates with feathering."""
    cdef Py_ssize_t N = coords.shape[0]
    
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    
    cdef int n_threads = num_threads
    if n_threads < 0:
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
        _corner_1ch_flat_feathered_kernel(
            tl, tr, bl, br, coords_mv, out_mv,
            border_constant, border_feathering,
            N, border_mode, distance_mode, n_threads
        )
    
    return out

# =============================================================================
# Public API - Multi-Channel (Per-Channel Border Modes)
# =============================================================================
def lerp_from_corners_multichannel_per_ch_border_feathered(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Multi-channel bilinear interpolation with per-channel border modes and feathering.
    
    Args:
        corners: Corner values, shape (4, C)
        coords: Coordinate grid, shape (H, W, 2) - same for all channels
        border_modes: Border mode for each channel, shape (C,)
        border_constants: Border constant for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric for 2D border computation
            - None: Use ALPHA_MAX (default)
            - int: Same for all channels
            - array: Per-channel distance modes, shape (C,)
        num_threads: Thread count (-1=auto)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t C = corners.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, C)")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")
    if border_constants.shape[0] != C:
        raise ValueError(f"border_constants must have length {C}")
    
    # Handle distance_mode
    cdef np.ndarray[i32, ndim=1] dm_arr
    if distance_mode is None:
        dm_arr = np.full(C, ALPHA_MAX, dtype=np.int32)
    elif isinstance(distance_mode, (int, np.integer)):
        dm_arr = np.full(C, int(distance_mode), dtype=np.int32)
    else:
        dm_arr = np.ascontiguousarray(distance_mode, dtype=np.int32)
        if dm_arr.shape[0] != C:
            raise ValueError(f"distance_mode array must have length {C}")
    
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)
    
    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, :, ::1] coords_mv = coords
    cdef i32[::1] bm_mv = border_modes
    cdef f64[::1] bc_mv = border_constants
    cdef i32[::1] dm_mv = dm_arr
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    with nogil:
        _corner_multichannel_per_ch_border_feathered_kernel(
            corners_mv, coords_mv, out_mv, bc_mv, bm_mv, dm_mv,
            border_feathering, H, W, C, n_threads
        )
    
    return out

def lerp_from_corners_multichannel_flat_per_ch_border_feathered(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """Flat multi-channel with per-channel border modes and feathering."""
    cdef Py_ssize_t C = corners.shape[1]
    cdef Py_ssize_t N = coords.shape[0]
    
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, C)")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")
    if border_constants.shape[0] != C:
        raise ValueError(f"border_constants must have length {C}")
    
    # Handle distance_mode
    cdef np.ndarray[i32, ndim=1] dm_arr
    if distance_mode is None:
        dm_arr = np.full(C, ALPHA_MAX, dtype=np.int32)
    elif isinstance(distance_mode, (int, np.integer)):
        dm_arr = np.full(C, int(distance_mode), dtype=np.int32)
    else:
        dm_arr = np.ascontiguousarray(distance_mode, dtype=np.int32)
        if dm_arr.shape[0] != C:
            raise ValueError(f"distance_mode array must have length {C}")
    
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)
    
    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, ::1] coords_mv = coords
    cdef i32[::1] bm_mv = border_modes
    cdef f64[::1] bc_mv = border_constants
    cdef i32[::1] dm_mv = dm_arr
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    with nogil:
        _corner_multichannel_flat_per_ch_border_feathered_kernel(
            corners_mv, coords_mv, out_mv, bc_mv, bm_mv, dm_mv,
            border_feathering, N, C, n_threads
        )
    
    return out

# =============================================================================
# Public API - Multi-Channel with Per-Channel Coordinates
# =============================================================================
def lerp_from_corners_multichannel_per_ch_coords_feathered(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=4] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Multi-channel bilinear interpolation with per-channel coordinates.
    
    Args:
        corners: Corner values, shape (4, C)
        coords: Per-channel coordinate grids, shape (C, H, W, 2)
        border_modes: Border mode for each channel, shape (C,)
        border_constants: Border constant for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric for 2D border computation
            - None: Use ALPHA_MAX (default)
            - int: Same for all channels
            - array: Per-channel distance modes, shape (C,)
        num_threads: Thread count (-1=auto)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t C = corners.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t H = coords.shape[1]
    cdef Py_ssize_t W = coords.shape[2]
    
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, C)")
    if coords.shape[3] != 2:
        raise ValueError("coords must have shape (C, H, W, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match corners ({C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")
    if border_constants.shape[0] != C:
        raise ValueError(f"border_constants must have length {C}")
    
    # Handle distance_mode
    cdef np.ndarray[i32, ndim=1] dm_arr
    if distance_mode is None:
        dm_arr = np.full(C, ALPHA_MAX, dtype=np.int32)
    elif isinstance(distance_mode, (int, np.integer)):
        dm_arr = np.full(C, int(distance_mode), dtype=np.int32)
    else:
        dm_arr = np.ascontiguousarray(distance_mode, dtype=np.int32)
        if dm_arr.shape[0] != C:
            raise ValueError(f"distance_mode array must have length {C}")
    
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)
    
    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, :, :, ::1] coords_mv = coords
    cdef i32[::1] bm_mv = border_modes
    cdef f64[::1] bc_mv = border_constants
    cdef i32[::1] dm_mv = dm_arr
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    with nogil:
        _corner_multichannel_per_ch_coords_feathered_kernel(
            corners_mv, coords_mv, out_mv, bc_mv, bm_mv, dm_mv,
            border_feathering, H, W, C, n_threads
        )
    
    return out

def lerp_from_corners_multichannel_flat_per_ch_coords_feathered(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """Flat multi-channel with per-channel coordinates."""
    cdef Py_ssize_t C = corners.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t N = coords.shape[1]
    
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, C)")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (C, N, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match corners ({C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")
    if border_constants.shape[0] != C:
        raise ValueError(f"border_constants must have length {C}")
    
    # Handle distance_mode
    cdef np.ndarray[i32, ndim=1] dm_arr
    if distance_mode is None:
        dm_arr = np.full(C, ALPHA_MAX, dtype=np.int32)
    elif isinstance(distance_mode, (int, np.integer)):
        dm_arr = np.full(C, int(distance_mode), dtype=np.int32)
    else:
        dm_arr = np.ascontiguousarray(distance_mode, dtype=np.int32)
        if dm_arr.shape[0] != C:
            raise ValueError(f"distance_mode array must have length {C}")
    
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)
    
    cdef f64[:, ::1] corners_mv = corners
    cdef f64[:, :, ::1] coords_mv = coords
    cdef i32[::1] bm_mv = border_modes
    cdef f64[::1] bc_mv = border_constants
    cdef i32[::1] dm_mv = dm_arr
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    with nogil:
        _corner_multichannel_flat_per_ch_coords_feathered_kernel(
            corners_mv, coords_mv, out_mv, bc_mv, bm_mv, dm_mv,
            border_feathering, N, C, n_threads
        )
    
    return out

# =============================================================================
# Smart Dispatcher
# =============================================================================
DEF MIN_PARALLEL_SIZE = 10000

def lerp_from_corners_full_feathered(
    corners,
    coords,
    object border_mode=None,
    object border_constant=None,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Smart dispatcher for fast corner interpolation with full feature support.
    
    Automatically selects the appropriate kernel based on input shapes.
    Supports all combinations of:
    - Single vs multi-channel
    - Grid vs flat coordinates
    - Same vs per-channel coordinates
    - Global vs per-channel border modes
    - Global vs per-channel distance modes
    - Feathering support
    
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
        border_mode: Border mode(s):
            - int: Same for all channels
            - list/array: Per-channel modes, shape (C,)
        border_constant: Border constant(s):
            - float: Same for all channels
            - list/array: Per-channel constants, shape (C,)
        border_feathering: Feathering width (0.0 = no feathering)
        distance_mode: Distance metric(s) for 2D border computation:
            - None: Use ALPHA_MAX (default)
            - int: Same for all channels
            - list/array: Per-channel modes, shape (C,)
        num_threads: Thread count (-1=auto, 0=serial, >0=specific)
    
    Returns:
        Interpolated values
    """
    cdef Py_ssize_t total_size
    cdef int use_threads = num_threads
    cdef Py_ssize_t C
    cdef np.ndarray[f64, ndim=1] bc_arr
    cdef f64 bc_scalar
    cdef np.ndarray[i32, ndim=1] bm_arr
    cdef np.ndarray[i32, ndim=1] dm_arr
    cdef int bm_scalar
    cdef int dm_scalar
    cdef int coords_ndim = coords.ndim
    
    # Convert corners
    cdef bint single_channel = False
    cdef np.ndarray corners_arr
    
    if isinstance(corners, (list, tuple)) and len(corners) == 4:
        if all(isinstance(c, (int, float)) for c in corners):
            single_channel = True
            corners_arr = np.array(corners, dtype=np.float64)
        else:
            corners_arr = np.asarray(corners, dtype=np.float64)
    else:
        corners_arr = np.asarray(corners, dtype=np.float64)
    
    if corners_arr.ndim == 1:
        if corners_arr.shape[0] != 4:
            raise ValueError("For single channel, corners must have 4 values")
        single_channel = True
    elif corners_arr.ndim == 2:
        if corners_arr.shape[0] != 4:
            raise ValueError("corners must have shape (4,) or (4, C)")
        single_channel = False
    else:
        raise ValueError("corners must be 1D or 2D array")
    
    # Convert coords
    if not isinstance(coords, np.ndarray):
        coords = np.asarray(coords, dtype=np.float64)
    if coords.dtype != np.float64:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    elif not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    coords_arr = coords
    
    # Calculate size for parallelization
    if coords_ndim == 3:
        if coords_arr.shape[2] == 2:  # (H, W, 2)
            total_size = coords_arr.shape[0] * coords_arr.shape[1]
        else:  # (C, N, 2)
            total_size = coords_arr.shape[1]
    elif coords_ndim == 4:  # (C, H, W, 2)
        total_size = coords_arr.shape[1] * coords_arr.shape[2]
    else:  # (N, 2)
        total_size = coords_arr.shape[0]
    
    if total_size < MIN_PARALLEL_SIZE and num_threads < 0:
        use_threads = 1
    
    # Handle border mode
    if border_mode is None:
        border_mode = BORDER_CLAMP
    
    # Handle border constant
    if border_constant is None:
        border_constant = 0.0
    
    # Handle distance mode default
    if distance_mode is None:
        distance_mode = ALPHA_MAX
    
    # Single channel
    if single_channel:
        bc_scalar = float(border_constant) if not isinstance(border_constant, np.ndarray) else float(border_constant[0])
        bm_scalar = int(border_mode) if not isinstance(border_mode, np.ndarray) else int(border_mode[0])
        dm_scalar = int(distance_mode) if not isinstance(distance_mode, np.ndarray) else int(distance_mode[0])
        
        if coords_ndim == 3 and coords_arr.shape[2] == 2:
            return lerp_from_corners_1ch_feathered(
                corners_arr[0], corners_arr[1], corners_arr[2], corners_arr[3],
                coords_arr, bm_scalar, bc_scalar, border_feathering, dm_scalar, use_threads
            )
        elif coords_ndim == 2 and coords_arr.shape[1] == 2:
            return lerp_from_corners_1ch_flat_feathered(
                corners_arr[0], corners_arr[1], corners_arr[2], corners_arr[3],
                coords_arr, bm_scalar, bc_scalar, border_feathering, dm_scalar, use_threads
            )
        else:
            raise ValueError(f"Invalid coords shape for single channel: {coords_arr.shape}")
    
    # Multi-channel
    else:
        C = corners_arr.shape[1]
        
        # Convert border_mode to array
        if isinstance(border_mode, (int, np.integer)):
            bm_arr = np.full(C, int(border_mode), dtype=np.int32)
        else:
            bm_arr = np.ascontiguousarray(border_mode, dtype=np.int32)
            if bm_arr.shape[0] != C:
                raise ValueError(f"border_mode must have length {C}")
        
        # Convert border_constant to array
        if isinstance(border_constant, (int, float, np.floating)):
            bc_arr = np.full(C, float(border_constant), dtype=np.float64)
        else:
            bc_arr = np.ascontiguousarray(border_constant, dtype=np.float64)
            if bc_arr.shape[0] != C:
                raise ValueError(f"border_constant must have length {C}")
        
        # Convert distance_mode to array
        if isinstance(distance_mode, (int, np.integer)):
            dm_arr = np.full(C, int(distance_mode), dtype=np.int32)
        else:
            dm_arr = np.ascontiguousarray(distance_mode, dtype=np.int32)
            if dm_arr.shape[0] != C:
                raise ValueError(f"distance_mode must have length {C}")
        
        # Grid coordinates (H, W, 2) - same for all channels
        if coords_ndim == 3 and coords_arr.shape[2] == 2:
            return lerp_from_corners_multichannel_per_ch_border_feathered(
                corners_arr, coords_arr, bm_arr, bc_arr,
                border_feathering, dm_arr, use_threads
            )
        
        # Flat coordinates (N, 2) - same for all channels
        elif coords_ndim == 2 and coords_arr.shape[1] == 2:
            return lerp_from_corners_multichannel_flat_per_ch_border_feathered(
                corners_arr, coords_arr, bm_arr, bc_arr,
                border_feathering, dm_arr, use_threads
            )
        
        # Per-channel grid coordinates (C, H, W, 2)
        elif coords_ndim == 4 and coords_arr.shape[3] == 2:
            if coords_arr.shape[0] != C:
                raise ValueError(f"coords channels ({coords_arr.shape[0]}) must match corners ({C})")
            
            return lerp_from_corners_multichannel_per_ch_coords_feathered(
                corners_arr, coords_arr, bm_arr, bc_arr,
                border_feathering, dm_arr, use_threads
            )
        
        # Per-channel flat coordinates (C, N, 2)
        elif coords_ndim == 3 and coords_arr.shape[2] == 2 and coords_arr.shape[0] == C:
            return lerp_from_corners_multichannel_flat_per_ch_coords_feathered(
                corners_arr, coords_arr, bm_arr, bc_arr,
                border_feathering, dm_arr, use_threads
            )
        
        else:
            raise ValueError(f"Invalid coords shape for multi-channel: {coords_arr.shape}")
    
    #raise ValueError(f"Unsupported corners dimensions: {corners_arr.ndim}")

# =============================================================================
# Legacy compatibility functions
# =============================================================================
def lerp_from_corners_fast(
    corners,
    coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
    int num_threads=-1,
):
    """Legacy compatibility wrapper (no feathering)."""
    return lerp_from_corners_full_feathered(
        corners, coords, border_mode, border_constant,
        0.0, ALPHA_MAX, num_threads
    )

# =============================================================================
# Export border mode constants
# =============================================================================
BORDER_REPEAT = 0
BORDER_MIRROR = 1
BORDER_CONSTANT = 2
BORDER_CLAMP = 3
BORDER_OVERFLOW = 4

# Export distance mode constants
DIST_MAX_NORM = MAX_NORM
DIST_MANHATTAN = MANHATTAN
DIST_SCALED_MANHATTAN = SCALED_MANHATTAN
DIST_ALPHA_MAX = ALPHA_MAX
DIST_ALPHA_MAX_SIMPLE = ALPHA_MAX_SIMPLE
DIST_TAYLOR = TAYLOR
DIST_EUCLIDEAN = ALPHA_MAX
DIST_WEIGHTED_MINMAX = WEIGHTED_MINMAX