# interp_hue_corners_array_border.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from cython.parallel cimport prange

from .interp_hue_utils cimport (
    f64,
    i32,
    HUE_SHORTEST,
    wrap_hue,
    lerp_hue_single,
)
from ..interp_utils cimport (
    BorderResult,
    EUCLIDEAN,
    ALPHA_MAX,
    process_border_2d,
)
from ..border_handling cimport (
    BORDER_CONSTANT,
)


# =============================================================================
# Hue Bilinear Interpolation (WITH PER-AXIS MODES)
# =============================================================================
cdef inline f64 _hue_bilinear_interp_1ch(
    f64 tl, f64 tr, f64 bl, f64 br,
    f64 u_x, f64 u_y,
    i32 mode_x, i32 mode_y
) noexcept nogil:
    """Hue-aware bilinear interpolation with per-axis modes."""
    cdef f64 left = lerp_hue_single(tl, bl, u_y, mode_y)
    cdef f64 right = lerp_hue_single(tr, br, u_y, mode_y)
    return lerp_hue_single(left, right, u_x, mode_x)


cdef inline f64 _hue_bilinear_interp_multichannel(
    const f64[:, ::1] corners_mv,
    f64 u_x, f64 u_y,
    Py_ssize_t ch,
    i32 mode_x, i32 mode_y
) noexcept nogil:
    """Hue-aware bilinear interpolation for multi-channel with per-axis modes."""
    cdef f64 tl = corners_mv[0, ch]
    cdef f64 tr = corners_mv[1, ch]
    cdef f64 bl = corners_mv[2, ch]
    cdef f64 br = corners_mv[3, ch]
    return _hue_bilinear_interp_1ch(tl, tr, bl, br, u_x, u_y, mode_x, mode_y)


# =============================================================================
# Kernels with Array Border
# =============================================================================
cdef inline void _hue_corner_1ch_array_border_kernel(
    const f64[::1] corners_mv,          # shape (4,)
    const f64[:, :, ::1] coords_mv,     # shape (H, W, 2)
    f64[:, ::1] out_mv,                 # shape (H, W)
    const f64[:, ::1] border_array_mv,  # shape (H, W)
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W,
    int border_mode, int num_threads,
    i32 mode_x, i32 mode_y,
    i32 feather_hue_mode,  # NEW: hue mode for feathering blend
    i32 distance_mode,
) noexcept nogil:
    """Single-channel hue corner with array border."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y
    cdef f64 interp_val, border_val_wrapped, final_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = coords_mv[h, w, 0]
            u_y = coords_mv[h, w, 1]
            
            border_res = process_border_2d(u_x, u_y, border_mode,
                                           border_feathering, distance_mode)
            
            if border_res.use_border_directly:
                out_mv[h, w] = wrap_hue(border_array_mv[h, w])
            else:
                interp_val = _hue_bilinear_interp_1ch(
                    corners_mv[0], corners_mv[1],
                    corners_mv[2], corners_mv[3],
                    border_res.u_x_final, border_res.u_y_final,
                    mode_x, mode_y
                )
                
                if border_res.blend_factor > 0.0:
                    border_val_wrapped = wrap_hue(border_array_mv[h, w])
                    final_val = lerp_hue_single(
                        interp_val, border_val_wrapped,
                        border_res.blend_factor, feather_hue_mode  # FIXED
                    )
                    out_mv[h, w] = final_val
                else:
                    out_mv[h, w] = interp_val


cdef inline void _hue_corner_1ch_flat_array_border_kernel(
    const f64[::1] corners_mv,       # shape (4,)
    const f64[:, ::1] coords_mv,     # shape (N, 2)
    f64[::1] out_mv,                 # shape (N,)
    const f64[::1] border_array_mv,  # shape (N,)
    f64 border_feathering,
    Py_ssize_t N,
    int border_mode, int num_threads,
    i32 mode_x, i32 mode_y,
    i32 feather_hue_mode,  # NEW: hue mode for feathering blend
    i32 distance_mode,
) noexcept nogil:
    """Single-channel hue corner with array border (flat coords)."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y
    cdef f64 interp_val, border_val_wrapped, final_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = coords_mv[n, 0]
        u_y = coords_mv[n, 1]
        
        border_res = process_border_2d(u_x, u_y, border_mode,
                                       border_feathering, distance_mode)
        
        if border_res.use_border_directly:
            out_mv[n] = wrap_hue(border_array_mv[n])
        else:
            interp_val = _hue_bilinear_interp_1ch(
                corners_mv[0], corners_mv[1],
                corners_mv[2], corners_mv[3],
                border_res.u_x_final, border_res.u_y_final,
                mode_x, mode_y
            )
            
            if border_res.blend_factor > 0.0:
                border_val_wrapped = wrap_hue(border_array_mv[n])
                final_val = lerp_hue_single(
                    interp_val, border_val_wrapped,
                    border_res.blend_factor, feather_hue_mode  # FIXED
                )
                out_mv[n] = final_val
            else:
                out_mv[n] = interp_val


cdef inline void _hue_corner_multichannel_array_border_kernel(
    const f64[:, ::1] corners_mv,          # shape (4, C)
    const f64[:, :, ::1] coords_mv,        # shape (H, W, 2)
    f64[:, :, ::1] out_mv,                 # shape (H, W, C)
    const f64[:, :, ::1] border_array_mv,  # shape (H, W, C)
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t C,
    int border_mode, int num_threads,
    i32 mode_x, i32 mode_y,
    i32 feather_hue_mode,  # NEW: hue mode for feathering blend
    i32 distance_mode,
) noexcept nogil:
    """Multi-channel hue corner with array border."""
    cdef Py_ssize_t h, w, c
    cdef f64 u_x, u_y
    cdef f64 interp_val, border_val_wrapped, final_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = coords_mv[h, w, 0]
            u_y = coords_mv[h, w, 1]
            
            border_res = process_border_2d(u_x, u_y, border_mode,
                                           border_feathering, distance_mode)
            
            if border_res.use_border_directly:
                for c in range(C):
                    out_mv[h, w, c] = wrap_hue(border_array_mv[h, w, c])
            else:
                for c in range(C):
                    interp_val = _hue_bilinear_interp_multichannel(
                        corners_mv,
                        border_res.u_x_final, border_res.u_y_final,
                        c, mode_x, mode_y
                    )
                    
                    if border_res.blend_factor > 0.0:
                        border_val_wrapped = wrap_hue(border_array_mv[h, w, c])
                        final_val = lerp_hue_single(
                            interp_val, border_val_wrapped,
                            border_res.blend_factor, feather_hue_mode  # FIXED
                        )
                        out_mv[h, w, c] = final_val
                    else:
                        out_mv[h, w, c] = interp_val


cdef inline void _hue_corner_multichannel_per_ch_modes_array_border_kernel(
    const f64[:, ::1] corners_mv,          # shape (4, C)
    const f64[:, :, ::1] coords_mv,        # shape (H, W, 2)
    f64[:, :, ::1] out_mv,                 # shape (H, W, C)
    const f64[:, :, ::1] border_array_mv,  # shape (H, W, C)
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t C,
    int border_mode, int num_threads,
    const i32[::1] modes_x,                # shape (C,)
    const i32[::1] modes_y,                # shape (C,)
    i32 feather_hue_mode,  # NEW: hue mode for feathering blend
    i32 distance_mode,
) noexcept nogil:
    """Multi-channel hue corner with per-channel modes and array border."""
    cdef Py_ssize_t h, w, c
    cdef f64 u_x, u_y
    cdef f64 interp_val, border_val_wrapped, final_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = coords_mv[h, w, 0]
            u_y = coords_mv[h, w, 1]
            
            border_res = process_border_2d(u_x, u_y, border_mode,
                                           border_feathering, distance_mode)
            
            if border_res.use_border_directly:
                for c in range(C):
                    out_mv[h, w, c] = wrap_hue(border_array_mv[h, w, c])
            else:
                for c in range(C):
                    interp_val = _hue_bilinear_interp_multichannel(
                        corners_mv,
                        border_res.u_x_final, border_res.u_y_final,
                        c, modes_x[c], modes_y[c]
                    )
                    
                    if border_res.blend_factor > 0.0:
                        border_val_wrapped = wrap_hue(border_array_mv[h, w, c])
                        final_val = lerp_hue_single(
                            interp_val, border_val_wrapped,
                            border_res.blend_factor, feather_hue_mode  # FIXED
                        )
                        out_mv[h, w, c] = final_val
                    else:
                        out_mv[h, w, c] = interp_val


cdef inline void _hue_corner_multichannel_flat_array_border_kernel(
    const f64[:, ::1] corners_mv,       # shape (4, C)
    const f64[:, ::1] coords_mv,        # shape (N, 2)
    f64[:, ::1] out_mv,                 # shape (N, C)
    const f64[:, ::1] border_array_mv,  # shape (N, C)
    f64 border_feathering,
    Py_ssize_t N, Py_ssize_t C,
    int border_mode, int num_threads,
    i32 mode_x, i32 mode_y,
    i32 feather_hue_mode,  # NEW: hue mode for feathering blend
    i32 distance_mode,
) noexcept nogil:
    """Multi-channel hue corner with array border (flat coords)."""
    cdef Py_ssize_t n, c
    cdef f64 u_x, u_y
    cdef f64 interp_val, border_val_wrapped, final_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = coords_mv[n, 0]
        u_y = coords_mv[n, 1]
        
        border_res = process_border_2d(u_x, u_y, border_mode,
                                       border_feathering, distance_mode)
        
        if border_res.use_border_directly:
            for c in range(C):
                out_mv[n, c] = wrap_hue(border_array_mv[n, c])
        else:
            for c in range(C):
                interp_val = _hue_bilinear_interp_multichannel(
                    corners_mv,
                    border_res.u_x_final, border_res.u_y_final,
                    c, mode_x, mode_y
                )
                
                if border_res.blend_factor > 0.0:
                    border_val_wrapped = wrap_hue(border_array_mv[n, c])
                    final_val = lerp_hue_single(
                        interp_val, border_val_wrapped,
                        border_res.blend_factor, feather_hue_mode  # FIXED
                    )
                    out_mv[n, c] = final_val
                else:
                    out_mv[n, c] = interp_val


# =============================================================================
# Public APIs
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_array_border(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    i32 mode_x = HUE_SHORTEST,
    i32 mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CONSTANT,
    f64 border_feathering = 0.0,
    i32 feather_hue_mode = HUE_SHORTEST,  # NEW PARAMETER
    i32 distance_mode = ALPHA_MAX,
    int num_threads = 1,
):
    """
    Single-channel hue corner with array border on grid.
    
    Args:
        corners: 4 corner hue values [tl, tr, bl, br]
        coords: Coordinate grid shape (H, W, 2)
        border_array: Border values grid shape (H, W)
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode (int enum)
        border_feathering: Feathering distance
        feather_hue_mode: Hue interpolation mode for feathering blend (int enum)
        distance_mode: Distance metric for feathering
        num_threads: Number of threads (-1 for auto)
    
    Returns:
        Interpolated hue grid (H, W)
    """
    if corners.shape[0] != 4:
        raise ValueError("corners must have 4 values")
    if coords.shape[2] != 2:
        raise ValueError("coords must be (H, W, 2)")
    if border_array.shape[0] != coords.shape[0] or border_array.shape[1] != coords.shape[1]:
        raise ValueError("border_array shape must match coords grid shape")
    
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1
    
    cdef np.ndarray[f64, ndim=2] result = np.empty((H, W), dtype=np.float64)
    
    _hue_corner_1ch_array_border_kernel(
        corners, coords, result, border_array,
        border_feathering, H, W, border_mode, n_threads,
        mode_x, mode_y, feather_hue_mode, distance_mode
    )
    
    return result


cpdef np.ndarray[f64, ndim=1] hue_lerp_from_corners_flat_array_border(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    i32 mode_x = HUE_SHORTEST,
    i32 mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CONSTANT,
    f64 border_feathering = 0.0,
    i32 feather_hue_mode = HUE_SHORTEST,  # NEW PARAMETER
    i32 distance_mode = ALPHA_MAX,
    int num_threads = 1,
):
    """
    Single-channel hue corner with array border (flat coords).
    
    Args:
        corners: 4 corner hue values [tl, tr, bl, br]
        coords: Flat coordinate array shape (N, 2)
        border_array: Border values array shape (N,)
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode (int enum)
        border_feathering: Feathering distance
        feather_hue_mode: Hue interpolation mode for feathering blend (int enum)
        distance_mode: Distance metric for feathering
        num_threads: Number of threads (-1 for auto)
    
    Returns:
        Interpolated hue values (N,)
    """
    if corners.shape[0] != 4:
        raise ValueError("corners must have 4 values")
    if coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")
    if border_array.shape[0] != coords.shape[0]:
        raise ValueError("border_array length must match coords length")
    
    cdef Py_ssize_t N = coords.shape[0]
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1
    
    cdef np.ndarray[f64, ndim=1] result = np.empty(N, dtype=np.float64)
    
    _hue_corner_1ch_flat_array_border_kernel(
        corners, coords, result, border_array,
        border_feathering, N, border_mode, n_threads,
        mode_x, mode_y, feather_hue_mode, distance_mode
    )
    
    return result


cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] border_array,
    i32 mode_x = HUE_SHORTEST,
    i32 mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CONSTANT,
    f64 border_feathering = 0.0,
    i32 feather_hue_mode = HUE_SHORTEST,  # NEW PARAMETER
    i32 distance_mode = ALPHA_MAX,
    int num_threads = 1,
):
    """
    Multi-channel hue corner with array border on grid.
    
    Args:
        corners: Corner hue values shape (4, C)
        coords: Coordinate grid shape (H, W, 2)
        border_array: Border values grid shape (H, W, C)
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode (int enum)
        border_feathering: Feathering distance
        feather_hue_mode: Hue interpolation mode for feathering blend (int enum)
        distance_mode: Distance metric for feathering
        num_threads: Number of threads (-1 for auto)
    
    Returns:
        Interpolated hue grid (H, W, C)
    """
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, C)")
    if coords.shape[2] != 2:
        raise ValueError("coords must be (H, W, 2)")
    if border_array.shape[0] != coords.shape[0] or border_array.shape[1] != coords.shape[1]:
        raise ValueError("border_array grid shape must match coords")
    if border_array.shape[2] != corners.shape[1]:
        raise ValueError("border_array channels must match corners channels")
    
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    cdef Py_ssize_t C = corners.shape[1]
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1
    
    cdef np.ndarray[f64, ndim=3] result = np.empty((H, W, C), dtype=np.float64)
    
    _hue_corner_multichannel_array_border_kernel(
        corners, coords, result, border_array,
        border_feathering, H, W, C, border_mode, n_threads,
        mode_x, mode_y, feather_hue_mode, distance_mode
    )
    
    return result


cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel_per_ch_modes_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] border_array,
    np.ndarray[i32, ndim=1] modes_x,
    np.ndarray[i32, ndim=1] modes_y,
    int border_mode = BORDER_CONSTANT,
    f64 border_feathering = 0.0,
    i32 feather_hue_mode = HUE_SHORTEST,  # NEW PARAMETER
    i32 distance_mode = ALPHA_MAX,
    int num_threads = 1,
):
    """
    Multi-channel hue corner with per-channel modes and array border.
    
    Args:
        corners: Corner hue values shape (4, C)
        coords: Coordinate grid shape (H, W, 2)
        border_array: Border values grid shape (H, W, C)
        modes_x: Per-channel X interpolation modes shape (C,)
        modes_y: Per-channel Y interpolation modes shape (C,)
        border_mode: Border handling mode (int enum)
        border_feathering: Feathering distance
        feather_hue_mode: Hue interpolation mode for feathering blend (int enum)
        distance_mode: Distance metric for feathering
        num_threads: Number of threads (-1 for auto)
    
    Returns:
        Interpolated hue grid (H, W, C)
    """
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, C)")
    if coords.shape[2] != 2:
        raise ValueError("coords must be (H, W, 2)")
    
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    cdef Py_ssize_t C = corners.shape[1]
    
    if border_array.shape[0] != H or border_array.shape[1] != W:
        raise ValueError("border_array grid shape must match coords")
    if border_array.shape[2] != C:
        raise ValueError("border_array channels must match corners channels")
    if modes_x.shape[0] != C or modes_y.shape[0] != C:
        raise ValueError("modes_x and modes_y must have length C")
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    if not modes_x.flags['C_CONTIGUOUS']:
        modes_x = np.ascontiguousarray(modes_x)
    if not modes_y.flags['C_CONTIGUOUS']:
        modes_y = np.ascontiguousarray(modes_y)
    
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1
    
    cdef np.ndarray[f64, ndim=3] result = np.empty((H, W, C), dtype=np.float64)
    
    _hue_corner_multichannel_per_ch_modes_array_border_kernel(
        corners, coords, result, border_array,
        border_feathering, H, W, C, border_mode, n_threads,
        modes_x, modes_y, feather_hue_mode, distance_mode
    )
    
    return result


cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_multichannel_flat_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=2] border_array,
    i32 mode_x = HUE_SHORTEST,
    i32 mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CONSTANT,
    f64 border_feathering = 0.0,
    i32 feather_hue_mode = HUE_SHORTEST,  # NEW PARAMETER
    i32 distance_mode = ALPHA_MAX,
    int num_threads = 1,
):
    """
    Multi-channel hue corner with array border (flat coords).
    
    Args:
        corners: Corner hue values shape (4, C)
        coords: Flat coordinate array shape (N, 2)
        border_array: Border values array shape (N, C)
        mode_x: Hue interpolation mode for X axis
        mode_y: Hue interpolation mode for Y axis
        border_mode: Border handling mode (int enum)
        border_feathering: Feathering distance
        feather_hue_mode: Hue interpolation mode for feathering blend (int enum)
        distance_mode: Distance metric for feathering
        num_threads: Number of threads (-1 for auto)
    
    Returns:
        Interpolated hue values (N, C)
    """
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, C)")
    if coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")
    if border_array.shape[0] != coords.shape[0]:
        raise ValueError("border_array length must match coords")
    if border_array.shape[1] != corners.shape[1]:
        raise ValueError("border_array channels must match corners")
    
    cdef Py_ssize_t N = coords.shape[0]
    cdef Py_ssize_t C = corners.shape[1]
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1
    
    cdef np.ndarray[f64, ndim=2] result = np.empty((N, C), dtype=np.float64)
    
    _hue_corner_multichannel_flat_array_border_kernel(
        corners, coords, result, border_array,
        border_feathering, N, C, border_mode, n_threads,
        mode_x, mode_y, feather_hue_mode, distance_mode
    )
    
    return result