# interp_hue_corners_array_border.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

"""
Hue corner interpolation with array-based border blending.

Provides feathered corner interpolation that blends against per-pixel border
values using shortest-path hue blending. Supports flat and grid coordinates.
"""

import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport floor

from .interp_hue_utils cimport (
    f64,
    i32,
    HUE_SHORTEST,
    wrap_hue,
    lerp_hue_single,
)
from ..interp_utils cimport (
    BorderResult,
    MAX_NORM,
    MANHATTAN,
    SCALED_MANHATTAN,
    ALPHA_MAX,
    ALPHA_MAX_SIMPLE,
    TAYLOR,
    EUCLIDEAN,
    WEIGHTED_MINMAX,
    clamp_01,
    process_border_2d,
)


# =============================================================================
# Hue Bilinear Interpolation Helper Functions
# =============================================================================
cdef inline f64 _hue_bilinear_interp_1ch(
    f64 tl, f64 tr, f64 bl, f64 br,
    f64 u_x, f64 u_y
) noexcept nogil:
    """Hue-aware bilinear interpolation using shortest paths."""
    cdef f64 left = lerp_hue_single(tl, bl, u_y, HUE_SHORTEST)
    cdef f64 right = lerp_hue_single(tr, br, u_y, HUE_SHORTEST)
    return lerp_hue_single(left, right, u_x, HUE_SHORTEST)


cdef inline f64 _hue_bilinear_interp_multichannel(
    const f64[:, ::1] corners_mv,
    f64 u_x, f64 u_y,
    Py_ssize_t ch
) noexcept nogil:
    """Hue-aware bilinear interpolation for multi-channel."""
    cdef f64 tl = corners_mv[0, ch]
    cdef f64 tr = corners_mv[1, ch]
    cdef f64 bl = corners_mv[2, ch]
    cdef f64 br = corners_mv[3, ch]
    return _hue_bilinear_interp_1ch(tl, tr, bl, br, u_x, u_y)


# =============================================================================
# Kernels with Array Border
# =============================================================================
cdef inline void _hue_corner_1ch_array_border_kernel(
    const f64[::1] corners_mv,       # shape (4,)
    const f64[:, :, ::1] coords_mv,  # shape (H, W, 2)
    f64[:, ::1] out_mv,              # shape (H, W)
    const f64[:, ::1] border_array_mv,  # shape (H, W)
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Single-channel hue corner with array border."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y
    cdef f64 interp_val, border_val_wrapped, final_val
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = clamp_01(coords_mv[h, w, 0])
            u_y = clamp_01(coords_mv[h, w, 1])
            
            interp_val = _hue_bilinear_interp_1ch(
                corners_mv[0], corners_mv[1],
                corners_mv[2], corners_mv[3],
                u_x, u_y
            )
            
            if coords_mv[h, w, 0] >= 0.0 and coords_mv[h, w, 0] <= 1.0 and \
               coords_mv[h, w, 1] >= 0.0 and coords_mv[h, w, 1] <= 1.0:
                out_mv[h, w] = wrap_hue(interp_val)
            else:
                border_val_wrapped = wrap_hue(border_array_mv[h, w])
                final_val = lerp_hue_single(
                    interp_val, border_val_wrapped,
                    0.5, HUE_SHORTEST
                )
                out_mv[h, w] = wrap_hue(final_val)


cdef inline void _hue_corner_1ch_flat_array_border_kernel(
    const f64[::1] corners_mv,      # shape (4,)
    const f64[:, ::1] coords_mv,    # shape (N, 2)
    f64[::1] out_mv,                # shape (N,)
    const f64[::1] border_array_mv, # shape (N,)
    f64 border_feathering,
    Py_ssize_t N,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Single-channel hue corner with array border (flat coords)."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y
    cdef f64 interp_val, border_val_wrapped, final_val
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = clamp_01(coords_mv[n, 0])
        u_y = clamp_01(coords_mv[n, 1])
        
        interp_val = _hue_bilinear_interp_1ch(
            corners_mv[0], corners_mv[1],
            corners_mv[2], corners_mv[3],
            u_x, u_y
        )
        
        if coords_mv[n, 0] >= 0.0 and coords_mv[n, 0] <= 1.0 and \
           coords_mv[n, 1] >= 0.0 and coords_mv[n, 1] <= 1.0:
            out_mv[n] = wrap_hue(interp_val)
        else:
            border_val_wrapped = wrap_hue(border_array_mv[n])
            final_val = lerp_hue_single(
                interp_val, border_val_wrapped,
                0.5, HUE_SHORTEST
            )
            out_mv[n] = wrap_hue(final_val)


cdef inline void _hue_corner_multichannel_array_border_kernel(
    const f64[:, ::1] corners_mv,    # shape (4, C)
    const f64[:, :, ::1] coords_mv,  # shape (H, W, 2)
    f64[:, :, ::1] out_mv,           # shape (H, W, C)
    const f64[:, :, ::1] border_array_mv,  # shape (H, W, C)
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t C,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Multi-channel hue corner with array border."""
    cdef Py_ssize_t h, w, c
    cdef f64 u_x, u_y
    cdef f64 interp_val, border_val_wrapped, final_val
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = clamp_01(coords_mv[h, w, 0])
            u_y = clamp_01(coords_mv[h, w, 1])
            
            for c in range(C):
                interp_val = _hue_bilinear_interp_multichannel(corners_mv, u_x, u_y, c)
                
                if coords_mv[h, w, 0] >= 0.0 and coords_mv[h, w, 0] <= 1.0 and \
                   coords_mv[h, w, 1] >= 0.0 and coords_mv[h, w, 1] <= 1.0:
                    out_mv[h, w, c] = wrap_hue(interp_val)
                else:
                    border_val_wrapped = wrap_hue(border_array_mv[h, w, c])
                    final_val = lerp_hue_single(
                        interp_val, border_val_wrapped,
                        0.5, HUE_SHORTEST
                    )
                    out_mv[h, w, c] = wrap_hue(final_val)


cdef inline void _hue_corner_multichannel_flat_array_border_kernel(
    const f64[:, ::1] corners_mv,    # shape (4, C)
    const f64[:, ::1] coords_mv,     # shape (N, 2)
    f64[:, ::1] out_mv,              # shape (N, C)
    const f64[:, ::1] border_array_mv,  # shape (N, C)
    f64 border_feathering,
    Py_ssize_t N, Py_ssize_t C,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Multi-channel hue corner with array border (flat coords)."""
    cdef Py_ssize_t n, c
    cdef f64 u_x, u_y
    cdef f64 interp_val, border_val_wrapped, final_val
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = clamp_01(coords_mv[n, 0])
        u_y = clamp_01(coords_mv[n, 1])
        
        for c in range(C):
            interp_val = _hue_bilinear_interp_multichannel(corners_mv, u_x, u_y, c)
            
            if coords_mv[n, 0] >= 0.0 and coords_mv[n, 0] <= 1.0 and \
               coords_mv[n, 1] >= 0.0 and coords_mv[n, 1] <= 1.0:
                out_mv[n, c] = wrap_hue(interp_val)
            else:
                border_val_wrapped = wrap_hue(border_array_mv[n, c])
                final_val = lerp_hue_single(
                    interp_val, border_val_wrapped,
                    0.5, HUE_SHORTEST
                )
                out_mv[n, c] = wrap_hue(final_val)


# =============================================================================
# Public APIs
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_array_border(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    f64 border_feathering=0.0,
    object border_mode='constant',
    i32 distance_mode=7,
    int num_threads=1,
):
    """Single-channel hue corner with array border on grid."""
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
    
    cdef np.ndarray[f64, ndim=2] result = np.empty((H, W), dtype=np.float64)
    cdef int bmode = 2 if isinstance(border_mode, str) and border_mode.lower() == 'constant' else 0
    
    _hue_corner_1ch_array_border_kernel(
        corners, coords, result, border_array,
        border_feathering, H, W, bmode, num_threads,
        distance_mode
    )
    
    return result


cpdef np.ndarray[f64, ndim=1] hue_lerp_from_corners_flat_array_border(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    f64 border_feathering=0.0,
    object border_mode='constant',
    i32 distance_mode=7,
    int num_threads=1,
):
    """Single-channel hue corner with array border (flat coords)."""
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
    
    cdef np.ndarray[f64, ndim=1] result = np.empty(N, dtype=np.float64)
    cdef int bmode = 2 if isinstance(border_mode, str) and border_mode.lower() == 'constant' else 0
    
    _hue_corner_1ch_flat_array_border_kernel(
        corners, coords, result, border_array,
        border_feathering, N, bmode, num_threads,
        distance_mode
    )
    
    return result


cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] border_array,
    f64 border_feathering=0.0,
    object border_mode='constant',
    i32 distance_mode=7,
    int num_threads=1,
):
    """Multi-channel hue corner with array border on grid."""
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
    
    cdef np.ndarray[f64, ndim=3] result = np.empty((H, W, C), dtype=np.float64)
    cdef int bmode = 2 if isinstance(border_mode, str) and border_mode.lower() == 'constant' else 0
    
    _hue_corner_multichannel_array_border_kernel(
        corners, coords, result, border_array,
        border_feathering, H, W, C, bmode, num_threads,
        distance_mode
    )
    
    return result


cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_multichannel_flat_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=2] border_array,
    f64 border_feathering=0.0,
    object border_mode='constant',
    i32 distance_mode=7,
    int num_threads=1,
):
    """Multi-channel hue corner with array border (flat coords)."""
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
    
    cdef np.ndarray[f64, ndim=2] result = np.empty((N, C), dtype=np.float64)
    cdef int bmode = 2 if isinstance(border_mode, str) and border_mode.lower() == 'constant' else 0
    
    _hue_corner_multichannel_flat_array_border_kernel(
        corners, coords, result, border_array,
        border_feathering, N, C, bmode, num_threads,
        distance_mode
    )
    
    return result
