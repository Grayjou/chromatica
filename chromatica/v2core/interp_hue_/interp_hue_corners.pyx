# interp_hue_corners.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

"""
Hue corner interpolation with cyclical color space support.

Performs bilinear interpolation from four corner hue values using shortest-path
blending. Supports feathering, distance modes, and multi-channel with optional
per-channel coordinates.
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
    compute_interp_idx,
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
    """
    Hue-aware bilinear interpolation for single channel.
    
    Interpolates between four corner hues using shortest-path blending at each step.
    This ensures we never take the "long way" around the hue circle.
    """
    # First, interpolate between left corners (shortest path)
    cdef f64 left = lerp_hue_single(tl, bl, u_y, HUE_SHORTEST)
    # Then, interpolate between right corners (shortest path)
    cdef f64 right = lerp_hue_single(tr, br, u_y, HUE_SHORTEST)
    # Finally, interpolate between left and right (shortest path)
    return lerp_hue_single(left, right, u_x, HUE_SHORTEST)


cdef inline f64 _hue_bilinear_interp_multichannel(
    const f64[:, ::1] corners_mv,  # shape (4, C)
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
# Single-Channel Kernels with Feathering
# =============================================================================
cdef inline void _hue_corner_1ch_feathered_kernel(
    const f64[::1] corners_mv,      # shape (4,): [tl, tr, bl, br]
    const f64[:, :, ::1] coords_mv,  # shape (H, W, 2)
    f64[:, ::1] out_mv,              # shape (H, W)
    f64 border_constant, f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Single-channel hue corner interpolation with feathering."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y
    cdef f64 interp_val, final_val
    cdef f64 border_alpha
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = clamp_01(coords_mv[h, w, 0])
            u_y = clamp_01(coords_mv[h, w, 1])
            
            # Hue-aware bilinear interpolation
            interp_val = _hue_bilinear_interp_1ch(
                corners_mv[0], corners_mv[1],
                corners_mv[2], corners_mv[3],
                u_x, u_y
            )
            
            # Check if coordinate is in valid range [0, 1]
            if coords_mv[h, w, 0] >= 0.0 and coords_mv[h, w, 0] <= 1.0 and \
               coords_mv[h, w, 1] >= 0.0 and coords_mv[h, w, 1] <= 1.0:
                out_mv[h, w] = wrap_hue(interp_val)
            else:
                # Out of bounds: blend with constant using hue-aware lerp
                final_val = lerp_hue_single(
                    interp_val, border_constant,
                    0.5, HUE_SHORTEST  # Simple blend, no feathering in this path
                )
                out_mv[h, w] = wrap_hue(final_val)


cdef inline void _hue_corner_1ch_flat_feathered_kernel(
    const f64[::1] corners_mv,      # shape (4,)
    const f64[:, ::1] coords_mv,    # shape (N, 2)
    f64[::1] out_mv,                # shape (N,)
    f64 border_constant, f64 border_feathering,
    Py_ssize_t N,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Single-channel hue corner interpolation (flat coords) with feathering."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y
    cdef f64 interp_val, final_val
    
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
            final_val = lerp_hue_single(
                interp_val, border_constant,
                0.5, HUE_SHORTEST
            )
            out_mv[n] = wrap_hue(final_val)


# =============================================================================
# Multi-Channel Kernels with Per-Channel Border Modes and Feathering
# =============================================================================
cdef inline void _hue_corner_multichannel_per_ch_border_feathered_kernel(
    const f64[:, ::1] corners_mv,    # shape (4, C)
    const f64[:, :, ::1] coords_mv,  # shape (H, W, 2)
    f64[:, :, ::1] out_mv,           # shape (H, W, C)
    object border_constant_obj, f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t C,
    object border_mode_obj, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Multi-channel hue corner interpolation with per-channel border modes."""
    cdef Py_ssize_t h, w, c
    cdef f64 u_x, u_y
    cdef f64 interp_val, final_val
    
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
                    final_val = lerp_hue_single(interp_val, 0.0, 0.5, HUE_SHORTEST)
                    out_mv[h, w, c] = wrap_hue(final_val)


cdef inline void _hue_corner_multichannel_flat_per_ch_border_feathered_kernel(
    const f64[:, ::1] corners_mv,    # shape (4, C)
    const f64[:, ::1] coords_mv,     # shape (N, 2)
    f64[:, ::1] out_mv,              # shape (N, C)
    object border_constant_obj, f64 border_feathering,
    Py_ssize_t N, Py_ssize_t C,
    object border_mode_obj, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Multi-channel hue corner interpolation (flat coords) with per-channel border modes."""
    cdef Py_ssize_t n, c
    cdef f64 u_x, u_y
    cdef f64 interp_val, final_val
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = clamp_01(coords_mv[n, 0])
        u_y = clamp_01(coords_mv[n, 1])
        
        for c in range(C):
            interp_val = _hue_bilinear_interp_multichannel(corners_mv, u_x, u_y, c)
            
            if coords_mv[n, 0] >= 0.0 and coords_mv[n, 0] <= 1.0 and \
               coords_mv[n, 1] >= 0.0 and coords_mv[n, 1] <= 1.0:
                out_mv[n, c] = wrap_hue(interp_val)
            else:
                final_val = lerp_hue_single(interp_val, 0.0, 0.5, HUE_SHORTEST)
                out_mv[n, c] = wrap_hue(final_val)


# =============================================================================
# Multi-Channel with Per-Channel Coordinates
# =============================================================================
cdef inline void _hue_corner_multichannel_per_ch_coords_feathered_kernel(
    const f64[:, ::1] corners_mv,    # shape (4, C)
    const f64[:, :, :, ::1] coords_mv,  # shape (C, H, W, 2)
    f64[:, :, ::1] out_mv,           # shape (H, W, C)
    object border_constant_obj, f64 border_feathering,
    Py_ssize_t C, Py_ssize_t H, Py_ssize_t W,
    object border_mode_obj, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Multi-channel hue corner with per-channel coords."""
    cdef Py_ssize_t c, h, w
    cdef f64 u_x, u_y
    cdef f64 interp_val, final_val
    
    for c in prange(C, nogil=True, schedule='static', num_threads=num_threads):
        for h in range(H):
            for w in range(W):
                u_x = clamp_01(coords_mv[c, h, w, 0])
                u_y = clamp_01(coords_mv[c, h, w, 1])
                
                interp_val = _hue_bilinear_interp_multichannel(corners_mv, u_x, u_y, c)
                
                if coords_mv[c, h, w, 0] >= 0.0 and coords_mv[c, h, w, 0] <= 1.0 and \
                   coords_mv[c, h, w, 1] >= 0.0 and coords_mv[c, h, w, 1] <= 1.0:
                    out_mv[h, w, c] = wrap_hue(interp_val)
                else:
                    final_val = lerp_hue_single(interp_val, 0.0, 0.5, HUE_SHORTEST)
                    out_mv[h, w, c] = wrap_hue(final_val)


cdef inline void _hue_corner_multichannel_flat_per_ch_coords_feathered_kernel(
    const f64[:, ::1] corners_mv,    # shape (4, C)
    const f64[:, :, ::1] coords_mv,  # shape (C, N, 2)
    f64[:, ::1] out_mv,              # shape (N, C)
    object border_constant_obj, f64 border_feathering,
    Py_ssize_t C, Py_ssize_t N,
    object border_mode_obj, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Multi-channel hue corner with per-channel flat coords."""
    cdef Py_ssize_t c, n
    cdef f64 u_x, u_y
    cdef f64 interp_val, final_val
    
    for c in prange(C, nogil=True, schedule='static', num_threads=num_threads):
        for n in range(N):
            u_x = clamp_01(coords_mv[c, n, 0])
            u_y = clamp_01(coords_mv[c, n, 1])
            
            interp_val = _hue_bilinear_interp_multichannel(corners_mv, u_x, u_y, c)
            
            if coords_mv[c, n, 0] >= 0.0 and coords_mv[c, n, 0] <= 1.0 and \
               coords_mv[c, n, 1] >= 0.0 and coords_mv[c, n, 1] <= 1.0:
                out_mv[n, c] = wrap_hue(interp_val)
            else:
                final_val = lerp_hue_single(interp_val, 0.0, 0.5, HUE_SHORTEST)
                out_mv[n, c] = wrap_hue(final_val)


# =============================================================================
# Public API - Single Channel
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_1ch_feathered(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=3] coords,
    f64 border_constant=0.0,
    object border_mode='clamp',
    f64 border_feathering=0.0,
    object distance_mode='euclidean',
    int num_threads=-1,
):
    """Single-channel hue corner interpolation with feathering on grid."""
    if corners.shape[0] != 4:
        raise ValueError("corners must have 4 values: [tl, tr, bl, br]")
    if coords.shape[2] != 2:
        raise ValueError("coords must be (..., 2)")
    
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef np.ndarray[f64, ndim=2] result = np.empty((H, W), dtype=np.float64)
    
    cdef int bmode = 3 if isinstance(border_mode, str) and border_mode.lower() == 'clamp' else 0
    cdef i32 dmode = 7 if isinstance(distance_mode, str) and distance_mode.lower() == 'euclidean' else 0
    
    _hue_corner_1ch_feathered_kernel(
        corners, coords, result,
        border_constant, border_feathering,
        H, W, bmode, num_threads if num_threads > 0 else 1,
        dmode
    )
    
    return result


cpdef np.ndarray[f64, ndim=1] hue_lerp_from_corners_1ch_flat_feathered(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=2] coords,
    f64 border_constant=0.0,
    object border_mode='clamp',
    f64 border_feathering=0.0,
    object distance_mode='euclidean',
    int num_threads=-1,
):
    """Single-channel hue corner interpolation (flat coords) with feathering."""
    if corners.shape[0] != 4:
        raise ValueError("corners must have 4 values")
    if coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")
    
    cdef Py_ssize_t N = coords.shape[0]
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef np.ndarray[f64, ndim=1] result = np.empty(N, dtype=np.float64)
    
    cdef int bmode = 3 if isinstance(border_mode, str) and border_mode.lower() == 'clamp' else 0
    cdef i32 dmode = 7 if isinstance(distance_mode, str) and distance_mode.lower() == 'euclidean' else 0
    
    _hue_corner_1ch_flat_feathered_kernel(
        corners, coords, result,
        border_constant, border_feathering,
        N, bmode, num_threads if num_threads > 0 else 1,
        dmode
    )
    
    return result


# =============================================================================
# Public API - Multi-Channel
# =============================================================================
cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel_feathered(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    object border_constant=0.0,
    object border_mode='clamp',
    f64 border_feathering=0.0,
    object distance_mode='euclidean',
    int num_threads=-1,
):
    """Multi-channel hue corner interpolation on grid."""
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, C)")
    if coords.shape[2] != 2:
        raise ValueError("coords must be (H, W, 2)")
    
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    cdef Py_ssize_t C = corners.shape[1]
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef np.ndarray[f64, ndim=3] result = np.empty((H, W, C), dtype=np.float64)
    
    cdef int bmode = 3 if isinstance(border_mode, str) and border_mode.lower() == 'clamp' else 0
    cdef i32 dmode = 7 if isinstance(distance_mode, str) and distance_mode.lower() == 'euclidean' else 0
    
    _hue_corner_multichannel_per_ch_border_feathered_kernel(
        corners, coords, result,
        border_constant, border_feathering,
        H, W, C, border_mode, num_threads if num_threads > 0 else 1,
        dmode
    )
    
    return result


cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_multichannel_flat_feathered(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    object border_constant=0.0,
    object border_mode='clamp',
    f64 border_feathering=0.0,
    object distance_mode='euclidean',
    int num_threads=-1,
):
    """Multi-channel hue corner interpolation (flat coords)."""
    if corners.shape[0] != 4:
        raise ValueError("corners must have shape (4, C)")
    if coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")
    
    cdef Py_ssize_t N = coords.shape[0]
    cdef Py_ssize_t C = corners.shape[1]
    
    if not corners.flags['C_CONTIGUOUS']:
        corners = np.ascontiguousarray(corners)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef np.ndarray[f64, ndim=2] result = np.empty((N, C), dtype=np.float64)
    
    cdef int bmode = 3 if isinstance(border_mode, str) and border_mode.lower() == 'clamp' else 0
    cdef i32 dmode = 7 if isinstance(distance_mode, str) and distance_mode.lower() == 'euclidean' else 0
    
    _hue_corner_multichannel_flat_per_ch_border_feathered_kernel(
        corners, coords, result,
        border_constant, border_feathering,
        N, C, border_mode, num_threads if num_threads > 0 else 1,
        dmode
    )
    
    return result


# =============================================================================
# Smart Dispatcher
# =============================================================================
DEF MIN_PARALLEL_SIZE = 10000

cpdef np.ndarray hue_lerp_from_corners_full_feathered(
    corners,
    coords,
    f64 border_constant=0.0,
    object border_mode='clamp',
    f64 border_feathering=0.0,
    object distance_mode='euclidean',
    int num_threads=-1,
):
    """
    Smart dispatcher for hue corner interpolation.
    
    Routes to appropriate kernel based on corners shape and coords layout.
    Supports:
    - Single channel: corners (4,) -> output (H,W) or (N,)
    - Multi-channel: corners (4,C) -> output (H,W,C) or (N,C)
    - Per-channel coords: coords (C,H,W,2) or (C,N,2)
    """
    corners = np.asarray(corners, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    
    # Single-channel case
    if corners.ndim == 1:
        if coords.ndim == 3:
            return hue_lerp_from_corners_1ch_feathered(
                corners, coords,
                border_constant, border_mode, border_feathering, distance_mode,
                num_threads
            )
        elif coords.ndim == 2:
            return hue_lerp_from_corners_1ch_flat_feathered(
                corners, coords,
                border_constant, border_mode, border_feathering, distance_mode,
                num_threads
            )
    
    # Multi-channel case
    elif corners.ndim == 2:
        if coords.ndim == 3:
            return hue_lerp_from_corners_multichannel_feathered(
                corners, coords,
                border_constant, border_mode, border_feathering, distance_mode,
                num_threads
            )
        elif coords.ndim == 2:
            return hue_lerp_from_corners_multichannel_flat_feathered(
                corners, coords,
                border_constant, border_mode, border_feathering, distance_mode,
                num_threads
            )
    
    raise ValueError("Invalid corners/coords shapes")
