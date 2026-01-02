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

from .border_handling cimport handle_border_1d
from ..interp_utils cimport (
    process_border_2d as _process_border_2d,
    BorderResult,
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
)




# =============================================================================
# Core Interpolation Functions (simplified)
# =============================================================================
cdef inline f64 _interp_line_1ch(
    const f64[::1] l0, const f64[::1] l1,
    f64 u_x, f64 u_y, Py_ssize_t L
) noexcept nogil:
    cdef f64 frac, v0, v1
    cdef Py_ssize_t idx_lo, idx_hi
    
    idx_lo = compute_interp_idx(u_x, L, &frac)
    idx_hi = idx_lo + 1
    
    v0 = l0[idx_lo] + frac * (l0[idx_hi] - l0[idx_lo])
    v1 = l1[idx_lo] + frac * (l1[idx_hi] - l1[idx_lo])
    return v0 + u_y * (v1 - v0)


cdef inline f64 _interp_line_multichannel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    f64 u_x, f64 u_y, Py_ssize_t L, Py_ssize_t ch
) noexcept nogil:
    cdef f64 frac, v0, v1
    cdef Py_ssize_t idx_lo, idx_hi
    
    idx_lo = compute_interp_idx(u_x, L, &frac)
    idx_hi = idx_lo + 1
    
    v0 = l0[idx_lo, ch] + frac * (l0[idx_hi, ch] - l0[idx_lo, ch])
    v1 = l1[idx_lo, ch] + frac * (l1[idx_hi, ch] - l1[idx_lo, ch])
    return v0 + u_y * (v1 - v0)


cdef inline f64 _interp_line_discrete_multichannel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    f64 u_x, f64 u_y, Py_ssize_t L, Py_ssize_t ch
) noexcept nogil:
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f = u_x * L_minus_1
    cdef Py_ssize_t idx = <Py_ssize_t>floor(idx_f + 0.5)
    
    if idx < 0:
        idx = 0
    elif idx >= L:
        idx = L - 1
    
    return l0[idx, ch] + u_y * (l1[idx, ch] - l0[idx, ch])


# =============================================================================
# Simplified Kernels (all use the same border processing logic)
# =============================================================================
cdef inline void _lerp_multichannel_per_ch_border_feathered_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, ::1] c, f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv, 
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,

) noexcept nogil:
    """Multi-channel with per-channel border modes and feathering (shared coords)."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            for ch in range(C):
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch]) 
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_const_mv[ch]
                else:
                    edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                        border_res.u_y_final, L, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_const_mv[ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val


cdef inline void _lerp_flat_multichannel_per_ch_border_feathered_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, ::1] c, f64[:, ::1] out_mv,
    const f64[::1] border_const_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv, 
    f64 border_feathering, Py_ssize_t N, Py_ssize_t L, Py_ssize_t C,
    int num_threads,

) noexcept nogil:
    """Flat coords with per-channel border modes and feathering."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        for ch in range(C):
            border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch])
            
            if border_res.use_border_directly:
                out_mv[n, ch] = border_const_mv[ch]
            else:
                edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                    border_res.u_y_final, L, ch)
                if border_res.blend_factor > 0.0:
                    border_val = border_const_mv[ch]
                    out_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[n, ch] = edge_val


cdef inline void _lerp_multichannel_per_ch_coords_border_feathered_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, :, ::1] coords_mv, f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv, 
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,

) noexcept nogil:
    """Per-channel coordinates with per-channel border modes and feathering."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            for ch in range(C):
                u_x = coords_mv[ch, h, w, 0]
                u_y = coords_mv[ch, h, w, 1]
                
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch])
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_const_mv[ch]
                else:
                    edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                        border_res.u_y_final, L, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_const_mv[ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val


cdef inline void _lerp_flat_multichannel_per_ch_coords_border_feathered_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, ::1] coords_mv, f64[:, ::1] out_mv,
    const f64[::1] border_const_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv, 
    f64 border_feathering, Py_ssize_t N, Py_ssize_t L, Py_ssize_t C,
    int num_threads,

) noexcept nogil:
    """Flat per-channel coords with per-channel border modes and feathering."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for ch in range(C):
            u_x = coords_mv[ch, n, 0]
            u_y = coords_mv[ch, n, 1]
            
            border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch])
            
            if border_res.use_border_directly:
                out_mv[n, ch] = border_const_mv[ch]
            else:
                edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                    border_res.u_y_final, L, ch)
                if border_res.blend_factor > 0.0:
                    border_val = border_const_mv[ch]
                    out_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[n, ch] = edge_val


cdef inline void _lerp_1ch_feathered_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, :, ::1] c, f64[:, ::1] out_mv,
    f64 border_const, f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Single-channel with feathering support."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            border_res = _process_border_2d(u_x, u_y, border_mode, border_feathering, distance_mode)
            
            if border_res.use_border_directly:
                out_mv[h, w] = border_const
            else:
                edge_val = _interp_line_1ch(l0, l1, border_res.u_x_final, 
                                           border_res.u_y_final, L)
                if border_res.blend_factor > 0.0:
                    out_mv[h, w] = edge_val + border_res.blend_factor * (border_const - edge_val)
                else:
                    out_mv[h, w] = edge_val


cdef inline void _lerp_flat_1ch_feathered_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, ::1] c, f64[::1] out_mv,
    f64 border_const, f64 border_feathering,
    Py_ssize_t N, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Flat single-channel with feathering."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, edge_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        border_res = _process_border_2d(u_x, u_y, border_mode, border_feathering, distance_mode)
        
        if border_res.use_border_directly:
            out_mv[n] = border_const
        else:
            edge_val = _interp_line_1ch(l0, l1, border_res.u_x_final, 
                                       border_res.u_y_final, L)
            if border_res.blend_factor > 0.0:
                out_mv[n] = edge_val + border_res.blend_factor * (border_const - edge_val)
            else:
                out_mv[n] = edge_val


# =============================================================================
# Discrete X-Sampling Kernels (simplified)
# =============================================================================
cdef inline void _lerp_x_discrete_multichannel_per_ch_border_feathered_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, ::1] c, f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv, 
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,

) noexcept nogil:
    """Discrete x-sampling with per-channel border modes and feathering."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            for ch in range(C):
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch])
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_const_mv[ch]
                else:
                    edge_val = _interp_line_discrete_multichannel(
                        l0, l1, border_res.u_x_final, border_res.u_y_final, L, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_const_mv[ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val


cdef inline void _lerp_x_discrete_per_ch_coords_feathered_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, :, ::1] coords_mv, f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv, 
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,

) noexcept nogil:
    """Discrete x-sampling with per-channel coords, border modes, and feathering."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            for ch in range(C):
                u_x = coords_mv[ch, h, w, 0]
                u_y = coords_mv[ch, h, w, 1]
                
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], border_feathering, distance_modes_mv[ch])
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_const_mv[ch]
                else:
                    edge_val = _interp_line_discrete_multichannel(
                        l0, l1, border_res.u_x_final, border_res.u_y_final, L, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_const_mv[ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val

# =============================================================================
# Public API - Single Channel
# =============================================================================
def lerp_between_lines_1ch_feathered(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,
    int num_threads=-1,
):
    """
    Fast parallel single-channel interpolation between two lines with feathering support.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
        border_mode: Border handling mode (BORDER_CLAMP, BORDER_CONSTANT, etc.)
        border_constant: Value for BORDER_CONSTANT mode
        border_feathering: Feathering width for BORDER_CONSTANT mode (0.0 = hard edge)
        distance_mode: Distance metric for 2D border computation (ALPHA_MAX, EUCLIDEAN, etc.)
        num_threads: Number of threads (-1 = auto, 0 = serial, >0 = specific count)
    
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

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

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

    with nogil:
        _lerp_1ch_feathered_kernel(l0, l1, c, out_mv, border_constant, 
                                   border_feathering, H, W, L, border_mode, 
                                   n_threads, distance_mode)

    return out


def lerp_between_lines_flat_1ch_feathered(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,
    int num_threads=-1,
):
    """
    Fast parallel flat single-channel interpolation with feathering support.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Flat coordinates, shape (N, 2)
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
        border_feathering: Feathering width for BORDER_CONSTANT mode
        distance_mode: Distance metric for 2D border computation
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values, shape (N,)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

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

    with nogil:
        _lerp_flat_1ch_feathered_kernel(l0, l1, c, out_mv, border_constant, 
                                        border_feathering, N, L, border_mode, 
                                        n_threads, distance_mode)

    return out


# =============================================================================
# Public API - Multi-Channel (Per-Channel Border Modes)
# =============================================================================
def lerp_between_lines_multichannel_per_ch_border_feathered(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Multi-channel interpolation with per-channel border modes and feathering.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2) - same coordinates for all channels
        border_modes: Border mode for each channel, shape (C,)
        border_constants: Border constant for each channel, shape (C,)
        border_feathering: Feathering width (same for all channels)
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    
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

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] bc = border_constants
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_multichannel_per_ch_border_feathered_kernel(
            l0, l1, c, out_mv, bc, bm, dm, border_feathering, 
            H, W, L, C, n_threads
        )

    return out


def lerp_between_lines_flat_multichannel_per_ch_border_feathered(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Flat multi-channel interpolation with per-channel border modes and feathering.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Flat coordinates, shape (N, 2) - same for all channels
        border_modes: Border mode for each channel, shape (C,)
        border_constants: Border constant for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    
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

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[::1] bc = border_constants
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_flat_multichannel_per_ch_border_feathered_kernel(
            l0, l1, c, out_mv, bc, bm, dm, border_feathering,
            N, L, C, n_threads
        )

    return out


# =============================================================================
# Public API - Multi-Channel with Per-Channel Coordinates
# =============================================================================
def lerp_between_lines_multichannel_per_ch_coords_feathered(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=4] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Multi-channel interpolation with per-channel coordinates and border modes.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Per-channel coordinate grids, shape (C, H, W, 2)
        border_modes: Border mode for each channel, shape (C,)
        border_constants: Border constant for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t H = coords.shape[1]
    cdef Py_ssize_t W = coords.shape[2]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[3] != 2:
        raise ValueError("coords must have shape (C, H, W, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match lines ({C})")
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

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, :, ::1] c = coords
    cdef f64[::1] bc = border_constants
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_multichannel_per_ch_coords_border_feathered_kernel(
            l0, l1, c, out_mv, bc, bm, dm, border_feathering,
            H, W, L, C, n_threads
        )

    return out


def lerp_between_lines_flat_multichannel_per_ch_coords_feathered(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Flat multi-channel interpolation with per-channel coordinates and border modes.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Per-channel flat coordinates, shape (C, N, 2)
        border_modes: Border mode for each channel, shape (C,)
        border_constants: Border constant for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values, shape (N, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t N = coords.shape[1]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (C, N, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match lines ({C})")
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

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] bc = border_constants
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_flat_multichannel_per_ch_coords_border_feathered_kernel(
            l0, l1, c, out_mv, bc, bm, dm, border_feathering,
            N, L, C, n_threads
        )

    return out


# =============================================================================
# Public API - Discrete X-Sampling
# =============================================================================
def lerp_between_lines_x_discrete_multichannel_per_ch_border_feathered(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Discrete x-sampling with per-channel border modes and feathering.
    
    Snaps u_x to nearest line index (no interpolation along x),
    interpolates continuously along y.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
        border_modes: Border mode for each channel, shape (C,)
        border_constants: Border constant for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    
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

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] bc = border_constants
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_x_discrete_multichannel_per_ch_border_feathered_kernel(
            l0, l1, c, out_mv, bc, bm, dm, border_feathering,
            H, W, L, C, n_threads
        )

    return out


def lerp_between_lines_x_discrete_per_ch_coords_feathered(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=4] coords,
    np.ndarray[i32, ndim=1] border_modes,
    np.ndarray[f64, ndim=1] border_constants,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Discrete x-sampling with per-channel coordinates, border modes, and feathering.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Per-channel coordinate grids, shape (C, H, W, 2)
        border_modes: Border mode for each channel, shape (C,)
        border_constants: Border constant for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t H = coords.shape[1]
    cdef Py_ssize_t W = coords.shape[2]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[3] != 2:
        raise ValueError("coords must have shape (C, H, W, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match lines ({C})")
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

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)
    if not border_constants.flags['C_CONTIGUOUS']:
        border_constants = np.ascontiguousarray(border_constants)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, :, ::1] c = coords
    cdef f64[::1] bc = border_constants
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_x_discrete_per_ch_coords_feathered_kernel(
            l0, l1, c, out_mv, bc, bm, dm, border_feathering,
            H, W, L, C, n_threads
        )

    return out


# =============================================================================
# Smart Dispatcher
# =============================================================================
DEF MIN_PARALLEL_SIZE = 10000

def lerp_between_lines_full_feathered(
    np.ndarray line0,
    np.ndarray line1,
    np.ndarray coords,
    object border_mode=None,
    object border_constant=None,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
    bint x_discrete=False,
):
    """
    Smart dispatcher for fast line interpolation with full feature support.
    
    Automatically selects the appropriate kernel based on input shapes.
    Supports all combinations of:
    - Single vs multi-channel
    - Grid vs flat coordinates
    - Same vs per-channel coordinates
    - Global vs per-channel border modes
    - Global vs per-channel distance modes
    - Feathering support
    
    Args:
        line0: First line, shape (L,) or (L, C)
        line1: Second line, shape (L,) or (L, C)
        coords: Coordinate array, shape:
            - (H, W, 2) for grid
            - (N, 2) for flat
            - (C, H, W, 2) for per-channel grid
            - (C, N, 2) for per-channel flat
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
            Options: MAX_NORM, MANHATTAN, SCALED_MANHATTAN, ALPHA_MAX, 
                     ALPHA_MAX_SIMPLE, TAYLOR, EUCLIDEAN, WEIGHTED_MINMAX
        num_threads: Number of threads (-1=auto, 0=serial, >0=specific)
        x_discrete: If True, use discrete x-sampling (nearest neighbor in x)
    
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

    # Convert to proper types
    if line0.dtype != np.float64:
        line0 = np.ascontiguousarray(line0, dtype=np.float64)
    if line1.dtype != np.float64:
        line1 = np.ascontiguousarray(line1, dtype=np.float64)
    if coords.dtype != np.float64:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    # Calculate size for parallelization
    if coords_ndim == 3:  # (H, W, 2) or (C, N, 2)
        if coords.shape[2] == 2:  # (H, W, 2)
            total_size = coords.shape[0] * coords.shape[1]
        else:  # (C, N, 2)
            total_size = coords.shape[1]
    elif coords_ndim == 4:  # (C, H, W, 2)
        total_size = coords.shape[1] * coords.shape[2]
    else:  # (N, 2)
        total_size = coords.shape[0]
    
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
    if line0.ndim == 1:
        C = 1
        bc_scalar = float(border_constant) if not isinstance(border_constant, np.ndarray) else float(border_constant[0])
        bm_scalar = int(border_mode) if not isinstance(border_mode, np.ndarray) else int(border_mode[0])
        dm_scalar = int(distance_mode) if not isinstance(distance_mode, np.ndarray) else int(distance_mode[0])
        
        if x_discrete:
            raise NotImplementedError("Single-channel discrete x-sampling not implemented")
        
        if coords_ndim == 3 and coords.shape[2] == 2:
            return lerp_between_lines_1ch_feathered(
                line0, line1, coords, bm_scalar, bc_scalar, 
                border_feathering, dm_scalar, use_threads
            )
        elif coords_ndim == 2 and coords.shape[1] == 2:
            return lerp_between_lines_flat_1ch_feathered(
                line0, line1, coords, bm_scalar, bc_scalar,
                border_feathering, dm_scalar, use_threads
            )
        else:
            raise ValueError(f"Invalid coords shape for single channel: {(<object>coords).shape}")
    
    # Multi-channel
    elif line0.ndim == 2:
        C = line0.shape[1]
        
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
        if coords_ndim == 3 and coords.shape[2] == 2:
            if x_discrete:
                return lerp_between_lines_x_discrete_multichannel_per_ch_border_feathered(
                    line0, line1, coords, bm_arr, bc_arr,
                    border_feathering, dm_arr, use_threads
                )
            else:
                return lerp_between_lines_multichannel_per_ch_border_feathered(
                    line0, line1, coords, bm_arr, bc_arr,
                    border_feathering, dm_arr, use_threads
                )
        
        # Flat coordinates (N, 2) - same for all channels
        elif coords_ndim == 2 and coords.shape[1] == 2:
            if x_discrete:
                raise NotImplementedError("Flat discrete x-sampling not implemented")
            return lerp_between_lines_flat_multichannel_per_ch_border_feathered(
                line0, line1, coords, bm_arr, bc_arr,
                border_feathering, dm_arr, use_threads
            )
        
        # Per-channel grid coordinates (C, H, W, 2)
        elif coords_ndim == 4 and coords.shape[3] == 2:
            if coords.shape[0] != C:
                raise ValueError(f"coords channels ({int(coords.shape[0])}) must match lines ({C})")
            
            if x_discrete:
                return lerp_between_lines_x_discrete_per_ch_coords_feathered(
                    line0, line1, coords, bm_arr, bc_arr,
                    border_feathering, dm_arr, use_threads
                )
            else:
                return lerp_between_lines_multichannel_per_ch_coords_feathered(
                    line0, line1, coords, bm_arr, bc_arr,
                    border_feathering, dm_arr, use_threads
                )
        
        # Per-channel flat coordinates (C, N, 2)
        elif coords_ndim == 3 and coords.shape[2] == 2 and coords.shape[0] == C:
            if x_discrete:
                raise NotImplementedError("Per-channel flat discrete x-sampling not implemented")
            return lerp_between_lines_flat_multichannel_per_ch_coords_feathered(
                line0, line1, coords, bm_arr, bc_arr,
                border_feathering, dm_arr, use_threads
            )
        
        else:
            raise ValueError(f"Invalid coords shape for multi-channel: {(<object>coords).shape}")
    
    raise ValueError(f"Unsupported line dimensions: {line0.ndim}")


# =============================================================================
# Legacy compatibility functions
# =============================================================================
def lerp_between_lines_1ch_fast(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    int num_threads=-1,
):
    """Legacy compatibility wrapper (no feathering)."""
    return lerp_between_lines_1ch_feathered(
        line0, line1, coords, border_mode, border_constant, 
        0.0, ALPHA_MAX, num_threads
    )


def lerp_between_lines_multichannel_fast(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """Legacy compatibility wrapper (same border mode for all channels)."""
    cdef Py_ssize_t C = line0.shape[1]
    
    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    
    cdef np.ndarray[i32, ndim=1] bm_arr = np.full(C, border_mode, dtype=np.int32)
    
    return lerp_between_lines_multichannel_per_ch_border_feathered(
        line0, line1, coords, bm_arr, border_constant,
        0.0, ALPHA_MAX, num_threads
    )


# =============================================================================
# Export constants
# =============================================================================
BORDER_REPEAT = 0
BORDER_MIRROR = 1
BORDER_CONSTANT = 2
BORDER_CLAMP = 3
BORDER_OVERFLOW = 4

# Distance mode exports
DIST_MAX_NORM = MAX_NORM
DIST_MANHATTAN = MANHATTAN
DIST_SCALED_MANHATTAN = SCALED_MANHATTAN
DIST_ALPHA_MAX = ALPHA_MAX
DIST_ALPHA_MAX_SIMPLE = ALPHA_MAX_SIMPLE
DIST_TAYLOR = TAYLOR
DIST_EUCLIDEAN = ALPHA_MAX
DIST_WEIGHTED_MINMAX = WEIGHTED_MINMAX