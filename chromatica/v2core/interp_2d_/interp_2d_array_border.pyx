# interp_2d_array_border.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

"""
Interpolation between lines with array-based border values.

This module allows blending interpolation results with an existing image/array
at the borders, rather than using a constant value. Useful for compositing
operations where out-of-bounds regions should show through to a background.
"""

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

from ..border_handling_ cimport handle_border_1d
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
    compute_extra_1d,
    compute_interp_idx
)






# =============================================================================
# Core Interpolation Functions
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
# Kernels with Array-Based Border Values (HxWxC)
# =============================================================================
cdef inline void _lerp_multichannel_array_border_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, ::1] c, f64[:, :, ::1] out_mv,
    const f64[:, :, ::1] border_array_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,  # NEW: distance mode array
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Multi-channel with per-pixel border values from array (HxWxC)."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            for ch in range(C):
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                               border_feathering, distance_modes_mv[ch])  # UPDATED
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_array_mv[h, w, ch]
                else:
                    edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                        border_res.u_y_final, L, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_array_mv[h, w, ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val


cdef inline void _lerp_flat_multichannel_array_border_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, ::1] c, f64[:, ::1] out_mv,
    const f64[:, ::1] border_array_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,  # NEW: distance mode array
    f64 border_feathering, Py_ssize_t N, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Flat coords with per-point border values from array (NxC)."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        for ch in range(C):
            border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                           border_feathering, distance_modes_mv[ch])  # UPDATED
            
            if border_res.use_border_directly:
                out_mv[n, ch] = border_array_mv[n, ch]
            else:
                edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                    border_res.u_y_final, L, ch)
                if border_res.blend_factor > 0.0:
                    border_val = border_array_mv[n, ch]
                    out_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[n, ch] = edge_val


cdef inline void _lerp_multichannel_per_ch_coords_array_border_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, :, ::1] coords_mv, f64[:, :, ::1] out_mv,
    const f64[:, :, ::1] border_array_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,  # NEW: distance mode array
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Per-channel coordinates with array-based border values (HxWxC)."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            for ch in range(C):
                u_x = coords_mv[ch, h, w, 0]
                u_y = coords_mv[ch, h, w, 1]
                
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                               border_feathering, distance_modes_mv[ch])  # UPDATED
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_array_mv[h, w, ch]
                else:
                    edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                        border_res.u_y_final, L, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_array_mv[h, w, ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val


cdef inline void _lerp_flat_multichannel_per_ch_coords_array_border_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, ::1] coords_mv, f64[:, ::1] out_mv,
    const f64[:, ::1] border_array_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,  # NEW: distance mode array
    f64 border_feathering, Py_ssize_t N, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Flat per-channel coords with array-based border values (NxC)."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for ch in range(C):
            u_x = coords_mv[ch, n, 0]
            u_y = coords_mv[ch, n, 1]
            
            border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                           border_feathering, distance_modes_mv[ch])  # UPDATED
            
            if border_res.use_border_directly:
                out_mv[n, ch] = border_array_mv[n, ch]
            else:
                edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                    border_res.u_y_final, L, ch)
                if border_res.blend_factor > 0.0:
                    border_val = border_array_mv[n, ch]
                    out_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[n, ch] = edge_val


cdef inline void _lerp_1ch_array_border_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, :, ::1] c, f64[:, ::1] out_mv,
    const f64[:, ::1] border_array_mv, f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,  # NEW: distance mode parameter
) noexcept nogil:
    """Single-channel with per-pixel border values from array (HxW)."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            border_res = _process_border_2d(u_x, u_y, border_mode, 
                                           border_feathering, distance_mode)  # UPDATED
            
            if border_res.use_border_directly:
                out_mv[h, w] = border_array_mv[h, w]
            else:
                edge_val = _interp_line_1ch(l0, l1, border_res.u_x_final, 
                                           border_res.u_y_final, L)
                if border_res.blend_factor > 0.0:
                    border_val = border_array_mv[h, w]
                    out_mv[h, w] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[h, w] = edge_val


cdef inline void _lerp_flat_1ch_array_border_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, ::1] c, f64[::1] out_mv,
    const f64[::1] border_array_mv, f64 border_feathering,
    Py_ssize_t N, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,  # NEW: distance mode parameter
) noexcept nogil:
    """Flat single-channel with per-point border values from array (N,)."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        border_res = _process_border_2d(u_x, u_y, border_mode, 
                                       border_feathering, distance_mode)  # UPDATED
        
        if border_res.use_border_directly:
            out_mv[n] = border_array_mv[n]
        else:
            edge_val = _interp_line_1ch(l0, l1, border_res.u_x_final, 
                                       border_res.u_y_final, L)
            if border_res.blend_factor > 0.0:
                border_val = border_array_mv[n]
                out_mv[n] = edge_val + border_res.blend_factor * (border_val - edge_val)
            else:
                out_mv[n] = edge_val


# =============================================================================
# Discrete X-Sampling Kernels with Array Border
# =============================================================================
cdef inline void _lerp_x_discrete_multichannel_array_border_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, ::1] c, f64[:, :, ::1] out_mv,
    const f64[:, :, ::1] border_array_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,  # NEW: distance mode array
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Discrete x-sampling with array-based border values (HxWxC)."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            for ch in range(C):
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                               border_feathering, distance_modes_mv[ch])  # UPDATED
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_array_mv[h, w, ch]
                else:
                    edge_val = _interp_line_discrete_multichannel(
                        l0, l1, border_res.u_x_final, border_res.u_y_final, L, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_array_mv[h, w, ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val


cdef inline void _lerp_x_discrete_per_ch_coords_array_border_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, :, ::1] coords_mv, f64[:, :, ::1] out_mv,
    const f64[:, :, ::1] border_array_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,  # NEW: distance mode array
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Discrete x-sampling with per-channel coords and array border (HxWxC)."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            for ch in range(C):
                u_x = coords_mv[ch, h, w, 0]
                u_y = coords_mv[ch, h, w, 1]
                
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                               border_feathering, distance_modes_mv[ch])  # UPDATED
                
                if border_res.use_border_directly:
                    out_mv[h, w, ch] = border_array_mv[h, w, ch]
                else:
                    edge_val = _interp_line_discrete_multichannel(
                        l0, l1, border_res.u_x_final, border_res.u_y_final, L, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = border_array_mv[h, w, ch]
                        out_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        out_mv[h, w, ch] = edge_val

# =============================================================================
# Discrete X-Sampling Kernels with Array Border (Single Channel)
# =============================================================================
cdef inline void _lerp_x_discrete_1ch_array_border_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, :, ::1] c, f64[:, ::1] out_mv,
    const f64[:, ::1] border_array_mv, f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Single-channel discrete x-sampling with per-pixel border values (HxW)."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val, border_val
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f
    cdef Py_ssize_t idx
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            border_res = _process_border_2d(u_x, u_y, border_mode, 
                                           border_feathering, distance_mode)
            
            if border_res.use_border_directly:
                out_mv[h, w] = border_array_mv[h, w]
            else:
                idx_f = border_res.u_x_final * L_minus_1
                idx = <Py_ssize_t>floor(idx_f + 0.5)
                
                if idx < 0:
                    idx = 0
                elif idx >= L:
                    idx = L - 1
                
                edge_val = l0[idx] + border_res.u_y_final * (l1[idx] - l0[idx])
                if border_res.blend_factor > 0.0:
                    border_val = border_array_mv[h, w]
                    out_mv[h, w] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[h, w] = edge_val


cdef inline void _lerp_x_discrete_flat_1ch_array_border_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, ::1] c, f64[::1] out_mv,
    const f64[::1] border_array_mv, f64 border_feathering,
    Py_ssize_t N, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Flat single-channel discrete x-sampling with per-point border values (N,)."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, edge_val, border_val
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f
    cdef Py_ssize_t idx
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        border_res = _process_border_2d(u_x, u_y, border_mode, 
                                       border_feathering, distance_mode)
        
        if border_res.use_border_directly:
            out_mv[n] = border_array_mv[n]
        else:
            idx_f = border_res.u_x_final * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)
            
            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1
            
            edge_val = l0[idx] + border_res.u_y_final * (l1[idx] - l0[idx])
            if border_res.blend_factor > 0.0:
                border_val = border_array_mv[n]
                out_mv[n] = edge_val + border_res.blend_factor * (border_val - edge_val)
            else:
                out_mv[n] = edge_val


# =============================================================================
# Discrete X-Sampling Kernels with Array Border (Flat Multi-Channel)
# =============================================================================
cdef inline void _lerp_x_discrete_flat_multichannel_array_border_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, ::1] c, f64[:, ::1] out_mv,
    const f64[:, ::1] border_array_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,
    f64 border_feathering, Py_ssize_t N, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Flat discrete x-sampling with per-point border values (NxC)."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f
    cdef Py_ssize_t idx
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        for ch in range(C):
            border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                           border_feathering, distance_modes_mv[ch])
            
            if border_res.use_border_directly:
                out_mv[n, ch] = border_array_mv[n, ch]
            else:
                idx_f = border_res.u_x_final * L_minus_1
                idx = <Py_ssize_t>floor(idx_f + 0.5)
                
                if idx < 0:
                    idx = 0
                elif idx >= L:
                    idx = L - 1
                
                edge_val = l0[idx, ch] + border_res.u_y_final * (l1[idx, ch] - l0[idx, ch])
                if border_res.blend_factor > 0.0:
                    border_val = border_array_mv[n, ch]
                    out_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[n, ch] = edge_val


cdef inline void _lerp_x_discrete_flat_per_ch_coords_array_border_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, ::1] coords_mv, f64[:, ::1] out_mv,
    const f64[:, ::1] border_array_mv, const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,
    f64 border_feathering, Py_ssize_t N, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Flat per-channel discrete x-sampling with per-point border values (NxC)."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f
    cdef Py_ssize_t idx
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for ch in range(C):
            u_x = coords_mv[ch, n, 0]
            u_y = coords_mv[ch, n, 1]
            
            border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                           border_feathering, distance_modes_mv[ch])
            
            if border_res.use_border_directly:
                out_mv[n, ch] = border_array_mv[n, ch]
            else:
                idx_f = border_res.u_x_final * L_minus_1
                idx = <Py_ssize_t>floor(idx_f + 0.5)
                
                if idx < 0:
                    idx = 0
                elif idx >= L:
                    idx = L - 1
                
                edge_val = l0[idx, ch] + border_res.u_y_final * (l1[idx, ch] - l0[idx, ch])
                if border_res.blend_factor > 0.0:
                    border_val = border_array_mv[n, ch]
                    out_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    out_mv[n, ch] = edge_val


# =============================================================================
# Public API - Single Channel with Array Border
# =============================================================================
def lerp_between_lines_1ch_array_border(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    int border_mode=BORDER_CONSTANT,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,  # NEW: distance mode parameter
    int num_threads=-1,
):
    """
    Single-channel interpolation with per-pixel border values from an array.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
        border_array: Background/border values, shape (H, W)
        border_mode: Border handling mode (typically BORDER_CONSTANT for compositing)
        border_feathering: Feathering width for smooth blending (0.0 = hard edge)
        distance_mode: Distance metric for 2D border computation (ALPHA_MAX, EUCLIDEAN, etc.)  # NEW
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (H, W)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if border_array.shape[0] != H or border_array.shape[1] != W:
        raise ValueError(f"border_array must have shape ({H}, {W})")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, ::1] ba = border_array

    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_1ch_array_border_kernel(l0, l1, c, out_mv, ba, border_feathering,
                                      H, W, L, border_mode, n_threads, distance_mode)  # UPDATED

    return out


def lerp_between_lines_flat_1ch_array_border(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    int border_mode=BORDER_CONSTANT,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,  # NEW: distance mode parameter
    int num_threads=-1,
):
    """
    Flat single-channel interpolation with per-point border values.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Flat coordinates, shape (N, 2)
        border_array: Background values, shape (N,)
        border_mode: Border handling mode
        border_feathering: Feathering width
        distance_mode: Distance metric for 2D border computation  # NEW
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (N,)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if border_array.shape[0] != N:
        raise ValueError(f"border_array must have shape ({N},)")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[::1] ba = border_array

    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out

    with nogil:
        _lerp_flat_1ch_array_border_kernel(l0, l1, c, out_mv, ba, border_feathering,
                                           N, L, border_mode, n_threads, distance_mode)  # UPDATED

    return out


# =============================================================================
# Public API - Multi-Channel with Array Border
# =============================================================================
def lerp_between_lines_multichannel_array_border(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] border_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: can be int or array
    int num_threads=-1,
):
    """
    Multi-channel interpolation with per-pixel border values from array.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2) - same for all channels
        border_array: Background image, shape (H, W, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)  # NEW
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if border_array.shape[0] != H or border_array.shape[1] != W or border_array.shape[2] != C:
        raise ValueError(f"border_array must have shape ({H}, {W}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, :, ::1] ba = border_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_multichannel_array_border_kernel(
            l0, l1, c, out_mv, ba, bm, dm, border_feathering,  # UPDATED
            H, W, L, C, n_threads
        )

    return out


def lerp_between_lines_flat_multichannel_array_border(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=2] border_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: can be int or array
    int num_threads=-1,
):
    """
    Flat multi-channel interpolation with per-point border values.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Flat coordinates, shape (N, 2)
        border_array: Background values, shape (N, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)  # NEW
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (N, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if border_array.shape[0] != N or border_array.shape[1] != C:
        raise ValueError(f"border_array must have shape ({N}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[:, ::1] ba = border_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_flat_multichannel_array_border_kernel(
            l0, l1, c, out_mv, ba, bm, dm, border_feathering,  # UPDATED
            N, L, C, n_threads
        )

    return out


# =============================================================================
# Public API - Per-Channel Coordinates with Array Border
# =============================================================================
def lerp_between_lines_multichannel_per_ch_coords_array_border(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=4] coords,
    np.ndarray[f64, ndim=3] border_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: can be int or array
    int num_threads=-1,
):
    """
    Multi-channel interpolation with per-channel coordinates and array border.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Per-channel coordinate grids, shape (C, H, W, 2)
        border_array: Background image, shape (H, W, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)  # NEW
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (H, W, C)
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
    if border_array.shape[0] != H or border_array.shape[1] != W or border_array.shape[2] != C:
        raise ValueError(f"border_array must have shape ({H}, {W}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, :, ::1] c = coords
    cdef f64[:, :, ::1] ba = border_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_multichannel_per_ch_coords_array_border_kernel(
            l0, l1, c, out_mv, ba, bm, dm, border_feathering,  # UPDATED
            H, W, L, C, n_threads
        )

    return out


def lerp_between_lines_flat_multichannel_per_ch_coords_array_border(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: can be int or array
    int num_threads=-1,
):
    """
    Flat multi-channel interpolation with per-channel coords and array border.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Per-channel flat coordinates, shape (C, N, 2)
        border_array: Background values, shape (N, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)  # NEW
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (N, C)
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
    if border_array.shape[0] != N or border_array.shape[1] != C:
        raise ValueError(f"border_array must have shape ({N}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, ::1] ba = border_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_flat_multichannel_per_ch_coords_array_border_kernel(
            l0, l1, c, out_mv, ba, bm, dm, border_feathering,  # UPDATED
            N, L, C, n_threads
        )

    return out


# =============================================================================
# Public API - Discrete X-Sampling with Array Border
# =============================================================================
def lerp_between_lines_x_discrete_multichannel_array_border(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] border_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: can be int or array
    int num_threads=-1,
):
    """
    Discrete x-sampling with array-based border values.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
        border_array: Background image, shape (H, W, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)  # NEW
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if border_array.shape[0] != H or border_array.shape[1] != W or border_array.shape[2] != C:
        raise ValueError(f"border_array must have shape ({H}, {W}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, :, ::1] ba = border_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_x_discrete_multichannel_array_border_kernel(
            l0, l1, c, out_mv, ba, bm, dm, border_feathering,  # UPDATED
            H, W, L, C, n_threads
        )

    return out


def lerp_between_lines_x_discrete_per_ch_coords_array_border(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=4] coords,
    np.ndarray[f64, ndim=3] border_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: can be int or array
    int num_threads=-1,
):
    """
    Discrete x-sampling with per-channel coords and array border.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Per-channel coordinate grids, shape (C, H, W, 2)
        border_array: Background image, shape (H, W, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)  # NEW
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (H, W, C)
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
    if border_array.shape[0] != H or border_array.shape[1] != W or border_array.shape[2] != C:
        raise ValueError(f"border_array must have shape ({H}, {W}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, :, ::1] c = coords
    cdef f64[:, :, ::1] ba = border_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_x_discrete_per_ch_coords_array_border_kernel(
            l0, l1, c, out_mv, ba, bm, dm, border_feathering,  # UPDATED
            H, W, L, C, n_threads
        )

    return out


# =============================================================================
# Public API - Discrete X-Sampling Single Channel with Array Border
# =============================================================================
def lerp_between_lines_x_discrete_1ch_array_border(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    int border_mode=BORDER_CONSTANT,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,
    int num_threads=-1,
):
    """
    Single-channel discrete x-sampling with per-pixel border values from array.
    
    Snaps u_x to nearest line index (no interpolation along x),
    interpolates continuously along y.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
        border_array: Background/border values, shape (H, W)
        border_mode: Border handling mode
        border_feathering: Feathering width for smooth blending (0.0 = hard edge)
        distance_mode: Distance metric for 2D border computation
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (H, W)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if border_array.shape[0] != H or border_array.shape[1] != W:
        raise ValueError(f"border_array must have shape ({H}, {W})")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, ::1] ba = border_array

    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_x_discrete_1ch_array_border_kernel(l0, l1, c, out_mv, ba, 
                                                border_feathering, H, W, L,
                                                border_mode, n_threads, distance_mode)

    return out


def lerp_between_lines_x_discrete_flat_1ch_array_border(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    int border_mode=BORDER_CONSTANT,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,
    int num_threads=-1,
):
    """
    Flat single-channel discrete x-sampling with per-point border values.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Flat coordinates, shape (N, 2)
        border_array: Background values, shape (N,)
        border_mode: Border handling mode
        border_feathering: Feathering width
        distance_mode: Distance metric for 2D border computation
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (N,)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if border_array.shape[0] != N:
        raise ValueError(f"border_array must have shape ({N},)")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[::1] ba = border_array

    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out

    with nogil:
        _lerp_x_discrete_flat_1ch_array_border_kernel(l0, l1, c, out_mv, ba,
                                                     border_feathering, N, L,
                                                     border_mode, n_threads, distance_mode)

    return out


# =============================================================================
# Public API - Discrete X-Sampling Flat Multi-Channel with Array Border
# =============================================================================
def lerp_between_lines_x_discrete_flat_multichannel_array_border(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=2] border_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Flat multi-channel discrete x-sampling with per-point border values.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Flat coordinates, shape (N, 2)
        border_array: Background values, shape (N, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (N, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if border_array.shape[0] != N or border_array.shape[1] != C:
        raise ValueError(f"border_array must have shape ({N}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[:, ::1] ba = border_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_x_discrete_flat_multichannel_array_border_kernel(
            l0, l1, c, out_mv, ba, bm, dm, border_feathering,
            N, L, C, n_threads
        )

    return out


def lerp_between_lines_x_discrete_flat_per_ch_coords_array_border(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Flat per-channel discrete x-sampling with per-point border values.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Per-channel flat coordinates, shape (C, N, 2)
        border_array: Background values, shape (N, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values blended with border_array, shape (N, C)
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
    if border_array.shape[0] != N or border_array.shape[1] != C:
        raise ValueError(f"border_array must have shape ({N}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, ::1] ba = border_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_x_discrete_flat_per_ch_coords_array_border_kernel(
            l0, l1, c, out_mv, ba, bm, dm, border_feathering,
            N, L, C, n_threads
        )

    return out


# =============================================================================
# Smart Dispatcher
# =============================================================================
DEF MIN_PARALLEL_SIZE = 10000

def lerp_between_lines_onto_array(
    np.ndarray line0,
    np.ndarray line1,
    np.ndarray coords,
    np.ndarray border_array,
    object border_mode=None,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: distance mode parameter
    int num_threads=-1,
    bint x_discrete=False,
):
    """
    Smart dispatcher for interpolation that composites onto a background array.
    
    Automatically selects the appropriate kernel based on input shapes.
    The interpolation result is blended with border_array at out-of-bounds regions.
    
    Args:
        line0: First line, shape (L,) or (L, C)
        line1: Second line, shape (L,) or (L, C)
        coords: Coordinate array, shape:
            - (H, W, 2) for grid
            - (N, 2) for flat
            - (C, H, W, 2) for per-channel grid
            - (C, N, 2) for per-channel flat
        border_array: Background to composite onto, shape:
            - (H, W) or (N,) for single channel
            - (H, W, C) or (N, C) for multi-channel
        border_mode: Border mode(s):
            - int: Same for all channels (default: BORDER_CONSTANT)
            - list/array: Per-channel modes, shape (C,)
        border_feathering: Feathering width for smooth blending (0.0 = hard edge)
        distance_mode: Distance metric for 2D border computation:  # NEW
            - None: Use ALPHA_MAX (default)
            - int: Same for all channels
            - list/array: Per-channel modes, shape (C,)
        num_threads: Number of threads (-1=auto, 0=serial, >0=specific)
        x_discrete: If True, use discrete x-sampling (nearest neighbor in x)
    
    Returns:
        Interpolated values composited onto border_array
    
    Example:
        # Composite interpolation onto existing image
        background = np.random.rand(100, 100, 3)
        result = lerp_between_lines_onto_array(
            line0, line1, coords, background,
            border_feathering=0.1  # Smooth 10% feathering at edges
        )
    """
    cdef Py_ssize_t total_size
    cdef int use_threads = num_threads
    cdef Py_ssize_t C
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
    if border_array.dtype != np.float64:
        border_array = np.ascontiguousarray(border_array, dtype=np.float64)
    
    # Calculate size for parallelization
    if coords_ndim == 3:
        if coords.shape[2] == 2:
            total_size = coords.shape[0] * coords.shape[1]
        else:
            total_size = coords.shape[1]
    elif coords_ndim == 4:
        total_size = coords.shape[1] * coords.shape[2]
    else:
        total_size = coords.shape[0]
    
    if total_size < MIN_PARALLEL_SIZE and num_threads < 0:
        use_threads = 1
    
    # Default border mode
    if border_mode is None:
        border_mode = BORDER_CONSTANT
    
    # Default distance mode
    if distance_mode is None:
        distance_mode = ALPHA_MAX
    
    # Single channel
    if line0.ndim == 1:
        bm_scalar = int(border_mode) if not isinstance(border_mode, np.ndarray) else int(border_mode[0])
        dm_scalar = int(distance_mode) if not isinstance(distance_mode, np.ndarray) else int(distance_mode[0])
        
        if x_discrete:
            if coords_ndim == 3 and coords.shape[2] == 2:
                return lerp_between_lines_x_discrete_1ch_array_border(
                    line0, line1, coords, border_array,
                    bm_scalar, border_feathering, dm_scalar, use_threads
                )
            elif coords_ndim == 2 and coords.shape[1] == 2:
                return lerp_between_lines_x_discrete_flat_1ch_array_border(
                    line0, line1, coords, border_array,
                    bm_scalar, border_feathering, dm_scalar, use_threads
                )
            else:
                raise ValueError(f"Invalid coords shape for single channel x_discrete: {(<object>coords).shape}")
        
        if coords_ndim == 3 and coords.shape[2] == 2:
            return lerp_between_lines_1ch_array_border(
                line0, line1, coords, border_array,
                bm_scalar, border_feathering, dm_scalar, use_threads  # UPDATED
            )
        elif coords_ndim == 2 and coords.shape[1] == 2:
            return lerp_between_lines_flat_1ch_array_border(
                line0, line1, coords, border_array,
                bm_scalar, border_feathering, dm_scalar, use_threads  # UPDATED
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
        
        # Convert distance_mode to array
        if isinstance(distance_mode, (int, np.integer)):
            dm_arr = np.full(C, int(distance_mode), dtype=np.int32)
        else:
            dm_arr = np.ascontiguousarray(distance_mode, dtype=np.int32)
            if dm_arr.shape[0] != C:
                raise ValueError(f"distance_mode must have length {C}")
        
        # Grid coordinates (H, W, 2)
        if coords_ndim == 3 and coords.shape[2] == 2:
            if x_discrete:
                return lerp_between_lines_x_discrete_multichannel_array_border(
                    line0, line1, coords, border_array, bm_arr,
                    border_feathering, dm_arr, use_threads  # UPDATED
                )
            else:
                return lerp_between_lines_multichannel_array_border(
                    line0, line1, coords, border_array, bm_arr,
                    border_feathering, dm_arr, use_threads  # UPDATED
                )
        
        # Flat coordinates (N, 2)
        elif coords_ndim == 2 and coords.shape[1] == 2:
            if x_discrete:
                return lerp_between_lines_x_discrete_flat_multichannel_array_border(
                    line0, line1, coords, border_array, bm_arr,
                    border_feathering, dm_arr, use_threads
                )
            return lerp_between_lines_flat_multichannel_array_border(
                line0, line1, coords, border_array, bm_arr,
                border_feathering, dm_arr, use_threads  # UPDATED
            )
        
        # Per-channel grid coordinates (C, H, W, 2)
        elif coords_ndim == 4 and coords.shape[3] == 2:
            if coords.shape[0] != C:
                raise ValueError(f"coords channels ({int(coords.shape[0])}) must match lines ({C})")
            
            if x_discrete:
                return lerp_between_lines_x_discrete_per_ch_coords_array_border(
                    line0, line1, coords, border_array, bm_arr,
                    border_feathering, dm_arr, use_threads  # UPDATED
                )
            else:
                return lerp_between_lines_multichannel_per_ch_coords_array_border(
                    line0, line1, coords, border_array, bm_arr,
                    border_feathering, dm_arr, use_threads  # UPDATED
                )
        
        # Per-channel flat coordinates (C, N, 2)
        elif coords_ndim == 3 and coords.shape[2] == 2 and coords.shape[0] == C:
            if x_discrete:
                return lerp_between_lines_x_discrete_flat_per_ch_coords_array_border(
                    line0, line1, coords, border_array, bm_arr,
                    border_feathering, dm_arr, use_threads
                )
            return lerp_between_lines_flat_multichannel_per_ch_coords_array_border(
                line0, line1, coords, border_array, bm_arr,
                border_feathering, dm_arr, use_threads  # UPDATED
            )
        
        else:
            raise ValueError(f"Invalid coords shape for multi-channel: {(<object>coords).shape}")
    
    raise ValueError(f"Unsupported line dimensions: {line0.ndim}")


# =============================================================================
# In-place Kernels (write directly to border_array)
# =============================================================================
cdef inline void _lerp_multichannel_inplace_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, ::1] c, f64[:, :, ::1] inout_mv,
    const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,  # NEW: distance mode array
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Multi-channel in-place: modifies the array directly."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            for ch in range(C):
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                               border_feathering, distance_modes_mv[ch])  # UPDATED
                
                if not border_res.use_border_directly:
                    edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                        border_res.u_y_final, L, ch)
                    if border_res.blend_factor > 0.0:
                        border_val = inout_mv[h, w, ch]
                        inout_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        inout_mv[h, w, ch] = edge_val
                # If use_border_directly, we leave inout_mv unchanged


cdef inline void _lerp_1ch_inplace_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, :, ::1] c, f64[:, ::1] inout_mv,
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,  # NEW: distance mode parameter
) noexcept nogil:
    """Single-channel in-place: modifies the array directly."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            border_res = _process_border_2d(u_x, u_y, border_mode, 
                                           border_feathering, distance_mode)  # UPDATED
            
            if not border_res.use_border_directly:
                edge_val = _interp_line_1ch(l0, l1, border_res.u_x_final, 
                                           border_res.u_y_final, L)
                if border_res.blend_factor > 0.0:
                    border_val = inout_mv[h, w]
                    inout_mv[h, w] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    inout_mv[h, w] = edge_val


cdef inline void _lerp_flat_multichannel_inplace_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, ::1] c, f64[:, ::1] inout_mv,
    const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,  # NEW: distance mode array
    f64 border_feathering, Py_ssize_t N, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Flat multi-channel in-place."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        for ch in range(C):
            border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                           border_feathering, distance_modes_mv[ch])  # UPDATED
            
            if not border_res.use_border_directly:
                edge_val = _interp_line_multichannel(l0, l1, border_res.u_x_final, 
                                                    border_res.u_y_final, L, ch)
                if border_res.blend_factor > 0.0:
                    border_val = inout_mv[n, ch]
                    inout_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    inout_mv[n, ch] = edge_val


cdef inline void _lerp_flat_1ch_inplace_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, ::1] c, f64[::1] inout_mv,
    f64 border_feathering,
    Py_ssize_t N, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,  # NEW: distance mode parameter
) noexcept nogil:
    """Flat single-channel in-place."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, edge_val, border_val
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        border_res = _process_border_2d(u_x, u_y, border_mode, 
                                       border_feathering, distance_mode)  # UPDATED
        
        if not border_res.use_border_directly:
            edge_val = _interp_line_1ch(l0, l1, border_res.u_x_final, 
                                       border_res.u_y_final, L)
            if border_res.blend_factor > 0.0:
                border_val = inout_mv[n]
                inout_mv[n] = edge_val + border_res.blend_factor * (border_val - edge_val)
            else:
                inout_mv[n] = edge_val


# =============================================================================
# Discrete X-Sampling In-place Kernels
# =============================================================================
cdef inline void _lerp_x_discrete_1ch_inplace_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, :, ::1] c, f64[:, ::1] inout_mv,
    f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Single-channel discrete x-sampling in-place."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val, border_val
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f
    cdef Py_ssize_t idx
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            border_res = _process_border_2d(u_x, u_y, border_mode, 
                                           border_feathering, distance_mode)
            
            if not border_res.use_border_directly:
                idx_f = border_res.u_x_final * L_minus_1
                idx = <Py_ssize_t>floor(idx_f + 0.5)
                
                if idx < 0:
                    idx = 0
                elif idx >= L:
                    idx = L - 1
                
                edge_val = l0[idx] + border_res.u_y_final * (l1[idx] - l0[idx])
                if border_res.blend_factor > 0.0:
                    border_val = inout_mv[h, w]
                    inout_mv[h, w] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    inout_mv[h, w] = edge_val


cdef inline void _lerp_x_discrete_flat_1ch_inplace_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, ::1] c, f64[::1] inout_mv,
    f64 border_feathering,
    Py_ssize_t N, Py_ssize_t L,
    int border_mode, int num_threads,
    i32 distance_mode,
) noexcept nogil:
    """Flat single-channel discrete x-sampling in-place."""
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, edge_val, border_val
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f
    cdef Py_ssize_t idx
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        border_res = _process_border_2d(u_x, u_y, border_mode, 
                                       border_feathering, distance_mode)
        
        if not border_res.use_border_directly:
            idx_f = border_res.u_x_final * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)
            
            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1
            
            edge_val = l0[idx] + border_res.u_y_final * (l1[idx] - l0[idx])
            if border_res.blend_factor > 0.0:
                border_val = inout_mv[n]
                inout_mv[n] = edge_val + border_res.blend_factor * (border_val - edge_val)
            else:
                inout_mv[n] = edge_val


cdef inline void _lerp_x_discrete_multichannel_inplace_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, :, ::1] c, f64[:, :, ::1] inout_mv,
    const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,
    f64 border_feathering, Py_ssize_t H, Py_ssize_t W, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Multi-channel discrete x-sampling in-place."""
    cdef Py_ssize_t h, w, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f
    cdef Py_ssize_t idx
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            for ch in range(C):
                border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                               border_feathering, distance_modes_mv[ch])
                
                if not border_res.use_border_directly:
                    idx_f = border_res.u_x_final * L_minus_1
                    idx = <Py_ssize_t>floor(idx_f + 0.5)
                    
                    if idx < 0:
                        idx = 0
                    elif idx >= L:
                        idx = L - 1
                    
                    edge_val = l0[idx, ch] + border_res.u_y_final * (l1[idx, ch] - l0[idx, ch])
                    if border_res.blend_factor > 0.0:
                        border_val = inout_mv[h, w, ch]
                        inout_mv[h, w, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                    else:
                        inout_mv[h, w, ch] = edge_val


cdef inline void _lerp_x_discrete_flat_multichannel_inplace_kernel(
    const f64[:, ::1] l0, const f64[:, ::1] l1,
    const f64[:, ::1] c, f64[:, ::1] inout_mv,
    const i32[::1] border_modes_mv,
    const i32[::1] distance_modes_mv,
    f64 border_feathering, Py_ssize_t N, Py_ssize_t L, Py_ssize_t C,
    int num_threads,
) noexcept nogil:
    """Flat multi-channel discrete x-sampling in-place."""
    cdef Py_ssize_t n, ch
    cdef f64 u_x, u_y, edge_val, border_val
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f
    cdef Py_ssize_t idx
    cdef BorderResult border_res
    
    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        for ch in range(C):
            border_res = _process_border_2d(u_x, u_y, border_modes_mv[ch], 
                                           border_feathering, distance_modes_mv[ch])
            
            if not border_res.use_border_directly:
                idx_f = border_res.u_x_final * L_minus_1
                idx = <Py_ssize_t>floor(idx_f + 0.5)
                
                if idx < 0:
                    idx = 0
                elif idx >= L:
                    idx = L - 1
                
                edge_val = l0[idx, ch] + border_res.u_y_final * (l1[idx, ch] - l0[idx, ch])
                if border_res.blend_factor > 0.0:
                    border_val = inout_mv[n, ch]
                    inout_mv[n, ch] = edge_val + border_res.blend_factor * (border_val - edge_val)
                else:
                    inout_mv[n, ch] = edge_val


# =============================================================================
# Public API - In-place Functions
# =============================================================================
def lerp_between_lines_1ch_inplace(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] target_array,
    int border_mode=BORDER_CONSTANT,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,  # NEW: distance mode parameter
    int num_threads=-1,
):
    """
    Single-channel in-place interpolation onto target array.
    
    Modifies target_array directly - in-bounds regions are overwritten,
    out-of-bounds regions are left unchanged (or blended with feathering).
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
        target_array: Array to modify in-place, shape (H, W)
        border_mode: Border handling mode
        border_feathering: Feathering width
        distance_mode: Distance metric for 2D border computation  # NEW
        num_threads: Number of threads (-1 = auto)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if target_array.shape[0] != H or target_array.shape[1] != W:
        raise ValueError(f"target_array must have shape ({H}, {W})")

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
    if not target_array.flags['C_CONTIGUOUS']:
        raise ValueError("target_array must be C-contiguous for in-place operation")

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, ::1] inout_mv = target_array

    with nogil:
        _lerp_1ch_inplace_kernel(l0, l1, c, inout_mv, border_feathering,
                                 H, W, L, border_mode, n_threads, distance_mode)  # UPDATED


def lerp_between_lines_multichannel_inplace(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] target_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: can be int or array
    int num_threads=-1,
):
    """
    Multi-channel in-place interpolation onto target array.
    
    Modifies target_array directly - in-bounds regions are overwritten,
    out-of-bounds regions are left unchanged (or blended with feathering).
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
        target_array: Array to modify in-place, shape (H, W, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)  # NEW
        num_threads: Number of threads (-1 = auto)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if target_array.shape[0] != H or target_array.shape[1] != W or target_array.shape[2] != C:
        raise ValueError(f"target_array must have shape ({H}, {W}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not target_array.flags['C_CONTIGUOUS']:
        raise ValueError("target_array must be C-contiguous for in-place operation")
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, :, ::1] inout_mv = target_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    with nogil:
        _lerp_multichannel_inplace_kernel(
            l0, l1, c, inout_mv, bm, dm, border_feathering,  # UPDATED
            H, W, L, C, n_threads
        )


def lerp_between_lines_flat_1ch_inplace(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] target_array,
    int border_mode=BORDER_CONSTANT,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,  # NEW: distance mode parameter
    int num_threads=-1,
):
    """
    Flat single-channel in-place interpolation.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Flat coordinates, shape (N, 2)
        target_array: Array to modify in-place, shape (N,)
        border_mode: Border handling mode
        border_feathering: Feathering width
        distance_mode: Distance metric for 2D border computation  # NEW
        num_threads: Number of threads (-1 = auto)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if target_array.shape[0] != N:
        raise ValueError(f"target_array must have shape ({N},)")

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
    if not target_array.flags['C_CONTIGUOUS']:
        raise ValueError("target_array must be C-contiguous for in-place operation")

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[::1] inout_mv = target_array

    with nogil:
        _lerp_flat_1ch_inplace_kernel(l0, l1, c, inout_mv, border_feathering,
                                      N, L, border_mode, n_threads, distance_mode)  # UPDATED


def lerp_between_lines_flat_multichannel_inplace(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=2] target_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: can be int or array
    int num_threads=-1,
):
    """
    Flat multi-channel in-place interpolation.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Flat coordinates, shape (N, 2)
        target_array: Array to modify in-place, shape (N, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)  # NEW
        num_threads: Number of threads (-1 = auto)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if target_array.shape[0] != N or target_array.shape[1] != C:
        raise ValueError(f"target_array must have shape ({N}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not target_array.flags['C_CONTIGUOUS']:
        raise ValueError("target_array must be C-contiguous for in-place operation")
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[:, ::1] inout_mv = target_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    with nogil:
        _lerp_flat_multichannel_inplace_kernel(
            l0, l1, c, inout_mv, bm, dm, border_feathering,  # UPDATED
            N, L, C, n_threads
        )


# =============================================================================
# Public API - Discrete X-Sampling In-place Functions
# =============================================================================
def lerp_between_lines_x_discrete_1ch_inplace(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] target_array,
    int border_mode=BORDER_CONSTANT,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,
    int num_threads=-1,
):
    """
    Single-channel discrete x-sampling in-place interpolation.
    
    Modifies target_array directly - in-bounds regions are overwritten,
    out-of-bounds regions are left unchanged (or blended with feathering).
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
        target_array: Array to modify in-place, shape (H, W)
        border_mode: Border handling mode
        border_feathering: Feathering width
        distance_mode: Distance metric for 2D border computation
        num_threads: Number of threads (-1 = auto)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if target_array.shape[0] != H or target_array.shape[1] != W:
        raise ValueError(f"target_array must have shape ({H}, {W})")

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
    if not target_array.flags['C_CONTIGUOUS']:
        raise ValueError("target_array must be C-contiguous for in-place operation")

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, ::1] inout_mv = target_array

    with nogil:
        _lerp_x_discrete_1ch_inplace_kernel(l0, l1, c, inout_mv, border_feathering,
                                           H, W, L, border_mode, n_threads, distance_mode)


def lerp_between_lines_x_discrete_flat_1ch_inplace(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] target_array,
    int border_mode=BORDER_CONSTANT,
    f64 border_feathering=0.0,
    int distance_mode=ALPHA_MAX,
    int num_threads=-1,
):
    """
    Flat single-channel discrete x-sampling in-place interpolation.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Flat coordinates, shape (N, 2)
        target_array: Array to modify in-place, shape (N,)
        border_mode: Border handling mode
        border_feathering: Feathering width
        distance_mode: Distance metric for 2D border computation
        num_threads: Number of threads (-1 = auto)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if target_array.shape[0] != N:
        raise ValueError(f"target_array must have shape ({N},)")

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
    if not target_array.flags['C_CONTIGUOUS']:
        raise ValueError("target_array must be C-contiguous for in-place operation")

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[::1] inout_mv = target_array

    with nogil:
        _lerp_x_discrete_flat_1ch_inplace_kernel(l0, l1, c, inout_mv, border_feathering,
                                                N, L, border_mode, n_threads, distance_mode)


def lerp_between_lines_x_discrete_multichannel_inplace(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] target_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Multi-channel discrete x-sampling in-place interpolation.
    
    Modifies target_array directly - in-bounds regions are overwritten,
    out-of-bounds regions are left unchanged (or blended with feathering).
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
        target_array: Array to modify in-place, shape (H, W, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")
    if target_array.shape[0] != H or target_array.shape[1] != W or target_array.shape[2] != C:
        raise ValueError(f"target_array must have shape ({H}, {W}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not target_array.flags['C_CONTIGUOUS']:
        raise ValueError("target_array must be C-contiguous for in-place operation")
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[:, :, ::1] inout_mv = target_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    with nogil:
        _lerp_x_discrete_multichannel_inplace_kernel(
            l0, l1, c, inout_mv, bm, dm, border_feathering,
            H, W, L, C, n_threads
        )


def lerp_between_lines_x_discrete_flat_multichannel_inplace(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=2] target_array,
    np.ndarray[i32, ndim=1] border_modes,
    f64 border_feathering=0.0,
    object distance_mode=None,
    int num_threads=-1,
):
    """
    Flat multi-channel discrete x-sampling in-place interpolation.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Flat coordinates, shape (N, 2)
        target_array: Array to modify in-place, shape (N, C)
        border_modes: Border mode for each channel, shape (C,)
        border_feathering: Feathering width
        distance_mode: Distance metric - int for all channels, or array shape (C,)
        num_threads: Number of threads (-1 = auto)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if target_array.shape[0] != N or target_array.shape[1] != C:
        raise ValueError(f"target_array must have shape ({N}, {C})")
    if border_modes.shape[0] != C:
        raise ValueError(f"border_modes must have length {C}")

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
    if not target_array.flags['C_CONTIGUOUS']:
        raise ValueError("target_array must be C-contiguous for in-place operation")
    if not border_modes.flags['C_CONTIGUOUS']:
        border_modes = np.ascontiguousarray(border_modes, dtype=np.int32)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[:, ::1] inout_mv = target_array
    cdef i32[::1] bm = border_modes
    cdef i32[::1] dm = dm_arr

    with nogil:
        _lerp_x_discrete_flat_multichannel_inplace_kernel(
            l0, l1, c, inout_mv, bm, dm, border_feathering,
            N, L, C, n_threads
        )


# =============================================================================
# Smart In-place Dispatcher
# =============================================================================
def lerp_between_lines_inplace(
    np.ndarray line0,
    np.ndarray line1,
    np.ndarray coords,
    np.ndarray target_array,
    object border_mode=None,
    f64 border_feathering=0.0,
    object distance_mode=None,  # NEW: distance mode parameter
    int num_threads=-1,
    bint x_discrete=False,  # NEW: x_discrete parameter
):
    """
    Smart dispatcher for in-place interpolation onto a target array.
    
    Modifies target_array directly - writes interpolated values where
    coordinates are in-bounds, leaves out-of-bounds regions unchanged
    (or blends them with feathering).
    
    This is more efficient than lerp_between_lines_onto_array when you
    want to modify an existing array without creating a copy.
    
    Args:
        line0: First line, shape (L,) or (L, C)
        line1: Second line, shape (L,) or (L, C)
        coords: Coordinate array, shape (H, W, 2) or (N, 2)
        target_array: Array to modify in-place (must be C-contiguous)
        border_mode: Border mode(s):
            - int: Same for all channels (default: BORDER_CONSTANT)
            - list/array: Per-channel modes, shape (C,)
        border_feathering: Feathering width for smooth blending
        distance_mode: Distance metric for 2D border computation:  # NEW
            - None: Use ALPHA_MAX (default)
            - int: Same for all channels
            - list/array: Per-channel modes, shape (C,)
        num_threads: Number of threads (-1=auto, 0=serial, >0=specific)
        x_discrete: If True, use discrete x-sampling (nearest neighbor in x)  # NEW
    
    Example:
        # Modify image in-place with discrete x-sampling
        image = np.random.rand(100, 100, 3)
        lerp_between_lines_inplace(
            line0, line1, coords, image,
            border_feathering=0.1,
            x_discrete=True
        )
        # image is now modified
    """
    cdef Py_ssize_t total_size
    cdef int use_threads = num_threads
    cdef Py_ssize_t C
    cdef np.ndarray[i32, ndim=1] bm_arr
    cdef np.ndarray[i32, ndim=1] dm_arr
    cdef int bm_scalar
    cdef int dm_scalar
    cdef int coords_ndim = coords.ndim
    
    # Convert inputs to proper types (but not target_array - it must stay as-is)
    if line0.dtype != np.float64:
        line0 = np.ascontiguousarray(line0, dtype=np.float64)
    if line1.dtype != np.float64:
        line1 = np.ascontiguousarray(line1, dtype=np.float64)
    if coords.dtype != np.float64:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    # Validate target_array
    if target_array.dtype != np.float64:
        raise ValueError("target_array must have dtype float64")
    if not target_array.flags['C_CONTIGUOUS']:
        raise ValueError("target_array must be C-contiguous for in-place operation")
    
    # Calculate size for parallelization
    if coords_ndim == 3 and coords.shape[2] == 2:
        total_size = coords.shape[0] * coords.shape[1]
    elif coords_ndim == 2 and coords.shape[1] == 2:
        total_size = coords.shape[0]
    else:
        raise ValueError(f"Invalid coords shape: {(<object>coords).shape}")
    
    if total_size < MIN_PARALLEL_SIZE and num_threads < 0:
        use_threads = 1
    
    # Default border mode
    if border_mode is None:
        border_mode = BORDER_CONSTANT
    
    # Default distance mode
    if distance_mode is None:
        distance_mode = ALPHA_MAX
    
    # Single channel
    if line0.ndim == 1:
        bm_scalar = int(border_mode) if not isinstance(border_mode, np.ndarray) else int(border_mode[0])
        dm_scalar = int(distance_mode) if not isinstance(distance_mode, np.ndarray) else int(distance_mode[0])
        
        if coords_ndim == 3 and coords.shape[2] == 2:
            if x_discrete:
                lerp_between_lines_x_discrete_1ch_inplace(
                    line0, line1, coords, target_array,
                    bm_scalar, border_feathering, dm_scalar, use_threads
                )
            else:
                lerp_between_lines_1ch_inplace(
                    line0, line1, coords, target_array,
                    bm_scalar, border_feathering, dm_scalar, use_threads  # UPDATED
                )
        elif coords_ndim == 2 and coords.shape[1] == 2:
            if x_discrete:
                lerp_between_lines_x_discrete_flat_1ch_inplace(
                    line0, line1, coords, target_array,
                    bm_scalar, border_feathering, dm_scalar, use_threads
                )
            else:
                lerp_between_lines_flat_1ch_inplace(
                    line0, line1, coords, target_array,
                    bm_scalar, border_feathering, dm_scalar, use_threads  # UPDATED
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
        
        # Convert distance_mode to array
        if isinstance(distance_mode, (int, np.integer)):
            dm_arr = np.full(C, int(distance_mode), dtype=np.int32)
        else:
            dm_arr = np.ascontiguousarray(distance_mode, dtype=np.int32)
            if dm_arr.shape[0] != C:
                raise ValueError(f"distance_mode must have length {C}")
        
        # Grid coordinates (H, W, 2)
        if coords_ndim == 3 and coords.shape[2] == 2:
            if x_discrete:
                lerp_between_lines_x_discrete_multichannel_inplace(
                    line0, line1, coords, target_array, bm_arr,
                    border_feathering, dm_arr, use_threads
                )
            else:
                lerp_between_lines_multichannel_inplace(
                    line0, line1, coords, target_array, bm_arr,
                    border_feathering, dm_arr, use_threads  # UPDATED
                )
        # Flat coordinates (N, 2)
        elif coords_ndim == 2 and coords.shape[1] == 2:
            if x_discrete:
                lerp_between_lines_x_discrete_flat_multichannel_inplace(
                    line0, line1, coords, target_array, bm_arr,
                    border_feathering, dm_arr, use_threads
                )
            else:
                lerp_between_lines_flat_multichannel_inplace(
                    line0, line1, coords, target_array, bm_arr,
                    border_feathering, dm_arr, use_threads  # UPDATED
                )
        else:
            raise ValueError(f"Invalid coords shape for multi-channel: {(<object>coords).shape}")
    
    else:
        raise ValueError(f"Unsupported line dimensions: {line0.ndim}")


# =============================================================================
# Export border mode constants
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
DIST_EUCLIDEAN = EUCLIDEAN
DIST_WEIGHTED_MINMAX = WEIGHTED_MINMAX