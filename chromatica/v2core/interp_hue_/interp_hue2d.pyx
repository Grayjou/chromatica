# interp_hue2d.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Hue interpolation with cyclical color space support.

Supports four interpolation modes per dimension:
- CW (0): Clockwise interpolation (always positive direction)
- CCW (1): Counterclockwise interpolation (always negative direction)
- SHORTEST (2): Shortest angular distance (≤180°)
- LONGEST (3): Longest angular distance (≥180°)

Functions in this module handle hue interpolation in multi-dimensional spaces
with proper handling of the cyclical nature of hue values (0-360° range).
"""

import numpy as np
cimport numpy as np
from libc.math cimport fmod, floor
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange

# Import border handling and modern features
from ..border_handling cimport (
    handle_border_lines_2d,
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)
from ..interp_utils cimport (
    BorderResult,
    process_border_2d,
    MAX_NORM,
    MANHATTAN,
    SCALED_MANHATTAN,
    ALPHA_MAX,
    ALPHA_MAX_SIMPLE,
    TAYLOR,
    EUCLIDEAN,
    WEIGHTED_MINMAX,
)
from .interp_hue_utils cimport (
    f64,                   
    i32,                   
    HueMode,
    HUE_CW,
    HUE_CCW,
    HUE_SHORTEST,
    HUE_LONGEST,
    wrap_hue,               
    adjust_end_for_mode,    
    lerp_hue_single,
    _interp_line_1ch_hue,
    _interp_line_discrete_hue,
    process_hue_border_2d,
)

# =============================================================================
# Modern Hue Kernels with Feathering and Border Support
# =============================================================================
cdef inline void _lerp_1ch_hue_feathered_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, :, ::1] c, f64[:, ::1] out_mv,
    f64 border_constant, f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t L,
    int border_mode, int num_threads,
    int mode_x, int mode_y,
    i32 distance_mode,
) noexcept nogil:
    """Single-channel hue interpolation with feathering and modern border support."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val, border_val_wrapped
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            border_res = process_hue_border_2d(u_x, u_y, border_mode, 
                                               border_feathering, distance_mode)
            
            if border_res.use_border_directly:
                out_mv[h, w] = wrap_hue(border_constant)
            else:
                edge_val = _interp_line_1ch_hue(l0, l1, border_res.u_x_final, 
                                               border_res.u_y_final, L,
                                               mode_x, mode_y)
                if border_res.blend_factor > 0.0:
                    # Blend with border using shortest hue path
                    border_val_wrapped = wrap_hue(border_constant)
                    out_mv[h, w] = lerp_hue_single(edge_val, border_val_wrapped,
                                                   border_res.blend_factor,
                                                   HUE_SHORTEST)
                else:
                    out_mv[h, w] = edge_val


cdef inline void _lerp_1ch_hue_array_border_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, :, ::1] c, f64[:, ::1] out_mv,
    const f64[:, ::1] border_array_mv, f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t L,
    int border_mode, int num_threads,
    int mode_x, int mode_y,
    i32 distance_mode,
) noexcept nogil:
    """Hue interpolation with array-based border values and feathering."""
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val, border_val_wrapped
    cdef BorderResult border_res
    
    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            border_res = process_hue_border_2d(u_x, u_y, border_mode, 
                                               border_feathering, distance_mode)
            
            if border_res.use_border_directly:
                out_mv[h, w] = wrap_hue(border_array_mv[h, w])
            else:
                edge_val = _interp_line_1ch_hue(l0, l1, border_res.u_x_final,
                                               border_res.u_y_final, L,
                                               mode_x, mode_y)
                if border_res.blend_factor > 0.0:
                    border_val_wrapped = wrap_hue(border_array_mv[h, w])
                    out_mv[h, w] = lerp_hue_single(edge_val, border_val_wrapped,
                                                   border_res.blend_factor,
                                                   HUE_SHORTEST)
                else:
                    out_mv[h, w] = edge_val


# =============================================================================
# 2D Hue Interpolation: coeffs (H, W, N), modes (N,) -> output (H, W)
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_spatial(   # <-- ADD RETURN TYPE
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=3] coeffs,
    np.ndarray[i32, ndim=1] modes,
):
    """
    Multi-dimensional hue interpolation over a 2D spatial grid.
    ... (rest of docstring)
    """
    # ... rest of function body unchanged ...
    cdef Py_ssize_t H = coeffs.shape[0]
    cdef Py_ssize_t W = coeffs.shape[1]
    cdef Py_ssize_t num_dims = coeffs.shape[2]
    cdef Py_ssize_t num_corners = starts.shape[0]
    
    if ends.shape[0] != num_corners:
        raise ValueError("starts and ends must have same length")
    if modes.shape[0] != num_dims:
        raise ValueError("modes must have length equal to num dimensions")
    if num_corners != (1 << (num_dims - 1)):
        raise ValueError(
            f"starts/ends length {num_corners} doesn't match "
            f"2^(num_dims-1)={1 << (num_dims - 1)} for num_dims={num_dims}"
        )
    
    if not starts.flags['C_CONTIGUOUS']:
        starts = np.ascontiguousarray(starts)
    if not ends.flags['C_CONTIGUOUS']:
        ends = np.ascontiguousarray(ends)
    if not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs)
    if not modes.flags['C_CONTIGUOUS']:
        modes = np.ascontiguousarray(modes)
    
    cdef f64[::1] starts_mv = starts
    cdef f64[::1] ends_mv = ends
    cdef f64[:, :, ::1] coeffs_mv = coeffs
    cdef i32[::1] modes_mv = modes
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t h, w, i, j, half, curr_size
    cdef f64 u, h0, h1, h1_adj
    cdef int mode
    
    cdef Py_ssize_t MAX_CORNERS = 256
    cdef f64 a_stack[256]
    cdef f64 b_stack[256]
    cdef f64* a = NULL
    cdef f64* b = NULL
    
    if num_corners <= MAX_CORNERS:
        for h in range(H):
            for w in range(W):
                memcpy(&a_stack[0], &starts_mv[0], num_corners * sizeof(f64))
                memcpy(&b_stack[0], &ends_mv[0], num_corners * sizeof(f64))
                
                curr_size = num_corners
                
                for i in range(num_dims):
                    u = coeffs_mv[h, w, i]
                    mode = modes_mv[i]
                    
                    for j in range(curr_size):
                        h0 = a_stack[j]
                        h1 = b_stack[j]
                        h1_adj = adjust_end_for_mode(h0, h1, mode)
                        a_stack[j] = wrap_hue(h0 + u * (h1_adj - h0))
                    
                    if curr_size > 1:
                        half = curr_size >> 1
                        memcpy(&b_stack[0], &a_stack[half], half * sizeof(f64))
                        curr_size = half
                
                out_mv[h, w] = a_stack[0]
        
        return out
    
    # Heap fallback
    a = <f64*>malloc(num_corners * sizeof(f64))
    b = <f64*>malloc(num_corners * sizeof(f64))
    if a == NULL or b == NULL:
        if a != NULL: free(a)
        if b != NULL: free(b)
        raise MemoryError("Failed to allocate working buffers")
    
    try:
        for h in range(H):
            for w in range(W):
                memcpy(a, &starts_mv[0], num_corners * sizeof(f64))
                memcpy(b, &ends_mv[0], num_corners * sizeof(f64))
                
                curr_size = num_corners
                
                for i in range(num_dims):
                    u = coeffs_mv[h, w, i]
                    mode = modes_mv[i]
                    
                    for j in range(curr_size):
                        h0 = a[j]
                        h1 = b[j]
                        h1_adj = adjust_end_for_mode(h0, h1, mode)
                        a[j] = wrap_hue(h0 + u * (h1_adj - h0))
                    
                    if curr_size > 1:
                        half = curr_size >> 1
                        memcpy(b, a + half, half * sizeof(f64))
                        curr_size = half
                
                out_mv[h, w] = a[0]
    finally:
        free(a)
        free(b)
    
    return out


# =============================================================================
# Hue interpolation between lines (Section 6 style)
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_x = HUE_SHORTEST,
    int mode_y = HUE_SHORTEST,
    object border_mode = 'clamp',
    f64 border_constant = 0.0,
    f64 border_feathering = 0.0,
    object distance_mode = 'euclidean',
    int num_threads = 1,
):
    """
    Interpolate hue between two 1D hue lines with 2D coordinate mapping.

    Adds border feathering, distance mode selection, and threaded execution
    using the modern border processor. Border modes accept either string
    identifiers ("constant", "clamp", "reflect", "periodic") or integer
    constants. Distance mode accepts strings or integer constants from
    `interp_utils`.
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

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    cdef int border_mode_int = _border_mode_to_int(border_mode)
    cdef i32 dist_mode_int = _distance_mode_to_int(distance_mode)
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    
    _lerp_1ch_hue_feathered_kernel(
        line0, line1, coords, out,
        border_constant, border_feathering,
        H, W, L, border_mode_int, n_threads,
        mode_x, mode_y, dist_mode_int,
    )
    
    return out


# =============================================================================
# Hue interpolation between lines with discrete x-sampling (Section 6 style)
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_x_discrete(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_y = HUE_SHORTEST,
    object border_mode = 'clamp',
    f64 border_constant = 0.0,
    f64 border_feathering = 0.0,
    object distance_mode = 'euclidean',
    int num_threads = 1,
):
    """
    Interpolate hue between two 1D hue lines with nearest-neighbor x-sampling.

    Upgraded to use modern border handling with feathering and distance modes.
    Accepts border/distance identifiers as strings or ints and supports
    threaded execution.
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

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    cdef int border_mode_int = _border_mode_to_int(border_mode)
    cdef i32 dist_mode_int = _distance_mode_to_int(distance_mode)
    cdef f64 wrapped_border = wrap_hue(border_constant)

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    cdef Py_ssize_t h, w
    cdef Py_ssize_t idx
    cdef f64 u_x, u_y, idx_f
    cdef f64 v0, v1, edge_val, border_val
    cdef BorderResult border_res
    cdef f64 L_minus_1 = <f64>(L - 1)

    if L == 1:
        v0 = l0[0]
        v1 = l1[0]
        for h in prange(H, nogil=True, schedule='static', num_threads=n_threads):
            for w in range(W):
                u_x = c[h, w, 0]
                u_y = c[h, w, 1]
                
                border_res = process_hue_border_2d(u_x, u_y, border_mode_int,
                                                   border_feathering, dist_mode_int)
                if border_res.use_border_directly:
                    out_mv[h, w] = wrapped_border
                    continue

                edge_val = lerp_hue_single(v0, v1, border_res.u_y_final, mode_y)
                if border_res.blend_factor > 0.0:
                    out_mv[h, w] = lerp_hue_single(edge_val, wrapped_border,
                                                   border_res.blend_factor,
                                                   HUE_SHORTEST)
                else:
                    out_mv[h, w] = edge_val
        return out

    for h in prange(H, nogil=True, schedule='static', num_threads=n_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            border_res = process_hue_border_2d(u_x, u_y, border_mode_int,
                                               border_feathering, dist_mode_int)
            if border_res.use_border_directly:
                out_mv[h, w] = wrapped_border
                continue
            
            idx_f = border_res.u_x_final * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)
            
            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1
            
            v0 = l0[idx]
            v1 = l1[idx]
            edge_val = lerp_hue_single(v0, v1, border_res.u_y_final, mode_y)

            if border_res.blend_factor > 0.0:
                border_val = wrapped_border
                out_mv[h, w] = lerp_hue_single(edge_val, border_val,
                                               border_res.blend_factor,
                                               HUE_SHORTEST)
            else:
                out_mv[h, w] = edge_val

    return out


# =============================================================================
# 2D Grid Hue Lerp with per-pixel mode
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_with_modes(
    np.ndarray[f64, ndim=2] h0_grid,
    np.ndarray[f64, ndim=2] h1_grid,
    np.ndarray[f64, ndim=2] coeffs,
    np.ndarray[i32, ndim=2] modes,
):
    """
    2D grid interpolation with per-pixel start, end, coefficient, and mode.
    
    Each pixel in the output is interpolated independently using its own
    start hue, end hue, interpolation coefficient, and interpolation mode.
    
    Args:
        h0_grid: Grid of start hue values, shape (H, W)
        h1_grid: Grid of end hue values, shape (H, W)
                 Must match shape of h0_grid
        coeffs: Grid of interpolation coefficients, shape (H, W)
                Each coefficient should typically be in [0, 1]
        modes: Grid of interpolation modes, shape (H, W)
               Each element must be one of:
               0=CW, 1=CCW, 2=SHORTEST, 3=LONGEST
    
    Returns:
        np.ndarray: Interpolated hue grid, shape (H, W)
                    Values are wrapped to [0, 360) range.
    
    Notes:
        - Useful for spatial varying interpolation parameters
        - Each pixel's interpolation is independent
    """
    cdef Py_ssize_t H = h0_grid.shape[0]
    cdef Py_ssize_t W = h0_grid.shape[1]
    
    if h1_grid.shape[0] != H or h1_grid.shape[1] != W:
        raise ValueError("h1_grid shape mismatch")
    if coeffs.shape[0] != H or coeffs.shape[1] != W:
        raise ValueError("coeffs shape mismatch")
    if modes.shape[0] != H or modes.shape[1] != W:
        raise ValueError("modes shape mismatch")
    
    if not h0_grid.flags['C_CONTIGUOUS']:
        h0_grid = np.ascontiguousarray(h0_grid)
    if not h1_grid.flags['C_CONTIGUOUS']:
        h1_grid = np.ascontiguousarray(h1_grid)
    if not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs)
    if not modes.flags['C_CONTIGUOUS']:
        modes = np.ascontiguousarray(modes)
    
    cdef f64[:, ::1] h0_mv = h0_grid
    cdef f64[:, ::1] h1_mv = h1_grid
    cdef f64[:, ::1] c_mv = coeffs
    cdef i32[:, ::1] m_mv = modes
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t h, w
    cdef f64 h0, h1, u, h1_adj
    cdef int mode
    
    for h in range(H):
        for w in range(W):
            h0 = h0_mv[h, w]
            h1 = h1_mv[h, w]
            u = c_mv[h, w]
            mode = m_mv[h, w]
            
            h1_adj = adjust_end_for_mode(h0, h1, mode)
            out_mv[h, w] = wrap_hue(h0 + u * (h1_adj - h0))
    
    return out

# =============================================================================
# Helper Functions for Mode Conversion
# =============================================================================
cdef i32 _distance_mode_to_int(object distance_mode):
    """Convert distance mode identifier (str/int) to integer constant."""
    if isinstance(distance_mode, str):
        if distance_mode == 'max_norm':
            return MAX_NORM
        elif distance_mode == 'manhattan':
            return MANHATTAN
        elif distance_mode == 'scaled_manhattan':
            return SCALED_MANHATTAN
        elif distance_mode == 'alpha_max':
            return ALPHA_MAX
        elif distance_mode == 'alpha_max_simple':
            return ALPHA_MAX_SIMPLE
        elif distance_mode == 'taylor':
            return TAYLOR
        elif distance_mode == 'euclidean':
            return EUCLIDEAN
        elif distance_mode == 'weighted_minmax':
            return WEIGHTED_MINMAX
        else:
            return ALPHA_MAX  # Default
    try:
        return <i32>distance_mode
    except Exception:
        return ALPHA_MAX


cdef int _border_mode_to_int(object border_mode):
    """Convert border mode identifier (str/int) to integer constant."""
    if isinstance(border_mode, str):
        if border_mode == 'constant':
            return BORDER_CONSTANT
        elif border_mode == 'clamp':
            return BORDER_CLAMP
        elif border_mode == 'reflect':
            return BORDER_MIRROR
        elif border_mode == 'periodic':
            return BORDER_REPEAT
        else:
            return BORDER_CONSTANT  # Default
    try:
        return <int>border_mode
    except Exception:
        return BORDER_CONSTANT


# =============================================================================
# Public API: Modern Feathered Hue Interpolation
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_feathered(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=3] c,
    f64 border_constant,
    f64 border_feathering=0.0,
    str border_mode='constant',
    int mode_x=HUE_SHORTEST,
    int mode_y=HUE_SHORTEST,
    str distance_mode='euclidean',
    int num_threads=1,
):
    """Hue interpolation with feathering and constant border value."""
    cdef Py_ssize_t H = c.shape[0], W = c.shape[1], L = l0.shape[0]
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef i32 dist_mode_int = _distance_mode_to_int(distance_mode)
    cdef int border_mode_int = _border_mode_to_int(border_mode)
    
    _lerp_1ch_hue_feathered_kernel(
        l0, l1, c, out,
        border_constant, border_feathering,
        H, W, L, border_mode_int, num_threads,
        mode_x, mode_y, dist_mode_int,
    )
    return out


cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_array_border(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=3] c,
    np.ndarray[f64, ndim=2] border_array,
    f64 border_feathering=0.0,
    str border_mode='constant',
    int mode_x=HUE_SHORTEST,
    int mode_y=HUE_SHORTEST,
    str distance_mode='euclidean',
    int num_threads=1,
):
    """Hue interpolation with array-based border values."""
    if border_array.shape[0] != c.shape[0] or border_array.shape[1] != c.shape[1]:
        raise ValueError("border_array shape must match c[:, :, 0]")
    
    cdef Py_ssize_t H = c.shape[0], W = c.shape[1], L = l0.shape[0]
    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef i32 dist_mode_int = _distance_mode_to_int(distance_mode)
    cdef int border_mode_int = _border_mode_to_int(border_mode)
    
    _lerp_1ch_hue_array_border_kernel(
        l0, l1, c, out, border_array, border_feathering,
        H, W, L, border_mode_int, num_threads,
        mode_x, mode_y, dist_mode_int,
    )
    return out
