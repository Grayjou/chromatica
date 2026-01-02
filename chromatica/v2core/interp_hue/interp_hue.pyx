# interp_hue.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Hue interpolation with cyclical color space support.

Supports four interpolation modes per dimension:
- CW (1): Clockwise interpolation (always positive direction)
- CCW (2): Counterclockwise interpolation (always negative direction)
- SHORTEST (3): Shortest angular distance (≤180°)
- LONGEST (4): Longest angular distance (≥180°)
"""

import numpy as np
cimport numpy as np
from libc.math cimport fmod, floor
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free

# Import border handling enums
from ..border_handling_ cimport (
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)
from ..interp_utils cimport (
    EUCLIDEAN,
    ALPHA_MAX,
    MANHATTAN,
    MAX_NORM,
)
from .interp_hue_utils cimport (
    f64,
    i32,
    HUE_CW,
    HUE_CCW,
    HUE_SHORTEST,
    HUE_LONGEST,
    wrap_hue,
    adjust_end_for_mode,
    lerp_hue_single,
)

# Import 2D functions
from .interp_hue2d cimport (
    hue_lerp_2d_spatial,
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
    hue_lerp_2d_with_modes,
)
from .interp_hue2d_array_border cimport (
    hue_lerp_between_lines_array_border,
    hue_lerp_between_lines_array_border_flat,
    hue_lerp_between_lines_array_border_x_discrete,
    hue_lerp_between_lines_array_border_flat_x_discrete,
)
from .interp_hue_corners cimport (
    hue_lerp_from_corners_1ch,
    hue_lerp_from_corners_1ch_flat,
    hue_lerp_from_corners_multichannel,
    hue_lerp_from_corners_multichannel_per_ch_modes,
    hue_lerp_from_corners_multichannel_flat,
    hue_lerp_from_corners_per_ch_coords,
)
from .interp_hue_corners_array_border cimport (
    hue_lerp_from_corners_array_border,
    hue_lerp_from_corners_flat_array_border,
    hue_lerp_from_corners_multichannel_array_border,
    hue_lerp_from_corners_multichannel_per_ch_modes_array_border,
    hue_lerp_from_corners_multichannel_flat_array_border,
)


# =============================================================================
# 1D Hue Interpolation: coeffs (L, N), modes (N,) -> output (L,)
# =============================================================================
cpdef np.ndarray[f64, ndim=1] hue_lerp_1d_spatial(
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=2] coeffs,
    np.ndarray[i32, ndim=1] modes,
):
    """
    Multi-dimensional hue interpolation along a 1D spatial grid.
    
    Performs recursive bilinear-like interpolation across N dimensions using
    hue-aware interpolation modes for each dimension.
    
    Args:
        starts: Start hue values for corner pairs, shape (2^{N-1},)
        ends: End hue values for corner pairs, shape (2^{N-1},)
        coeffs: Interpolation coefficients for L points across N dimensions,
                shape (L, N).
        modes: Interpolation mode for each dimension, shape (N,)
               1=CW, 2=CCW, 3=SHORTEST, 4=LONGEST
    
    Returns:
        np.ndarray: Interpolated hue values, shape (L,), wrapped to [0, 360)
    """
    cdef Py_ssize_t num_points = coeffs.shape[0]
    cdef Py_ssize_t num_dims = coeffs.shape[1]
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
    cdef f64[:, ::1] coeffs_mv = coeffs
    cdef i32[::1] modes_mv = modes
    
    cdef np.ndarray[f64, ndim=1] result = np.empty(num_points, dtype=np.float64)
    cdef f64[::1] result_mv = result
    
    cdef Py_ssize_t p, i, j, half, curr_size
    cdef f64 u, h0, h1, h1_adj
    cdef int mode
    
    cdef Py_ssize_t MAX_CORNERS = 256
    cdef f64 a_stack[256]
    cdef f64 b_stack[256]
    cdef f64* a = NULL
    cdef f64* b = NULL
    
    if num_corners <= MAX_CORNERS:
        for p in range(num_points):
            memcpy(&a_stack[0], &starts_mv[0], num_corners * sizeof(f64))
            memcpy(&b_stack[0], &ends_mv[0], num_corners * sizeof(f64))
            
            curr_size = num_corners
            
            for i in range(num_dims):
                u = coeffs_mv[p, i]
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
            
            result_mv[p] = a_stack[0]
        
        return result
    
    # Heap fallback
    a = <f64*>malloc(num_corners * sizeof(f64))
    b = <f64*>malloc(num_corners * sizeof(f64))
    if a == NULL or b == NULL:
        if a != NULL: free(a)
        if b != NULL: free(b)
        raise MemoryError("Failed to allocate working buffers")
    
    try:
        for p in range(num_points):
            memcpy(a, &starts_mv[0], num_corners * sizeof(f64))
            memcpy(b, &ends_mv[0], num_corners * sizeof(f64))
            
            curr_size = num_corners
            
            for i in range(num_dims):
                u = coeffs_mv[p, i]
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
            
            result_mv[p] = a[0]
    finally:
        free(a)
        free(b)
    
    return result

from ..interp_utils cimport (
    BORDER_CLAMP,
    BORDER_OVERFLOW,
    BorderResult1D,
    clamp_01,
    handle_border_1d,
    process_border_1d,
)

# =============================================================================
# Simple 1D Hue Lerp
# =============================================================================
cpdef np.ndarray[f64, ndim=1] hue_lerp_simple(
    f64 h0,
    f64 h1,
    np.ndarray[f64, ndim=1] coeffs,
    i32 mode = HUE_SHORTEST,
):
    """
    Simple 1D hue interpolation between two values for multiple coefficients.
    """
    cdef Py_ssize_t N = coeffs.shape[0]
    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out
    cdef f64[::1] c = coeffs
    
    cdef f64 h1_adj = adjust_end_for_mode(h0, h1, mode)
    cdef Py_ssize_t i
    cdef f64 u
    
    for i in range(N):
        u = c[i]
        out_mv[i] = wrap_hue(h0 + u * (h1_adj - h0))
    
    return out

cpdef np.ndarray[f64, ndim=1] hue_lerp_1d_full(
    f64 start_hue,
    f64 end_hue,
    np.ndarray[f64, ndim=1] coords,
    i32 hue_direction = HUE_SHORTEST,
    int border_mode = BORDER_CLAMP,
    f64 border_constant = 0.0,
    object border_array = None,  # <- Changed from np.ndarray[f64, ndim=1]
    f64 border_feathering = 0.0,
    i32 feather_hue_mode = HUE_SHORTEST,
):
    """
    Full-featured 1D hue interpolation with border handling and feathering.
    
    ...docstring...
    """
    cdef Py_ssize_t N = coords.shape[0]
    cdef Py_ssize_t i
    cdef f64 t, interp_val, border_val
    cdef f64 h0 = wrap_hue(start_hue)
    cdef f64 h1 = wrap_hue(end_hue)
    cdef f64 h1_adj = adjust_end_for_mode(h0, h1, hue_direction)
    cdef bint has_border_array = border_array is not None
    cdef bint use_feathering = border_feathering > 0.0
    cdef BorderResult1D br
    
    cdef f64[::1] coords_mv
    cdef f64[::1] border_mv
    cdef f64[::1] out_mv
    
    # Typed local variable for border_array
    cdef np.ndarray[f64, ndim=1] border_arr_typed
    
    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    coords_mv = coords
    out_mv = out
    
    if has_border_array:
        border_arr_typed = np.ascontiguousarray(border_array, dtype=np.float64)
        if border_arr_typed.shape[0] != N:
            raise ValueError(
                f"border_array length ({border_arr_typed.shape[0]}) must match "
                f"coords length ({N})"
            )
        border_mv = border_arr_typed
    
    # Fast path: CLAMP without feathering
    if border_mode == BORDER_CLAMP and not use_feathering:
        for i in range(N):
            t = clamp_01(coords_mv[i])
            out_mv[i] = wrap_hue(h0 + t * (h1_adj - h0))
        return out
    
    # Fast path: REPEAT/MIRROR without feathering
    if (border_mode == BORDER_REPEAT or border_mode == BORDER_MIRROR) and not use_feathering:
        for i in range(N):
            t = handle_border_1d(coords_mv[i], border_mode)
            out_mv[i] = wrap_hue(h0 + t * (h1_adj - h0))
        return out
    
    # Full path: all border modes with potential feathering
    for i in range(N):
        br = process_border_1d(coords_mv[i], border_mode, border_feathering)
        
        # Handle OVERFLOW -> NaN
        if br.use_border_directly and border_mode == BORDER_OVERFLOW:
            out_mv[i] = np.nan
            continue
        
        # Pure border value (no interpolation contribution)
        if br.use_border_directly and border_mode == BORDER_CONSTANT:
            if has_border_array:
                out_mv[i] = wrap_hue(border_mv[i])
            else:
                out_mv[i] = wrap_hue(border_constant)
            continue
        
        # Compute interpolated value
        interp_val = wrap_hue(h0 + br.u_final * (h1_adj - h0))
        
        # Apply feathering blend if needed
        if br.blend_factor > 0.0:
            if has_border_array:
                border_val = wrap_hue(border_mv[i])
            else:
                border_val = wrap_hue(border_constant)
            
            # Blend: border_val -> interp_val as blend_factor goes 1.0 -> 0.0
            out_mv[i] = lerp_hue_single(interp_val, border_val, br.blend_factor, feather_hue_mode)
        else:
            out_mv[i] = interp_val
    
    return out


cpdef np.ndarray[f64, ndim=1] hue_lerp_1d(
    f64 start_hue,
    f64 end_hue,
    np.ndarray[f64, ndim=1] coords,
    i32 hue_direction = HUE_SHORTEST,
):
    """
    Simple 1D hue interpolation without border handling.
    
    Coordinates are clamped to [0, 1]. For more control over boundary
    behavior, use hue_lerp_1d_full().
    
    Args:
        start_hue: Starting hue (degrees)
        end_hue: Ending hue (degrees)
        coords: Interpolation coordinates, shape (N,)
        hue_direction: HUE_CW, HUE_CCW, HUE_SHORTEST, or HUE_LONGEST
    
    Returns:
        Interpolated hue values, shape (N,)
    """
    return hue_lerp_1d_full(
        start_hue, end_hue, coords,
        hue_direction=hue_direction,
        border_mode=BORDER_CLAMP,
        border_constant=0.0,
        border_array=None,
        border_feathering=0.0,
        feather_hue_mode=HUE_SHORTEST,
    )

# =============================================================================
# Vectorized Array Lerp
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_arrays(
    np.ndarray[f64, ndim=1] h0_arr,
    np.ndarray[f64, ndim=1] h1_arr,
    np.ndarray[f64, ndim=1] coeffs,
    i32 mode = HUE_SHORTEST,
):
    """
    Vectorized 1D hue interpolation for multiple hue pairs and coefficients.
    """
    cdef Py_ssize_t M = h0_arr.shape[0]
    cdef Py_ssize_t N = coeffs.shape[0]
    
    if h1_arr.shape[0] != M:
        raise ValueError("h0_arr and h1_arr must have same length")
    
    if not h0_arr.flags['C_CONTIGUOUS']:
        h0_arr = np.ascontiguousarray(h0_arr)
    if not h1_arr.flags['C_CONTIGUOUS']:
        h1_arr = np.ascontiguousarray(h1_arr)
    if not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs)
    
    cdef f64[::1] h0_mv = h0_arr
    cdef f64[::1] h1_mv = h1_arr
    cdef f64[::1] c = coeffs
    
    cdef np.ndarray[f64, ndim=2] out = np.empty((N, M), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out
    
    cdef Py_ssize_t i, j
    cdef f64 u, h0, h1, h1_adj
    
    for j in range(M):
        h0 = h0_mv[j]
        h1 = h1_mv[j]
        h1_adj = adjust_end_for_mode(h0, h1, mode)
        
        for i in range(N):
            u = c[i]
            out_mv[i, j] = wrap_hue(h0 + u * (h1_adj - h0))
    
    return out


# =============================================================================
# Dispatcher for line-based hue interpolation
# =============================================================================
cpdef np.ndarray hue_lerp_between_lines_dispatch(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray coords,
    i32 mode_x = HUE_SHORTEST,
    i32 mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CLAMP,
    f64 border_constant = 0.0,
    np.ndarray border_array = None,
    f64 border_feathering = 0.0,
    i32 feather_hue_mode = HUE_SHORTEST,  # NEW: hue mode for feathering blend
    i32 distance_mode = ALPHA_MAX,
    int num_threads = 1,
    bint x_discrete = False,
):
    """
    Route hue line interpolation to array-border or constant-border paths.
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        coords: Coordinate array, shape (H, W, 2) or (N, 2)
        mode_x: Hue interpolation mode for X axis (int enum)
        mode_y: Hue interpolation mode for Y axis (int enum)
        border_mode: Border handling mode (int enum)
        border_constant: Constant value for BORDER_CONSTANT mode
        border_array: Optional per-pixel border values
        border_feathering: Feathering distance
        feather_hue_mode: Hue interpolation mode for feathering blend (int enum)
        distance_mode: Distance metric for feathering (int enum)
        num_threads: Number of threads (-1 for auto)
        x_discrete: If True, use nearest-neighbor x-sampling
    
    Returns:
        Interpolated hue values with shape matching coords grid
    """
    # Store ndim in cdef variable to avoid Python object conversion
    cdef int coords_ndim = coords.ndim
    cdef Py_ssize_t last_dim


    # Validate coords shape
    if coords_ndim == 3:
        last_dim = coords.shape[2]
        if last_dim != 2:
            raise ValueError(f"coords last dim must be 2, got {last_dim}")
    elif coords_ndim == 2:
        last_dim = coords.shape[1]
        if last_dim != 2:
            raise ValueError(f"coords last dim must be 2, got {last_dim}")
    else:
        raise ValueError(f"coords must be (H, W, 2) or (N, 2), got ndim={coords_ndim}")
    
    if border_array is not None:
        if coords_ndim == 3:
            if x_discrete:
                return hue_lerp_between_lines_array_border_x_discrete(
                    line0, line1, coords, border_array,
                    mode_y=mode_y,
                    border_mode=border_mode,
                    border_feathering=border_feathering,
                    feather_hue_mode=feather_hue_mode,  # Pass through
                    distance_mode=distance_mode,
                    num_threads=num_threads,
                )
            return hue_lerp_between_lines_array_border(
                line0, line1, coords, border_array,
                mode_x=mode_x,
                mode_y=mode_y,
                border_mode=border_mode,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,  # Pass through
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        else:  # coords_ndim == 2
            if x_discrete:
                return hue_lerp_between_lines_array_border_flat_x_discrete(
                    line0, line1, coords, border_array,
                    mode_y=mode_y,
                    border_mode=border_mode,
                    border_feathering=border_feathering,
                    feather_hue_mode=feather_hue_mode,  # Pass through
                    distance_mode=distance_mode,
                    num_threads=num_threads,
                )
            return hue_lerp_between_lines_array_border_flat(
                line0, line1, coords, border_array,
                mode_x=mode_x,
                mode_y=mode_y,
                border_mode=border_mode,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,  # Pass through
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
    
    # No border array - use constant border
    if x_discrete:
        return hue_lerp_between_lines_x_discrete(
            line0, line1, coords,
            mode_y=mode_y,
            border_mode=border_mode,
            border_constant=border_constant,
            border_feathering=border_feathering,
            feather_hue_mode=feather_hue_mode,  # Pass through
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    
    return hue_lerp_between_lines(
        line0, line1, coords,
        mode_x=mode_x,
        mode_y=mode_y,
        border_mode=border_mode,
        border_constant=border_constant,
        border_feathering=border_feathering,
        feather_hue_mode=feather_hue_mode,  # Pass through
        distance_mode=distance_mode,
        num_threads=num_threads,
    )


# =============================================================================
# Dispatcher for corner-based hue interpolation
# =============================================================================
cpdef np.ndarray hue_lerp_from_corners_dispatch(
    np.ndarray corners,
    np.ndarray coords,
    i32 mode_x = HUE_SHORTEST,
    i32 mode_y = HUE_SHORTEST,
    np.ndarray modes_x = None,
    np.ndarray modes_y = None,
    int border_mode = BORDER_CLAMP,
    f64 border_constant = 0.0,
    np.ndarray border_array = None,
    f64 border_feathering = 0.0,
    i32 feather_hue_mode = HUE_SHORTEST,  # NEW: hue mode for feathering blend
    i32 distance_mode = ALPHA_MAX,
    int num_threads = 1,
):
    """
    Smart dispatcher for hue corner interpolation.
    
    Automatically routes to the appropriate implementation based on:
    - corners shape: (4,) for single-channel, (4, C) for multi-channel
    - coords shape: (H, W, 2), (N, 2), or (C, H, W, 2) for per-channel coords
    - border_array: if provided, uses array-border variant
    - modes_x/modes_y arrays: if provided, uses per-channel modes
    
    Args:
        corners: Corner hue values, shape (4,) or (4, C)
        coords: Coordinate array, shape (H, W, 2), (N, 2), or (C, H, W, 2)
        mode_x: Hue interpolation mode for X axis (int enum)
        mode_y: Hue interpolation mode for Y axis (int enum)
        modes_x: Optional per-channel X modes, shape (C,)
        modes_y: Optional per-channel Y modes, shape (C,)
        border_mode: Border handling mode (int enum)
        border_constant: Constant value for BORDER_CONSTANT mode
        border_array: Optional per-pixel border values
        border_feathering: Feathering distance
        feather_hue_mode: Hue interpolation mode for feathering blend (int enum)
        distance_mode: Distance metric for feathering (int enum)
        num_threads: Number of threads (-1 for auto)
    
    Returns:
        Interpolated hue values with shape matching coords grid
    """
    # ALL cdef declarations at top
    cdef int corners_ndim = corners.ndim
    cdef int coords_ndim = coords.ndim
    cdef Py_ssize_t corners_dim0 = corners.shape[0]
    cdef Py_ssize_t corners_dim1 = 0
    cdef Py_ssize_t coords_last_dim
    cdef Py_ssize_t C = 0
    cdef bint use_array_border = border_array is not None
    cdef bint use_per_ch_modes = modes_x is not None and modes_y is not None

    # Get corners second dimension if multi-channel
    if corners_ndim == 2:
        corners_dim1 = corners.shape[1]
        C = corners_dim1
    
    # Ensure contiguous
    corners = np.ascontiguousarray(corners, dtype=np.float64)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    if use_array_border:
        border_array = np.ascontiguousarray(border_array, dtype=np.float64)
    
    if use_per_ch_modes:
        modes_x = np.ascontiguousarray(modes_x, dtype=np.int32)
        modes_y = np.ascontiguousarray(modes_y, dtype=np.int32)
    
    # Validate corners has 4 in first dimension
    if corners_dim0 != 4:
        raise ValueError(f"corners first dimension must be 4, got {corners_dim0}")
    
    # Single-channel: corners shape (4,)
    if corners_ndim == 1:
        if coords_ndim == 3:  # (H, W, 2)
            coords_last_dim = coords.shape[2]
            if coords_last_dim != 2:
                raise ValueError(f"coords last dim must be 2, got {coords_last_dim}")
            
            if use_array_border:
                return hue_lerp_from_corners_array_border(
                    corners, coords, border_array,
                    mode_x=mode_x, mode_y=mode_y,
                    border_mode=border_mode,
                    border_feathering=border_feathering,
                    feather_hue_mode=feather_hue_mode,  # Pass through
                    distance_mode=distance_mode,
                    num_threads=num_threads,
                )

            return hue_lerp_from_corners_1ch(
                corners, coords,
                mode_x=mode_x, mode_y=mode_y,
                border_mode=border_mode,
                border_constant=border_constant,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,  # Pass through
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        
        elif coords_ndim == 2:  # (N, 2)
            coords_last_dim = coords.shape[1]
            if coords_last_dim != 2:
                raise ValueError(f"coords last dim must be 2, got {coords_last_dim}")
            
            if use_array_border:
                return hue_lerp_from_corners_flat_array_border(
                    corners, coords, border_array,
                    mode_x=mode_x, mode_y=mode_y,
                    border_mode=border_mode,
                    border_feathering=border_feathering,
                    feather_hue_mode=feather_hue_mode,  # Pass through
                    distance_mode=distance_mode,
                    num_threads=num_threads,
                )
            return hue_lerp_from_corners_1ch_flat(
                corners, coords,
                mode_x=mode_x, mode_y=mode_y,
                border_mode=border_mode,
                border_constant=border_constant,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,  # Pass through
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        
        else:
            raise ValueError(f"Single-channel coords must be (H,W,2) or (N,2), got ndim={coords_ndim}")
    
    # Multi-channel: corners shape (4, C)
    elif corners_ndim == 2:
        # Per-channel coordinates: (C, H, W, 2)
        if coords_ndim == 4:
            coords_last_dim = coords.shape[3]
            if coords_last_dim != 2:
                raise ValueError(f"coords last dim must be 2, got {coords_last_dim}")
            if coords.shape[0] != C:
                raise ValueError(f"coords channels ({coords.shape[0]}) must match corners ({C})")
            
            return hue_lerp_from_corners_per_ch_coords(
                corners, coords,
                mode_x=mode_x, mode_y=mode_y,
                border_mode=border_mode,
                border_constant=border_constant,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,  # Pass through
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        
        # Grid coords: (H, W, 2)
        elif coords_ndim == 3:
            coords_last_dim = coords.shape[2]
            if coords_last_dim != 2:
                raise ValueError(f"coords last dim must be 2, got {coords_last_dim}")
            
            if use_array_border:
                if use_per_ch_modes:
                    return hue_lerp_from_corners_multichannel_per_ch_modes_array_border(
                        corners, coords, border_array,
                        modes_x, modes_y,
                        border_mode=border_mode,
                        border_feathering=border_feathering,
                        feather_hue_mode=feather_hue_mode,  # Pass through
                        distance_mode=distance_mode,
                        num_threads=num_threads,
                    )
                return hue_lerp_from_corners_multichannel_array_border(
                    corners, coords, border_array,
                    mode_x=mode_x, mode_y=mode_y,
                    border_mode=border_mode,
                    border_feathering=border_feathering,
                    feather_hue_mode=feather_hue_mode,  # Pass through
                    distance_mode=distance_mode,
                    num_threads=num_threads,
                )
            
            if use_per_ch_modes:
                return hue_lerp_from_corners_multichannel_per_ch_modes(
                    corners, coords,
                    modes_x, modes_y,
                    border_mode=border_mode,
                    border_constant=border_constant,
                    border_feathering=border_feathering,
                    feather_hue_mode=feather_hue_mode,  # Pass through
                    distance_mode=distance_mode,
                    num_threads=num_threads,
                )
            return hue_lerp_from_corners_multichannel(
                corners, coords,
                mode_x=mode_x, mode_y=mode_y,
                border_mode=border_mode,
                border_constant=border_constant,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,  # Pass through
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        
        # Flat coords: (N, 2)
        elif coords_ndim == 2:
            coords_last_dim = coords.shape[1]
            if coords_last_dim != 2:
                raise ValueError(f"coords last dim must be 2, got {coords_last_dim}")
            
            if use_array_border:
                return hue_lerp_from_corners_multichannel_flat_array_border(
                    corners, coords, border_array,
                    mode_x=mode_x, mode_y=mode_y,
                    border_mode=border_mode,
                    border_feathering=border_feathering,
                    feather_hue_mode=feather_hue_mode,  # Pass through
                    distance_mode=distance_mode,
                    num_threads=num_threads,
                )
            return hue_lerp_from_corners_multichannel_flat(
                corners, coords,
                mode_x=mode_x, mode_y=mode_y,
                border_mode=border_mode,
                border_constant=border_constant,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,  # Pass through
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        
        else:
            raise ValueError(f"Multi-channel coords must be (C,H,W,2), (H,W,2), or (N,2), got ndim={coords_ndim}")
    
    else:
        raise ValueError(f"corners must be (4,) or (4,C), got ndim={corners_ndim}")


# =============================================================================
# Dispatcher for multi-dim hue lerp
# =============================================================================
cpdef np.ndarray hue_multidim_lerp(
    np.ndarray starts,
    np.ndarray ends,
    np.ndarray coeffs,
    np.ndarray modes,
):
    """
    Dispatcher for multi-dimensional hue interpolation.
    
    Automatically routes to appropriate implementation based on spatial
    dimensions of the coefficient grid.
    """
    # Store ndim in cdef variable
    cdef int coeffs_ndim = coeffs.ndim
    cdef int spatial_ndims = coeffs_ndim - 1
    
    if starts.dtype != np.float64 or not starts.flags['C_CONTIGUOUS']:
        starts = np.ascontiguousarray(starts, dtype=np.float64)
    if ends.dtype != np.float64 or not ends.flags['C_CONTIGUOUS']:
        ends = np.ascontiguousarray(ends, dtype=np.float64)
    if coeffs.dtype != np.float64 or not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
    if modes.dtype != np.int32 or not modes.flags['C_CONTIGUOUS']:
        modes = np.ascontiguousarray(modes, dtype=np.int32)
    
    if spatial_ndims == 1:
        return hue_lerp_1d_spatial(starts, ends, coeffs, modes)
    elif spatial_ndims == 2:
        return hue_lerp_2d_spatial(starts, ends, coeffs, modes)
    else:
        raise NotImplementedError(
            f"Hue lerp for {spatial_ndims}D spatial grid not implemented. "
            f"Only 1D (L, N) and 2D (H, W, N) coefficient grids are supported."
        )