# interp_hue_utils.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Hue interpolation utilities with cyclical color space support.

Modernized to support feathering, distance modes, and advanced border handling.
"""

from libc.math cimport fmod, floor

# Import from consolidated interp_utils
from ..interp_utils cimport (
    process_border_2d,
    BorderResult,
    MAX_NORM,
    MANHATTAN,
    SCALED_MANHATTAN,
    ALPHA_MAX,
    ALPHA_MAX_SIMPLE,
    TAYLOR,
    EUCLIDEAN,
    WEIGHTED_MINMAX,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
    compute_interp_idx,
)

# =============================================================================
# Inline Helpers - Hue Operations
# =============================================================================
cdef inline f64 wrap_hue(f64 h) noexcept nogil:
    """Wrap hue to [0, 360)."""
    h = fmod(h, 360.0)
    if h < 0.0:
        h += 360.0
    return h


cdef inline f64 adjust_end_for_mode(f64 h0, f64 h1, int mode) noexcept nogil:
    """
    Adjust h1 relative to h0 based on interpolation mode.
    Returns adjusted h1 (may be outside [0, 360)).
    """
    cdef f64 d = h1 - h0
    
    if mode == HUE_CW:
        if h0 > h1:
            return h1 + 360.0
        return h1
        
    elif mode == HUE_CCW:
        if h0 < h1:
            return h1 - 360.0
        return h1
        
    elif mode == HUE_SHORTEST:
        if d > 180.0:
            return h1 - 360.0
        elif d < -180.0:
            return h1 + 360.0
        return h1
        
    elif mode == HUE_LONGEST:
        if d >= 0.0 and d < 180.0:
            return h1 - 360.0
        elif d < 0.0 and d > -180.0:
            return h1 + 360.0
        return h1
    
    return h1


cdef inline f64 lerp_hue_single(f64 h0, f64 h1, f64 u, int mode) noexcept nogil:
    """Lerp between two hues with mode, returning wrapped result."""
    
    cdef f64 h1_adj = adjust_end_for_mode(h0, h1, mode)
    cdef f64 result = h0 + u * (h1_adj - h0)

    return wrap_hue(result)


# =============================================================================
# Hue-Specific Interpolation Functions
# =============================================================================
cdef inline f64 _interp_line_1ch_hue(
    const f64[::1] l0, const f64[::1] l1,
    f64 u_x, f64 u_y, Py_ssize_t L,
    int mode_x, int mode_y
) noexcept nogil:
    """Single-channel hue interpolation with hue-specific wrapping."""
    cdef f64 frac, v0, v1
    cdef Py_ssize_t idx_lo, idx_hi
    
    idx_lo = compute_interp_idx(u_x, L, &frac)
    idx_hi = idx_lo + 1
    
    # Hue interpolation for both lines
    v0 = lerp_hue_single(l0[idx_lo], l0[idx_hi], frac, mode_x)
    v1 = lerp_hue_single(l1[idx_lo], l1[idx_hi], frac, mode_x)
    
    # Interpolate between lines with hue wrapping
    return lerp_hue_single(v0, v1, u_y, mode_y)


cdef inline f64 _interp_line_discrete_hue(
    const f64[::1] l0, const f64[::1] l1,
    f64 u_x, f64 u_y, Py_ssize_t L,
    int mode_y
) noexcept nogil:
    """Discrete x-sampling hue interpolation."""
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f = u_x * L_minus_1
    cdef Py_ssize_t idx = <Py_ssize_t>floor(idx_f + 0.5)
    
    if idx < 0:
        idx = 0
    elif idx >= L:
        idx = L - 1
    
    # Direct interpolation between two lines (discrete in x)
    return lerp_hue_single(l0[idx], l1[idx], u_y, mode_y)


# =============================================================================
# Convenience Wrapper for hue code with modern features
# =============================================================================
cdef inline BorderResult process_hue_border_2d(
    f64 u_x, f64 u_y, int bmode, f64 feathering, i32 distance_mode
) noexcept nogil:
    """Wrapper for hue code - uses specified distance mode."""
    return process_border_2d(u_x, u_y, bmode, feathering, distance_mode)