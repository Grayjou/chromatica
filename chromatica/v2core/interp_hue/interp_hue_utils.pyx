# interp_hue_utils.pyx
"""
Hue interpolation with cyclical color space support.
"""

from libc.math cimport fmod, floor
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free

# Import border handling
from ..border_handling cimport (
    handle_border_lines_2d,
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)
 
# Types and enum come from the .pxd automatically
# (Cython implicitly cimports the corresponding .pxd)

# =============================================================================
# Inline Helpers
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
        if h0 >= h1:
            return h1 + 360.0
        return h1
        
    elif mode == HUE_CCW:
        if h0 <= h1:
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