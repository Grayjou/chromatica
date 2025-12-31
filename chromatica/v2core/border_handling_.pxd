# border_handling_.pxd
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Border handling modes for 2D interpolation (nogil compatible).
"""

from libc.math cimport fmod, fabs

ctypedef double f64

# =============================================================================
# Border Mode Constants (enum allows values in .pxd)
# =============================================================================
cdef enum BorderMode:
    BORDER_REPEAT = 0
    BORDER_MIRROR = 1
    BORDER_CONSTANT = 2
    BORDER_CLAMP = 3
    BORDER_OVERFLOW = 4

# =============================================================================
# Helper Functions
# =============================================================================
cdef inline f64 tri2(f64 x) noexcept nogil:
    """Triangle wave for mirror mode: maps any value to [0, 1]."""
    cdef f64 m = fmod(x, 2.0)
    if m < 0.0:
        m += 2.0
    return 1.0 - fabs(m - 1.0)


cdef inline bint is_out_of_bounds_2d(f64 x, f64 y) noexcept nogil:
    """Check if coordinates are outside [0, 1] range."""
    return x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0


cdef inline f64 handle_border_1d(f64 t, int border_mode) noexcept nogil:
    """
    Apply border handling to a single coordinate.
    
    Only handles REPEAT, MIRROR, and CLAMP modes.
    Caller should handle CONSTANT and OVERFLOW separately.
    
    Args:
        t: Coordinate value (normalized, may be outside [0, 1])
        border_mode: BORDER_REPEAT, BORDER_MIRROR, or BORDER_CLAMP
        
    Returns:
        Transformed coordinate in [0, 1] range
    """
    cdef f64 result
    
    if border_mode == BORDER_REPEAT:
        result = fmod(t, 1.0)
        if result < 0.0:
            result += 1.0
        return result
    
    elif border_mode == BORDER_MIRROR:
        return tri2(t)
    
    else:  # BORDER_CLAMP (default fallback)
        if t < 0.0:
            return 0.0
        elif t > 1.0:
            return 1.0
        return t