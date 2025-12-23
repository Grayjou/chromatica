# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Common hue interpolation helpers and constants.

Supports four interpolation modes:
- CW (0): Clockwise
- CCW (1): Counterclockwise  
- SHORTEST (2): Shortest path (≤180°)
- LONGEST (3): Longest path (≥180°)
"""

import numpy as np
cimport numpy as np
from libc.math cimport fmod, floor

ctypedef np.float64_t f64
ctypedef np.int32_t i32

# =============================================================================
# Hue Mode Constants
# =============================================================================
DEF HUE_CW = 0
DEF HUE_CCW = 1
DEF HUE_SHORTEST = 2
DEF HUE_LONGEST = 3

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
        # Clockwise: ensure h1 > h0
        if h0 >= h1:
            return h1 + 360.0
        return h1
        
    elif mode == HUE_CCW:
        # Counterclockwise: ensure h1 < h0
        if h0 <= h1:
            return h1 - 360.0
        return h1
        
    elif mode == HUE_SHORTEST:
        # Shortest path: |d| <= 180
        if d > 180.0:
            return h1 - 360.0
        elif d < -180.0:
            return h1 + 360.0
        return h1
        
    elif mode == HUE_LONGEST:
        # Longest path: |d| >= 180
        if d >= 0.0 and d < 180.0:
            return h1 - 360.0
        elif d < 0.0 and d > -180.0:
            return h1 + 360.0
        return h1
    
    # Default: no adjustment
    return h1


cdef inline f64 lerp_hue_single(f64 h0, f64 h1, f64 u, int mode) noexcept nogil:
    """Lerp between two hues with mode, returning wrapped result."""
    cdef f64 h1_adj = adjust_end_for_mode(h0, h1, mode)
    cdef f64 result = h0 + u * (h1_adj - h0)
    return wrap_hue(result)
