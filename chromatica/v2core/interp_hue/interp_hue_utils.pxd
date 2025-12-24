# interp_hue_utils.pxd
"""
Cython declarations for hue interpolation utilities.
"""

cimport numpy as np

# =============================================================================
# Type Definitions
# =============================================================================
ctypedef np.float64_t f64
ctypedef np.int32_t i32

# =============================================================================
# Hue Mode Constants (C-level enum)
# =============================================================================
cdef enum HueMode:
    HUE_CW = 0
    HUE_CCW = 1
    HUE_SHORTEST = 2
    HUE_LONGEST = 3
 
# =============================================================================
# Function Declarations
# =============================================================================
cdef  f64 wrap_hue(f64 h) noexcept nogil
cdef  f64 adjust_end_for_mode(f64 h0, f64 h1, int mode) noexcept nogil
cdef  f64 lerp_hue_single(f64 h0, f64 h1, f64 u, int mode) noexcept nogil