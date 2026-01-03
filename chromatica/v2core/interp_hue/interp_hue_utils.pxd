# interp_hue_utils.pxd
# Cython declarations for hue interpolation utilities

from ..interp_utils cimport BorderResult

# Type definitions (also defined in .pyx)
ctypedef double f64
ctypedef int i32

# Hue interpolation modes
cdef enum HueDirection:
    HUE_CW = 1
    HUE_CCW = 2
    HUE_SHORTEST = 3
    HUE_LONGEST = 4

# Public function declarations
cdef f64 wrap_hue(f64 h) noexcept nogil
cdef f64 adjust_end_for_mode(f64 h0, f64 h1, int mode) noexcept nogil
cdef f64 lerp_hue_single(f64 h0, f64 h1, f64 t, int mode) noexcept nogil
cdef f64 _interp_line_1ch_hue(const f64[::1] l0, const f64[::1] l1, f64 u_x, f64 u_y, Py_ssize_t L, int mode_x, int mode_y) noexcept nogil
cdef f64 _interp_line_discrete_hue(const f64[::1] l0, const f64[::1] l1, f64 u_x, f64 u_y, Py_ssize_t L, int mode_y) noexcept nogil
cdef BorderResult process_hue_border_2d(f64 u_x, f64 u_y, int border_mode, f64 border_feathering, i32 distance_mode) noexcept nogil
