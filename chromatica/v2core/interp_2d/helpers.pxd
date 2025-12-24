



import numpy as np
cimport numpy as np
from libc.math cimport floor, fmod, fabs
from libc.stdlib cimport malloc, free
from ..border_handling cimport (
    handle_border_edges_2d,
    handle_border_lines_2d,
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)
ctypedef np.float64_t f64

cdef f64 handle_border_1d(f64 u, int border_mode) noexcept nogil

cdef bint is_out_of_bounds_1d(f64 u) noexcept nogil

cdef bint is_out_of_bounds_2d(f64 u_x, f64 u_y) noexcept nogil

cdef bint is_out_of_bounds_3d(f64 u_x, f64 u_y, f64 u_z) noexcept nogil

cdef np.ndarray[f64, ndim=1] prepare_border_constant_array(object border_constant, Py_ssize_t C)

