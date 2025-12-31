# border_handling.pxd
ctypedef double f64

# Border mode constants
cdef int BORDER_REPEAT
cdef int BORDER_MIRROR
cdef int BORDER_CONSTANT
cdef int BORDER_CLAMP
cdef int BORDER_OVERFLOW

# Function declarations (NO inline keyword in .pxd!)
cdef bint is_out_of_bounds_2d(f64 x, f64 y) noexcept nogil
cdef f64 handle_border_1d(f64 t, int border_mode) noexcept nogil