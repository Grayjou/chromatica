# border_handling.pxd

ctypedef double f64

# Exported border mode constants (C-level)
cdef int BORDER_REPEAT
cdef int BORDER_MIRROR
cdef int BORDER_CONSTANT
cdef int BORDER_CLAMP
cdef int BORDER_OVERFLOW

# Cython-callable functions - use cpdef
cpdef handle_border_edges_2d(f64 x, f64 y, int border_mode)
cpdef handle_border_lines_2d(f64 x, f64 y, int border_mode)