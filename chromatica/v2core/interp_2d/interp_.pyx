# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

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