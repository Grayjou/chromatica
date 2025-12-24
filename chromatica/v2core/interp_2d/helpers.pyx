
import numpy as np
cimport numpy as np
# =============================================================================
# Helper: Inline border handling for a single coordinate
# =============================================================================
cdef inline f64 handle_border_1d(f64 u, int border_mode) noexcept nogil:
    """Handle border for a single coordinate in [0, 1] range."""
    cdef f64 result
    
    if border_mode == BORDER_CLAMP:
        if u < 0.0:
            return 0.0
        elif u > 1.0:
            return 1.0
        return u
    elif border_mode == BORDER_REPEAT:
        result = fmod(u, 1.0)
        if result < 0.0:
            result += 1.0
        return result
    elif border_mode == BORDER_MIRROR:
        result = fmod(fabs(u), 2.0)
        if result > 1.0:
            result = 2.0 - result
        return result
    else:
        return u



cdef inline bint is_out_of_bounds_1d(f64 u) noexcept nogil:
    """Check if coordinate is out of [0, 1] bounds."""
    return u < 0.0 or u > 1.0


cdef inline bint is_out_of_bounds_2d(f64 u_x, f64 u_y) noexcept nogil:
    """Check if 2D coordinates are out of [0, 1] bounds."""
    return u_x < 0.0 or u_x > 1.0 or u_y < 0.0 or u_y > 1.0


cdef inline bint is_out_of_bounds_3d(f64 u_x, f64 u_y, f64 u_z) noexcept nogil:
    """Check if 3D coordinates are out of [0, 1] bounds."""
    return u_x < 0.0 or u_x > 1.0 or u_y < 0.0 or u_y > 1.0 or u_z < 0.0 or u_z > 1.0


# =============================================================================
# Helper: Prepare border constants for multi-channel data
# =============================================================================
cdef np.ndarray[f64, ndim=1] prepare_border_constant_array(object border_constant, Py_ssize_t C):
    """
    Prepare border constants array for multi-channel data.
    
    Args:
        border_constant: Can be:
            - None: returns zeros array of shape (C,)
            - scalar (int/float): returns array filled with that value
            - array-like of shape (C,): returns that array
        C: Number of channels
    
    Returns:
        Contiguous float64 array of shape (C,)
    """
    cdef np.ndarray[f64, ndim=1] result
    
    if border_constant is None:
        result = np.zeros(C, dtype=np.float64)
    elif isinstance(border_constant, (int, float)):
        result = np.full(C, <f64>border_constant, dtype=np.float64)
    else:
        # Assume array-like (list, tuple, ndarray)
        result = np.asarray(border_constant, dtype=np.float64)
        if result.ndim != 1:
            raise ValueError(f"border_constant must be 1D, got {result.ndim}D")
        if result.shape[0] != C:
            raise ValueError(f"border_constant must have length {C}, got {result.shape[0]}")
    
    if not result.flags['C_CONTIGUOUS']:
        result = np.ascontiguousarray(result)
    
    return result