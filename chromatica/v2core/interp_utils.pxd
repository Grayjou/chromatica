# interp_utils.pxd

cimport numpy as np
from .border_handling_ cimport (handle_border_1d,
                                    BorderMode,
                                    BORDER_REPEAT,
                                    BORDER_MIRROR,
                                    BORDER_CONSTANT,
                                    BORDER_CLAMP,
                                    BORDER_OVERFLOW)

# =============================================================================
# Type Definitions
# =============================================================================
ctypedef double f64
ctypedef np.int32_t i32
from libc.math cimport floor
# =============================================================================
# Constants (using enum allows values directly in .pxd)
# =============================================================================


cdef enum DistanceMode:
    MAX_NORM = 1
    MANHATTAN = 2
    SCALED_MANHATTAN = 3
    ALPHA_MAX = 4
    ALPHA_MAX_SIMPLE = 5
    TAYLOR = 6
    EUCLIDEAN = 7
    WEIGHTED_MINMAX = 8

# =============================================================================
# Structs
# =============================================================================
cdef struct BorderResult:
    f64 u_x_final
    f64 u_y_final
    f64 blend_factor
    bint use_border_directly

# =============================================================================
# Inline Function Implementations
# =============================================================================
cdef inline f64 compute_extra_1d(f64 u) noexcept nogil:
    return -u if u < 0.0 else (u - 1.0 if u > 1.0 else 0.0)


cdef inline f64 compute_extra_2d(f64 u_x, f64 u_y, i32 distance_mode) noexcept nogil:
    cdef f64 extra_x = compute_extra_1d(u_x)
    cdef f64 extra_y = compute_extra_1d(u_y)
    cdef f64 ex2, ey2
    
    if distance_mode == MAX_NORM:
        return extra_x if extra_x > extra_y else extra_y
    elif distance_mode == MANHATTAN:
        return extra_x + extra_y
    elif distance_mode == SCALED_MANHATTAN:
        return (extra_x + extra_y) * 0.7071
    elif distance_mode == ALPHA_MAX:
        return max(extra_x, extra_y) + 0.4142 * min(extra_x, extra_y)
    elif distance_mode == ALPHA_MAX_SIMPLE:
        return max(extra_x, extra_y) + 0.5 * min(extra_x, extra_y)
    elif distance_mode == TAYLOR:
        ex2 = extra_x * extra_x
        ey2 = extra_y * extra_y
        return max(extra_x, extra_y) + 0.5 * min(extra_x, extra_y) - 0.25 * ex2 - 0.25 * ey2
    elif distance_mode == WEIGHTED_MINMAX:
        return 0.9604 * max(extra_x, extra_y) + 0.3981 * min(extra_x, extra_y)
    else:  # EUCLIDEAN
        return (extra_x * extra_x + extra_y * extra_y) ** 0.5


cdef inline f64 clamp_01(f64 u) noexcept nogil:
    return 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)


cdef inline BorderResult process_border_2d(
    f64 u_x, f64 u_y, int bmode, f64 feathering, i32 distance_mode
) noexcept nogil:
    cdef BorderResult res
    cdef f64 extra
    
    res.use_border_directly = False
    
    if bmode == BORDER_CONSTANT:
        extra = compute_extra_2d(u_x, u_y, distance_mode)
        if extra <= 0.0:
            res.u_x_final = u_x
            res.u_y_final = u_y
            res.blend_factor = 0.0
        elif feathering <= 0.0 or extra >= feathering:
            res.use_border_directly = True
            res.blend_factor = 1.0
        else:
            res.u_x_final = clamp_01(u_x)
            res.u_y_final = clamp_01(u_y)
            res.blend_factor = extra / feathering
    
    elif bmode == BORDER_CLAMP:
        res.u_x_final = clamp_01(u_x)
        res.u_y_final = clamp_01(u_y)
        res.blend_factor = 0.0
    
    elif bmode == BORDER_OVERFLOW:
        res.u_x_final = u_x
        res.u_y_final = u_y
        res.blend_factor = 0.0
    
    else:  # REPEAT or MIRROR
        res.u_x_final = handle_border_1d(u_x, bmode)
        res.u_y_final = handle_border_1d(u_y, bmode)
        res.blend_factor = 0.0
    
    return res

# =============================================================================
# Interpolation Index Helper
# =============================================================================
cdef inline Py_ssize_t compute_interp_idx(f64 u_x, Py_ssize_t L, f64* frac) noexcept nogil:
    """Shared index calculation for bilinear interpolation."""
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f = u_x * L_minus_1
    cdef Py_ssize_t idx_lo = <Py_ssize_t>floor(idx_f)
    
    if idx_lo < 0:
        idx_lo = 0
    elif idx_lo >= L - 1:
        idx_lo = L - 2
    
    frac[0] = idx_f - <f64>idx_lo
    if frac[0] < 0.0:
        frac[0] = 0.0
    elif frac[0] > 1.0:
        frac[0] = 1.0
    
    return idx_lo


cdef struct BorderResult1D:
    f64 u_final
    f64 blend_factor
    bint use_border_directly


cdef inline BorderResult1D process_border_1d(
    f64 u, int bmode, f64 feathering
) noexcept nogil:
    """
    Process 1D border handling and feathering.
    
    Uses compute_extra_1d and handle_border_1d from interp_utils.
    """
    cdef BorderResult1D res
    cdef f64 extra
    
    res.use_border_directly = False
    res.blend_factor = 0.0
    
    if bmode == BORDER_CONSTANT:
        extra = compute_extra_1d(u)
        if extra <= 0.0:
            res.u_final = u
        elif feathering <= 0.0 or extra >= feathering:
            res.use_border_directly = True
            res.blend_factor = 1.0
        else:
            res.u_final = clamp_01(u)
            res.blend_factor = extra / feathering
    
    elif bmode == BORDER_CLAMP:
        res.u_final = clamp_01(u)
    
    elif bmode == BORDER_OVERFLOW:
        res.u_final = u
        res.use_border_directly = (u < 0.0 or u > 1.0)
    
    else:  # REPEAT or MIRROR
        res.u_final = handle_border_1d(u, bmode)
    
    return res