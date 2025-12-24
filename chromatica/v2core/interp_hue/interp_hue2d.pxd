# interp_hue2d.pxd
import numpy as np
cimport numpy as np

from ..border_handling cimport (
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)
from .interp_hue_utils cimport (
    f64,
    i32,
    HUE_CW,
    HUE_CCW,
    HUE_SHORTEST,
    HUE_LONGEST,
    wrap_hue,
    adjust_end_for_mode,
    lerp_hue_single,
)

cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_spatial(
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=3] coeffs,
    np.ndarray[i32, ndim=1] modes,
)

cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_x=*,          # Has default in .pyx
    int mode_y=*,          # Has default in .pyx
    int border_mode=*,     # Has default in .pyx
    f64 border_constant=*, # Has default in .pyx
)

cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_x_discrete(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int mode_y=*,          # Has default in .pyx
    int border_mode=*,     # Has default in .pyx
    f64 border_constant=*, # Has default in .pyx
)

 
cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_with_modes(
    np.ndarray[f64, ndim=2] h0_grid,
    np.ndarray[f64, ndim=2] h1_grid,
    np.ndarray[f64, ndim=2] coeffs,
    np.ndarray[i32, ndim=2] modes,
)