# interp_hue2d.pxd
# Cython declarations for 2D hue interpolation

import numpy as np
cimport numpy as np

from .interp_hue_utils cimport f64, i32, HueMode, HUE_CW, HUE_CCW, HUE_SHORTEST, HUE_LONGEST

# Public function declarations
cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_spatial(np.ndarray[f64, ndim=1] starts, np.ndarray[f64, ndim=1] ends, np.ndarray[f64, ndim=3] coeffs, np.ndarray[i32, ndim=1] modes)
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines(np.ndarray[f64, ndim=1] line0, np.ndarray[f64, ndim=1] line1, np.ndarray[f64, ndim=3] coords, int mode_x=*, int mode_y=*, object border_mode=*, f64 border_constant=*, f64 border_feathering=*, object distance_mode=*, int num_threads=*)
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_x_discrete(np.ndarray[f64, ndim=1] line0, np.ndarray[f64, ndim=1] line1, np.ndarray[f64, ndim=3] coords, int mode_y=*, object border_mode=*, f64 border_constant=*, f64 border_feathering=*, object distance_mode=*, int num_threads=*)
cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_with_modes(np.ndarray[f64, ndim=2] h0_grid, np.ndarray[f64, ndim=2] h1_grid, np.ndarray[f64, ndim=2] coeffs, np.ndarray[i32, ndim=2] modes)
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_feathered(np.ndarray[f64, ndim=1] l0, np.ndarray[f64, ndim=1] l1, np.ndarray[f64, ndim=3] c, f64 border_constant, f64 border_feathering=*, str border_mode=*, int mode_x=*, int mode_y=*, str distance_mode=*, int num_threads=*)
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_array_border(np.ndarray[f64, ndim=1] l0, np.ndarray[f64, ndim=1] l1, np.ndarray[f64, ndim=3] c, np.ndarray[f64, ndim=2] border_array, f64 border_feathering=*, str border_mode=*, int mode_x=*, int mode_y=*, str distance_mode=*, int num_threads=*)





