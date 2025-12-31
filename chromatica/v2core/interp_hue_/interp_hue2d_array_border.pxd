# interp_hue2d_array_border.pxd
import numpy as np
cimport numpy as np
from .interp_hue_utils cimport f64, i32

cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_array_border(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    f64 border_feathering=*,
    object border_mode=*,
    int mode_x=*,
    int mode_y=*,
    object distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray[f64, ndim=1] hue_lerp_between_lines_array_border_flat(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    f64 border_feathering=*,
    object border_mode=*,
    int mode_x=*,
    int mode_y=*,
    object distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_array_border_x_discrete(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    f64 border_feathering=*,
    object border_mode=*,
    int mode_y=*,
    object distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray[f64, ndim=1] hue_lerp_between_lines_array_border_flat_x_discrete(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    f64 border_feathering=*,
    object border_mode=*,
    int mode_y=*,
    object distance_mode=*,
    int num_threads=*,
)
