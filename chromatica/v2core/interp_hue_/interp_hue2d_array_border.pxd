# interp_hue2d_array_border.pxd
import numpy as np
cimport numpy as np
from .interp_hue_utils cimport f64, i32, HUE_SHORTEST
from ..interp_utils cimport EUCLIDEAN
from ..border_handling_ cimport BORDER_CONSTANT

# Grid coords (H, W, 2)
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_array_border(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

# Flat coords (N, 2)
cpdef np.ndarray[f64, ndim=1] hue_lerp_between_lines_array_border_flat(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

# Discrete X variants
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_array_border_x_discrete(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

cpdef np.ndarray[f64, ndim=1] hue_lerp_between_lines_array_border_flat_x_discrete(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)