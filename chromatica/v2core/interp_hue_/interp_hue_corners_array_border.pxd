# interp_hue_corners_array_border.pxd
import numpy as np
cimport numpy as np
from .interp_hue_utils cimport f64, i32

cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_array_border(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    f64 border_feathering=*,
    object border_mode=*,
    i32 distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray[f64, ndim=1] hue_lerp_from_corners_flat_array_border(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    f64 border_feathering=*,
    object border_mode=*,
    i32 distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] border_array,
    f64 border_feathering=*,
    object border_mode=*,
    i32 distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_multichannel_flat_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=2] border_array,
    f64 border_feathering=*,
    object border_mode=*,
    i32 distance_mode=*,
    int num_threads=*,
)
