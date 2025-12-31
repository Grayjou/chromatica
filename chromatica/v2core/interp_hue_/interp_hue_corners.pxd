# interp_hue_corners.pxd
import numpy as np
cimport numpy as np
from .interp_hue_utils cimport f64, i32

cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_1ch_feathered(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=3] coords,
    f64 border_constant=*,
    object border_mode=*,
    f64 border_feathering=*,
    object distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray[f64, ndim=1] hue_lerp_from_corners_1ch_flat_feathered(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=2] coords,
    f64 border_constant=*,
    object border_mode=*,
    f64 border_feathering=*,
    object distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel_feathered(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    object border_constant=*,
    object border_mode=*,
    f64 border_feathering=*,
    object distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_multichannel_flat_feathered(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    object border_constant=*,
    object border_mode=*,
    f64 border_feathering=*,
    object distance_mode=*,
    int num_threads=*,
)
cpdef np.ndarray hue_lerp_from_corners_full_feathered(
    corners,
    coords,
    f64 border_constant=*,
    object border_mode=*,
    f64 border_feathering=*,
    object distance_mode=*,
    int num_threads=*,
)
