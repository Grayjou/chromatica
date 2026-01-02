# interp_hue_corners_array_border.pxd
import numpy as np
cimport numpy as np
from .interp_hue_utils cimport f64, i32, HUE_SHORTEST
from ..interp_utils cimport EUCLIDEAN
from ..border_handling_ cimport BORDER_CONSTANT

# Single-channel with array border
cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_array_border(
    np.ndarray[f64, ndim=1] corners,
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

cpdef np.ndarray[f64, ndim=1] hue_lerp_from_corners_flat_array_border(
    np.ndarray[f64, ndim=1] corners,
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

# Multi-channel with array border (uniform modes)
cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] border_array,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

# Multi-channel with per-channel modes and array border
cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel_per_ch_modes_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=3] border_array,
    np.ndarray[i32, ndim=1] modes_x,
    np.ndarray[i32, ndim=1] modes_y,
    int border_mode=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_multichannel_flat_array_border(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=2] border_array,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)