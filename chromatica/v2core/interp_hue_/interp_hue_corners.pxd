# interp_hue_corners.pxd
import numpy as np
cimport numpy as np
from .interp_hue_utils cimport f64, i32, HUE_SHORTEST
from ..interp_utils cimport EUCLIDEAN
from ..border_handling_ cimport BORDER_CLAMP, BORDER_CONSTANT

# Single-channel
cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_1ch(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=3] coords,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_constant=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

cpdef np.ndarray[f64, ndim=1] hue_lerp_from_corners_1ch_flat(
    np.ndarray[f64, ndim=1] corners,
    np.ndarray[f64, ndim=2] coords,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_constant=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

# Multi-channel with uniform modes
cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_constant=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

# Multi-channel with per-channel modes
cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_multichannel_per_ch_modes(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[i32, ndim=1] modes_x,
    np.ndarray[i32, ndim=1] modes_y,
    int border_mode=*,
    f64 border_constant=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

cpdef np.ndarray[f64, ndim=2] hue_lerp_from_corners_multichannel_flat(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=2] coords,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_constant=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)

# Per-channel coordinates
cpdef np.ndarray[f64, ndim=3] hue_lerp_from_corners_per_ch_coords(
    np.ndarray[f64, ndim=2] corners,
    np.ndarray[f64, ndim=4] coords,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_constant=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW
    i32 distance_mode=*,
    int num_threads=*,
)