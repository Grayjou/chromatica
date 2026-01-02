# interp_hue2d.pxd
# Cython declarations for 2D hue interpolation

import numpy as np
cimport numpy as np

from .interp_hue_utils cimport f64, i32, HUE_SHORTEST
from ..interp_utils cimport EUCLIDEAN
from ..border_handling_ cimport BORDER_CLAMP, BORDER_CONSTANT

# Multi-dimensional spatial interpolation
cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_spatial(
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=3] coeffs,
    np.ndarray[i32, ndim=1] modes,
)

# Line-based interpolation with feathering
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    i32 mode_x=*,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_constant=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW: hue mode for feathering blend
    i32 distance_mode=*,
    int num_threads=*,
)

cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_x_discrete(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    i32 mode_y=*,
    int border_mode=*,
    f64 border_constant=*,
    f64 border_feathering=*,
    i32 feather_hue_mode=*,  # NEW: hue mode for feathering blend
    i32 distance_mode=*,
    int num_threads=*,
)

# Per-pixel modes
cpdef np.ndarray[f64, ndim=2] hue_lerp_2d_with_modes(
    np.ndarray[f64, ndim=2] h0_grid,
    np.ndarray[f64, ndim=2] h1_grid,
    np.ndarray[f64, ndim=2] coeffs,
    np.ndarray[i32, ndim=2] modes,
)