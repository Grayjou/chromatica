# interp_hue_array_border.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Dispatcher for hue array-border interpolation.

Routes to grid/flat and discrete/continuous x variants while keeping hue-aware
blending along the shortest path for border mixing.
"""

import numpy as np
cimport numpy as np
from .interp_hue_utils cimport f64, i32, HUE_SHORTEST
from .interp_hue2d_array_border cimport (
    hue_lerp_between_lines_array_border,
    hue_lerp_between_lines_array_border_flat,
    hue_lerp_between_lines_array_border_x_discrete,
    hue_lerp_between_lines_array_border_flat_x_discrete,
)


cpdef np.ndarray hue_lerp_between_lines_array_border_dispatch(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray coords,
    np.ndarray border_array,
    int mode_x=HUE_SHORTEST,
    int mode_y=HUE_SHORTEST,
    object border_mode='constant',
    f64 border_feathering=0.0,
    object distance_mode='euclidean',
    int num_threads=1,
    bint x_discrete=False,
):
    """Dispatch hue array-border interpolation based on coord layout."""
    if coords.ndim == 3:
        if x_discrete:
            return hue_lerp_between_lines_array_border_x_discrete(
                line0, line1, coords, border_array,
                border_feathering=border_feathering,
                border_mode=border_mode,
                mode_y=mode_y,
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        return hue_lerp_between_lines_array_border(
            line0, line1, coords, border_array,
            border_feathering=border_feathering,
            border_mode=border_mode,
            mode_x=mode_x,
            mode_y=mode_y,
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    elif coords.ndim == 2:
        if x_discrete:
            return hue_lerp_between_lines_array_border_flat_x_discrete(
                line0, line1, coords, border_array,
                border_feathering=border_feathering,
                border_mode=border_mode,
                mode_y=mode_y,
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        return hue_lerp_between_lines_array_border_flat(
            line0, line1, coords, border_array,
            border_feathering=border_feathering,
            border_mode=border_mode,
            mode_x=mode_x,
            mode_y=mode_y,
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    else:
        raise ValueError("coords must be (H, W, 2) or (N, 2)")
