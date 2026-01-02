# interp_hue_array_border.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Dispatcher for hue array-border interpolation.

Routes to grid/flat and discrete/continuous x variants while keeping hue-aware
blending with configurable hue mode for border mixing.
"""

import numpy as np
cimport numpy as np
from .interp_hue_utils cimport f64, i32, HUE_SHORTEST
from ..border_handling cimport BORDER_CONSTANT
from ..interp_utils cimport EUCLIDEAN, ALPHA_MAX
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
    i32 mode_x = HUE_SHORTEST,
    i32 mode_y = HUE_SHORTEST,
    int border_mode = BORDER_CONSTANT,
    f64 border_feathering = 0.0,
    i32 feather_hue_mode = HUE_SHORTEST,  # NEW: hue mode for feathering blend
    i32 distance_mode = ALPHA_MAX,
    int num_threads = 1,
    bint x_discrete = False,
):
    """
    Dispatch hue array-border interpolation based on coord layout.
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        coords: Coordinate array, shape (H, W, 2) or (N, 2)
        border_array: Border values, shape matching coords grid
        mode_x: Hue interpolation mode for X axis (int enum)
        mode_y: Hue interpolation mode for Y axis (int enum)
        border_mode: Border handling mode (int enum)
        border_feathering: Feathering distance
        feather_hue_mode: Hue interpolation mode for feathering blend (int enum)
        distance_mode: Distance metric for feathering (int enum)
        num_threads: Number of threads (-1 for auto)
        x_discrete: If True, use nearest-neighbor x-sampling
    
    Returns:
        Interpolated hue values with shape matching coords grid
    
    Note:
        All mode parameters are int enums - conversion from strings/user-friendly
        types should happen in the Python wrapper layer.
    """
    # Store ndim to avoid Python object conversion issues
    cdef int coords_ndim = coords.ndim
    
    if coords_ndim == 3:
        if x_discrete:
            return hue_lerp_between_lines_array_border_x_discrete(
                line0, line1, coords, border_array,
                mode_y=mode_y,
                border_mode=border_mode,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,  # Pass through
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        return hue_lerp_between_lines_array_border(
            line0, line1, coords, border_array,
            mode_x=mode_x,
            mode_y=mode_y,
            border_mode=border_mode,
            border_feathering=border_feathering,
            feather_hue_mode=feather_hue_mode,  # Pass through
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    elif coords_ndim == 2:
        if x_discrete:
            return hue_lerp_between_lines_array_border_flat_x_discrete(
                line0, line1, coords, border_array,
                mode_y=mode_y,
                border_mode=border_mode,
                border_feathering=border_feathering,
                feather_hue_mode=feather_hue_mode,  # Pass through
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        return hue_lerp_between_lines_array_border_flat(
            line0, line1, coords, border_array,
            mode_x=mode_x,
            mode_y=mode_y,
            border_mode=border_mode,
            border_feathering=border_feathering,
            feather_hue_mode=feather_hue_mode,  # Pass through
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    else:
        raise ValueError(f"coords must be (H, W, 2) or (N, 2), got ndim={coords_ndim}")