#chromatica\gradients\gradient2dv2\helpers\interpolation\corners.py
import numpy as np
from boundednumbers import BoundType
from typing import List, Optional
from .....v2core.core import HueMode
from .....v2core.core2d import multival2d_lerp_from_corners, hue_gradient_2d as hue_gradient2d_from_corners
from .....types.color_types import is_hue_space, ColorSpace
from .utils import prepare_hue_and_rest_channels, combine_hue_and_rest_channels


def _prepare_corner_arrays(c_tl, c_tr, c_bl, c_br, is_hue=True):
    """Prepare corner arrays for interpolation."""
    hue_tl, rest_tl = prepare_hue_and_rest_channels(c_tl, is_hue)
    hue_tr, rest_tr = prepare_hue_and_rest_channels(c_tr, is_hue)
    hue_bl, rest_bl = prepare_hue_and_rest_channels(c_bl, is_hue)
    hue_br, rest_br = prepare_hue_and_rest_channels(c_br, is_hue)
    
    return {
        'hue': (hue_tl, hue_tr, hue_bl, hue_br),
        'rest': np.array([rest_tl, rest_tr, rest_bl, rest_br])
    }


def _interp_transformed_non_hue_space_2d_corners(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    transformed: List[np.ndarray],
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Interpolate non-hue values from 4 corner values using transformed coordinates in multidimensional space.
    """
    corners_array = np.array([c_tl, c_tr, c_bl, c_br])
    return multival2d_lerp_from_corners(
        corners=corners_array,
        coords=transformed,
        bound_types=bound_types,
    )


def _interp_transformed_hue_space_2d_corners(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    transformed: List[np.ndarray],
    huemode_x: HueMode,
    huemode_y: HueMode,
) -> np.ndarray:
    """
    Interpolate hue values from 4 corner hues using transformed coordinates in multidimensional space.
    """
    # Prepare corner data
    corner_data = _prepare_corner_arrays(c_tl, c_tr, c_bl, c_br, is_hue=True)
    h_tl, h_tr, h_bl, h_br = corner_data['hue']

    # Interpolate hue channel
    result_hue = hue_gradient2d_from_corners(
        corners=(h_tl, h_tr, h_bl, h_br),
        shape=transformed[0].shape[:2],
        modes=(huemode_x, huemode_y)
    )

    # Interpolate rest channels (skip first channel which is hue)
    result_rest = multival2d_lerp_from_corners(
        corners=corner_data['rest'],
        coords=transformed[1:],  # Skip hue channel coordinates
        bound_types=BoundType.CLAMP,
    )
    
    return combine_hue_and_rest_channels(result_hue, result_rest)


def interp_transformed_2d_from_corners(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    transformed: List[np.ndarray],
    color_space: ColorSpace,
    huemode_x: Optional[HueMode] = None,
    huemode_y: Optional[HueMode] = None,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Interpolate values from 4 corner values using transformed coordinates in multidimensional space.
    """
    if is_hue_space(color_space):
        if huemode_x is None or huemode_y is None:
            raise ValueError("Hue modes must be provided for hue color spaces.")
        return _interp_transformed_hue_space_2d_corners(
            c_tl, c_tr, c_bl, c_br, transformed, huemode_x, huemode_y
        )
    else:
        return _interp_transformed_non_hue_space_2d_corners(
            c_tl, c_tr, c_bl, c_br, transformed, bound_types
        )