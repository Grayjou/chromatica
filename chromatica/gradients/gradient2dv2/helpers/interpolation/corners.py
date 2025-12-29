#chromatica\gradients\gradient2dv2\helpers\interpolation\corners.py
from unittest import result
import numpy as np
from boundednumbers import BoundType
from typing import List, Optional
from .....v2core.core import HueMode, hue_lerp_2d_spatial
from .....v2core.core2d import multival2d_lerp_from_corners
from .....types.color_types import is_hue_space, ColorSpaces
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

def interp_hue_2d_from_corners(
    h_tl: np.ndarray,
    h_tr: np.ndarray,
    h_bl: np.ndarray,
    h_br: np.ndarray,
    coeffs: np.ndarray,
    huemode_x: HueMode,
    huemode_y: HueMode,
) -> np.ndarray:
    """
    Interpolate hue values from 4 corner hues using 2D spatial coefficients.

    Parameters
    ----------
    h_tl, h_tr, h_bl, h_br
        Scalar hue values for the four corners.
    coeffs
        3D array (H, W, 2) of interpolation coefficients.
    huemode_x, huemode_y
        Hue interpolation modes for X and Y axes.

    Returns
    -------
    np.ndarray
        Interpolated hue values.
    """
    coeffs_f64 = coeffs.astype(np.float64)

    # For 2D bilinear hue interpolation:
    # starts = [top_left, bottom_left]
    # ends   = [top_right, bottom_right]
    starts = np.array([h_tl, h_bl], dtype=np.float64)
    ends = np.array([h_tr, h_br], dtype=np.float64)

    return hue_lerp_2d_spatial(
        start_hues=starts,
        end_hues=ends,
        coeffs=coeffs_f64,
        modes=(int(huemode_x), int(huemode_y)),
    )


def _interp_transformed_non_hue_space_2d_corners(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    transformed: List[np.ndarray],
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
    border_mode: Optional[int] = None,
    border_value: Optional[float] = None,
) -> np.ndarray:
    """
    Interpolate non-hue values from 4 corner values using transformed coordinates in multidimensional space.
    """

    corners_array = np.array([c_tl, c_tr, c_bl, c_br])
    if isinstance(transformed, np.ndarray) and transformed.ndim == 3:
        # Same coords for all channels
        transformed = [transformed for _ in range(c_tl.shape[0])]
    elif isinstance(transformed, list) and len(transformed) == 1 and transformed[0].ndim == 3:
        # Same coords for all channels
        transformed = [transformed[0] for _ in range(c_tl.shape[0])]
    else:
        # Assume transformed is already a list of per-channel coords
        pass
    return multival2d_lerp_from_corners(
        corners=corners_array,
        coords=transformed,
        bound_types=bound_types,
        border_mode=border_mode,
        border_constant=border_value,
    )


def _interp_transformed_hue_space_2d_corners(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    transformed: List[np.ndarray],
    huemode_x: HueMode,
    huemode_y: HueMode,
    border_mode: Optional[int] = None,
    border_value: Optional[float] = None,
) -> np.ndarray:
    """
    Interpolate hue values from 4 corner hues using transformed coordinates in multidimensional space.
    """
    # Prepare corner data
    # Prepare channels
    num_channels = len(c_tl)
    corner_data = _prepare_corner_arrays(c_tl, c_tr, c_bl, c_br, is_hue=True)
    h_tl, h_tr, h_bl, h_br = corner_data['hue']

    # Get the actual coordinates for hue interpolation
    # transformed[0] contains the coordinates for the hue channel
    if isinstance(transformed, np.ndarray) and transformed.ndim == 3:
        # Same coords
        hue_coords = transformed.copy()
        rest_coords = [transformed.copy() for _ in range(num_channels - 1)]
    elif isinstance(transformed, list) and len(transformed) == 1 and transformed[0].ndim == 3:
        # Same coords
        hue_coords = transformed[0].copy()
        rest_coords = [transformed[0].copy() for _ in range(num_channels - 1)]

    else:
        hue_coords = transformed[0]
        rest_coords = transformed[1:]

    
    # Convert to float64 for the hue interpolation function
    result_hue = interp_hue_2d_from_corners(
        h_tl.astype(np.float64),
        h_tr.astype(np.float64),
        h_bl.astype(np.float64),
        h_br.astype(np.float64),
        hue_coords.astype(np.float64),
        huemode_x,
        huemode_y,
    )
    
    # Prepare starts and ends for 2D interpolation
    # For 2D bilinear: starts = [top_left, bottom_left], ends = [top_right, bottom_right]
    

    # Interpolate rest channels (skip first channel which is hue)
    result_rest = multival2d_lerp_from_corners(
        corners=corner_data['rest'],
        coords=rest_coords,  # Skip hue channel coordinates
        bound_types=BoundType.CLAMP,
        border_mode=border_mode,
        border_constant=border_value,
    )
    
    return combine_hue_and_rest_channels(result_hue, result_rest)


def interp_transformed_2d_from_corners(
    c_tl: np.ndarray,
    c_tr: np.ndarray,
    c_bl: np.ndarray,
    c_br: np.ndarray,
    transformed: List[np.ndarray],
    color_space: ColorSpaces,
    huemode_x: Optional[HueMode] = None,
    huemode_y: Optional[HueMode] = None,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
    border_mode: Optional[int] = None,
    border_value: Optional[float] = None,
) -> np.ndarray:
    """
    Interpolate values from 4 corner values using transformed coordinates in multidimensional space.
    """

    if is_hue_space(color_space):
        if huemode_x is None or huemode_y is None:
            raise ValueError("Hue modes must be provided for hue color spaces in corner interpolation.")
        return _interp_transformed_hue_space_2d_corners(
            c_tl, c_tr, c_bl, c_br, transformed, huemode_x, huemode_y,
            border_mode, border_value
        )
    else:
        return _interp_transformed_non_hue_space_2d_corners(
            c_tl, c_tr, c_bl, c_br, transformed, bound_types,
            border_mode, border_value
        )
    
