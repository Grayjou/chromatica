#chromatica\gradients\gradient2dv2\helpers\interpolation\corners.py
import numpy as np
from boundednumbers import BoundType
from typing import List, Optional, Union

from .....v2core.interp_2d import (
    lerp_from_corners,
    lerp_from_corners_array_border,
)
from .....v2core.interp_hue import (
    hue_lerp_from_corners,
    hue_lerp_from_corners_array_border,
)
from .....types.color_types import is_hue_space, ColorSpaces, HueMode
from .utils import prepare_hue_and_rest_channels, combine_hue_and_rest_channels
from .....v2core.border_handler import (
    BorderMode,
    BorderModeInput,
    BorderConstant,
    DistanceMode,
    BorderArrayInput,
    DistanceModeInput,
)


def _interp_transformed_non_hue_space_2d_corners(
    corners: np.ndarray,
    transformed: np.ndarray,
    *,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: Optional[BorderConstant] = None,
    border_array: BorderArrayInput = None,
    border_feathering: float = 0.0,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
    border_value: Optional[BorderConstant] = None,
) -> np.ndarray:
    """
    Interpolate non-hue values from four corners using transformed coordinates.
    
    Args:
        corners: Corner values, shape (4,) or (4, C)
            Order: [top_left, top_right, bottom_left, bottom_right]
        transformed: Transformed coordinates, shape (H, W, 2) or (N, 2)
        border_mode: Border handling mode
        border_constant: Border constant value for CONSTANT mode
        border_array: Optional array to use as border/background
        border_feathering: Amount of feathering at borders
        distance_mode: Distance calculation mode for feathering
        num_threads: Thread count (-1=auto)
        bound_types: Legacy parameter for backward compatibility
        border_value: Legacy parameter, use border_constant instead
        
    Returns:
        Interpolated values
    """
    # Handle legacy border_value parameter
    border_constant = next(
        (i for i in [border_value, border_constant] if i is not None), None
    )
    
    if border_array is not None:
        return lerp_from_corners_array_border(
            corners=corners,
            coords=transformed,
            background_array=border_array,
            border_mode=border_mode,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    else:
        return lerp_from_corners(
            corners=corners,
            coords=transformed,
            border_mode=border_mode,
            border_constant=border_constant,
            border_feathering=border_feathering,
            num_threads=num_threads,
        )


def _interp_transformed_hue_space_2d_corners(
    corners: np.ndarray,
    transformed: np.ndarray,
    huemode_x: HueMode,
    huemode_y: HueMode,
    *,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstant = 0,
    border_array: BorderArrayInput = None,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    num_channels: Optional[int] = 3,
    border_value: Optional[BorderConstant] = None,
) -> np.ndarray:
    """
    Interpolate hue values from four corners using transformed coordinates.
    
    Args:
        corners: Corner values, shape (4, C) where C is number of channels
            Order: [top_left, top_right, bottom_left, bottom_right]
        transformed: Transformed coordinates, shape (H, W, 2) or list of coords
        huemode_x: Hue interpolation mode for x-axis
        huemode_y: Hue interpolation mode for y-axis
        bound_types: Legacy parameter for backward compatibility
        border_mode: Border handling mode
        border_constant: Border constant value for CONSTANT mode
        border_array: Optional array to use as border/background
        border_feathering: Amount of feathering at borders
        feather_hue_mode: Hue mode for feathering
        distance_mode: Distance calculation mode for feathering
        num_threads: Thread count (-1=auto)
        num_channels: Number of channels
        border_value: Legacy parameter, use border_constant instead
        
    Returns:
        Interpolated values, shape (H, W, C)
    """
    # Handle legacy border_value parameter
    border_constant = next(
        (i for i in [border_value, border_constant] if i is not None), None
    )
    
    # Prepare coordinates
    if isinstance(transformed, np.ndarray) and transformed.ndim == 3:
        # Same coords for all channels
        transformed_h = transformed.copy()
        num_channels = corners.shape[-1] if corners.ndim > 1 else 1
        transformed_r = [transformed.copy() for _ in range(num_channels - 1)]
    elif isinstance(transformed, list) and len(transformed) == 1 and transformed[0].ndim == 3:
        # Same coords for all channels
        transformed_h = transformed[0].copy()
        num_channels = corners.shape[-1] if corners.ndim > 1 else 1
        transformed_r = [transformed[0].copy() for _ in range(num_channels - 1)]
    else:
        transformed_h = transformed[0]
        transformed_r = transformed[1:]

    # Split corners into hue and rest channels
    # corners shape: (4, C) -> hue corners: (4,), rest corners: (4, C-1)
    if corners.ndim == 1:
        # Single channel, assume it's hue only
        hue_corners = corners
        rest_corners = None
    else:
        hue_corners = corners[:, 0]  # Shape (4,)
        rest_corners = corners[:, 1:]  # Shape (4, C-1)

    # Handle border array splitting
    if border_array is not None:
        if border_array.ndim == 2:
            hue_border_array = border_array
            rest_border_array = border_array
        elif border_array.ndim == 3:
            hue_border_array = border_array[..., 0]
            rest_border_array = border_array[..., 1:]
        else:
            raise ValueError("border_array must be 2D or 3D array.")
    else:
        hue_border_array = None
        rest_border_array = None

    # Interpolate hue channel
    hresult = hue_lerp_from_corners(
        corners=hue_corners,
        coords=transformed_h,
        mode_x=huemode_x,
        mode_y=huemode_y,
        border_mode=border_mode,
        border_constant=border_constant,
        border_array=hue_border_array,
        border_feathering=border_feathering,
        feather_hue_mode=feather_hue_mode,
        distance_mode=distance_mode,
        num_threads=num_threads,
    )

    # If no rest channels, return hue result with added dimension
    if rest_corners is None:
        return hresult[..., np.newaxis]

    # Handle rest border constant
    if isinstance(border_value, (list, np.ndarray)):
        rest_border_value = border_value[1:]
    else:
        rest_border_value = border_value

    # Interpolate rest channels
    if rest_border_array is not None:
        rresult = lerp_from_corners_array_border(
            corners=rest_corners,
            coords=transformed_r,
            background_array=rest_border_array,
            border_mode=border_mode,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    else:
        rresult = lerp_from_corners(
            corners=rest_corners,
            coords=transformed_r,
            border_mode=border_mode,
            border_constant=rest_border_value,
            border_feathering=border_feathering,
            num_threads=num_threads,
        )

    return combine_hue_and_rest_channels(hresult, rresult)


def interp_transformed_2d_corners(
    corners: np.ndarray,
    transformed: np.ndarray,
    color_space: ColorSpaces,
    huemode_x: Optional[HueMode] = None,
    huemode_y: Optional[HueMode] = None,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstant = 0,
    border_array: BorderArrayInput = None,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    num_channels: Optional[int] = None,
    *,
    border_value: Optional[BorderConstant] = None,  # Legacy parameter
) -> np.ndarray:
    """
    Interpolate values from four corners using transformed coordinates.
    
    Args:
        corners: Corner values, shape (4,) or (4, C)
            Order: [top_left, top_right, bottom_left, bottom_right]
        transformed: Transformed coordinates, shape (H, W, 2)
        color_space: Color space of the data
        huemode_x: Hue interpolation mode for x-axis (required for hue spaces)
        huemode_y: Hue interpolation mode for y-axis (required for hue spaces)
        bound_types: List of BoundType for each channel or a single BoundType
        border_mode: Border handling mode (e.g., BorderMode.CLAMP, BorderMode.REPEAT)
        border_constant: Border constant value for BORDER_CONSTANT mode
        border_array: Optional array to use as border/background
        border_feathering: Amount of feathering to apply at borders
        feather_hue_mode: Hue mode to use when feathering hue values
        distance_mode: Mode for calculating distance (for feathering)
        num_threads: Number of threads to use (-1 for auto)
        num_channels: Number of channels (inferred from color_space if not provided)
        border_value: Legacy parameter, use border_constant instead
        
    Returns:
        Interpolated values, shape (H, W) or (H, W, C)
    """
    # Handle legacy border_value parameter
    border_constant = next(
        (i for i in [border_value, border_constant] if i is not None), 0
    )
    
    num_channels = num_channels or len(color_space)

    if is_hue_space(color_space):
        if huemode_x is None:
            raise ValueError(
                "huemode_x must be specified for hue space interpolation from corners."
            )
        if huemode_y is None:
            raise ValueError(
                "huemode_y must be specified for hue space interpolation from corners."
            )
        
        return _interp_transformed_hue_space_2d_corners(
            corners,
            transformed,
            huemode_x=huemode_x,
            huemode_y=huemode_y,
            bound_types=bound_types,
            border_mode=border_mode,
            border_constant=border_constant,
            border_array=border_array,
            border_feathering=border_feathering,
            feather_hue_mode=feather_hue_mode,
            distance_mode=distance_mode,
            num_threads=num_threads,
            num_channels=num_channels,
        )
    else:
        return _interp_transformed_non_hue_space_2d_corners(
            corners,
            transformed,
            border_mode=border_mode,
            border_constant=border_constant,
            border_array=border_array,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
            bound_types=bound_types,
        )


def interp_transformed_2d_corners_unpacked(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
    transformed: np.ndarray,
    color_space: ColorSpaces,
    huemode_x: Optional[HueMode] = None,
    huemode_y: Optional[HueMode] = None,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstant = 0,
    border_array: BorderArrayInput = None,
    border_feathering: float = 0.0,
    feather_hue_mode: HueMode = HueMode.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    num_channels: Optional[int] = None,
    *,
    border_value: Optional[BorderConstant] = None,  # Legacy parameter
) -> np.ndarray:
    """
    Interpolate values from unpacked corner values using transformed coordinates.
    
    Convenience wrapper around interp_transformed_2d_corners that accepts
    individual corner arrays instead of a packed array.
    
    Args:
        top_left: Top-left corner value(s), shape () or (C,)
        top_right: Top-right corner value(s), shape () or (C,)
        bottom_left: Bottom-left corner value(s), shape () or (C,)
        bottom_right: Bottom-right corner value(s), shape () or (C,)
        transformed: Transformed coordinates, shape (H, W, 2)
        color_space: Color space of the data
        huemode_x: Hue interpolation mode for x-axis
        huemode_y: Hue interpolation mode for y-axis
        bound_types: List of BoundType for each channel or single BoundType
        border_mode: Border handling mode
        border_constant: Border constant value
        border_array: Optional array to use as border/background
        border_feathering: Amount of feathering at borders
        feather_hue_mode: Hue mode for feathering
        distance_mode: Distance calculation mode
        num_threads: Thread count (-1=auto)
        num_channels: Number of channels
        border_value: Legacy parameter, use border_constant instead
        
    Returns:
        Interpolated values
    """
    # Pack corners into array
    top_left = np.atleast_1d(np.asarray(top_left))
    top_right = np.atleast_1d(np.asarray(top_right))
    bottom_left = np.atleast_1d(np.asarray(bottom_left))
    bottom_right = np.atleast_1d(np.asarray(bottom_right))
    
    corners = np.stack(
        [top_left, top_right, bottom_left, bottom_right], axis=0
    ).astype(np.float64)
    
    return interp_transformed_2d_corners(
        corners=corners,
        transformed=transformed,
        color_space=color_space,
        huemode_x=huemode_x,
        huemode_y=huemode_y,
        bound_types=bound_types,
        border_mode=border_mode,
        border_constant=border_constant,
        border_array=border_array,
        border_feathering=border_feathering,
        feather_hue_mode=feather_hue_mode,
        distance_mode=distance_mode,
        num_threads=num_threads,
        num_channels=num_channels,
        border_value=border_value,
    )