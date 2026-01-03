#chromatica\gradients\gradient2dv2\helpers\interpolation\lines.py  # NEW
import numpy as np
from boundednumbers import BoundType
from typing import List, Optional

from .....v2core.interp_2d import (
    lerp_between_lines,
    lerp_between_lines_x_discrete,
    lerp_between_lines_onto_array,
    lerp_between_lines_onto_array_x_discrete,

)
from .....v2core.interp_hue import(
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
    hue_lerp_between_lines_array_border,
    hue_lerp_between_lines_array_border_x_discrete,
    hue_lerp_between_lines_inplace_x_discrete,
    hue_lerp_between_lines_inplace,
)
from .....types.color_types import is_hue_space, ColorModes, HueDirection
from enum import Enum
from .utils import prepare_hue_and_rest_channels, combine_hue_and_rest_channels
from .....v2core.border_handler import BorderMode, BorderModeInput, BorderConstant, DistanceMode, BorderArrayInput, DistanceModeInput
 
class LineInterpMethods(Enum):
    """Methods for interpolating between lines."""
    LINES_CONTINUOUS = 1
    LINES_DISCRETE = 0


def _get_line_method(line_method: LineInterpMethods, hue_direction_x: Optional[HueDirection] = None):
    """
    Determine the line interpolation line_method to use.
    
    Args:
        line_method: Requested interpolation line_method
        hue_direction_x: Hue mode for x-axis (if None, uses discrete)
        
    Returns:
        LineInterpMethods enum value
    """
    if hue_direction_x is None:
        return LineInterpMethods.LINES_DISCRETE
    return line_method


def _interp_transformed_non_hue_space_2d_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: List[np.ndarray],
    
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_value: Optional[BorderConstant] = None,    
    *,
    border_constant: Optional[BorderConstant] = None,
    border_array: BorderArrayInput = None,
    border_feathering: float = 0.0,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Interpolate non-hue values between two lines using discrete x-sampling in multidimensional space.
    """
    # Bound types is legacy for border mode
    border_constant = next((i for i in [border_value, border_constant] if i is not None), None)
    
    if border_array is not None:
        return lerp_between_lines_onto_array_x_discrete(
            line0=line0,
            line1=line1,
            coords=transformed,
            background_array=border_array,
            border_mode=border_mode,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    else:
        return lerp_between_lines_x_discrete(
            line0=line0,
            line1=line1,
            coords=transformed,
            border_mode=border_mode,
            border_constant=border_constant,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
        )


def _interp_transformed_non_hue_space_2d_lines_continuous(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: List[np.ndarray],
    
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_value: Optional[BorderConstant] = None,    
    *,
    border_constant: Optional[BorderConstant] = None,
    border_array: BorderArrayInput = None,
    border_feathering: float = 0.0,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Interpolate non-hue values between two lines using continuous sampling in multidimensional space.
    """
    # Bound types is legacy for border mode
    border_constant = next((i for i in [border_value, border_constant] if i is not None), None)
    
    if border_array is not None:
        return lerp_between_lines_onto_array(
            line0=line0,
            line1=line1,
            coords=transformed,
            background_array=border_array,
            border_mode=border_mode,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
            x_discrete=False,
        )
    else:
        return lerp_between_lines(
            line0=line0,
            line1=line1,
            coords=transformed,
            border_mode=border_mode,
            border_constant=border_constant,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    

def _interp_transformed_hue_space_2d_lines_continuous(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: List[np.ndarray],
    hue_direction_x: HueDirection,
    hue_direction_y: HueDirection,
    *,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstant = 0,
    border_array: BorderArrayInput = None,
    border_feathering: float = 0,
    feather_hue_mode: HueDirection = HueDirection.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    x_discrete: bool = False,
    num_channels: Optional[int] = 3,
    border_value: Optional[BorderConstant] = None,
) -> np.ndarray:
    """
    Interpolate hue values between two lines using continuous sampling in multidimensional space.
    """
    from .....v2core.core import _prepare_bound_types

    # Prepare channels
    if isinstance(transformed, np.ndarray) and transformed.ndim == 3:
        # Same coords
        transformed_h = transformed.copy()
        num_channels = line0.shape[-1]
        transformed_r = [transformed.copy() for _ in range(num_channels - 1)]
    elif isinstance(transformed, list) and len(transformed) == 1 and transformed[0].ndim == 3:
        # Same coords
        transformed_h = transformed[0].copy()
        num_channels = line0.shape[-1]
        transformed_r = [transformed[0].copy() for _ in range(num_channels - 1)]
    else:
        transformed_h = transformed[0]
        transformed_r = transformed[1:]

    hline0, rline0 = prepare_hue_and_rest_channels(line0, is_hue=True)
    hline1, rline1 = prepare_hue_and_rest_channels(line1, is_hue=True)
    #legacy
    #btypes_list = _prepare_bound_types(bound_types, num_channels=num_channels)

    # Interpolate hue channel
    border_constant = next((i for i in [border_value, border_constant] if i is not None), None)
    if border_array is not None:
        # If border_array is 3D
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
    hresult = hue_lerp_between_lines(
        line0=hline0,
        line1=hline1,
        coords=transformed_h,
        mode_x=hue_direction_x,
        mode_y=hue_direction_y,

        border_mode=border_mode,
        border_constant=border_constant,
        border_array=hue_border_array,
        border_feathering=border_feathering,
        feather_hue_mode=feather_hue_mode,
        distance_mode=distance_mode,
        num_threads=num_threads,
    )

    # Interpolate rest channels
    rlines0 = rline0
    rlines1 = rline1
    if isinstance(border_value, (list, np.ndarray)):
        rest_border_value = border_value[1:]
    else:
        rest_border_value = border_value

    if rest_border_array is not None:
        rresult = lerp_between_lines_onto_array(
            line0=rlines0,
            line1=rlines1,
            coords=transformed_r,
            background_array=rest_border_array,
            border_mode=border_mode,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
            x_discrete=False,
        )
    else:
        rresult = lerp_between_lines(
            line0=rlines0,
            line1=rlines1,
            coords=transformed_r,
            border_mode=border_mode,
            border_constant=rest_border_value,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
            x_discrete=False,
        )

    return combine_hue_and_rest_channels(hresult, rresult)


def _interp_transformed_hue_space_2d_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: List[np.ndarray],
    hue_direction_y: HueDirection,
    *,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstant = 0,
    border_array: BorderArrayInput = None,
    border_feathering: float = 0,
    feather_hue_mode: HueDirection = HueDirection.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    num_channels: Optional[int] = 3,
    border_value: Optional[BorderConstant] = None,
) -> np.ndarray:
    """
    Interpolate hue values between two lines using discrete x-sampling in multidimensional space.
    """
    from .....v2core.core import _prepare_bound_types
    
    # Prepare channels
    if isinstance(transformed, np.ndarray) and transformed.ndim == 3:
        # Same coords
        transformed_h = transformed.copy()
        num_channels = line0.shape[-1]
        transformed_r = [transformed.copy() for _ in range(num_channels - 1)]
    elif isinstance(transformed, list) and len(transformed) == 1 and transformed[0].ndim == 3:
        # Same coords
        transformed_h = transformed[0].copy()
        num_channels = line0.shape[-1]
        transformed_r = [transformed[0].copy() for _ in range(num_channels - 1)]
    else:
        transformed_h = transformed[0]
        transformed_r = transformed[1:]

    hline0, rline0 = prepare_hue_and_rest_channels(line0, is_hue=True)
    hline1, rline1 = prepare_hue_and_rest_channels(line1, is_hue=True)
    btypes_list = _prepare_bound_types(bound_types, num_channels=num_channels)

    # Interpolate hue channel
    border_constant = next((i for i in [border_value, border_constant] if i is not None), None)
    if border_array is not None:
        # If border_array is 3D
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

    hresult = hue_lerp_between_lines_x_discrete(
        line0=hline0,
        line1=hline1,
        coords=transformed_h,
        mode_y=hue_direction_y,
        border_mode=border_mode,
        border_constant=border_constant,
        border_array=hue_border_array,
        border_feathering=border_feathering,
        feather_hue_mode=feather_hue_mode,
        distance_mode=distance_mode,
        num_threads=num_threads,
    )

    # Interpolate rest channels
    rlines0 = rline0
    rlines1 = rline1
    if isinstance(border_value, (list, np.ndarray)):
        rest_border_value = border_value[1:]
    else:
        rest_border_value = border_value

    if rest_border_array is not None:
        rresult = lerp_between_lines_onto_array_x_discrete(
            line0=rlines0,
            line1=rlines1,
            coords=transformed_r,
            background_array=rest_border_array,
            border_mode=border_mode,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
        )
    else:
        rresult = lerp_between_lines_x_discrete(
            line0=rlines0,
            line1=rlines1,
            coords=transformed_r,
            border_mode=border_mode,
            border_constant=rest_border_value,
            border_feathering=border_feathering,
            distance_mode=distance_mode,
            num_threads=num_threads,
        )

    return combine_hue_and_rest_channels(hresult, rresult)


def interp_transformed_2d_lines(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
    color_mode: ColorModes,
    hue_direction_x: Optional[HueDirection] = None,
    hue_direction_y: Optional[HueDirection] = None,
    line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
    border_mode: BorderModeInput = BorderMode.CLAMP,
    border_constant: BorderConstant = 0,
    border_array: BorderArrayInput = None,
    border_feathering: float = 0.0,
    feather_hue_mode: HueDirection = HueDirection.SHORTEST,
    distance_mode: DistanceModeInput = DistanceMode.ALPHA_MAX,
    num_threads: int = -1,
    num_channels: Optional[int] = None,
    *,
    border_value: Optional[BorderConstant] = None,  # Legacy parameter
) -> np.ndarray:
    """
    Interpolate values between two lines using transformed coordinates.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        transformed: Transformed coordinates, shape (H, W, 2)
        color_mode: Color space of the data
        hue_direction_x: Hue interpolation mode for x-axis (if is_hue_space is True)
        hue_direction_y: Hue interpolation mode for y-axis (if is_hue_space is True)
        line_method: Interpolation line_method to use
        bound_types: List of BoundType for each channel or a single BoundType
        border_mode: Border handling mode (e.g., BorderMode.CLAMP, BorderMode.REPEAT)
        border_constant: Border constant value for BORDER_CONSTANT mode
        border_array: Optional array to use as border/background
        border_feathering: Amount of feathering to apply at borders
        feather_hue_mode: Hue mode to use when feathering hue values
        distance_mode: Mode for calculating distance (for feathering)
        num_threads: Number of threads to use (-1 for auto)
        num_channels: Number of channels (inferred from color_mode if not provided)
        border_value: Legacy parameter, use border_constant instead
        
    Returns:
        Interpolated values, shape (H, W, C)
    """
    # Handle legacy border_value parameter
    border_constant = next((i for i in [border_value, border_constant] if i is not None), 0)
    
    num_channels = num_channels or len(color_mode)

    if is_hue_space(color_mode):
        line_method = _get_line_method(line_method, hue_direction_x)

        if hue_direction_y is None:
            raise ValueError("hue_direction_y must be specified for hue space interpolation between lines.")
        
        if line_method == LineInterpMethods.LINES_CONTINUOUS:
            return _interp_transformed_hue_space_2d_lines_continuous(
                line0,
                line1,
                transformed,
                hue_direction_x=hue_direction_x,
                hue_direction_y=hue_direction_y,
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
            return _interp_transformed_hue_space_2d_lines_discrete(
                line0,
                line1,
                transformed,
                hue_direction_y=hue_direction_y,
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
        if line_method == LineInterpMethods.LINES_CONTINUOUS:
            return _interp_transformed_non_hue_space_2d_lines_continuous(
                line0,
                line1,
                transformed,
                bound_types=bound_types,
                border_mode=border_mode,
                border_constant=border_constant,
                border_array=border_array,
                border_feathering=border_feathering,
                distance_mode=distance_mode,
                num_threads=num_threads,
            )
        else:
            return _interp_transformed_non_hue_space_2d_lines_discrete(
                line0,
                line1,
                transformed,
                bound_types=bound_types,
                border_mode=border_mode,
                border_constant=border_constant,
                border_array=border_array,
                border_feathering=border_feathering,
                distance_mode=distance_mode,
                num_threads=num_threads,
            )