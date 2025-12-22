import numpy as np
from boundednumbers import BoundType
from typing import List, Optional
from .....v2core.core import HueMode
from .....v2core.core2d import (
    sample_hue_between_lines_continuous,
    sample_hue_between_lines_discrete,
    multival2d_lerp_between_lines_continuous,
    multival2d_lerp_between_lines_discrete,
)
from .....types.color_types import is_hue_space, ColorSpace
from enum import Enum
from .utils import prepare_hue_and_rest_channels, combine_hue_and_rest_channels


class LineInterpMethods(Enum):
    """Methods for interpolating between lines."""
    LINES_CONTINUOUS = 1
    LINES_DISCRETE = 0


def _get_line_method(line_method: LineInterpMethods, huemode_x: Optional[HueMode] = None):
    """
    Determine the line interpolation line_method to use.
    
    Args:
        line_method: Requested interpolation line_method
        huemode_x: Hue mode for x-axis (if None, uses discrete)
        
    Returns:
        LineInterpMethods enum value
    """
    if huemode_x is None:
        return LineInterpMethods.LINES_DISCRETE
    return line_method


def _interp_transformed_non_hue_space_2d_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: List[np.ndarray],
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Interpolate non-hue values between two lines using discrete x-sampling in multidimensional space.
    """
    # multival2d_lerp_between_lines_discrete expects list of lines, one per channel
    lines0 = [line0[..., i] for i in range(line0.shape[-1])]
    lines1 = [line1[..., i] for i in range(line1.shape[-1])]
    return multival2d_lerp_between_lines_discrete(
        lines0,
        lines1,
        transformed,
        bound_types=bound_types,
    )


def _interp_transformed_non_hue_space_2d_lines_continuous(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: List[np.ndarray],
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Interpolate non-hue values between two lines using continuous sampling in multidimensional space.
    """
    lines0 = [line0[..., i] for i in range(line0.shape[-1])]
    lines1 = [line1[..., i] for i in range(line1.shape[-1])]
    return multival2d_lerp_between_lines_continuous(
        lines0,
        lines1,
        transformed,
        bound_types=bound_types,
    )


def _interp_transformed_hue_space_2d_lines_continuous(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: List[np.ndarray],
    huemode_x: HueMode,
    huemode_y: HueMode,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Interpolate hue values between two lines using continuous sampling in multidimensional space.
    """
    from .....v2core.core import _prepare_bound_types
    
    # Prepare channels
    hline0, rline0 = prepare_hue_and_rest_channels(line0, is_hue=True)
    hline1, rline1 = prepare_hue_and_rest_channels(line1, is_hue=True)
    btypes_list = _prepare_bound_types(bound_types)
    
    # Interpolate hue channel
    hresult = sample_hue_between_lines_continuous(
        hline0,
        hline1,
        transformed,
        mode_x=huemode_x,
        mode_y=huemode_y,
        bound_type=btypes_list[0],
    )
    
    # Interpolate rest channels
    rlines0 = [rline0[..., i] for i in range(rline0.shape[-1])]
    rlines1 = [rline1[..., i] for i in range(rline1.shape[-1])]
    rresult = multival2d_lerp_between_lines_continuous(
        rlines0,
        rlines1,
        transformed,
        bound_types=btypes_list[1:],
    )
    
    return combine_hue_and_rest_channels(hresult, rresult)


def _interp_transformed_hue_space_2d_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: List[np.ndarray],
    huemode_x: HueMode,
    huemode_y: HueMode,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Interpolate hue values between two lines using discrete x-sampling in multidimensional space.
    """
    from .....v2core.core import _prepare_bound_types
    
    # Prepare channels
    hline0, rline0 = prepare_hue_and_rest_channels(line0, is_hue=True)
    hline1, rline1 = prepare_hue_and_rest_channels(line1, is_hue=True)
    btypes_list = _prepare_bound_types(bound_types)
    
    # Interpolate hue channel
    hresult = sample_hue_between_lines_discrete(
        hline0,
        hline1,
        transformed,
        mode_x=huemode_x,
        mode_y=huemode_y,
        bound_type=btypes_list[0],
    )
    
    # Interpolate rest channels
    # multival2d_lerp_between_lines_discrete expects list of lines, one per channel
    rlines0 = [rline0[..., i] for i in range(rline0.shape[-1])]
    rlines1 = [rline1[..., i] for i in range(rline1.shape[-1])]

    rresult = multival2d_lerp_between_lines_discrete(
        rlines0,
        rlines1,
        transformed,
        bound_types=btypes_list[1:],
    )
    
    return combine_hue_and_rest_channels(hresult, rresult)


def interp_transformed_2d_lines(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
    color_space: ColorSpace,
    huemode_x: Optional[HueMode] = None,
    huemode_y: Optional[HueMode] = None,
    line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
    bound_types: List[BoundType] | BoundType = BoundType.CLAMP,

) -> np.ndarray:
    """
    Interpolate values between two lines using transformed coordinates.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        transformed: Transformed coordinates, shape (H, W, 2)
        is_hue_space: Whether the data is in hue space
        huemode_x: Hue interpolation mode for x-axis (if is_hue_space is True)
        huemode_y: Hue interpolation mode for y-axis (if is_hue_space is True)
        line_method: Interpolation line_method to use
        bound_types: List of BoundType for each channel or a single BoundType
        
    Returns:
        Interpolated values, shape (H, W, C)
    """
    
    
    if is_hue_space(color_space):
        line_method = _get_line_method(line_method, huemode_x)
        if huemode_x is None or huemode_y is None:
            raise ValueError("Hue modes must be provided for hue color spaces.")
        
        if line_method == LineInterpMethods.LINES_CONTINUOUS:
            return _interp_transformed_hue_space_2d_lines_continuous(
                line0,
                line1,
                transformed,
                huemode_x=huemode_x,
                huemode_y=huemode_y,
                bound_types=bound_types,
            )
        else:
            return _interp_transformed_hue_space_2d_lines_discrete(
                line0,
                line1,
                transformed,
                huemode_x=huemode_x,
                huemode_y=huemode_y,
                bound_types=bound_types,
            )
    else:
        if line_method == LineInterpMethods.LINES_CONTINUOUS:
            return _interp_transformed_non_hue_space_2d_lines_continuous(
                line0,
                line1,
                transformed,
                bound_types=bound_types,
            )
        else:
            return _interp_transformed_non_hue_space_2d_lines_discrete(
                line0,
                line1,
                transformed,
                bound_types=bound_types,
            )