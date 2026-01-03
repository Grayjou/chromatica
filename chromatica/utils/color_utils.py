"""Utility functions for color space conversion in gradient operations."""

from typing import Union, List
import numpy as np
from numpy import ndarray as NDArray
from ..colors import unified_tuple_to_class
from ..colors.color_base import ColorBase
from ..types.format_type import FormatType
from ..types.color_types import ColorMode
from ..types.array_types import ndarray_1d


def convert_to_space_float(
    color: Union[ColorBase, tuple, List, ndarray_1d, NDArray],
    from_space: ColorMode,
    format_type: FormatType,
    to_space: ColorMode,
) -> ColorBase:
    """
    Convert a color to a specified color space in float format.
    
    Args:
        color: Input color in various formats
        from_space: Source color space
        format_type: Format type of input color
        to_space: Target color space
        
    Returns:
        ColorBase object in target space with float format
    """

    from_space, to_space = ColorMode(from_space), ColorMode(to_space)
    from_class = unified_tuple_to_class[(from_space, format_type)]
    to_float_class = unified_tuple_to_class[(to_space, FormatType.FLOAT)]
    return to_float_class(from_class(color))

def is_hue_color_grayscale(color:np.ndarray, thresh=1e-5) -> bool:
    """Check if a color in HSV space is grayscale (zero saturation)."""
    return color[1] < thresh

def is_hue_color_arr_grayscale(color:np.ndarray, thresh=1e-5) -> np.ndarray:
    """Masks for colors in HSV space that are grayscale (zero saturation)."""
    return color[:, 1] < thresh