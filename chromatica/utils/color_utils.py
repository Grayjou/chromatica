"""Utility functions for color space conversion in gradient operations."""

from typing import Union, List
import numpy as np
from numpy import ndarray as NDArray
from ..colors import unified_tuple_to_class
from ..colors.color_base import ColorBase
from ..types.format_type import FormatType
from ..types.color_types import ColorSpace
from ..types.array_types import ndarray_1d


def convert_to_space_float(
    color: Union[ColorBase, tuple, List, ndarray_1d, NDArray],
    from_space: ColorSpace,
    format_type: FormatType,
    to_space: ColorSpace,
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

    from_class = unified_tuple_to_class[(from_space, format_type)]
    to_float_class = unified_tuple_to_class[(to_space, FormatType.FLOAT)]
    return to_float_class(from_class(color))
