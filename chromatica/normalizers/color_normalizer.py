from ..colors import ColorBase
from ..types.color_types import ColorElement
from typing import Optional, Tuple
import numpy as np


ColorInput = ColorElement | ColorBase | np.ndarray
def validate_and_return_1d_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")
    return arr
def normalize_color_input(color_input: ColorInput) -> Tuple: # type: ignore
    if isinstance(color_input, ColorBase):
        if isinstance(color_input.value, np.ndarray):
            return tuple(validate_and_return_1d_array(color_input.value).tolist())
        elif isinstance(color_input.value, tuple):
            return color_input.value
    elif isinstance(color_input, np.ndarray):
        return tuple(validate_and_return_1d_array(color_input).tolist())
    elif isinstance(color_input, tuple):
        return color_input
    elif isinstance(color_input, (int, float)):
        return (color_input,)
    elif isinstance(color_input, list):
        return tuple(color_input)
    else:
        raise TypeError("Unsupported color input type.")