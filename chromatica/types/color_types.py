from __future__ import annotations
from typing import Literal, Tuple, Union
import numpy as np
from numpy import ndarray

Scalar = int | float
IntVector = Tuple[int, ...]
ScalarVector = Tuple[Scalar, ...]
IntElement = Union[int, IntVector]
FloatElement = Union[float, Tuple[float, ...]]
ColorElement = Union[IntElement, FloatElement]
ColorValue = Union[ColorElement, ndarray]  # Includes array support
ColorSpace = Literal["rgb","rgba","hsv","hsva","hsl","hsla"]
HUE_SPACES = {"hsl", "hsla", "hsv", "hsva"}
HueDirection = Literal["cw", "ccw", "shortest", "longest"]

def element_to_array(element: Union[ColorElement, ndarray]) -> np.ndarray:
    """
    Convert a color element to a numpy array.
    
    Args:
        element: Scalar, tuple, or already an ndarray
        
    Returns:
        numpy array representation
    """
    if isinstance(element, ndarray):
        return element
    if isinstance(element, (int, float)):
        return np.array([element])
    return np.array(element)

def is_hue_space(color_space: ColorSpace) -> bool:
    """
    Check if the given color space is a hue-based space (HSV or HSL).
    
    Args:
        color_space: Color space string
    Returns:
        True if hue-based, False otherwise
    """
    return color_space.lower() in HUE_SPACES