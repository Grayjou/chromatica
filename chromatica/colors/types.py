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
ColorModes = Literal["rgb","rgba","hsv","hsva","hsl","hsla"]
HUE_SPACES = {"hsl", "hsla", "hsv", "hsva"}

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


