"""
GradientV1 - Legacy gradient implementation.

This module contains the original gradient implementation.
New code should use gradient1dv2 and gradient2dv2 instead.
"""

from ...colors.color import convert_color, get_color_class
from .gradient1d import Gradient1D
from .gradient2d import Gradient2D
from .radial import radial_gradient
from .examples import (
    example,
    example_2d_gradient,
    example_radial_gradient,
    example_gradient_rotate,
    example_arr_rotate,
)

__all__ = [
    "convert_color",
    "get_color_class",
    "Gradient1D",
    "Gradient2D",
    "radial_gradient",
    "example",
    "example_2d_gradient",
    "example_radial_gradient",
    "example_gradient_rotate",
    "example_arr_rotate",
]
