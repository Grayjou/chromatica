from ..colors.color import convert_color, get_color_class
from .gradientv1.gradient1d import Gradient1D
from .gradientv1.gradient2d import Gradient2D
from .gradientv1.radial import radial_gradient
from .gradientv1.examples import (
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
