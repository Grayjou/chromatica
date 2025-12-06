"""Chromatica: advanced color manipulation utilities."""

from .colors.rgb import (
    ColorRGBINT,
    ColorRGBAINT,
    ColorUnitRGB,
    ColorUnitRGBA,
    ColorPercentageRGB,
    ColorPercentageRGBA,
)
from .colors.hsv import (
    ColorHSVINT,
    ColorHSVAINT,
    UnitHSV,
    UnitHSVA,
    PercentageHSV,
    PercentageHSVA,
)
from .colors.hsl import (
    ColorHSLINT,
    ColorHSLAINT,
    UnitHSL,
    UnitHSLA,
    PercentageHSL,
    PercentageHSLA,
)
from .colors.color_base import ColorBase
from .colors.color import color_convert

# Friendly aliases for common integer variants
ColorRGB = ColorRGBINT
ColorRGBA = ColorRGBAINT
ColorHSV = ColorHSVINT
ColorHSL = ColorHSLINT

from .color_arr import Color1DArr, Color2DArr
from .gradient import (
    Gradient1D,
    Gradient2D,
    radial_gradient,
    example,
    example_2d_gradient,
    example_radial_gradient,
    example_gradient_rotate,
    example_arr_rotate,
)
from .conversions import (
    hsl_to_hsv,
    hsv_to_hsl,
    unit_rgb_to_hsv,
    unit_rgb_to_hsl,
    hsv_to_unit_rgb,
    hsl_to_unit_rgb,
    np_hsl_to_hsv,
    np_hsv_to_hsl,
    np_unit_rgb_to_hsv,
    np_unit_rgb_to_hsl,
    np_hsv_to_unit_rgb,
    np_hsl_to_unit_rgb,
    convert,
    np_convert,
)

__all__ = [
    # core color types
    "ColorBase",
    "ColorRGBINT",
    "ColorRGBAINT",
    "ColorUnitRGB",
    "ColorUnitRGBA",
    "ColorPercentageRGB",
    "ColorPercentageRGBA",
    "ColorHSVINT",
    "ColorHSVAINT",
    "UnitHSV",
    "UnitHSVA",
    "PercentageHSV",
    "PercentageHSVA",
    "ColorHSLINT",
    "ColorHSLAINT",
    "UnitHSL",
    "UnitHSLA",
    "PercentageHSL",
    "PercentageHSLA",
    "ColorRGB",
    "ColorRGBA",
    "ColorHSV",
    "ColorHSL",
    "color_convert",
    # arrays and gradients
    "Color1DArr",
    "Color2DArr",
    "Gradient1D",
    "Gradient2D",
    "radial_gradient",
    "example",
    "example_2d_gradient",
    "example_radial_gradient",
    "example_gradient_rotate",
    "example_arr_rotate",
    # conversions
    "hsl_to_hsv",
    "hsv_to_hsl",
    "unit_rgb_to_hsv",
    "unit_rgb_to_hsl",
    "hsv_to_unit_rgb",
    "hsl_to_unit_rgb",
    "np_hsl_to_hsv",
    "np_hsv_to_hsl",
    "np_unit_rgb_to_hsv",
    "np_unit_rgb_to_hsl",
    "np_hsv_to_unit_rgb",
    "np_hsl_to_unit_rgb",
    "convert",
    "np_convert",
]
