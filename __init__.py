"""
Chromatica - Advanced Color Manipulation Library
=================================================

A powerful Python library for color manipulation, gradient generation, and color space conversions.
Designed for graphics programming, data visualization, and image processing.

Key Features
------------
- Multi-format color support (RGB, HSV, HSL with INT/FLOAT/PERCENTAGE formats)
- Alpha channel support (RGBA, HSVA, HSLA)
- Advanced gradient generation (1D, 2D, radial, angular)
- Vectorized color array operations
- Color space conversions with high precision
- Arithmetic operations on colors with overflow control
- Immutable color instances for safe sharing

Quick Start
-----------
>>> from chromatica import ColorUnitRGB, Gradient1D, FormatType
>>> 
>>> # Create colors
>>> red = ColorUnitRGB((1.0, 0.0, 0.0))
>>> blue = ColorUnitRGB((0.0, 0.0, 1.0))
>>> 
>>> # Generate gradient
>>> gradient = Gradient1D.from_colors(
...     red, blue, steps=10, 
...     color_space='rgb', 
...     format_type=FormatType.FLOAT
... )
>>> 
>>> # Convert color spaces
>>> hsv = red.convert('hsv', FormatType.FLOAT)

Modules
-------
- colors: Color classes for RGB, HSV, HSL with various formats
- color_arr: Color array classes with generation methods
- gradient: Gradient generation (1D, 2D, radial)
- conversions: Color space conversion functions
- arithmetic: Arithmetic operations with overflow control
"""

from .chromatica.colors.rgb import (
    ColorRGBINT, ColorRGBAINT,
    ColorUnitRGB, ColorUnitRGBA,
    ColorPercentageRGB, ColorPercentageRGBA
)
from .chromatica.colors.hsv import (
    ColorHSVINT, ColorHSVAINT,
    UnitHSV, UnitHSVA,
    PercentageHSV, PercentageHSVA
)
from .chromatica.colors.hsl import (
    ColorHSLINT, ColorHSLAINT,
    UnitHSL, UnitHSLA,
    PercentageHSL, PercentageHSLA
)
from .chromatica.colors.color_base import ColorBase
from .chromatica.colors.arithmetic import make_arithmetic

from .chromatica.color_arr import Color1DArr, Color2DArr

from .chromatica.gradient import (
    Gradient1D, Gradient2D,
    radial_gradient,
    example, example_2d_gradient, example_radial_gradient,
    example_gradient_rotate, example_arr_rotate
)

from .chromatica.conversions import (
    convert, np_convert,
    ColorSpace, FormatType
)

from boundednumbers.functions import clamp, bounce, cyclic_wrap_float

__version__ = "1.0.0"

__all__ = [
    # Color classes - RGB
    "ColorRGBINT", "ColorRGBAINT",
    "ColorUnitRGB", "ColorUnitRGBA",
    "ColorPercentageRGB", "ColorPercentageRGBA",
    
    # Color classes - HSV
    "ColorHSVINT", "ColorHSVAINT",
    "UnitHSV", "UnitHSVA",
    "PercentageHSV", "PercentageHSVA",
    
    # Color classes - HSL
    "ColorHSLINT", "ColorHSLAINT",
    "UnitHSL", "UnitHSLA",
    "PercentageHSL", "PercentageHSLA",
    
    # Base and utilities
    "ColorBase",
    "make_arithmetic",
    
    # Color arrays
    "Color1DArr", "Color2DArr",
    
    # Gradients
    "Gradient1D", "Gradient2D",
    "radial_gradient",
    
    # Example functions
    "example", "example_2d_gradient", "example_radial_gradient",
    "example_gradient_rotate", "example_arr_rotate",
    
    # Conversions
    "convert", "np_convert",
    "ColorSpace", "FormatType",
    
    # Utility functions
    "clamp", "bounce", "cyclic_wrap_float",
    
    # Version
    "__version__",
]
