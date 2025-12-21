"""
Chromatica Color Classes
========================

This module provides immutable color classes for RGB, HSV, and HSL color spaces
with support for both scalar colors and vectorized color arrays.

Features
--------
- Immutable color instances (frozen after initialization)
- Scalar colors (single color values)
- Array colors (batch operations on multiple colors)
- Automatic dtype validation and enforcement
- Value clamping to valid ranges
- Type conversion between color spaces and formats
- Alpha channel support with WithAlpha mixin

Scalar Usage
-----------
>>> from chromatica.colors.rgb import RGB, RGBA

>>> 
>>> # Create a single color
>>> color = RGB((255, 128, 0))
>>> print(color.value)  # (255, 128, 0)
>>> print(color.is_array)  # False
>>> 
>>> # Convert to HSV
>>> hsv_color = color.convert("hsv")
>>> 
>>> # Work with alpha
>>> rgba = RGBA((255, 128, 0, 255))
>>> print(rgba.alpha)  # 255
>>> semi_transparent = rgba.with_alpha(128)

Array Usage
-----------
>>> import numpy as np
>>> from chromatica.colors.rgb import RGB
>>> 
>>> # Create array of colors
>>> colors = RGB(np.array([
...     [255, 128, 0],
...     [100, 200, 50],
...     [0, 0, 255]
... ], dtype=np.uint16))
>>> 
>>> print(colors.is_array)  # True
>>> print(colors.shape)  # (3, 3)
>>> 
>>> # Convert entire array
>>> hsv_colors = colors.convert("hsv", use_css_algo=True)
>>> 
>>> # Work with alpha channel on arrays
>>> rgba = RGBA(np.array([
...     [255, 128, 0, 255],
...     [100, 200, 50, 128]
... ]))
>>> print(rgba.alpha)  # array([255, 128])
>>> 
>>> # Set alpha for entire array
>>> opaque = rgba.with_alpha(255)
>>> # Or set different alpha per element
>>> variable_alpha = rgba.with_alpha(np.array([200, 100]))

Color Classes
-------------
RGB variants:
    - ColorRGBINT: Integer RGB (0-255)
    - ColorRGBAINT: Integer RGBA with alpha
    - ColorUnitRGB: Float RGB (0.0-1.0)
    - ColorUnitRGBA: Float RGBA with alpha
    - ColorPercentageRGB: Percentage RGB (0-100)
    - ColorPercentageRGBA: Percentage RGBA with alpha

HSV variants:
    - ColorHSVINT: Integer HSV
    - ColorHSVAINT: Integer HSVA with alpha
    - ColorUnitHSV: Float HSV
    - ColorUnitHSVA: Float HSVA with alpha
    - ColorPercentageHSV: Percentage HSV
    - ColorPercentageHSVA: Percentage HSVA with alpha

HSL variants:
    - ColorHSLINT: Integer HSL
    - ColorHSLAINT: Integer HSLA with alpha
    - ColorUnitHSL: Float HSL
    - ColorUnitHSLA: Float HSLA with alpha
    - ColorPercentageHSL: Percentage HSL
    - ColorPercentageHSLA: Percentage HSLA with alpha

Notes
-----
- Array dtypes are validated against format_valid_dtypes
- Arrays must have last dimension equal to num_channels
- Conversion automatically detects scalar vs array and uses appropriate method
- WithAlpha mixin uses [..., -1] indexing for array compatibility
- All values are clamped to maxima during initialization
"""

from .color import color_convert
from .color_base import ColorBase
from .color import unified_tuple_to_class


__all__ = ['color_convert']