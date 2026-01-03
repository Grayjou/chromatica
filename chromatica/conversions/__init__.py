"""
Chromatica Color Space Conversions
===================================

This module provides comprehensive color space conversion utilities supporting RGB, HSV, and HSL
color spaces with both scalar and vectorized (numpy) implementations.

Features
--------
- Bidirectional conversions: RGB ↔ HSV ↔ HSL
- Scalar functions for single color conversions
- Vectorized numpy functions for batch processing
- CSS Color 4 / Culori-compatible algorithms
- Multiple format types: integer (0-255), float (0.0-1.0), percentage (0-100)

Conversion Functions
-------------------

RGB → HSV:
    unit_rgb_to_hsv(r, g, b, use_css_algo=False)
        Scalar RGB to HSV conversion
    np_unit_rgb_to_hsv(r, g, b, use_css_algo=False)
        Vectorized RGB to HSV conversion
    unit_rgb_to_hsv_analytic(r, g, b)
        Analytical algorithm only (no CSS option)

RGB → HSL:
    unit_rgb_to_hsl(r, g, b, use_css_algo=False)
        Scalar RGB to HSL conversion
    np_unit_rgb_to_hsl(r, g, b, use_css_algo=False)
        Vectorized RGB to HSL conversion
    unit_rgb_to_hsl_analytic(r, g, b)
        Analytical algorithm only (no CSS option)

HSV → RGB:
    hsv_to_unit_rgb(h, s, v, use_css_algo=False)
        Scalar HSV to RGB conversion
    np_hsv_to_unit_rgb(h, s, v, use_css_algo=False)
        Vectorized HSV to RGB conversion

HSL → RGB:
    hsl_to_unit_rgb(h, s, l, use_css_algo=False)
        Scalar HSL to RGB conversion
    np_hsl_to_unit_rgb(h, s, l, use_css_algo=False)
        Vectorized HSL to RGB conversion
    hsl_to_unit_rgb_fast(h, s, l)
        Fast approximate conversion using analytical functions

HSV ↔ HSL:
    hsv_to_hsl(h, s, v)
        Convert HSV to HSL
    hsl_to_hsv(h, s, l)
        Convert HSL to HSV
    np_hsv_to_hsl(h, s, v)
        Vectorized HSV to HSL
    np_hsl_to_hsv(h, s, l)
        Vectorized HSL to HSV

High-Level API
-------------
    convert(color, from_space, to_space, input_type, output_type, use_css_algo=False)
        Universal color space converter with format handling
    np_convert(color, from_space, to_space, input_type, output_type, use_css_algo=False)
        Vectorized universal converter

Types & Enums
------------
    ColorMode: Type alias for color space names ("rgb", "hsv", "hsl", "rgba", "hsva", "hsla")
    FormatType: Enum for value formats (INT, FLOAT, PERCENTAGE)

Algorithm Selection
------------------
All RGB↔HSV and RGB↔HSL functions accept `use_css_algo` parameter:
    - use_css_algo=False (default): Standard analytical algorithm
    - use_css_algo=True: CSS Color 4 / Culori-compatible algorithm with linear RGB and atan2-based hue

Examples
--------
>>> from chromatica.conversions import unit_rgb_to_hsv, hsv_to_unit_rgb

>>> 
>>> # Convert RGB to HSV
>>> h, s, v = unit_rgb_to_hsv(UnitFloat(1.0), UnitFloat(0.5), UnitFloat(0.0))
>>> print(f"Hue: {h}°, Saturation: {s}, Value: {v}")
>>> 
>>> # Convert back to RGB
>>> r, g, b = hsv_to_unit_rgb(h, s, v)
>>> 
>>> # Use CSS algorithm
>>> h_css, s_css, v_css = unit_rgb_to_hsv(r, g, b, use_css_algo=True)
>>> 
>>> # Vectorized conversion
>>> import numpy as np
>>> from chromatica.conversions import np_unit_rgb_to_hsv
>>> rgb_array = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
>>> hsv_array = np_unit_rgb_to_hsv(rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2])
"""

# RGB → HSV conversions
from .to_hsv import (
    unit_rgb_to_hsv,
    unit_rgb_to_hsv_analytic,
    unit_rgb_to_hsv_picker,
    np_unit_rgb_to_hsv,
)

# RGB → HSL conversions
from .to_hsl import (
    unit_rgb_to_hsl,
    unit_rgb_to_hsl_analytic,
    np_unit_rgb_to_hsl,
)

# HSV → RGB conversions
from .to_rgb import (
    hsv_to_unit_rgb,
    hsv_to_unit_rgb_analytical,
    np_hsv_to_unit_rgb,
)

# HSL → RGB conversions
from .to_rgb import (
    hsl_to_unit_rgb,
    hsl_to_unit_rgb_analytical,
    hsl_to_unit_rgb_fast,
    np_hsl_to_unit_rgb,
    np_hsl_to_unit_rgb_fast,
)

# HSV ↔ HSL conversions
from .to_hsv import hsl_to_hsv, np_hsl_to_hsv
from .to_hsl import hsv_to_hsl, np_hsv_to_hsl

# High-level API
from .wrapper import convert, np_convert, ColorMode

# Types and enums
from ..types.format_type import FormatType

__all__ = [
    # RGB → HSV
    'unit_rgb_to_hsv',
    'unit_rgb_to_hsv_analytic',
    'unit_rgb_to_hsv_picker',
    'np_unit_rgb_to_hsv',
    
    # RGB → HSL
    'unit_rgb_to_hsl',
    'unit_rgb_to_hsl_analytic',
    'np_unit_rgb_to_hsl',
    
    # HSV → RGB
    'hsv_to_unit_rgb',
    'hsv_to_unit_rgb_analytical',
    'np_hsv_to_unit_rgb',
    
    # HSL → RGB
    'hsl_to_unit_rgb',
    'hsl_to_unit_rgb_analytical',
    'hsl_to_unit_rgb_fast',
    'np_hsl_to_unit_rgb',
    'np_hsl_to_unit_rgb_fast',
    
    # HSV ↔ HSL
    'hsv_to_hsl',
    'hsl_to_hsv',
    'np_hsv_to_hsl',
    'np_hsl_to_hsv',
    
    # High-level API
    'convert',
    'np_convert',
    'ColorMode',
    
    # Types
    'FormatType',
]