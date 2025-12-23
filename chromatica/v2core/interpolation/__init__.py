"""
Interpolation submodule for chromatica.v2core.

This module contains Cython-accelerated interpolation functions.
"""

# Import interpolation functions from Cython modules
try:
    from .interp import (
        lerp_bounded_2d_spatial_fast,
    )
except ImportError:
    # If Cython extension is not built
    lerp_bounded_2d_spatial_fast = None

try:
    from .interp_2d import (
        lerp_between_lines,
        lerp_between_planes,
        lerp_between_lines_x_discrete,
    )
except ImportError:
    # If Cython extension is not built
    lerp_between_lines = None
    lerp_between_planes = None
    lerp_between_lines_x_discrete = None

try:
    from .interp_hue import (
        hue_lerp_between_lines,
        hue_lerp_between_lines_x_discrete,
        hue_multidim_lerp,
    )
except ImportError:
    # If Cython extension is not built
    hue_lerp_between_lines = None
    hue_lerp_between_lines_x_discrete = None
    hue_multidim_lerp = None

try:
    from .corner_interp_2d import (
        lerp_from_corners,
    )
except ImportError:
    # If Cython extension is not built
    lerp_from_corners = None

__all__ = [
    'lerp_bounded_2d_spatial_fast',
    'lerp_between_lines',
    'lerp_between_planes',
    'lerp_between_lines_x_discrete',
    'hue_lerp_between_lines',
    'hue_lerp_between_lines_x_discrete',
    'hue_multidim_lerp',
    'lerp_from_corners',
]
