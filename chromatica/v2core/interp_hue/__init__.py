"""
Hue Interpolation Module

Provides type-hinted Python wrappers for Cython-optimized hue interpolation functions.
Functions are imported from wrappers.py which adds type hints to the Cython implementations.
"""

from .wrappers import (
    hue_lerp_1d_spatial,
    hue_lerp_simple,
    hue_lerp_arrays,
    hue_multidim_lerp,
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
    hue_lerp_2d_spatial,
    hue_lerp_2d_with_modes,
)

__all__ = [
    'hue_lerp_1d_spatial',
    'hue_lerp_simple',
    'hue_lerp_arrays',
    'hue_multidim_lerp',
    'hue_lerp_between_lines',
    'hue_lerp_between_lines_x_discrete',
    'hue_lerp_2d_spatial',
    'hue_lerp_2d_with_modes',
]
