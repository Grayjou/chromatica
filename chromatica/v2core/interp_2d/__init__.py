"""
2D Interpolation Module

Provides type-hinted Python wrappers for Cython-optimized 2D interpolation functions.
Functions are imported from wrappers.py which adds type hints to the Cython implementations.
"""

from .wrappers import (
    lerp_between_lines,
    lerp_between_lines_x_discrete_1ch,
    lerp_between_lines_multichannel,
    lerp_between_lines_x_discrete_multichannel,
    lerp_from_corners,
    lerp_from_corners_1ch_flat,
    lerp_from_corners_multichannel,
    lerp_from_corners_multichannel_same_coords,
    lerp_from_corners_multichannel_flat,
    lerp_from_corners_multichannel_flat_same_coords,
    lerp_between_planes,
)

__all__ = [
    'lerp_between_lines',
    'lerp_between_lines_x_discrete_1ch',
    'lerp_between_lines_multichannel',
    'lerp_between_lines_x_discrete_multichannel',
    'lerp_from_corners',
    'lerp_from_corners_1ch_flat',
    'lerp_from_corners_multichannel',
    'lerp_from_corners_multichannel_same_coords',
    'lerp_from_corners_multichannel_flat',
    'lerp_from_corners_multichannel_flat_same_coords',
    'lerp_between_planes',
]
