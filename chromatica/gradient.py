"""
Gradient Generation Module
==========================

This module provides gradient generation classes and functions for creating
1D and 2D color gradients with advanced interpolation options.

The implementation is split across multiple submodules in ``chromatica.gradients``
for readability and easier testing.
"""

from .gradients import (
    Gradient1D,
    Gradient2D,
    radial_gradient,
    example,
    example_2d_gradient,
    example_radial_gradient,
    example_gradient_rotate,
    example_arr_rotate,
)

__all__ = [
    "Gradient1D",
    "Gradient2D",
    "radial_gradient",
    "example",
    "example_2d_gradient",
    "example_radial_gradient",
    "example_gradient_rotate",
    "example_arr_rotate",
]
