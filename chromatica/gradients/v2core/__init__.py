"""
V2 Core interpolation module.

This module provides high-performance interpolation functions with Cython acceleration
for 1D and 2D gradient operations.
"""

# Import main API from core module
from .core import (
    multival1d_lerp,
    multival2d_lerp,
    multival1d_lerp_uniform,
    multival2d_lerp_uniform,
    single_channel_multidim_lerp,
    bound_coeffs,
    bound_coeffs_fused,
)

# Import 2D interpolation functions from interp_2d
try:
    from .interp_2d import (
        lerp_between_lines,
        lerp_between_planes,
    )
except ImportError:
    # If Cython extension is not built, these will not be available
    lerp_between_lines = None
    lerp_between_planes = None

__all__ = [
    # 1D/2D multi-value interpolation
    "multival1d_lerp",
    "multival2d_lerp",
    "multival1d_lerp_uniform",
    "multival2d_lerp_uniform",
    "single_channel_multidim_lerp",
    # Utility functions
    "bound_coeffs",
    "bound_coeffs_fused",
    # 2D interpolation between lines/planes
    "lerp_between_lines",
    "lerp_between_planes",
]
