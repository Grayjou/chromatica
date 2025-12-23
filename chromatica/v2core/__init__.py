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
    hue_lerp,
    hue_lerp_multi,
    hue_multidim_lerp_bounded,
    hue_gradient_1d,
    hue_gradient_2d,
    sample_hue_between_lines,
    make_hue_line_sampler,
    sample_between_lines,
    HueMode,
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

# Import 2D wrappers from core2d
from .core2d import (
    sample_between_lines_continuous,
    sample_between_lines_discrete,
    sample_hue_between_lines_continuous,
    sample_hue_between_lines_discrete,
    multival2d_lerp_between_lines_continuous,
    multival2d_lerp_between_lines_discrete,
    multival2d_lerp_from_corners,
    sample_between_planes,
)

# Import base class
from .subgradient import SubGradient

# Import border handling

from .border_handler import (
    handle_border_edges_2d,
    handle_border_lines_2d,
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)


__all__ = [
    # 1D/2D multi-value interpolation
    "multival1d_lerp",
    "multival2d_lerp",
    "multival1d_lerp_uniform",
    "multival2d_lerp_uniform",
    "single_channel_multidim_lerp",
    
    # Hue interpolation
    "hue_lerp",
    "hue_lerp_multi",
    "hue_multidim_lerp_bounded",
    "hue_gradient_1d",
    "hue_gradient_2d",
    "sample_hue_between_lines",
    "make_hue_line_sampler",
    
    # Utility functions
    "bound_coeffs",
    "bound_coeffs_fused",
    "HueMode",
    
    # 2D interpolation between lines/planes
    "lerp_between_lines",
    "lerp_between_planes",
    "sample_between_lines",
    
    # 2D wrappers
    "sample_between_lines_continuous",
    "sample_between_lines_discrete",
    "sample_hue_between_lines_continuous",
    "sample_hue_between_lines_discrete",
    "multival2d_lerp_between_lines_continuous",
    "multival2d_lerp_between_lines_discrete",
    "multival2d_lerp_from_corners",
    "sample_between_planes",
    
    # Base classes
    "SubGradient",
    
    # Border handling
    "handle_border_edges_2d",
    "handle_border_lines_2d",
    "BORDER_REPEAT",
    "BORDER_MIRROR",
    "BORDER_CONSTANT",
    "BORDER_CLAMP",
    "BORDER_OVERFLOW",
]
