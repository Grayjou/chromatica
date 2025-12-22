"""
Helper utilities for gradient2dv2 operations.
"""

from .interpolation import (
    LineInterpMethods,
    get_line_method,
    interp_transformed_hue_2d_corners,
    interp_transformed_hue_2d_lines_continuous,
    interp_transformed_hue_2d_lines_discrete,
    interp_transformed_non_hue_2d_corners,
    interp_transformed_non_hue_2d_lines_continuous,
    interp_transformed_non_hue_2d_lines_discrete,
)

from .cell_utils import (
    CellMode,
    apply_per_channel_transforms_2d,
    separate_hue_and_non_hue_transforms,
)

# Re-export HueMode from v2core for convenience
from ...v2core.core import HueMode

__all__ = [
    # Interpolation
    'LineInterpMethods',
    'get_line_method',
    'interp_transformed_hue_2d_corners',
    'interp_transformed_hue_2d_lines_continuous',
    'interp_transformed_hue_2d_lines_discrete',
    'interp_transformed_non_hue_2d_corners',
    'interp_transformed_non_hue_2d_lines_continuous',
    'interp_transformed_non_hue_2d_lines_discrete',
    
    # Cell utilities
    'CellMode',
    'apply_per_channel_transforms_2d',
    'separate_hue_and_non_hue_transforms',
    
    # Re-exports
    'HueMode',
]
