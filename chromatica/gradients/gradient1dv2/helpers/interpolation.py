#chromatica\gradients\gradient1dv2\helpers\interpolation.py
"""
Interpolation utilities for 1D gradient segments.
"""

from typing import List, Optional
import numpy as np

from ....types.color_types import ColorSpaces, is_hue_space
from boundednumbers import BoundType
from ....v2core import multival1d_lerp


def interpolate_transformed_non_hue(
    starts: np.ndarray,
    ends: np.ndarray,
    per_channel_coords: List[np.ndarray],
    bound_types: List[BoundType] | BoundType,
) -> np.ndarray:
    """
    Interpolate non-hue channels with transforms applied.
    
    Args:
        starts: Start values for each channel
        ends: End values for each channel
        per_channel_coords: Per-channel interpolation parameters (already transformed)
        bound_types: Bound types for each channel
        
    Returns:
        Interpolated values, shape (N, num_channels)
    """
    return multival1d_lerp(
        starts=[starts],
        ends=[ends],
        coeffs=per_channel_coords,
        bound_types=bound_types,
    )


def interpolate_transformed_hue_space(
    start: np.ndarray,
    end: np.ndarray,
    per_channel_coords: List[np.ndarray],
    hue_direction: str,
    bound_types: List[BoundType] | BoundType,
) -> np.ndarray:
    """
    Interpolate colors in a hue-based space (HSV/HSL).
    
    Args:
        start: Start color in hue space
        end: End color in hue space
        per_channel_coords: Per-channel interpolation parameters (already transformed)
        hue_direction: Direction for hue interpolation
        bound_types: Bound types for each channel
        
    Returns:
        Interpolated colors, shape (N, num_channels)
    """
    from ....utils.interpolate_hue import interpolate_hue
    
    hue = interpolate_hue(start[0], end[0], per_channel_coords[0], hue_direction)
    
    rest = interpolate_transformed_non_hue(
        starts=start[1:], ends=end[1:], per_channel_coords=per_channel_coords[1:], bound_types=bound_types
    )
    return np.column_stack((hue, rest))


def transform_1dchannels(
    per_channel_coords: List[np.ndarray],
    per_channel_transforms: Optional[dict],
    indices: range,
) -> List[np.ndarray]:
    """
    Apply per-channel transforms to non-hue channels.
    
    Args:
        per_channel_coords: List of interpolation parameters for each channel
        per_channel_transforms: Dictionary mapping channel index to transform function
        indices: Range of channel indices to process
        
    Returns:
        List of transformed interpolation parameters
    """
    if per_channel_transforms:
        return [
            transform(u) if (transform := per_channel_transforms.get(i)) else u
            for u, i in zip(per_channel_coords, indices)
        ]
    else:
        return per_channel_coords



__all__ = [
    'interpolate_transformed_non_hue',
    'interpolate_transformed_hue_space',
    'transform_1dchannels',

]
