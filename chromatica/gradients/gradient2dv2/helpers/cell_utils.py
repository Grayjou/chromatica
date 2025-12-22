"""
Utilities for gradient cell operations.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from enum import Enum

from ....types.color_types import ColorSpace
from ....v2core.core import HueMode
from boundednumbers import BoundType


class CellMode(Enum):
    """Modes for defining a gradient cell."""
    CORNERS = 1
    LINES = 2
    CORNERS_DUAL = 3

def apply_per_channel_transforms_2d(
    coords: np.ndarray,
    per_channel_transforms: Optional[dict] = None,
    num_channels: int = 3,
) -> List[np.ndarray]:
    """
    Apply per-channel transforms to 2D coordinates.
    
    This function takes a single 2D coordinate grid and applies different
    transformations to each channel, producing per-channel transformed coordinates.
    
    Args:
        coords: Base coordinate grid, shape (H, W, 2) with u_x and u_y
        per_channel_transforms: Dictionary mapping channel index to transform function.
                               Each function should take coords and return transformed coords.
        num_channels: Number of color channels
        
    Returns:
        List of transformed coordinate arrays, one per channel, each shape (H, W, 2)
        
    Example:
        >>> # Define transforms
        >>> transforms = {
        ...     0: lambda c: np.stack([c[..., 0]**2, c[..., 1]], axis=-1),  # Square x for channel 0
        ...     1: lambda c: np.stack([c[..., 0], c[..., 1]**0.5], axis=-1),  # Sqrt y for channel 1
        ... }
        >>> # Apply to coordinates
        >>> coords = upbm_2d(100, 100)
        >>> transformed = apply_per_channel_transforms_2d(coords, transforms, 3)
        >>> # Channel 2 uses untransformed coords
    """
    if per_channel_transforms is None:
        # No transforms, use same coords for all channels
        return [coords] * num_channels
    
    transformed_coords = []
    for ch in range(num_channels):
        if ch in per_channel_transforms:
            transform = per_channel_transforms[ch]
            transformed = transform(coords)
            transformed_coords.append(transformed)
        else:
            # No transform for this channel, use original coords
            transformed_coords.append(coords)
    
    return transformed_coords




__all__ = [
    'CellMode',
    'apply_per_channel_transforms_2d',
    'separate_hue_and_non_hue_transforms',
]
