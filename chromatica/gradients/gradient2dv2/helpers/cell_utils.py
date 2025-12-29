"""
Utilities for gradient cell operations.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from enum import Enum

from ....types.color_types import ColorSpaces
from ....v2core.core import HueMode
from boundednumbers import BoundType
 

class CellMode(Enum):
    """Modes for defining a gradient cell."""
    CORNERS = 1
    LINES = 2
    CORNERS_DUAL = 3

def apply_per_channel_transforms_2d_single(
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
            transformed = transform(coords.copy())
            transformed_coords.append(transformed)
        else:
            # No transform for this channel, use original coords
            transformed_coords.append(coords.copy())
    
    return transformed_coords


def apply_per_channel_transforms_2d(
    coords: Union[List[np.ndarray], np.ndarray],
    per_channel_transforms: Optional[dict] = None,
    num_channels: int = 3,
) -> List[np.ndarray]:

    """
    Apply per-channel transforms to 2D coordinates.
    
    This function handles both single coordinate grids and lists of coordinate grids,
    applying the specified transformations to each channel.
    
    Args:
        coords: Either a single base coordinate grid (H, W, 2) or a list of such grids.
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
        >>> # Apply to single coordinate grid
        >>> coords = upbm_2d(100, 100)
        >>> transformed = apply_per_channel_transforms_2d(coords, transforms, 3)
        >>> # Apply to list of coordinate grids
        >>> coords_list = [upbm_2d(100, 100) for _ in range(3)]
        >>> transformed_list = apply_per_channel_transforms_2d(coords_list, transforms, 3)
    """
    if isinstance(coords, list):
        # List of coordinate grids provided
        transformed_coords = []
        for ch in range(num_channels):
            if ch < len(coords):
                base_coords = coords[ch]
            else:
                base_coords = coords[0]  # Fallback to first if not enough provided
            if per_channel_transforms and ch in per_channel_transforms:
                transform = per_channel_transforms[ch]
                transformed = transform(base_coords)
                transformed_coords.append(transformed)
            else:
                transformed_coords.append(base_coords)
        return transformed_coords
    else:
        # Single coordinate grid provided
        return apply_per_channel_transforms_2d_single(
            coords,
            per_channel_transforms,
            num_channels
        )

__all__ = [
    'CellMode',
    'apply_per_channel_transforms_2d',
]
