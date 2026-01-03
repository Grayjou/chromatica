"""
Utilities for gradient cell operations.
"""

from typing import List, Optional, Union, Tuple, Callable
import numpy as np
from enum import Enum

from ....types.color_types import ColorModes
from ....types.transform_types import TransformOutput
from ....v2core.core import HueDirection
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
    transform_output: TransformOutput = TransformOutput.LIST,
    omni_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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
        if transform_output == TransformOutput.LIST:
            return [coords.copy()] * num_channels
        elif transform_output == TransformOutput.ARRAY4D:
            return np.array([coords.copy()] * num_channels)
        elif transform_output == TransformOutput.ARRAY3D:
            return coords.copy()
    
    transformed_coords = []
    coords = _apply_omni_transform_2d(coords, omni_transform)
    any_transform = False
    for ch in range(num_channels):
        if ch in per_channel_transforms:
            transform = per_channel_transforms[ch]
            transformed = transform(coords.copy())
            transformed_coords.append(transformed)
            any_transform = True
        else:
            # No transform for this channel, use original coords
            transformed_coords.append(coords.copy())
    if transform_output == TransformOutput.ARRAY4D:
        return np.array(transformed_coords)
    elif transform_output == TransformOutput.ARRAY3D and not any_transform:
        return coords.copy()
    return transformed_coords

def _apply_omni_transform_2d(
    coords: np.ndarray,
    omni_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Apply an omni-channel transform to 2D coordinates if provided."""
    if omni_transform is not None:
        return omni_transform(coords)
    return coords

def apply_per_channel_transforms_2d(
    coords: Union[List[np.ndarray], np.ndarray],
    per_channel_transforms: Optional[dict] = None,
    num_channels: int = 3,
    transform_output: TransformOutput = TransformOutput.LIST,
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
    if per_channel_transforms is None:
        # No transforms, return original coords
        if transform_output == TransformOutput.LIST:
            return coords if isinstance(coords, list) else [coords.copy()] * num_channels
        elif transform_output == TransformOutput.ARRAY4D:
            if isinstance(coords, list):
                return np.array(coords)
            else:
                if coords.ndim == 3:
                    return np.array([coords.copy()] * num_channels)
                else:
                    return coords.copy()
        else: # TransformOutput.ARRAY3D
            return coords.copy()
    
    omni_transform = per_channel_transforms.get(-1)

    if isinstance(coords, list):
        # List of coordinate grids provided
        transformed_coords = []
        for ch in range(num_channels):
            if ch < len(coords):
                base_coords = coords[ch]
            else:
                base_coords = coords[0]  # Fallback to first if not enough provided
            base_coords = _apply_omni_transform_2d(base_coords, omni_transform)
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
            num_channels,
            transform_output=transform_output,
            omni_transform=omni_transform,
        )

__all__ = [
    'CellMode',
    'apply_per_channel_transforms_2d',
]
