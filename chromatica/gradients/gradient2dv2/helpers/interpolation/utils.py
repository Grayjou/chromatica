#chromatica\gradients\gradient2dv2\helpers\interpolation\utils.py
import numpy as np
from boundednumbers import BoundType
from typing import List, Optional
from .....v2core.core import HueDirection, _prepare_bound_types
from .....types.color_types import is_hue_space, ColorModes


def prepare_hue_and_rest_channels(data: np.ndarray, is_hue: bool = True):
    """
    Split data into hue and rest channels.
    
    Args:
        data: Input data array
        is_hue: Whether to treat first channel as hue
        
    Returns:
        Tuple of (hue_channel, rest_channels) or (None, all_channels)
    """
    if is_hue and data.ndim > 0:
        return data[..., 0], data[..., 1:]
    return None, data


def combine_hue_and_rest_channels(hue_data: Optional[np.ndarray], 
                                  rest_data: np.ndarray) -> np.ndarray:
    """
    Combine hue and rest channels back together.
    
    Args:
        hue_data: Hue channel data or None
        rest_data: Rest of the channels
        
    Returns:
        Combined array
    """
    if hue_data is not None:
        return np.concatenate([hue_data[..., np.newaxis], rest_data], axis=-1)
    return rest_data


