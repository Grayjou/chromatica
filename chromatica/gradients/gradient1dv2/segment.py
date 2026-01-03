
from __future__ import annotations
from typing import List, Optional, Union, Tuple
import numpy as np
from ...types.color_types import ColorModes, is_hue_space
from abc import abstractmethod
from ...v2core.subgradient import SubGradient
from boundednumbers import BoundType
from ...conversions import np_convert
from ...types.format_type import FormatType
from ...colors.color_base import ColorBase
from ...types.array_types import ndarray_1d
from .helpers import (
    interpolate_transformed_non_hue,
    interpolate_transformed_hue_space,
    transform_1dchannels,

    get_segments_from_scaled_u,
)


class SegmentBase(SubGradient):
    """Base class for 1D gradient segments, extending SubGradient."""
    
    __slots__ = ('start_color', 'end_color', '_value')
    
    def __init__(self):
        """Initialize with no cached value."""
        super().__init__()
    
    @abstractmethod
    def start_as_color_mode(self, color_mode: ColorModes) -> np.ndarray:
        """Get start color in specified color space."""
        pass
    
    @abstractmethod
    def end_as_color_mode(self, color_mode: ColorModes) -> np.ndarray:
        """Get end color in specified color space."""
        pass


def get_transformed_segment(
    already_converted_start_color: Optional[np.ndarray] = None,
    already_converted_end_color: Optional[np.ndarray] = None,
    per_channel_coords: List[np.ndarray] = None,
    color_mode: ColorModes = None,
    hue_direction: Optional[str] = None,
    per_channel_transforms: Optional[dict] = None,
    bound_types: Optional[List[BoundType] | BoundType] = BoundType.CLAMP,
    homogeneous_per_channel_coords: bool = False,
    *,
    value: Optional[np.ndarray] = None,
    # New parameters for conversion
    start_color: Optional[Union[ColorBase, Tuple, List, ndarray_1d]] = None,
    end_color: Optional[Union[ColorBase, Tuple, List, ndarray_1d]] = None,
    start_color_mode: Optional[ColorModes] = None,
    end_color_mode: Optional[ColorModes] = None,
    format_type: Optional[FormatType] = None,
) -> TransformedGradientSegment:
    """
    Create a transformed gradient segment.
    
    Can be called in two ways:
    1. Legacy mode: Pass already_converted_start_color and already_converted_end_color
    2. New mode: Pass start_color, end_color, start_color_mode, end_color_mode, format_type
    
    Args:
        already_converted_start_color: Pre-converted start color (legacy)
        already_converted_end_color: Pre-converted end color (legacy)
        per_channel_coords: Local u parameters for interpolation
        color_mode: Target color space for interpolation
        hue_direction: Hue direction for hue spaces
        per_channel_transforms: Per-channel transforms
        bound_types: Bound types for interpolation
        homogeneous_per_channel_coords: If True, per_channel_coords is a single array used for all channels.
                              If False, per_channel_coords is a list of arrays (one per channel).
                              When per_channel_transforms is None, this determines whether
                              to return UniformGradientSegment (True) or TransformedGradientSegment (False).
        value: Pre-computed values (optional)
        start_color: Unconverted start color (new mode)
        end_color: Unconverted end color (new mode)
        start_color_mode: Color space of start_color (new mode)
        end_color_mode: Color space of end_color (new mode)
        format_type: Format type of input colors (new mode)
        
    Returns:
        TransformedGradientSegment or UniformGradientSegment
    """
    # Handle conversion if new parameters are provided
    if start_color is not None and already_converted_start_color is None:
        from ...utils.color_utils import convert_to_space_float
        already_converted_start_color = convert_to_space_float(
            start_color, start_color_mode, format_type, color_mode
        ).value
    
    if end_color is not None and already_converted_end_color is None:
        from ...utils.color_utils import convert_to_space_float
        already_converted_end_color = convert_to_space_float(
            end_color, end_color_mode, format_type, color_mode
        ).value

    if per_channel_transforms is not None:
        transformed_us = transform_1dchannels(
            per_channel_coords, per_channel_transforms, range(len(color_mode)))
        return TransformedGradientSegment(
            already_converted_start_color,
            already_converted_end_color,
            transformed_us,
            color_mode,
            hue_direction,
            bound_types,
            value=value,
        )
    else:
        # When no per_channel_transforms, check if per_channel_coords is homogeneous
        # Auto-detect: if per_channel_coords is a list with single element, treat as homogeneous
        # unless explicitly specified otherwise
        if homogeneous_per_channel_coords or (isinstance(per_channel_coords, list) and len(per_channel_coords) == 1):
            # Extract single u array for uniform segment
            transformed_us = per_channel_coords[0] if isinstance(per_channel_coords, list) else per_channel_coords
            return UniformGradientSegment(
                already_converted_start_color,
                already_converted_end_color,
                transformed_us,
                color_mode,
                hue_direction,
                bound_types,
                value=value,
            )
        else:
            # per_channel_coords already contains per-channel arrays
            return TransformedGradientSegment(
                already_converted_start_color,
                already_converted_end_color,
                per_channel_coords,
                color_mode,
                hue_direction,
                bound_types,
                value=value,
            )    
    

class TransformedGradientSegment(SegmentBase):
    __slots__ = ('start_color', 'end_color', 'per_channel_coords', 'color_mode', '_value', 'hue_direction', 'bound_types')
    def __init__(self, 
                 already_converted_start_color:np.ndarray, 
                 already_converted_end_color:np.ndarray, 
                 per_channel_coords:np.ndarray, 
                 color_mode:ColorModes, 
                 hue_direction: Optional[str]=None, 
                 bound_types: Optional[List[BoundType] | BoundType]=BoundType.CLAMP, *, value: Optional[np.ndarray]=None):
        self.start_color = already_converted_start_color
        self.end_color = already_converted_end_color
        self.per_channel_coords = per_channel_coords
        self.color_mode = color_mode
        self._value = value
        self.hue_direction = hue_direction
        self.bound_types = bound_types
    def _render_value(self) -> np.ndarray:
        if is_hue_space(self.color_mode):
            return interpolate_transformed_hue_space(
                self.start_color,
                self.end_color,
                self.per_channel_coords,
                self.hue_direction,
                self.bound_types,
            )
        else:
            return interpolate_transformed_non_hue(
                self.start_color,
                self.end_color,
                self.per_channel_coords,
                self.bound_types,
            )
    def convert_to_space(self, color_mode: ColorModes) -> TransformedGradientSegment:
        if self.color_mode == color_mode:
            return self
        converted_start = np_convert(self.start_color, self.color_mode, color_mode, fmt="float", output_type='float')
        converted_end = np_convert(self.end_color, self.color_mode, color_mode, fmt="float", output_type='float')
        converted_value = np_convert(self.get_value(), self.color_mode, color_mode, fmt="float", output_type='float') if self._value is not None else None
        return TransformedGradientSegment(
            converted_start,
            converted_end,
            self.per_channel_coords,
            color_mode,
            self.hue_direction,
            self.bound_types,
            value=converted_value,
        )
    
    def start_as_color_mode(self, color_mode: ColorModes) -> np.ndarray:
        if self.color_mode == color_mode:
            return self.start_color
        return np_convert(self.start_color, self.color_mode, color_mode, fmt="float", output_type='float')
    def end_as_color_mode(self, color_mode: ColorModes) -> np.ndarray:
        if self.color_mode == color_mode:
            return self.end_color
        return np_convert(self.end_color, self.color_mode, color_mode, fmt="float", output_type='float')


    def build_next(self, new_end_color:np.ndarray, new_u_local:np.ndarray, new_end_color_mode:ColorModes, new_end_hue_direction:Optional[str]=None, new_bound_types:Optional[List[BoundType] | BoundType]=BoundType.CLAMP) -> TransformedGradientSegment:
        return TransformedGradientSegment(
            already_converted_start_color=self.end_as_color_mode(new_end_color_mode),
            already_converted_end_color=new_end_color,
            u_local=new_u_local,
            color_mode=new_end_color_mode,
            hue_direction=new_end_hue_direction or self.hue_direction,
            bound_types=new_bound_types or self.bound_types,
        )
    
class UniformGradientSegment(TransformedGradientSegment):
    def __init__(self, 
                 already_converted_start_color:np.ndarray, 
                 already_converted_end_color:np.ndarray, 
                 u_local:np.ndarray, color_mode:ColorModes, 
                 hue_direction: Optional[str]=None, 
                 bound_types: Optional[List[BoundType] | BoundType]=BoundType.CLAMP, *, value: Optional[np.ndarray]=None):
        per_channel_coords = [u_local]*len(color_mode)
        super().__init__(
            already_converted_start_color,
            already_converted_end_color,
            per_channel_coords,
            color_mode,
            hue_direction,
            bound_types,
            value=value,
        )


