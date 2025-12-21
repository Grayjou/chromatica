
from __future__ import annotations
from typing import List, Optional
import numpy as np
from ...types.color_types import ColorSpace, is_hue_space
from abc import ABC, abstractmethod
from ...utils.interpolate_hue import interpolate_hue
from ..v2_core import multival1d_lerp
from boundednumbers import BoundType
from ...conversions import np_convert


def get_segments_from_scaled_u(arr: np.ndarray, max_value: float) -> List[tuple[int, np.ndarray]]:
    if not arr.size:
        raise ValueError("Input array must not be empty.")
    current_low = int(arr[0])
    current_segment = [arr[0]]
    segments = []
    for value in arr[1:]:
        value_floor = int(np.floor(value))
        if value_floor == current_low or (value_floor == max_value and current_low == max_value - 1):
            current_segment.append(value)
        else:
            segments.append((current_low,np.array(current_segment)-current_low))
            current_segment = [value]
            current_low = value_floor
    segments.append((current_low,np.array(current_segment)-current_low))
    return segments


def interpolate_transformed_non_hue(
        starts: np.ndarray,
        ends: np.ndarray,
        local_us: List[np.ndarray],
        bound_types: List[BoundType] | BoundType,
) -> np.ndarray:
        return multival1d_lerp(
            starts=[starts],
            ends=[ends],
            coeffs=local_us,
            bound_types=bound_types,
        )

def interpolate_transformed_hue_space(
        start: np.ndarray,
        end: np.ndarray,
        local_us: List[np.ndarray],
        hue_direction: str,
        bound_types: List[BoundType] | BoundType,
    ) -> np.ndarray:
        """Interpolate colors in a hue-based space (HSV/HSL)."""
        hue = interpolate_hue(start[0], end[0], local_us[0], hue_direction)
        
        rest = interpolate_transformed_non_hue(
            starts=start[1:], ends=end[1:], local_us=local_us[1:], bound_types=bound_types
        )
        return np.column_stack((hue, rest))

def transform_non_hue_channels(
    local_us: List[np.ndarray],
    per_channel_transforms: Optional[dict],
    indices: range,
) -> list[np.ndarray]:
    """Apply per-channel transforms to non-hue channels."""
    if per_channel_transforms:
        return [
            transform(u) if (transform := per_channel_transforms.get(i)) else u
            for u, i in zip(local_us, indices)
        ]
    else:
        return local_us

def transform_hue_space(
    local_us: List[np.ndarray],
    per_channel_transforms: Optional[dict],
    num_channels: int = 3,
    ) -> np.ndarray:
    """Apply per-channel transform to hue channel."""
    if not per_channel_transforms:
        return np.column_stack(local_us)
    hue_transform = per_channel_transforms.get(0) if per_channel_transforms else None
    rest_transformed = transform_non_hue_channels(
        local_us[1:]
        , per_channel_transforms, range(1, num_channels)
    )
    if hue_transform:
        hue_transformed = hue_transform(local_us[0])
    else:
        hue_transformed = local_us[0]
    return np.column_stack((hue_transformed, *rest_transformed))


class SegmentBase(ABC):
    __slots__ = ('start_color', 'end_color', '_value')
    def get_value(self) -> np.ndarray:
        if self._value is None:
            self._value = self._render_value()
        return self._value
    @abstractmethod
    def _render_value(self) -> np.ndarray:
        pass
    @property
    def format_type(self) -> str:
        return "float"
    @abstractmethod
    def convert_to_space(self, color_space: ColorSpace) -> SegmentBase:
        pass
    @abstractmethod
    def start_as_color_space(self, color_space: ColorSpace) -> np.ndarray:
        pass
    @abstractmethod
    def end_as_color_space(self, color_space: ColorSpace) -> np.ndarray:
        pass


def get_transformed_segment(
    already_converted_start_color:np.ndarray, 
    already_converted_end_color:np.ndarray, 
    local_us:List[np.ndarray], 
    color_space:ColorSpace, 
    hue_direction: Optional[str]=None, 
    per_channel_transforms: Optional[dict]=None,
    bound_types: Optional[List[BoundType] | BoundType]=BoundType.CLAMP, *, value: Optional[np.ndarray]=None
) -> TransformedGradientSegment:

    if per_channel_transforms is not None:
        if is_hue_space(color_space):
            transformed_us = transform_hue_space(local_us, per_channel_transforms, num_channels=len(color_space))
        else:
            transformed_us = transform_non_hue_channels(
                local_us, per_channel_transforms, range(len(color_space)))
    else:
        transformed_us = local_us[0] #if isinstance(local_us, list) and len(local_us) == 1 else local_us
        return UniformGradientSegment(
            already_converted_start_color,
            already_converted_end_color,
            transformed_us,
            color_space,
            hue_direction,
            bound_types,
            value=value,
        )
    return TransformedGradientSegment(
        already_converted_start_color,
        already_converted_end_color,
        transformed_us,
        color_space,
        hue_direction,
        bound_types,
        value=value,
    )    
    
#Let's make a get_transformed_segment function

class TransformedGradientSegment(SegmentBase):
    __slots__ = ('start_color', 'end_color', 'local_us', 'color_space', '_value', 'hue_direction', 'bound_types')
    def __init__(self, 
                 already_converted_start_color:np.ndarray, 
                 already_converted_end_color:np.ndarray, 
                 local_us:np.ndarray, 
                 color_space:ColorSpace, 
                 hue_direction: Optional[str]=None, 
                 bound_types: Optional[List[BoundType] | BoundType]=BoundType.CLAMP, *, value: Optional[np.ndarray]=None):
        self.start_color = already_converted_start_color
        self.end_color = already_converted_end_color
        self.local_us = local_us
        self.color_space = color_space
        self._value = value
        self.hue_direction = hue_direction
        self.bound_types = bound_types
    def _render_value(self) -> np.ndarray:
        if is_hue_space(self.color_space):
            return interpolate_transformed_hue_space(
                self.start_color,
                self.end_color,
                self.local_us,
                self.hue_direction,
                self.bound_types,
            )
        else:
            return interpolate_transformed_non_hue(
                self.start_color,
                self.end_color,
                self.local_us,
                self.bound_types,
            )
    def convert_to_space(self, color_space: ColorSpace) -> TransformedGradientSegment:
        if self.color_space == color_space:
            return self
        converted_start = np_convert(self.start_color, self.color_space, color_space, fmt="float", output_type='float')
        converted_end = np_convert(self.end_color, self.color_space, color_space, fmt="float", output_type='float')
        converted_value = np_convert(self.get_value(), self.color_space, color_space, fmt="float", output_type='float') if self._value is not None else None
        return TransformedGradientSegment(
            converted_start,
            converted_end,
            self.local_us,
            color_space,
            self.hue_direction,
            self.bound_types,
            value=converted_value,
        )
    
    def start_as_color_space(self, color_space: ColorSpace) -> np.ndarray:
        if self.color_space == color_space:
            return self.start_color
        return np_convert(self.start_color, self.color_space, color_space, fmt="float", output_type='float')
    def end_as_color_space(self, color_space: ColorSpace) -> np.ndarray:
        if self.color_space == color_space:
            return self.end_color
        return np_convert(self.end_color, self.color_space, color_space, fmt="float", output_type='float')


    def build_next(self, new_end_color:np.ndarray, new_u_local:np.ndarray, new_end_color_space:ColorSpace, new_end_hue_direction:Optional[str]=None, new_bound_types:Optional[List[BoundType] | BoundType]=BoundType.CLAMP) -> TransformedGradientSegment:
        return TransformedGradientSegment(
            already_converted_start_color=self.end_as_color_space(new_end_color_space),
            already_converted_end_color=new_end_color,
            u_local=new_u_local,
            color_space=new_end_color_space,
            hue_direction=new_end_hue_direction or self.hue_direction,
            bound_types=new_bound_types or self.bound_types,
        )
    
class UniformGradientSegment(TransformedGradientSegment):
    def __init__(self, 
                 already_converted_start_color:np.ndarray, 
                 already_converted_end_color:np.ndarray, 
                 u_local:np.ndarray, color_space:ColorSpace, 
                 hue_direction: Optional[str]=None, 
                 bound_types: Optional[List[BoundType] | BoundType]=BoundType.CLAMP, *, value: Optional[np.ndarray]=None):
        local_us = [u_local]*len(color_space)
        super().__init__(
            already_converted_start_color,
            already_converted_end_color,
            local_us,
            color_space,
            hue_direction,
            bound_types,
            value=value,
        )


