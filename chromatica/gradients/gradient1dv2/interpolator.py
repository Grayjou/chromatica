# ===================== MODIFIED: interpolator.py =====================
from __future__ import annotations
import numpy as np
from numpy import ndarray as NDArray
from typing import Optional, Tuple, Union, List, Callable
from ...types.format_type import FormatType
from ...utils.interpolate_hue import interpolate_hue
from ...types.transform_types import PerChannelTransform
from ...types.color_types import ColorMode, HueDirection, is_hue_space
from ...utils.dimension import most_common_element
from ...v2core import multival1d_lerp
from ...colors import ColorBase 
from boundednumbers import BoundType
from .unit_builder import _Gradient1DUnitBuilder
from enum import Enum
from .gradient_segment_builder import GradientSegmentBuilder
from ...utils.color_utils import convert_to_space_float as _convert_to_space_float


class SequenceMethod(Enum):
    MASK = "mask"
    SEGMENT = "segment"


def get_transform(
    transforms: Optional[PerChannelTransform],
    index: int,
) -> Optional[Callable]:
    """Safely retrieve a transform for a given channel index."""
    return transforms.get(index) if transforms else None


class _Gradient1DInterpolator(_Gradient1DUnitBuilder):
    """Handles gradient interpolation logic with various color spaces."""
    
    @classmethod
    def _interpolate_all_segments_scaled_u(
        cls,
        colors: List,
        u_scaled: np.ndarray,
        num_segments: int,
        total_steps: int,
        input_color_modes: List[ColorMode],
        color_modes: List[ColorMode],
        output_color_mode: Optional[ColorMode],
        format_type: FormatType,
        hue_directions: List[HueDirection],
        per_channel_transforms: List[Optional[PerChannelTransform]],
        bound_type: BoundType,
        *,
        method: SequenceMethod = SequenceMethod.MASK,
    ) -> np.ndarray:
        """Interpolate all segments and combine into result array."""
        if output_color_mode is None:
            output_color_mode = most_common_element(color_modes)
        
        if method == SequenceMethod.MASK:
            return cls._interpolate_with_mask_method(
                colors, u_scaled, num_segments, total_steps,
                input_color_modes, color_modes, output_color_mode,
                format_type, hue_directions, per_channel_transforms, bound_type
            )
        else:
            return cls._interpolate_with_segment_method(
                colors, u_scaled, num_segments, total_steps,
                input_color_modes, color_modes, output_color_mode,
                format_type, hue_directions, per_channel_transforms, bound_type
            )
    
    @classmethod
    def _interpolate_with_mask_method(
        cls,
        colors: List,
        u_scaled: np.ndarray,
        num_segments: int,
        total_steps: int,
        input_color_modes: List[ColorMode],
        color_modes: List[ColorMode],
        output_color_mode: ColorMode,
        format_type: FormatType,
        hue_directions: List[HueDirection],
        per_channel_transforms: List[Optional[PerChannelTransform]],
        bound_type: BoundType,
    ) -> np.ndarray:
        """Interpolate using mask method."""
        first_color_converted = _convert_to_space_float(
            colors[0], input_color_modes[0], format_type, output_color_mode
        )
        
        current_color_mode = color_modes[0]
        float_color_left = _convert_to_space_float(
            colors[0], input_color_modes[0], format_type, current_color_mode
        )
        float_color_right = _convert_to_space_float(
            colors[1], input_color_modes[1], format_type, current_color_mode
        )
        
        num_channels = len(float_color_left.value)
        result = np.full((total_steps, num_channels), first_color_converted.value, dtype=float)
        
        for seg_idx in range(num_segments):
            mask = (seg_idx < u_scaled) & (u_scaled <= seg_idx + 1)
            
            if not np.any(mask):
                continue
            
            this_chunk = cls._interpolate(
                start=float_color_left.value,
                end=float_color_right.value,
                u=u_scaled[mask] - seg_idx,
                is_hue=is_hue_space(current_color_mode),
                hue_direction=hue_directions[seg_idx],
                per_channel_transforms=per_channel_transforms[seg_idx],
                bound_type=bound_type,
            )
            
            if current_color_mode != output_color_mode:
                this_chunk = _convert_to_space_float(
                    this_chunk, current_color_mode, FormatType.FLOAT, output_color_mode
                ).value
            
            result[mask] = this_chunk
            
            if seg_idx < num_segments - 1:
                float_color_left, float_color_right, current_color_mode = cls._prepare_next_segment_colors(
                    seg_idx, colors, current_color_mode, float_color_right, 
                    input_color_modes, color_modes
                )
        
        return result
    
    @classmethod
    def _interpolate_with_segment_method(
        cls,
        colors: List,
        u_scaled: np.ndarray,
        num_segments: int,
        total_steps: int,
        input_color_modes: List[ColorMode],
        color_modes: List[ColorMode],
        output_color_mode: ColorMode,
        format_type: FormatType,
        hue_directions: List[HueDirection],
        per_channel_transforms: List[Optional[PerChannelTransform]],
        bound_type: BoundType,
    ) -> np.ndarray:
        """Interpolate using segment method."""
        segments = GradientSegmentBuilder.build_segments_from_scaled_u(
            u_scaled=u_scaled,
            num_segments=num_segments,
            colors=colors,
            input_color_modes=input_color_modes,
            color_modes=color_modes,
            format_type=format_type,
            hue_directions=hue_directions,
            per_channel_transforms=per_channel_transforms,
            bound_type=bound_type,
        )
        
        # Convert segments to output space and combine
        segment_values = [
            segment.convert_to_space(output_color_mode).get_value() 
            for segment in segments
        ]
        return np.vstack(segment_values)
    
    @classmethod
    def _prepare_next_segment_colors(
        cls,
        seg_idx: int,
        colors: List,
        current_color_mode: ColorMode,
        float_color_right: ColorBase,
        input_color_modes: List[ColorMode],
        color_modes: List[ColorMode],
    ) -> Tuple[ColorBase, ColorBase, ColorMode]:
        """Prepare left and right colors for the next segment."""
        next_seg_idx = seg_idx + 1
        next_color_mode = color_modes[next_seg_idx]
        
        float_color_left = float_color_right
        if next_color_mode != current_color_mode:
            float_color_left = float_color_left.convert(
                to_format=FormatType.FLOAT,
                to_space=next_color_mode,
            )
            current_color_mode = next_color_mode
        
        float_color_right = _convert_to_space_float(
            colors[next_seg_idx + 1],
            input_color_modes[next_seg_idx + 1],
            FormatType.FLOAT,
            current_color_mode,
        )
        
        return float_color_left, float_color_right, current_color_mode
    
    @classmethod
    def _interpolate(
        cls,
        start: NDArray,
        end: NDArray,
        u: NDArray,
        is_hue: bool,
        hue_direction: HueDirection,
        per_channel_transforms: Optional[PerChannelTransform],
        bound_type: BoundType,
    ) -> NDArray:
        """Unified interpolation dispatcher."""
        if is_hue:
            return cls._interpolate_hue_space(
                start, end, u, hue_direction, per_channel_transforms, bound_type
            )
        return cls._interpolate_channels(
            start, end, u, per_channel_transforms, range(len(start)), bound_type
        )
    
    @classmethod
    def _interpolate_hue_space(
        cls,
        start: NDArray,
        end: NDArray,
        u: NDArray,
        hue_direction: HueDirection,
        per_channel_transforms: Optional[PerChannelTransform],
        bound_type: BoundType,
    ) -> NDArray:
        """Interpolate colors in a hue-based space (HSV/HSL)."""
        hue_transform = cls._get_transform(per_channel_transforms, 0)
        hue_u = hue_transform(u) if hue_transform else u
        hue = interpolate_hue(start[0], end[0], hue_u, hue_direction)
        
        rest = cls._interpolate_channels(
            start[1:], end[1:], u, per_channel_transforms, range(1, len(start)), bound_type
        )
        
        return np.column_stack((hue, rest))
    
    @classmethod
    def _interpolate_channels(
        cls,
        starts: NDArray,
        ends: NDArray,
        u: NDArray,
        per_channel_transforms: Optional[PerChannelTransform],
        indices: range,
        bound_type: BoundType,
    ) -> NDArray:
        """Interpolate non-hue channels with optional transforms."""
        if per_channel_transforms:
            coeffs = [
                transform(u) if (transform := cls._get_transform(per_channel_transforms, i)) else u
                for i in indices
            ]
        else:
            coeffs = [u for _ in indices]
        
        return multival1d_lerp(
            starts=[starts],
            ends=[ends],
            coeffs=coeffs,
            bound_types=bound_type,
        )
    
    @staticmethod
    def _get_transform(
        transforms: Optional[PerChannelTransform],
        index: int,
    ) -> Optional[Callable]:
        """Safely retrieve a transform for a given channel index."""
        return transforms.get(index) if transforms else None