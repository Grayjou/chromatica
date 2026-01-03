#Chromatica\chromatica\gradients\gradient_1dv2.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union, List
from ...colors import unified_tuple_to_class
from ...colors.color_base import ColorBase
from ...types.format_type import FormatType
from ...types.transform_types import PerChannelTransform, UnitTransform
from ...types.array_types import ndarray_1d
from ...types.color_types import ColorMode, HueDirection, is_hue_space
from ...utils.dimension import most_common_element
from unitfield import flat_1d_upbm
from boundednumbers import BoundType
from .interpolator import _Gradient1DInterpolator, SequenceMethod

 

# ===================== Gradient1D Class =====================

class Gradient1D(_Gradient1DInterpolator):
    """
    Represents a 1D gradient of colors with advanced interpolation.

    Supports:
    - Hue direction control for HSV/HSL (cw/ccw/shortest)
    - Per-channel transforms
    - Multi-segment gradients with varying color spaces
    - Custom segment lengths and endpoint handling
    """

    @classmethod
    def from_colors(
        cls,
        left_color: Union[ColorBase, Tuple, List, ndarray_1d],
        right_color: Union[ColorBase, Tuple, List, ndarray_1d],
        steps: int,
        color_mode: ColorMode = "rgb",
        format_type: FormatType = FormatType.INT,
        hue_direction: HueDirection = "shortest",
        per_channel_transforms: Optional[PerChannelTransform] = None,
        bound_type: BoundType = BoundType.CLAMP,
        unit_transform: Optional[UnitTransform] = None,
    ) -> Gradient1D:
        """Create a Gradient1D from two colors with specified steps and options."""
        color_class = unified_tuple_to_class[(color_mode, format_type)]
        float_color_class = unified_tuple_to_class[(color_mode, FormatType.FLOAT)]
        
        left = float_color_class(color_class(left_color))
        right = float_color_class(color_class(right_color))
        
        color_array = cls._interpolate(
            start=left.value,
            end=right.value,
            u= unit_transform(flat_1d_upbm(steps)) if unit_transform else flat_1d_upbm(steps),
            is_hue=is_hue_space(color_mode),
            hue_direction=hue_direction,
            per_channel_transforms=per_channel_transforms,
            bound_type=bound_type,
        )
        
        return cls(color_class(float_color_class(color_array)))

    @classmethod
    def _gradient_sequence_no_global_transform(
        cls,
        colors: List[Union[ColorBase, Tuple, List, ndarray_1d]],
        total_steps: Optional[int],
        input_color_modes: List[ColorMode],
        color_modes: List[ColorMode],
        format_type: FormatType,
        hue_directions: List[HueDirection],
        per_channel_transforms: List[Optional[PerChannelTransform]],
        bound_type: BoundType,
        output_color_mode: Optional[ColorMode],
        segment_lengths: Optional[List[int]],
        offset: int = 1,):
        if len(colors) < 2:
            raise ValueError("At least 2 colors are required for gradient_sequence")

        num_segments = len(colors) - 1

        input_color_modes, color_modes, hue_directions, per_channel_transforms = cls._normalize_gradient_sequence_settings(
            colors,
            color_modes,
            input_color_modes,
            hue_directions,
            per_channel_transforms,
            num_segments,
        )
        per_channel_coords = cls._construct_per_channel_coords_no_transform(
            total_steps=total_steps,
            segment_lengths=segment_lengths,
            num_segments=num_segments,
            offset=offset,)

        # Run the interpolation loop
        # Prepare the first two colors

        first_color_converted = cls._convert_to_space_float(
            colors[0], input_color_modes[0], format_type, color_modes[0]
        )

        second_color_converted = cls._convert_to_space_float(
            colors[1], input_color_modes[1], format_type, color_modes[0]
        )
        current_color_mode = color_modes[0]

        first_gradient = cls._interpolate(
            start=first_color_converted.value,
            end=second_color_converted.value,
            u=per_channel_coords[0],
            is_hue=is_hue_space(current_color_mode),
            hue_direction=hue_directions[0],
            per_channel_transforms=per_channel_transforms[0],
            bound_type=bound_type,
        )

        if output_color_mode is None:
            output_color_mode = most_common_element(color_modes)

        if current_color_mode != output_color_mode:
            first_gradient = cls._convert_to_space_float(
                first_gradient,
                current_color_mode,
                FormatType.FLOAT,
                output_color_mode,
            ).value

        for seg_idx in range(num_segments-1):
            # Prepare next segment's colors

            float_color_left, float_color_right, current_color_mode = cls._prepare_next_segment_colors(
                seg_idx,
                colors,
                current_color_mode,
                second_color_converted,
                input_color_modes,
                color_modes,
            )

            next_gradient = cls._interpolate(
                start=float_color_left.value,
                end=float_color_right.value,
                u=per_channel_coords[seg_idx+1],
                is_hue=is_hue_space(current_color_mode),
                hue_direction=hue_directions[seg_idx],
                per_channel_transforms=per_channel_transforms[seg_idx],
                bound_type=bound_type,
            )

            # Convert to output space if needed
            if current_color_mode != output_color_mode:
                next_gradient = cls._convert_to_space_float(
                    next_gradient,
                    current_color_mode,
                    FormatType.FLOAT,
                    output_color_mode,
                ).value
            first_gradient = np.concatenate((first_gradient, next_gradient))

        output_color_class = unified_tuple_to_class[(output_color_mode, format_type)]
        float_output_class = unified_tuple_to_class[(output_color_mode, FormatType.FLOAT)]
        return output_color_class(float_output_class(first_gradient))


        
    @classmethod
    def gradient_sequence(
        cls,
        colors: List[Union[ColorBase, Tuple, List, ndarray_1d]],
        total_steps: Optional[int] = None,
        input_color_modes: Optional[Union[ColorMode, List[ColorMode]]] = None,
        color_modes: Optional[Union[ColorMode, List[ColorMode]]] = None,
        format_type: FormatType = FormatType.INT,
        hue_directions: Optional[List[HueDirection]] = None,
        per_channel_transforms: Optional[List[PerChannelTransform]] = None,
        global_unit_transform: Optional[UnitTransform] = None,
        bound_type: BoundType = BoundType.CLAMP,
        output_color_mode: Optional[ColorMode] = None,
        segment_lengths: Optional[List[int]] = None,
        offset: int = 1,
        *,
        method: SequenceMethod = SequenceMethod.MASK,
    ) -> Gradient1D:
        """
        Create a gradient from a sequence of colors with segment-specific options.

        Args:
            colors: List of colors to interpolate between (minimum 2).
            total_steps: Total steps in gradient (used if segment_lengths not provided).
            input_color_modes: Color space(s) of input colors.
            color_modes: Interpolation color space for each segment.
            format_type: Output format type (INT or FLOAT).
            hue_directions: Hue direction per segment ('cw', 'ccw', 'shortest').
            per_channel_transforms: Per-channel transforms for each segment.
            global_unit_transform: Transform applied to global interpolation parameter.
            bound_type: How to handle out-of-bound values.
            output_color_mode: Final output color space (defaults to most common).
            segment_lengths: Explicit length for each segment.
            offset: Points between segments (0=merge, 1=adjacent, >1=gap).

        Returns:
            Gradient1D instance with interpolated colors.
        """
        if len(colors) < 2:
            raise ValueError("At least 2 colors are required for gradient_sequence")
        
        if global_unit_transform is None:
            return cls._gradient_sequence_no_global_transform(
                colors, total_steps, input_color_modes or [], color_modes or [],
                format_type, hue_directions or [], per_channel_transforms or [],
                bound_type, output_color_mode, segment_lengths, offset
            )
        
        num_segments = len(colors) - 1
        
        input_color_modes, color_modes, hue_directions, per_channel_transforms = cls._normalize_gradient_sequence_settings(
            colors, color_modes, input_color_modes, hue_directions, 
            per_channel_transforms, num_segments
        )
        
        u_scaled, actual_total_steps = cls._construct_scaled_u_and_steps(
            num_segments=num_segments,
            total_steps=total_steps,
            segment_lengths=segment_lengths,
            offset=offset,
            global_unit_transform=global_unit_transform,
        )
        result = cls._interpolate_all_segments_scaled_u(
            colors=colors, u_scaled=u_scaled, num_segments=num_segments,
            total_steps=actual_total_steps, input_color_modes=input_color_modes,
            color_modes=color_modes, output_color_mode=output_color_mode,
            format_type=format_type, hue_directions=hue_directions,
            per_channel_transforms=per_channel_transforms, bound_type=bound_type,
            method=method
        )
        if output_color_mode is None:
            output_color_mode = most_common_element(color_modes)
        
        output_color_class = unified_tuple_to_class[(output_color_mode, format_type)]
        float_output_class = unified_tuple_to_class[(output_color_mode, FormatType.FLOAT)]
        
        return cls(output_color_class(float_output_class(result)))


