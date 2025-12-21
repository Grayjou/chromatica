# ===================== NEW FILE: gradient_segment_builder.py =====================
"""
Module for building gradient segments from scaled u parameters.
"""

import numpy as np
from typing import List, Tuple
from .segment import get_transformed_segment


class GradientSegmentBuilder:
    """Builds gradient segments from scaled u parameters."""
    
    @staticmethod
    def build_segments_from_scaled_u(
        u_scaled: np.ndarray,
        num_segments: int,
        colors: List,
        input_color_spaces: List[str],
        color_spaces: List[str],
        format_type: str,
        hue_directions: List[str],
        per_channel_transforms: List,
        bound_type: str,
        conversion_func
    ) -> List:
        """
        Build gradient segments from scaled u array.
        
        Args:
            u_scaled: Scaled u parameters for the entire gradient
            num_segments: Number of segments to build
            colors: List of color points
            input_color_spaces: Color spaces of input colors
            color_spaces: Interpolation color spaces for each segment
            format_type: Output format type
            hue_directions: Hue direction for each segment
            per_channel_transforms: Per-channel transforms for each segment
            bound_type: Bound type for interpolation
            conversion_func: Function to convert between color spaces
            
        Returns:
            List of gradient segments
        """
        from .segment import get_segments_from_scaled_u
        
        index_local_us = get_segments_from_scaled_u(u_scaled, num_segments)
        
        # Build first segment
        first_segment = get_transformed_segment(
            already_converted_start_color=conversion_func(
                colors[0], input_color_spaces[0], format_type, color_spaces[0]
            ).value,
            already_converted_end_color=conversion_func(
                colors[1], input_color_spaces[1], format_type, color_spaces[0]
            ).value,
            local_us=[index_local_us[0][1]],
            color_space=color_spaces[0],
            hue_direction=hue_directions[0],
            per_channel_transforms=per_channel_transforms[0],
            bound_types=bound_type,
        )
        
        segments = [first_segment]
        previous_segment = first_segment
        
        # Build remaining segments
        for seg_idx in range(1, num_segments):
            new_segment = get_transformed_segment(
                already_converted_start_color=previous_segment.end_as_color_space(
                    color_spaces[seg_idx]
                ),
                already_converted_end_color=conversion_func(
                    colors[seg_idx + 1],
                    input_color_spaces[seg_idx + 1],
                    format_type,
                    color_spaces[seg_idx],
                ).value,
                local_us=[index_local_us[seg_idx][1]],
                color_space=color_spaces[seg_idx],
                hue_direction=hue_directions[seg_idx],
                per_channel_transforms=per_channel_transforms[seg_idx],
                bound_types=bound_type,
            )
            segments.append(new_segment)
            previous_segment = new_segment
            
        return segments