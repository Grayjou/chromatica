from __future__ import annotations

import numpy as np
from numpy import ndarray as NDArray
from typing import Callable, Optional, Tuple, Union, Dict, Iterable, List

from ...color_arr import Color1DArr
from ...colors.color_base import ColorBase
from ...types.format_type import FormatType
from .color_utils import convert_color, get_color_class
from ...utils.interpolate_hue import interpolate_hue
from ...utils.list_mismatch import handle_list_size_mismatch
from ...utils.multiple_envelope import global_envelope_multiple_interp
from ...types.transform_types import (
    UnitTransform,
    PerChannelTransform,
)


class Gradient1D(Color1DArr):
    """
    Represents a 1D gradient of colors with advanced interpolation.

    Extends Color1DArr with gradient-specific creation methods including:
    - Hue direction control for HSV/HSL (cw/ccw/shortest)
    - Custom interpolation transforms
    - Per-channel transforms
    - Multiple color interpolation
    - Automatic color space handling
    """

    @classmethod
    def from_colors(
        cls,
        color1: Union[ColorBase, Tuple, int],
        color2: Union[ColorBase, Tuple, int],
        steps: int,
        color_space: str = "rgb",
        format_type: FormatType = FormatType.FLOAT,
        unit_transform: Optional[UnitTransform] = None,
        hue_direction: Optional[str] = None,
        per_channel_transforms: Optional[PerChannelTransform] = None,
    ) -> "Gradient1D":
        """
        Create a 1D gradient from two colors with optional transforms.

        Args:
            color1: First color (ColorBase instance or tuple/int)
            color2: Second color (ColorBase instance or tuple/int)
            steps: Number of steps in the gradient
            color_space: Target color space ('rgb', 'hsv', 'hsl', etc.)
            format_type: Format type (INT or FLOAT)
            unit_transform: Optional function to transform interpolation parameter
            hue_direction: Hue direction for HSV/HSL - 'cw' (clockwise), 
                          'ccw' (counter-clockwise), 'shortest', or None
            per_channel_transforms: Channel-specific easing functions

        Returns:
            Gradient1D instance with interpolated colors
        """
        color_space = color_space.lower()
        is_hue_space = color_space in ("hsv", "hsl", "hsva", "hsla")
        
        color_class = get_color_class(color_space, format_type)

        # Convert colors to target color space
        c1 = convert_color(color1, color_space, format_type)
        c2 = convert_color(color2, color_space, format_type)

        start = np.array(c1.value, dtype=float)
        end = np.array(c2.value, dtype=float)
        num_channels = start.shape[0]

        # Create normalized interpolation parameter
        u = np.linspace(0.0, 1.0, steps, dtype=float)
        if unit_transform is not None:
            u = unit_transform(u)

        # Check if we need per-channel processing
        needs_per_channel_processing = (
            per_channel_transforms is not None or
            (is_hue_space and hue_direction is not None)
        )

        if needs_per_channel_processing:
            colors = cls._interpolate_with_transforms(
                start, end, u, 
                is_hue_space, hue_direction,
                per_channel_transforms, num_channels
            )
        else:
            # Fast path: simple linear interpolation
            if is_hue_space:
                # Handle hue channel separately for shortest path
                h0 = start[0] % 360.0
                h1 = end[0] % 360.0
                
                # Calculate shortest path for hue
                delta = h1 - h0
                if delta > 180.0:
                    h1 -= 360.0
                elif delta < -180.0:
                    h1 += 360.0
                
                dh = h1 - h0
                hues = (h0 + u * dh) % 360.0
                rest = start[1:] * (1 - u[:, None]) + end[1:] * u[:, None]
                colors = np.concatenate([hues[:, None], rest], axis=1)
            else:
                colors = start * (1 - u[:, None]) + end * u[:, None]

        # Format and wrap hue values
        if format_type == FormatType.INT:
            colors = np.round(colors).astype(np.uint16)
            if is_hue_space:
                colors[:, 0] = colors[:, 0] % 360
        else:
            colors = colors.astype(np.float32)
            if is_hue_space:
                colors[:, 0] = colors[:, 0] % 360.0

        gradient_color = color_class(colors)
        return cls(gradient_color)

    @staticmethod
    def _interpolate_with_transforms(
        start: NDArray,
        end: NDArray,
        u: NDArray,
        is_hue_space: bool,
        hue_direction: Optional[str],
        per_channel_transforms: Optional[Dict[int, Callable[[NDArray], NDArray]]],
        num_channels: int
    ) -> NDArray:
        """
        Perform interpolation with all transforms.
        
        Args:
            start, end: Start and end color arrays
            u: Interpolation parameter array
            is_hue_space: Whether color space is hue-based
            hue_direction: Hue interpolation direction
            per_channel_transforms: Channel-specific easing functions
            num_channels: Number of color channels
            
        Returns:
            Interpolated color array with all transforms applied
        """
        colors = np.zeros((len(u), num_channels), dtype=float)
        
        # Determine hue channel index (usually 0 for HSV/HSL)
        hue_channel = 0 if is_hue_space else -1
        
        # Process each channel
        for ch in range(num_channels):
            u_ch = u.copy()
            
            # Apply per-channel transform if specified
            if per_channel_transforms and ch in per_channel_transforms:
                u_ch = per_channel_transforms[ch](u_ch)
            
            # Special handling for hue channel with direction control
            if ch == hue_channel and is_hue_space and hue_direction is not None:
                h0 = start[ch] % 360.0
                h1 = end[ch] % 360.0
                
                # Use interpolate_hue utility for directional control
                colors[:, ch] = interpolate_hue(h0, h1, u_ch, hue_direction)
            else:
                # Standard linear interpolation
                colors[:, ch] = start[ch] * (1 - u_ch) + end[ch] * u_ch
        
        return colors

    @classmethod
    def gradient_sequence(
        cls,
        colors: List[Union[ColorBase, Tuple, int]],
        steps: int,
        color_spaces: Optional[List[str]] = None,
        format_type: FormatType = FormatType.FLOAT,
        unit_transforms: Optional[List[UnitTransform]] = None,
        hue_directions: Optional[List[str]] = None,
        per_channel_transforms: Optional[List[PerChannelTransform]] = None,
        global_unit_transform: Optional[UnitTransform] = None,
    ) -> "Gradient1D":
        """
        Create a gradient from a sequence of colors with segment-specific transforms.

        Args:
            colors: List of colors to interpolate between
            steps: Total number of steps in the gradient
            color_spaces: Optional list of color spaces for each segment
            format_type: Format type (INT or FLOAT)
            unit_transforms: Optional list of unit transforms for each segment
            hue_directions: Optional list of hue directions for each segment
            per_channel_transforms: Optional list of per-channel transforms for each segment
            global_unit_transform: Optional global unit transform applied to entire gradient

        Returns:
            Gradient1D instance with interpolated colors
        """
        if len(colors) < 2:
            raise ValueError("At least 2 colors are required for gradient_sequence")
        
        # Handle optional lists with defaults
        num_segments = len(colors) - 1
        
        if color_spaces is None:
            color_spaces = ["rgb"] * num_segments
        color_spaces = handle_list_size_mismatch(color_spaces, num_segments)
        
        if hue_directions is None:
            hue_directions = [None] * num_segments
        hue_directions = handle_list_size_mismatch(hue_directions, num_segments)
        
        if unit_transforms is None:
            unit_transforms = [None] * num_segments
        unit_transforms = handle_list_size_mismatch(unit_transforms, num_segments)
        
        if per_channel_transforms is None:
            per_channel_transforms = [None] * num_segments
        per_channel_transforms = handle_list_size_mismatch(per_channel_transforms, num_segments)
        
        # Create global interpolation parameter
        u_global = np.linspace(0.0, 1.0, steps, dtype=float)
        if global_unit_transform is not None:
            u_global = global_unit_transform(u_global)
        
        # Scale to segment index space
        u_segments = u_global * num_segments
        
        # Convert all colors to arrays in their respective color spaces
        color_arrays = []
        for i in range(len(colors)):
            # Determine which color space to use (use the segment's color space)
            seg_idx = min(i, num_segments - 1)
            color_space = color_spaces[seg_idx].lower()
            
            color_obj = convert_color(colors[i], color_space, format_type)
            color_arrays.append(np.array(color_obj.value, dtype=float))
        
        # Determine output color space (use first segment's color space)
        output_color_space = color_spaces[0].lower()
        color_class = get_color_class(output_color_space, format_type)
        is_hue_space = output_color_space in ("hsv", "hsl", "hsva", "hsla")
        
        # Interpolate using global envelope function
        colors_result = global_envelope_multiple_interp(
            values=color_arrays,
            t=u_segments,
            unit_envelopes=unit_transforms,
            cyclic=False,
            direction=None,
        )
        
        # Format and wrap hue values
        if format_type == FormatType.INT:
            colors_result = np.round(colors_result).astype(np.uint16)
            if is_hue_space:
                colors_result[:, 0] = colors_result[:, 0] % 360
        else:
            colors_result = colors_result.astype(np.float32)
            if is_hue_space:
                colors_result[:, 0] = colors_result[:, 0] % 360.0
        
        gradient_color = color_class(colors_result)
        return cls(gradient_color)