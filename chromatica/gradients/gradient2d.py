from __future__ import annotations

import numpy as np
from numpy import ndarray as NDArray
from typing import Callable, Optional, Tuple, Union, Dict, Iterable, List

from ..color_arr import Color1DArr

from ..color_arr import Color2DArr
from ..colors.color_base import ColorBase
from ..types.format_type import FormatType
from .color_utils import convert_color, get_color_class
from ..utils.interpolate_hue import interpolate_hue
from ..utils.list_mismatch import handle_list_size_mismatch
from ..utils.multiple_envelope import global_envelope_multiple_interp
from ..types.transform_types import (
    UnitTransform,
    BiVariableSpaceTransform,
    BiVariableColorTransform,
    PerChannelTransform,
    get_bivar_space_transforms
)
from ..normalizers.rows_and_portions import normalize_portions, normalize_2d_rows, normalize_width_portions, normalize_height_portions
from .partitions import PerpendicularPartition, HuePartition
from .gradient1d import Gradient1D
from unitfield import Unit2DMappedEndomorphism


class Gradient2D(Color2DArr):
    """
    Represents a 2D gradient from four corner colors with bilinear interpolation.
    Extends Color2DArr with gradient-specific creation methods.
    """

    @classmethod
    def from_colors(
        cls,
        color_tl: Union[ColorBase, Tuple, int],
        color_tr: Union[ColorBase, Tuple, int],
        color_bl: Union[ColorBase, Tuple, int],
        color_br: Union[ColorBase, Tuple, int],
        width: int,
        height: int,
        color_space: str = "rgb",
        format_type: FormatType = FormatType.FLOAT,
        unit_transform_x: Optional[UnitTransform] = None,
        unit_transform_y: Optional[UnitTransform] = None,
        hue_direction_x: Optional[str] = None,
        hue_direction_y: Optional[str] = None,
        hue_partition_y: Optional[HuePartition] = None,  # NEW: Vertical hue partitions
        bivariable_space_transforms: Optional[BiVariableSpaceTransform] = None,
        bivariable_color_transforms: Optional[BiVariableColorTransform] = None,
        easing_x: Optional[PerChannelTransform] = None,
        easing_y: Optional[PerChannelTransform] = None,
    ) -> "Gradient2D":
        """
        Create a 2D gradient from four corner colors with optional transforms.

        Args:
            color_tl: Top-left color
            color_tr: Top-right color
            color_bl: Bottom-left color
            color_br: Bottom-right color
            width: Number of columns
            height: Number of rows
            color_space: Target color space ('rgb', 'hsv', 'hsl', etc.)
            format_type: Format type (INT or FLOAT)
            unit_transform_x: Optional transformation of x interpolation factors
            unit_transform_y: Optional transformation of y interpolation factors
            hue_direction_x: Hue interpolation direction along x-axis for hue-based 
                            color spaces ('cw', 'ccw', 'shortest', or None)
            hue_direction_y: Hue interpolation direction along y-axis for hue-based 
                            color spaces ('cw', 'ccw', 'shortest', or None)
            hue_partition_y: Optional HuePartition for multiple vertical hue directions.
                           If provided, overrides hue_direction_y.
            bivariable_space_transforms: Channel-specific (x, y) → (x', y') transforms
            bivariable_color_transforms: Channel-specific (x, y, color) → color' transforms
            easing_x: Channel-specific x-axis easing functions
            easing_y: Channel-specific y-axis easing functions

        Returns:
            Gradient2D instance with bilinearly interpolated colors
        """
        color_space = color_space.lower()
        is_hue_space = color_space in ("hsv", "hsl", "hsva", "hsla")
        
        color_class = get_color_class(color_space, format_type)

        # Convert corner colors to target color space
        corners = [
            convert_color(corner_color, color_space, format_type)
            for corner_color in [color_tl, color_tr, color_bl, color_br]
        ]
        
        tl, tr, bl, br = [np.array(c.value, dtype=float) for c in corners]
        num_channels = tl.shape[0]
        
        # Create normalized coordinate grids
        x_norm = np.linspace(0.0, 1.0, width, dtype=float)
        y_norm = np.linspace(0.0, 1.0, height, dtype=float)

        if unit_transform_x is not None:
            x_norm = unit_transform_x(x_norm)
        if unit_transform_y is not None:
            y_norm = unit_transform_y(y_norm)

        xx, yy = np.meshgrid(x_norm, y_norm)
        
        # Check if we need per-channel processing
        needs_per_channel_processing = (
            bivariable_space_transforms is not None or
            bivariable_color_transforms is not None or
            easing_x is not None or
            easing_y is not None or
            (is_hue_space and (
                hue_direction_x is not None or 
                hue_direction_y is not None or 
                hue_partition_y is not None  # NEW: Include partition check
            ))
        )
        
        if needs_per_channel_processing:
            colors = cls._interpolate_with_transforms(
                tl, tr, bl, br, xx, yy, x_norm, y_norm,
                is_hue_space, hue_direction_x, hue_direction_y,
                hue_partition_y,  # NEW: Pass hue partition
                bivariable_space_transforms, bivariable_color_transforms,
                easing_x, easing_y, num_channels
            )
        else:
            # Original bilinear interpolation (fast path)
            colors = (
                (1 - xx)[:, :, None] * (1 - yy)[:, :, None] * tl
                + xx[:, :, None] * (1 - yy)[:, :, None] * tr
                + (1 - xx)[:, :, None] * yy[:, :, None] * bl
                + xx[:, :, None] * yy[:, :, None] * br
            )
        
        # Format and wrap hue values
        if format_type == FormatType.INT:
            colors = np.round(colors).astype(np.uint16)
            if is_hue_space:
                colors[..., 0] = colors[..., 0] % 360
        else:
            colors = colors.astype(np.float32)
            if is_hue_space:
                colors[..., 0] = colors[..., 0] % 360.0

        gradient_color = color_class(colors)
        return cls(gradient_color)

    @staticmethod
    def _interpolate_with_transforms(
        tl: NDArray,
        tr: NDArray,
        bl: NDArray,
        br: NDArray,
        xx: NDArray,
        yy: NDArray,
        x_norm: NDArray,
        y_norm: NDArray,
        is_hue_space: bool,
        hue_direction_x: Optional[str],
        hue_direction_y: Optional[str],
        hue_partition_y: Optional[HuePartition],  # NEW: Vertical hue partition
        bivariable_space_transforms: Optional[Dict[int, Callable[[NDArray, NDArray], Tuple[NDArray, NDArray]]]],
        bivariable_color_transforms: Optional[Dict[int, Callable[[NDArray, NDArray, NDArray], NDArray]]],
        easing_x: Optional[Dict[int, Callable[[NDArray], NDArray]]],
        easing_y: Optional[Dict[int, Callable[[NDArray], NDArray]]],
        num_channels: int
    ) -> NDArray:
        """
        Perform bilinear interpolation with all transforms.
        
        Args:
            tl, tr, bl, br: Corner color arrays
            xx, yy: Meshgrid of interpolation factors
            x_norm, y_norm: Original normalized coordinate arrays
            is_hue_space: Whether color space is hue-based
            hue_direction_x: Hue direction along x-axis
            hue_direction_y: Hue direction along y-axis
            hue_partition_y: Optional HuePartition for vertical hue directions
            bivariable_space_transforms: Channel-specific space transforms
            bivariable_color_transforms: Channel-specific color transforms
            easing_x: Channel-specific x-axis easing
            easing_y: Channel-specific y-axis easing
            num_channels: Number of color channels
            
        Returns:
            Interpolated color array with all transforms applied
        """
        height, width = xx.shape
        result = np.zeros((height, width, num_channels), dtype=float)
        
        # Determine hue channel index (usually 0 for HSV/HSL)
        hue_channel = 0 if is_hue_space else -1
        
        # Pre-process corner hue values if needed
        if is_hue_space and hue_channel < num_channels:
            h_tl, h_tr, h_bl, h_br = (
                tl[hue_channel], tr[hue_channel], 
                bl[hue_channel], br[hue_channel]
            )
        
        # Process each channel
        for ch in range(num_channels):
            # Apply easing transforms if specified
            if easing_x and ch in easing_x:
                x_eased = easing_x[ch](xx.copy())
            else:
                x_eased = xx
            
            if easing_y and ch in easing_y:
                y_eased = easing_y[ch](yy.copy())
            else:
                y_eased = yy
            
            # Apply bivariable space transforms if specified
            if bivariable_space_transforms and ch in bivariable_space_transforms:
                x_transformed, y_transformed = bivariable_space_transforms[ch](x_eased, y_eased)
            else:
                x_transformed, y_transformed = x_eased, y_eased
            
            # Clamp transformed coordinates to avoid numerical issues
            x_transformed = np.clip(x_transformed, -1.0, 2.0)
            y_transformed = np.clip(y_transformed, -1.0, 2.0)
            
            # Special handling for hue channel with direction control
            if ch == hue_channel and is_hue_space:
                channel_result = Gradient2D._interpolate_hue_channel_with_transforms(
                    h_tl, h_tr, h_bl, h_br,
                    x_transformed, y_transformed,
                    hue_direction_x, hue_direction_y,
                    hue_partition_y  # NEW: Pass partition
                )
            else:
                # Standard bilinear interpolation with transformed coordinates
                channel_result = (
                    (1 - x_transformed) * (1 - y_transformed) * tl[ch]
                    + x_transformed * (1 - y_transformed) * tr[ch]
                    + (1 - x_transformed) * y_transformed * bl[ch]
                    + x_transformed * y_transformed * br[ch]
                )
            
            # Apply bivariable color transforms if specified
            if bivariable_color_transforms and ch in bivariable_color_transforms:
                channel_result = bivariable_color_transforms[ch](x_norm, y_norm, channel_result)
            
            result[..., ch] = channel_result
        
        return result

    @staticmethod
    def _interpolate_hue_channel_with_transforms(
        h_tl: float,
        h_tr: float,
        h_bl: float,
        h_br: float,
        x_transformed: NDArray,
        y_transformed: NDArray,
        hue_direction_x: Optional[str],
        hue_direction_y: Optional[str],
        hue_partition_y: Optional[HuePartition] = None  # NEW: Optional partition
    ) -> NDArray:
        """
        Interpolate hue channel with directional control and transformed coordinates.
        
        Uses a two-step bilinear approach:
        1. First interpolate horizontally along top and bottom edges
        2. Then interpolate vertically between the results
        
        If hue_partition_y is provided, it overrides hue_direction_y.
        """
        height, width = x_transformed.shape
        
        # Default to shortest path if direction not specified
        dir_x = hue_direction_x if hue_direction_x is not None else 'shortest'
        
        # Clamp x coordinates for hue interpolation
        x_clamped = np.clip(x_transformed, 0.0, 1.0)
        y_clamped = np.clip(y_transformed, 0.0, 1.0)
        
        # Step 1: Interpolate along top edge (horizontal)
        top_hues = np.zeros((height, width), dtype=float)
        for i in range(height):
            for j in range(width):
                x_val = x_clamped[i, j]
                top_hues[i, j] = interpolate_hue(h_tl, h_tr, x_val, dir_x)
        
        # Step 1: Interpolate along bottom edge (horizontal)
        bottom_hues = np.zeros((height, width), dtype=float)
        for i in range(height):
            for j in range(width):
                x_val = x_clamped[i, j]
                bottom_hues[i, j] = interpolate_hue(h_bl, h_br, x_val, dir_x)
        
        # Step 2: Interpolate vertically between top and bottom results
        final_hues = np.zeros((height, width), dtype=float)
        
        for i in range(height):
            # Get hue direction for this row
            y_val = y_clamped[i, 0]  # y is constant across a row
            
            # Use partition if provided, otherwise use constant direction
            if hue_partition_y is not None:
                dir_y = hue_partition_y.get_hue_direction(y_val)
            else:
                dir_y = hue_direction_y if hue_direction_y is not None else 'shortest'
            
            for j in range(width):
                # y_val is the same for entire row, but we use the clamped value
                final_hues[i, j] = interpolate_hue(
                    top_hues[i, j], 
                    bottom_hues[i, j], 
                    y_clamped[i, j], 
                    dir_y
                )
        
        return final_hues
    @classmethod
    def from_1d_arrays(
        cls,
        top_color_arr: ColorBase,
        bottom_color_arr: ColorBase,
        height: int,
        color_space: str = "rgb",
        format_type: FormatType = FormatType.FLOAT,
        unit_transform_y: Optional[UnitTransform] = None,
        hue_direction_y: Optional[str] = None,
        hue_partition_y: Optional[HuePartition] = None,
        bivariable_space_transforms: Optional[BiVariableSpaceTransform] = None,
        bivariable_color_transforms: Optional[BiVariableColorTransform] = None,
        easing_y: Optional[PerChannelTransform] = None,
    ) -> "Gradient2D":
        """
        Create a 2D gradient by interpolating between two 1D gradients vertically.
        
        Args:
            top_color_arr: 1D gradient for the top row (shape: [width, channels])
            bottom_color_arr: 1D gradient for the bottom row (shape: [width, channels])
            height: Number of rows in the output 2D gradient
            color_space: Target color space ('rgb', 'hsv', 'hsl', etc.)
            format_type: Format type (INT or FLOAT)
            unit_transform_y: Optional transformation of y interpolation factors
            hue_direction_y: Hue interpolation direction along y-axis for hue-based 
                            color spaces ('cw', 'ccw', 'shortest', or None)
            hue_partition_y: Optional HuePartition for multiple vertical hue directions.
                           If provided, overrides hue_direction_y.
            bivariable_space_transforms: Channel-specific (x, y) → (x', y') transforms
            bivariable_color_transforms: Channel-specific (x, y, color) → color' transforms
            easing_y: Channel-specific y-axis easing functions
            
        Returns:
            Gradient2D instance with vertically interpolated colors
            
        Raises:
            ValueError: If gradients have different widths or incompatible color spaces
        """
        # Validate that gradients have the same width
        top_width = top_color_arr.value.shape[0]
        bottom_width = bottom_color_arr.value.shape[0]
        
        if top_width != bottom_width:
            raise ValueError(
                f"Top gradient width ({top_width}) must equal bottom gradient width ({bottom_width})"
            )
        
        width = top_width
        
        # Get color classes to check compatibility
        top_color_class = type(top_color_arr.value)
        bottom_color_class = type(bottom_color_arr.value)


        # Convert gradients to target color space if needed
        if top_color_arr.mode != color_space or top_color_arr.format_type != format_type:
            converted_top_color = top_color_arr.convert(color_space, format_type)

            top_arr = converted_top_color.value

        else:
            top_arr = top_color_arr.value

            
        if bottom_color_arr.mode != color_space or bottom_color_arr.format_type != format_type:
            converted_bottom_color = bottom_color_arr.convert(color_space, format_type)
            bottom_arr = converted_bottom_color.value
        else:
            bottom_arr = bottom_color_arr.value
        
        # Ensure both arrays have the same shape
        if top_arr.shape != bottom_arr.shape:
            raise ValueError(
                f"Gradient arrays have different shapes: {top_arr.shape} vs {bottom_arr.shape}"
            )
        
        num_channels = top_arr.shape[1]
        is_hue_space = color_space.lower() in ("hsv", "hsl", "hsva", "hsla")

        color_class = get_color_class(color_space, format_type)
        
        # Create normalized y coordinate grid
        y_norm = np.linspace(0.0, 1.0, height, dtype=float)
        
        if unit_transform_y is not None:
            y_norm = unit_transform_y(y_norm)
        
        # Create meshgrid for x and y
        x_norm = np.linspace(0.0, 1.0, width, dtype=float)
        xx, yy = np.meshgrid(x_norm, y_norm)
        
        # Check if we need per-channel processing
        needs_per_channel_processing = (
            bivariable_space_transforms is not None or
            bivariable_color_transforms is not None or
            easing_y is not None or
            (is_hue_space and (
                hue_direction_y is not None or 
                hue_partition_y is not None
            ))
        )
        
        if needs_per_channel_processing:
            colors = cls._interpolate_1d_arrays_with_transforms(
                top_arr, bottom_arr, xx, yy, x_norm, y_norm,
                is_hue_space, hue_direction_y, hue_partition_y,
                bivariable_space_transforms, bivariable_color_transforms,
                easing_y, num_channels
            )
        else:
            # Fast path: simple vertical interpolation
            # Reshape for broadcasting: top/bottom_arr are (width, channels)
            # yy is (height, width), we want to broadcast to (height, width, channels)
            top_2d = top_arr[np.newaxis, :, :]  # (1, width, channels)
            bottom_2d = bottom_arr[np.newaxis, :, :]  # (1, width, channels)
            
            # yy[:, :, np.newaxis] expands to (height, width, 1) for broadcasting
            colors = (1 - yy[:, :, np.newaxis]) * top_2d + yy[:, :, np.newaxis] * bottom_2d
        
        # Format and wrap hue values
        if format_type == FormatType.INT:
            colors = np.round(colors).astype(np.uint16)
            if is_hue_space:
                colors[..., 0] = colors[..., 0] % 360
        else:
            colors = colors.astype(np.float32)
            if is_hue_space:
                colors[..., 0] = colors[..., 0] % 360.0
        
        gradient_color = color_class(colors)
        return cls(gradient_color)
    
    @staticmethod
    def _interpolate_1d_arrays_with_transforms(
        top_arr: NDArray,
        bottom_arr: NDArray,
        xx: NDArray,
        yy: NDArray,
        x_norm: NDArray,
        y_norm: NDArray,
        is_hue_space: bool,
        hue_direction_y: Optional[str],
        hue_partition_y: Optional[HuePartition],
        bivariable_space_transforms: Optional[BiVariableSpaceTransform],
        bivariable_color_transforms: Optional[BiVariableColorTransform],
        easing_y: Optional[PerChannelTransform],
        num_channels: int
    ) -> NDArray:
        """
        Perform vertical interpolation between two 1D gradients with all transforms.
        
        Args:
            top_arr: Top gradient array (width, channels)
            bottom_arr: Bottom gradient array (width, channels)
            xx, yy: Meshgrid of interpolation factors
            x_norm, y_norm: Original normalized coordinate arrays
            is_hue_space: Whether color space is hue-based
            hue_direction_y: Hue direction along y-axis
            hue_partition_y: Optional HuePartition for vertical hue directions
            bivariable_space_transforms: Channel-specific space transforms
            bivariable_color_transforms: Channel-specific color transforms
            easing_y: Channel-specific y-axis easing
            num_channels: Number of color channels
            
        Returns:
            Interpolated color array with all transforms applied
        """
        height, width = xx.shape
        result = np.zeros((height, width, num_channels), dtype=float)
        
        # Determine hue channel index
        hue_channel = 0 if is_hue_space else -1
        
        # Process each channel

        for ch in range(num_channels):
            # Get channel arrays
            top_channel = top_arr[:, ch]  # (width,)
            bottom_channel = bottom_arr[:, ch]  # (width,)
            
            # Apply easing transform to y if specified
            if easing_y and ch in easing_y:
                y_eased = easing_y[ch](yy.copy())
            else:
                y_eased = yy
            
            # Apply bivariable space transforms if specified
            if bivariable_space_transforms and ch in bivariable_space_transforms:
                x_transformed, y_transformed = bivariable_space_transforms[ch](xx, y_eased)
            else:
                x_transformed, y_transformed = xx, y_eased
            
            # Clamp transformed coordinates
            x_transformed = np.clip(x_transformed, -1.0, 2.0)
            y_transformed = np.clip(y_transformed, -1.0, 2.0)
            
            # Special handling for hue channel
            if ch == hue_channel and is_hue_space:

                channel_result = Gradient2D._interpolate_hue_vertical(
                    top_channel, bottom_channel,
                    x_transformed, y_transformed,
                    hue_direction_y, hue_partition_y
                )
            else:
                # Linear interpolation in y direction
                # We need to broadcast top_channel and bottom_channel to match y_transformed shape
                top_2d = np.tile(top_channel[np.newaxis, :], (height, 1))  # (height, width)
                bottom_2d = np.tile(bottom_channel[np.newaxis, :], (height, 1))  # (height, width)
                
                # Perform vertical interpolation
                channel_result = (1 - y_transformed) * top_2d + y_transformed * bottom_2d
            
            # Apply bivariable color transforms if specified
            if bivariable_color_transforms and ch in bivariable_color_transforms:
                channel_result = bivariable_color_transforms[ch](x_norm, y_norm, channel_result)
            
            result[..., ch] = channel_result
        
        return result
    
    @staticmethod
    def _interpolate_hue_vertical(
        top_hues: NDArray,
        bottom_hues: NDArray,
        x_transformed: NDArray,
        y_transformed: NDArray,
        hue_direction_y: Optional[str],
        hue_partition_y: Optional[HuePartition]
    ) -> NDArray:
        """
        Interpolate hue channel vertically between two hue arrays.
        
        Args:
            top_hues: Hue values for top row (width,)
            bottom_hues: Hue values for bottom row (width,)
            x_transformed: Transformed x coordinates (height, width)
            y_transformed: Transformed y coordinates (height, width)
            hue_direction_y: Hue direction for y-axis
            hue_partition_y: Optional partition for hue directions
            
        Returns:
            Interpolated hue values (height, width)
        """
        height, width = y_transformed.shape
        
        # Clamp y coordinates for hue interpolation
        y_clamped = np.clip(y_transformed, 0.0, 1.0)
        
        # Prepare result array
        result = np.zeros((height, width), dtype=float)
        
        # For each row
        for i in range(height):
            # Get hue direction for this row
            if hue_partition_y is not None:
                dir_y = hue_partition_y.get_hue_direction(y_clamped[i, 0])
            else:
                dir_y = hue_direction_y if hue_direction_y is not None else 'shortest'
            
            # For each column
            for j in range(width):
                y_val = y_clamped[i, j]
                result[i, j] = interpolate_hue(top_hues[j], bottom_hues[j], y_val, dir_y)
        
        return result

    # Alternative: simpler version without x transforms (if you only want vertical interpolation)
    @classmethod
    def from_1d_arrays_simple(
        cls,
        top_color_arr: Color1DArr,
        bottom_color_arr: Color1DArr,
        height: int,
        color_space: str = "rgb",
        format_type: FormatType = FormatType.FLOAT,
        unit_transform_y: Optional[UnitTransform] = None,
        hue_direction_y: Optional[str] = None,
        hue_partition_y: Optional[HuePartition] = None,
    ) -> "Gradient2D":
        """
        Create a 2D gradient by interpolating between two 1D gradients vertically.
        Simplified version without bivariable transforms.
        
        Args:
            top_color_arr: 1D gradient for the top row
            bottom_color_arr: 1D gradient for the bottom row
            height: Number of rows in the output
            color_space: Target color space
            format_type: Format type (INT or FLOAT)
            unit_transform_y: Optional transformation of y interpolation factors
            hue_direction_y: Hue interpolation direction along y-axis
            hue_partition_y: Optional HuePartition for multiple vertical hue directions
            
        Returns:
            Gradient2D instance
        """
        # Validate gradients
        top_color_arr = Color1DArr(top_color_arr)
        bottom_color_arr = Color1DArr(bottom_color_arr)
        top_width = top_color_arr._color.value.shape[0]
        bottom_width = bottom_color_arr._color.value.shape[0]
        
        if top_width != bottom_width:
            raise ValueError(
                f"Top gradient width ({top_width}) must equal bottom gradient width ({bottom_width})"
            )
        
        width = top_width
        
        # Convert to target color space
        if top_color_arr.mode != color_space or top_color_arr.format_type != format_type:
            top_arr = top_color_arr.convert(color_space, format_type).value
        else:
            top_arr = top_color_arr.value
            
        if bottom_color_arr.mode != color_space or bottom_color_arr.format_type != format_type:
            bottom_arr = bottom_color_arr.convert(color_space, format_type).value
        else:
            bottom_arr = bottom_color_arr.value

        # Create normalized y coordinates
        y_norm = np.linspace(0.0, 1.0, height, dtype=float)
        if unit_transform_y is not None:
            y_norm = unit_transform_y(y_norm)
        
        # Get color class and check if hue space
        color_class = get_color_class(color_space, format_type)
        is_hue_space = color_space.lower() in ("hsv", "hsl", "hsva", "hsla")
        
        # Prepare result array
        if is_hue_space and (hue_direction_y is not None or hue_partition_y is not None):
            # Special handling for hue with direction
            result = cls._interpolate_hue_vertical_simple(
                top_arr, bottom_arr, y_norm, hue_direction_y, hue_partition_y
            )
        else:
            # Simple vertical interpolation
            result = np.zeros((height, width, top_arr.shape[1]), dtype=float)
            for i, y in enumerate(y_norm):
                result[i] = (1 - y) * top_arr + y * bottom_arr
        
        # Format and wrap hue values
        if format_type == FormatType.INT:
            result = np.round(result).astype(np.uint16)
            if is_hue_space:
                result[..., 0] = result[..., 0] % 360
        else:
            result = result.astype(np.float32)
            if is_hue_space:
                result[..., 0] = result[..., 0] % 360.0
        
        gradient_color = color_class(result)
        return cls(gradient_color)
    
    @staticmethod
    def _interpolate_hue_vertical_simple(
        top_arr: NDArray,
        bottom_arr: NDArray,
        y_norm: NDArray,
        hue_direction_y: Optional[str],
        hue_partition_y: Optional[HuePartition]
    ) -> NDArray:
        """
        Simplified hue interpolation for vertical gradients.
        """
        height = len(y_norm)
        width, num_channels = top_arr.shape
        
        result = np.zeros((height, width, num_channels), dtype=float)
        
        # Copy non-hue channels
        if num_channels > 1:
            for i, y in enumerate(y_norm):
                result[i, :, 1:] = (1 - y) * top_arr[:, 1:] + y * bottom_arr[:, 1:]
        
        # Interpolate hue channel
        for i, y in enumerate(y_norm):
            # Get hue direction for this y value
            if hue_partition_y is not None:
                dir_y = hue_partition_y.get_hue_direction(y)
            else:
                dir_y = hue_direction_y if hue_direction_y is not None else 'shortest'
            
            # Interpolate hue for each column
            for j in range(width):
                result[i, j, 0] = interpolate_hue(
                    top_arr[j, 0], 
                    bottom_arr[j, 0], 
                    y, 
                    dir_y
                )
        
        return result
    @classmethod
    def heterogenous_from_colors(
        color_tl: List[Union[ColorBase, Tuple, int]],
        color_tr: List[Union[ColorBase, Tuple, int]],
        color_bl: List[Union[ColorBase, Tuple, int]],
        color_br: List[Union[ColorBase, Tuple, int]],
        width: int,
        height: int,
        color_space: str = "rgb",
        format_type: FormatType = FormatType.FLOAT,
        unit_transform_x: Optional[UnitTransform] = None,
        unit_transform_y: Optional[UnitTransform] = None,
        hue_direction_x: Optional[str] = None,
        hue_direction_y: Optional[str] = None,
        perpendicular_partition_y: Optional[PerpendicularPartition] = None,
        bivariable_space_transforms: Optional[BiVariableSpaceTransform] = None,
        bivariable_color_transforms: Optional[BiVariableColorTransform] = None,
        easing_x: Optional[PerChannelTransform] = None,
        easing_y: Optional[PerChannelTransform] = None,
    ):
        """
        Create multiple 2D gradients from lists of corner colors with bilinear interpolation.
        Handles heterogeneous lists by normalizing their lengths.

        Args:
            color_tl: List of top-left colors
            color_tr: List of top-right colors
            color_bl: List of bottom-left colors
            color_br: List of bottom-right colors
            width: Number of columns
            height: Number of rows
            color_space: Target color space ('rgb', 'hsv', 'hsl', etc.)
            format_type: Format type (INT or FLOAT)
            unit_transform_x: Optional transformation of x interpolation factors
            unit_transform_y: Optional transformation of y interpolation factors
            hue_direction_x: Hue interpolation direction along x-axis for hue-based 
                            color spaces ('cw', 'ccw', 'shortest', or None)
            hue_direction_y: Hue interpolation direction along y-axis for hue-based 
                            color spaces ('cw', 'ccw', 'shortest', or None)
            perpendicular_partition_y: Optional PerpendicularPartition for multiple vertical hue directions.
                           If provided, overrides hue_direction_y.
            bivariable_space_transforms: Channel-specific (x, y) → (x', y') transforms
            bivariable_color_transforms: Channel-specific (x, y, color) → color' transforms
            easing_x: Channel-specific x-axis easing functions
            easing_y: Channel-specific y-axis easing functions
        Returns:
            List of Gradient2D instances
        """
        #First, make 1D gradients from the corner colors
        top_gradient = Gradient1D.from_colors(
            color_tl,
            color_tr,
            width,
            color_space,
            format_type,
            unit_transform_x,
            hue_direction_x,
        )
    @classmethod
    def gradient_stack(
        cls,
        colors: List[List[Union[ColorBase, Tuple, int]]], # 2D List of colors for each gradient
        width: int,
        height: int,
        width_portions: Optional[List[List[float]]] = None, # 2D list width portions between gradient colors. Must sum 1.0 in total per row.
        height_portions: Optional[List[float]] = None, # 1D list height portions between gradient rows. Must sum 1.0 in total. #
                                                            #Can't have both width_portions and height_portions as heterogeneous.
        color_spaces_x: Optional[List[List[str]]] = None, #2D list matching colors
        color_spaces_y: Optional[List[List[str]]] = None, #2D list matching colors
        format_type: FormatType = FormatType.FLOAT,
        unit_transforms_x: Optional[List[List[UnitTransform]]] = None, #2D list matching colors
        unit_transforms_y: Optional[List[List[UnitTransform]]] = None, #2D list matching colors
        hue_directions_x: Optional[List[List[str]]] = None, #2D list matching colors
        hue_directions_y: Optional[List[str]] = None, #1D list matching colors
        hue_partitions_y: Optional[List[HuePartition]] = None, #1D list matching colors
        bivariable_space_transforms: Optional[List[BiVariableSpaceTransform]] = None, #1D list one between each row
        bivariable_color_transforms: Optional[List[BiVariableColorTransform]] = None, #1D list one between each row
        global_easing_x: Optional[PerChannelTransform] = None,
        global_easing_y: Optional[PerChannelTransform] = None,

    )  -> Gradient2D:
        """
        Create a stack of 2D gradients from a grid of corner colors.

        Args:
            colors: 2D list of corner colors for each gradient
            width: Number of columns in each gradient
            height: Number of rows in each gradient
            color_spaces_x: List of color spaces for each gradient along x-axis
            color_spaces_y: List of color spaces for each gradient along y-axis
            format_type: Format type (INT or FLOAT)
            unit_transforms_x: List of x-axis unit transforms for each gradient
            unit_transforms_y: List of y-axis unit transforms for each gradient
            hue_directions_x: List of hue directions along x-axis for each gradient
            hue_directions_y: List of hue directions along y-axis for each gradient
            bivariable_space_transforms: List of bivariable space transforms for each gradient
            bivariable_color_transforms: List of bivariable color transforms for each gradient
        Returns:
            List of Gradient2D instances
        """
        num_rows = len(colors)
        if num_rows < 2:
            raise ValueError("At least 2 rows are required for gradient_stack")
        #colors is a 2D list: rows of gradients
        between_color_lengths = [len(row) - 1 for row in colors]
        between_color_height = len(colors) - 1

        # Portions
        width_portions = normalize_width_portions(
            width_portions,
            between_color_lengths,
        )

        height_portions = normalize_height_portions(
            height_portions,
            between_color_height,
        )

        # Color spaces
        color_spaces_x = normalize_2d_rows(
            color_spaces_x,
            between_color_lengths,
            default="rgb",
        )

        color_spaces_y = normalize_2d_rows(
            color_spaces_y,
            between_color_lengths,
            default="rgb",
        )

        # Unit transforms
        unit_transforms_x = normalize_2d_rows(
            unit_transforms_x,
            between_color_lengths,
            default=None,
        )

        unit_transforms_y = normalize_2d_rows(
            unit_transforms_y,
            between_color_lengths,
            default=None,
        )

        # Hue directions
        hue_directions_x = normalize_2d_rows(
            hue_directions_x,
            between_color_lengths,
            default="shortest",
        )

        hue_directions_y = normalize_2d_rows(
            hue_directions_y,
            between_color_lengths,
            default="shortest",
        )

        # Bivariable transforms
        bivariable_space_transforms = normalize_2d_rows(
            [bivariable_space_transforms],
            between_color_lengths,
            default=None,
        )[0]

        bivariable_color_transforms = normalize_2d_rows(
            [bivariable_color_transforms],
            between_color_lengths,
            default=None,
        )[0]

        hue_partitions_y = normalize_2d_rows(
            [hue_partitions_y],
            between_color_lengths,
            default=None,
        )[0]
