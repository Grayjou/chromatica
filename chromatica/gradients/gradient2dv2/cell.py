"""
GradientCell module for 2D gradient interpolation.

This module provides the GradientCell class which is similar to GradientSegment but for 2D operations.
"""

from __future__ import annotations
from typing import List, Optional, Union, Tuple
import numpy as np
from ...types.color_types import ColorSpace, is_hue_space
from abc import ABC, abstractmethod
from ...utils.interpolate_hue import interpolate_hue
from ..v2core import multival2d_lerp, lerp_between_lines
from boundednumbers import BoundType
from ...conversions import np_convert
from ...types.format_type import FormatType
from ...colors.color_base import ColorBase
from ...types.array_types import ndarray_1d


class CellBase(ABC):
    """Abstract base class for gradient cells."""
    __slots__ = ('_value',)
    
    def get_value(self) -> np.ndarray:
        """Get or compute the cell values."""
        if self._value is None:
            self._value = self._render_value()
        return self._value
    
    @abstractmethod
    def _render_value(self) -> np.ndarray:
        """Render the cell values."""
        pass
    
    @property
    def format_type(self) -> str:
        """Format type of the cell."""
        return "float"
    
    @abstractmethod
    def convert_to_space(self, color_space: ColorSpace) -> CellBase:
        """Convert the cell to a different color space."""
        pass


def interpolate_2d_non_hue(
    starts: np.ndarray,
    ends: np.ndarray,
    coords: List[np.ndarray],  # List of H x W arrays, one per channel
    bound_types: Union[List[BoundType], BoundType],
) -> np.ndarray:
    """
    Interpolate non-hue channels in 2D.
    
    Args:
        starts: Start values, shape (num_channels,)
        ends: End values, shape (num_channels,)
        coords: List of H x W arrays, one per channel
        bound_types: Bound types for interpolation
        
    Returns:
        Interpolated values, shape (H, W, num_channels)
    """
    return multival2d_lerp(
        starts=starts,
        ends=ends,
        coeffs=coords,
        bound_types=bound_types,
    )


def interpolate_2d_hue_space(
    start: np.ndarray,
    end: np.ndarray,
    coords: List[np.ndarray],  # List of H x W arrays, one per channel
    hue_direction: str,
    bound_types: Union[List[BoundType], BoundType],
) -> np.ndarray:
    """
    Interpolate colors in a hue-based space (HSV/HSL) in 2D.
    
    Note: This implementation uses nested loops for clarity. For production use
    with large images, consider vectorizing or using Cython for better performance.
    
    Args:
        start: Start color values, shape (num_channels,)
        end: End color values, shape (num_channels,)
        coords: List of H x W arrays for each channel
        hue_direction: Hue interpolation direction
        bound_types: Bound types for interpolation
        
    Returns:
        Interpolated values, shape (H, W, num_channels)
    """
    # Interpolate hue channel
    hue_u = coords[0]
    H, W = hue_u.shape
    hue_result = np.zeros((H, W))
    
    # For each pixel, interpolate the hue
    for h in range(H):
        for w in range(W):
            hue_result[h, w] = interpolate_hue(start[0], end[0], hue_u[h, w], hue_direction)
    
    # Interpolate other channels
    rest = interpolate_2d_non_hue(
        starts=start[1:],
        ends=end[1:],
        coords=coords[1:],
        bound_types=bound_types,
    )
    
    # Stack hue with other channels
    hue_expanded = hue_result[:, :, np.newaxis]
    return np.concatenate([hue_expanded, rest], axis=2)


class GradientCell(CellBase):
    """
    A 2D gradient cell for spatial interpolation.
    
    Similar to GradientSegment but operates on 2D spatial grids.
    Supports per-channel 2D coordinate remaps (WxHx2 per channel).
    """
    __slots__ = (
        'top_left_color', 'top_right_color', 'bottom_left_color', 'bottom_right_color',
        'coords', 'color_space', '_value', 'hue_direction_horizontal', 'hue_direction_vertical',
        'bound_types'
    )
    
    def __init__(
        self,
        top_left_color: np.ndarray,
        top_right_color: np.ndarray,
        bottom_left_color: np.ndarray,
        bottom_right_color: np.ndarray,
        coords: List[np.ndarray],  # List of H x W x 2 arrays, one per channel
        color_space: ColorSpace,
        hue_direction_horizontal: Optional[str] = None,
        hue_direction_vertical: Optional[str] = None,
        bound_types: Optional[Union[List[BoundType], BoundType]] = BoundType.CLAMP,
        *,
        value: Optional[np.ndarray] = None
    ):
        """
        Initialize a GradientCell.
        
        Args:
            top_left_color: Color at top-left corner
            top_right_color: Color at top-right corner
            bottom_left_color: Color at bottom-left corner
            bottom_right_color: Color at bottom-right corner
            coords: List of H x W x 2 coordinate arrays, one per channel
            color_space: Color space for interpolation
            hue_direction_horizontal: Hue direction for horizontal interpolation
            hue_direction_vertical: Hue direction for vertical interpolation
            bound_types: Bound types for interpolation
            value: Pre-computed values (optional)
        """
        self.top_left_color = top_left_color
        self.top_right_color = top_right_color
        self.bottom_left_color = bottom_left_color
        self.bottom_right_color = bottom_right_color
        self.coords = coords
        self.color_space = color_space
        self.hue_direction_horizontal = hue_direction_horizontal
        self.hue_direction_vertical = hue_direction_vertical
        self.bound_types = bound_types
        self._value = value
    
    def _render_value(self) -> np.ndarray:
        """Render the cell by interpolating between corners."""
        # For now, implement a simple bilinear interpolation
        # This can be extended to use the per-channel coords
        
        if is_hue_space(self.color_space):
            return self._render_hue_space()
        else:
            return self._render_non_hue_space()
    
    def _render_non_hue_space(self) -> np.ndarray:
        """Render the cell in non-hue color space."""
        # Get the first coord array to determine dimensions
        H, W, _ = self.coords[0].shape
        num_channels = len(self.coords)
        
        # Extract u and v coordinates per channel
        u_coords = [self.coords[i][:, :, 0] for i in range(num_channels)]
        v_coords = [self.coords[i][:, :, 1] for i in range(num_channels)]
        
        # First interpolate horizontally (top edge)
        top_edge = interpolate_2d_non_hue(
            starts=self.top_left_color,
            ends=self.top_right_color,
            coords=u_coords,
            bound_types=self.bound_types,
        )
        
        # Interpolate horizontally (bottom edge)
        bottom_edge = interpolate_2d_non_hue(
            starts=self.bottom_left_color,
            ends=self.bottom_right_color,
            coords=u_coords,
            bound_types=self.bound_types,
        )
        
        # Now interpolate vertically between top and bottom edges
        result = np.zeros((H, W, num_channels))
        for ch in range(num_channels):
            v = v_coords[ch]
            result[:, :, ch] = top_edge[:, :, ch] * (1 - v) + bottom_edge[:, :, ch] * v
        
        return result
    
    def _render_hue_space(self) -> np.ndarray:
        """
        Render the cell in hue-based color space.
        
        Note: This implementation uses nested loops for clarity. For production use
        with large images, consider vectorizing or using Cython for better performance.
        """
        # Similar to non-hue but with special hue handling
        # This is a simplified implementation
        H, W, _ = self.coords[0].shape
        num_channels = len(self.coords)
        
        u_coords = [self.coords[i][:, :, 0] for i in range(num_channels)]
        v_coords = [self.coords[i][:, :, 1] for i in range(num_channels)]
        
        # Interpolate top edge
        top_edge = interpolate_2d_hue_space(
            start=self.top_left_color,
            end=self.top_right_color,
            coords=u_coords,
            hue_direction=self.hue_direction_horizontal or "shortest",
            bound_types=self.bound_types,
        )
        
        # Interpolate bottom edge
        bottom_edge = interpolate_2d_hue_space(
            start=self.bottom_left_color,
            end=self.bottom_right_color,
            coords=u_coords,
            hue_direction=self.hue_direction_horizontal or "shortest",
            bound_types=self.bound_types,
        )
        
        # Vertically interpolate between edges
        result = np.zeros((H, W, num_channels))
        for h in range(H):
            for w in range(W):
                for ch in range(num_channels):
                    v = v_coords[ch][h, w]
                    if ch == 0:  # Hue channel
                        result[h, w, ch] = interpolate_hue(
                            top_edge[h, w, ch],
                            bottom_edge[h, w, ch],
                            v,
                            self.hue_direction_vertical or "shortest"
                        )
                    else:
                        result[h, w, ch] = top_edge[h, w, ch] * (1 - v) + bottom_edge[h, w, ch] * v
        
        return result
    
    def convert_to_space(self, color_space: ColorSpace) -> GradientCell:
        """Convert the cell to a different color space."""
        if self.color_space == color_space:
            return self
        
        # Convert corner colors
        converted_tl = np_convert(self.top_left_color, self.color_space, color_space, fmt="float", output_type='float')
        converted_tr = np_convert(self.top_right_color, self.color_space, color_space, fmt="float", output_type='float')
        converted_bl = np_convert(self.bottom_left_color, self.color_space, color_space, fmt="float", output_type='float')
        converted_br = np_convert(self.bottom_right_color, self.color_space, color_space, fmt="float", output_type='float')
        
        # Convert pre-computed value if exists
        converted_value = None
        if self._value is not None:
            converted_value = np_convert(self.get_value(), self.color_space, color_space, fmt="float", output_type='float')
        
        return GradientCell(
            top_left_color=converted_tl,
            top_right_color=converted_tr,
            bottom_left_color=converted_bl,
            bottom_right_color=converted_br,
            coords=self.coords,
            color_space=color_space,
            hue_direction_horizontal=self.hue_direction_horizontal,
            hue_direction_vertical=self.hue_direction_vertical,
            bound_types=self.bound_types,
            value=converted_value,
        )
    
    @classmethod
    def from_color_arrays(
        cls,
        top_line: np.ndarray,
        bottom_line: np.ndarray,
        coords: np.ndarray,  # H x W x 2
        color_space: ColorSpace,
        hue_direction: Optional[str] = None,
        bound_types: Optional[Union[List[BoundType], BoundType]] = BoundType.CLAMP,
    ) -> GradientCell:
        """
        Create a GradientCell from two 1D color arrays (top and bottom lines).
        
        Args:
            top_line: Top line colors, shape (L, C) where L is length and C is channels
            bottom_line: Bottom line colors, shape (L, C)
            coords: Coordinate grid, shape (H, W, 2)
            color_space: Color space for the cell
            hue_direction: Hue direction if applicable
            bound_types: Bound types for interpolation
            
        Returns:
            GradientCell instance
        """
        # Use the 2D interpolation from v2core
        if lerp_between_lines is not None:
            value = lerp_between_lines(top_line, bottom_line, coords)
        else:
            # Fallback if Cython extension not available
            raise NotImplementedError("lerp_between_lines requires Cython extension to be built")
        
        # Extract corner colors
        top_left = top_line[0]
        top_right = top_line[-1]
        bottom_left = bottom_line[0]
        bottom_right = bottom_line[-1]
        
        # Create coords list (one per channel, all using the same coords for now)
        num_channels = top_line.shape[1] if len(top_line.shape) > 1 else 1
        coords_list = [coords] * num_channels
        
        return cls(
            top_left_color=top_left,
            top_right_color=top_right,
            bottom_left_color=bottom_left,
            bottom_right_color=bottom_right,
            coords=coords_list,
            color_space=color_space,
            hue_direction_horizontal=hue_direction,
            hue_direction_vertical=hue_direction,
            bound_types=bound_types,
            value=value,
        )
    
    @classmethod
    def from_corners(
        cls,
        top_left_color: Union[ColorBase, Tuple, List, ndarray_1d],
        top_right_color: Union[ColorBase, Tuple, List, ndarray_1d],
        bottom_left_color: Union[ColorBase, Tuple, List, ndarray_1d],
        bottom_right_color: Union[ColorBase, Tuple, List, ndarray_1d],
        width: int,
        height: int,
        input_color_spaces: Tuple[ColorSpace, ColorSpace, ColorSpace, ColorSpace],
        format_type: FormatType,
        color_space: ColorSpace,
        hue_direction_horizontal: Optional[str] = None,
        hue_direction_vertical: Optional[str] = None,
        per_channel_coords: Optional[List[np.ndarray]] = None,
        bound_types: Optional[Union[List[BoundType], BoundType]] = BoundType.CLAMP,
    ) -> GradientCell:
        """
        Create a GradientCell from four corner colors.
        
        Args:
            top_left_color: Top-left corner color
            top_right_color: Top-right corner color
            bottom_left_color: Bottom-left corner color
            bottom_right_color: Bottom-right corner color
            width: Width of the cell
            height: Height of the cell
            input_color_spaces: Color spaces of the four input colors (TL, TR, BL, BR)
            format_type: Format type of input colors
            color_space: Target color space for interpolation
            hue_direction_horizontal: Hue direction for horizontal interpolation
            hue_direction_vertical: Hue direction for vertical interpolation
            per_channel_coords: Optional per-channel coordinate remaps
            bound_types: Bound types for interpolation
            
        Returns:
            GradientCell instance
        """
        from ..gradient1dv2.color_conversion_utils import convert_to_space_float
        
        # Convert corner colors to target space
        tl = convert_to_space_float(top_left_color, input_color_spaces[0], format_type, color_space).value
        tr = convert_to_space_float(top_right_color, input_color_spaces[1], format_type, color_space).value
        bl = convert_to_space_float(bottom_left_color, input_color_spaces[2], format_type, color_space).value
        br = convert_to_space_float(bottom_right_color, input_color_spaces[3], format_type, color_space).value
        
        # Create default coords if not provided
        if per_channel_coords is None:
            # Create a simple grid
            u = np.linspace(0, 1, width)
            v = np.linspace(0, 1, height)
            uu, vv = np.meshgrid(u, v)
            coords_2d = np.stack([uu, vv], axis=2)
            
            # Use same coords for all channels
            num_channels = len(tl)
            per_channel_coords = [coords_2d] * num_channels
        
        return cls(
            top_left_color=tl,
            top_right_color=tr,
            bottom_left_color=bl,
            bottom_right_color=br,
            coords=per_channel_coords,
            color_space=color_space,
            hue_direction_horizontal=hue_direction_horizontal,
            hue_direction_vertical=hue_direction_vertical,
            bound_types=bound_types,
        )
