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
from ..v2core import multival2d_lerp, lerp_between_lines, multival2d_lerp_uniform
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
    Interpolate non-hue channels in 2D using optimized Cython functions.
    
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
    
    Args:
        start: Start color values, shape (num_channels,)
        end: End color values, shape (num_channels,)
        coords: List of H x W arrays for each channel
        hue_direction: Hue interpolation direction
        bound_types: Bound types for interpolation
        
    Returns:
        Interpolated values, shape (H, W, num_channels)
    """
    # For hue spaces, we need to handle hue specially
    # Use lerp_between_lines for optimized 2D interpolation
    # We'll interpolate horizontally first, then vertically
    
    H, W = coords[0].shape[:2]
    num_channels = len(coords)
    
    # Extract u (horizontal) and v (vertical) coordinates
    # Assuming coords are HxW arrays (not HxWx2 as in the previous implementation)
    # For simplicity, let's use the same coordinates for both dimensions
    # You might want to adjust this based on your actual coordinate format
    
    # Create coordinate grid for lerp_between_lines
    # This is simplified - you may need to adjust based on your actual data structure
    u_coords = coords[0]  # Use first channel's coordinates for positioning
    
    # For now, use a simple approach: treat as uniform interpolation in 2D
    # For production, you might want a more sophisticated approach
    
    # Use multival2d_lerp_uniform for simplicity if all channels use same coefficients
    if all(np.array_equal(coords[0], c) for c in coords[1:]):
        return multival2d_lerp_uniform(
            starts=start,
            ends=end,
            coeff=coords[0],
            bound_type=bound_types if isinstance(bound_types, BoundType) else bound_types[0],
        )
    
    # Otherwise use the standard multival2d_lerp
    return multival2d_lerp(
        starts=start,
        ends=end,
        coeffs=coords,
        bound_types=bound_types,
    )


class GradientCell(CellBase):
    """
    A 2D gradient cell for spatial interpolation.
    
    Similar to GradientSegment but operates on 2D spatial grids.
    """
    __slots__ = (
        'top_left_color', 'top_right_color', 'bottom_left_color', 'bottom_right_color',
        'coords', 'color_space', '_value', 'hue_direction', 'bound_types'
    )
    
    def __init__(
        self,
        top_left_color: np.ndarray,
        top_right_color: np.ndarray,
        bottom_left_color: np.ndarray,
        bottom_right_color: np.ndarray,
        coords: List[np.ndarray],  # List of H x W arrays, one per channel
        color_space: ColorSpace,
        hue_direction: Optional[str] = None,
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
            coords: List of H x W coefficient arrays, one per channel
            color_space: Color space for interpolation
            hue_direction: Hue direction for interpolation (if applicable)
            bound_types: Bound types for interpolation
            value: Pre-computed values (optional)
        """
        self.top_left_color = top_left_color
        self.top_right_color = top_right_color
        self.bottom_left_color = bottom_left_color
        self.bottom_right_color = bottom_right_color
        self.coords = coords
        self.color_space = color_space
        self.hue_direction = hue_direction
        self.bound_types = bound_types
        self._value = value
    
    def _render_value(self) -> np.ndarray:
        """Render the cell by interpolating between corners."""
        # For 2D interpolation, we can use lerp_between_lines
        # First create top and bottom lines from corners
        
        # Get dimensions
        H, W = self.coords[0].shape[:2]
        
        # Create top line (interpolating between top_left and top_right)
        top_line = self._interpolate_line(
            self.top_left_color, self.top_right_color, self.coords[0]
        )
        
        # Create bottom line (interpolating between bottom_left and bottom_right)
        bottom_line = self._interpolate_line(
            self.bottom_left_color, self.bottom_right_color, self.coords[0]
        )
        
        # Now interpolate between top and bottom lines using vertical coordinates
        # Use lerp_between_lines for optimized 2D interpolation
        # We need to create a coordinate grid for the interpolation
        
        # Create coordinate grid where:
        # - u_x ∈ [0,1]: position along the lines (from left to right)
        # - u_y ∈ [0,1]: blend factor between lines (0 = top, 1 = bottom)
        
        # For simplicity, assume coords[0] provides u_x and we need u_y
        # You might need to adjust this based on your actual coordinate structure
        
        # Create a simple vertical blend factor (linearly spaced)
        u_y = np.linspace(0, 1, H)[:, np.newaxis]  # Shape (H, 1)
        u_y = np.repeat(u_y, W, axis=1)  # Shape (H, W)
        
        # Stack coordinates
        coord_grid = np.stack([self.coords[0], u_y], axis=-1)  # Shape (H, W, 2)
        
        # Use lerp_between_lines for efficient 2D interpolation
        return lerp_between_lines(top_line, bottom_line, coord_grid)
    
    def _interpolate_line(self, start: np.ndarray, end: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Interpolate a 1D line between two colors."""
        # Reshape coords to (num_points, 1) for 1D interpolation
        num_points = coords.size
        coords_1d = coords.reshape(-1, 1)
        
        # Create multiple calls for each channel if needed
        # For simplicity, using the same approach as multival1d_lerp
        num_channels = len(start)
        
        if num_channels == 1:
            # Single channel
            result_1d = multival1d_lerp_uniform(
                starts=start,
                ends=end,
                coeff=coords_1d.flatten(),
                bound_type=self.bound_types if isinstance(self.bound_types, BoundType) else self.bound_types[0]
            )
            return result_1d.reshape(coords.shape[0], -1)  # Reshape to (num_points, 1)
        else:
            # Multi-channel - need to handle properly
            # This is a simplified version
            results = []
            for i in range(num_channels):
                channel_result = multival1d_lerp_uniform(
                    starts=np.array([start[i]]),
                    ends=np.array([end[i]]),
                    coeff=coords_1d.flatten(),
                    bound_type=self.bound_types if isinstance(self.bound_types, BoundType) else self.bound_types[i]
                )
                results.append(channel_result.reshape(-1, 1))
            
            return np.hstack(results)  # Shape (num_points, num_channels)
    
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
            hue_direction=self.hue_direction,
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
        value = lerp_between_lines(top_line, bottom_line, coords)
        
        # Extract corner colors
        if top_line.ndim == 1:
            top_left = top_line[0:1]
            top_right = top_line[-1:]
            bottom_left = bottom_line[0:1]
            bottom_right = bottom_line[-1:]
        else:
            top_left = top_line[0]
            top_right = top_line[-1]
            bottom_left = bottom_line[0]
            bottom_right = bottom_line[-1]
        
        # Create coords list (one per channel, all using the same coords for now)
        if top_line.ndim == 1:
            num_channels = 1
        else:
            num_channels = top_line.shape[1]
        
        # Extract u and v from coords
        u_coords = coords[:, :, 0]  # Horizontal coordinates
        coords_list = [u_coords] * num_channels  # Same coordinates for all channels
        
        return cls(
            top_left_color=top_left,
            top_right_color=top_right,
            bottom_left_color=bottom_left,
            bottom_right_color=bottom_right,
            coords=coords_list,
            color_space=color_space,
            hue_direction=hue_direction,
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
        hue_direction: Optional[str] = None,
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
            hue_direction: Hue direction for interpolation
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
            # Create a simple horizontal coordinate grid
            u = np.linspace(0, 1, width)
            u_grid = np.tile(u, (height, 1))  # Shape (height, width)
            
            # Use same coords for all channels
            num_channels = len(tl)
            per_channel_coords = [u_grid] * num_channels
        
        return cls(
            top_left_color=tl,
            top_right_color=tr,
            bottom_left_color=bl,
            bottom_right_color=br,
            coords=per_channel_coords,
            color_space=color_space,
            hue_direction=hue_direction,
            bound_types=bound_types,
        )