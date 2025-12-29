# chromatica/gradients/gradient2dv2/cell/lines_base.py
"""Base class for line-based 2D gradient cells."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

from boundednumbers import BoundType

from ....types.color_types import ColorSpaces
from .base import CellBase
from ._cell_coords import get_shape, extract_edge
from ._descriptors import CellPropertyDescriptor
from ..helpers import LineInterpMethods


class LinesBase(CellBase, ABC):
    """Abstract base class for line-based 2D gradient cells.
    
    Line-based cells interpolate vertically between a top line and bottom line,
    where each line is a 1D array of colors (shape: width x channels).
    
    Attributes:
        top_line: Color values along the top edge
        bottom_line: Color values along the bottom edge
        per_channel_coords: Coordinate arrays for each channel
        color_space: Color space for interpolation
        hue_direction_x: Hue interpolation direction along X axis
        hue_direction_y: Hue interpolation direction along Y axis
        line_method: Method for interpolating along lines
        boundtypes: Boundary handling types per channel
        border_mode: OpenCV border mode for coordinate mapping
        border_value: Value used for constant border mode
    """
    
    # === Line Properties ===
    top_line: np.ndarray = CellPropertyDescriptor('top_line')
    bottom_line: np.ndarray = CellPropertyDescriptor('bottom_line')
    
    # === Coordinate Property ===
    per_channel_coords: Union[List[np.ndarray], np.ndarray] = CellPropertyDescriptor('per_channel_coords')
    
    # === Interpolation Properties ===
    hue_direction_x: Optional[str] = CellPropertyDescriptor('hue_direction_x')
    hue_direction_y: Optional[str] = CellPropertyDescriptor('hue_direction_y')
    line_method: LineInterpMethods = CellPropertyDescriptor('line_method')
    boundtypes: Union[List[BoundType], BoundType] = CellPropertyDescriptor('boundtypes')
    border_mode: Optional[int] = CellPropertyDescriptor('border_mode')
    border_value: Optional[float] = CellPropertyDescriptor('border_value')
    
    # === Read-only Properties ===
    color_space: ColorSpaces = CellPropertyDescriptor('color_space', readonly=True)
    
    def __init__(
        self,
        top_line: np.ndarray,
        bottom_line: np.ndarray,
        per_channel_coords: Union[List[np.ndarray], np.ndarray],
        color_space: ColorSpaces,
        hue_direction_y: Optional[str] = None,
        hue_direction_x: Optional[str] = None,
        line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
        boundtypes: Union[List[BoundType], BoundType] = BoundType.CLAMP,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        *,
        value: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        
        # Validate line shapes match
        if top_line.shape != bottom_line.shape:
            raise ValueError(
                f"Top and bottom lines must have same shape. "
                f"Got top: {top_line.shape}, bottom: {bottom_line.shape}"
            )
        
        # Line data
        self._top_line = top_line
        self._bottom_line = bottom_line
        
        # Coordinates and color space
        self._per_channel_coords = per_channel_coords
        self._color_space = color_space
        
        # Interpolation settings
        self._hue_direction_y = hue_direction_y
        self._hue_direction_x = hue_direction_x
        self._line_method = line_method
        self._boundtypes = boundtypes
        self._border_mode = border_mode
        self._border_value = border_value
        
        # Cached value
        self._value = value
    
    # === Computed Dimensions ===
    
    @property
    def width(self) -> int:
        """Width of the gradient in pixels."""
        return get_shape(self._per_channel_coords)[1]
    
    @property
    def height(self) -> int:
        """Height of the gradient in pixels."""
        return get_shape(self._per_channel_coords)[0]
    
    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the gradient as (height, width)."""
        return get_shape(self._per_channel_coords)
    
    @property
    def line_width(self) -> int:
        """Width of the lines (number of color samples)."""
        return self._top_line.shape[0]
    
    @property
    def num_channels(self) -> int:
        """Number of color channels."""
        return self._top_line.shape[1] if self._top_line.ndim > 1 else 1
    
    # === Edge Coordinates ===
    
    @property
    def top_edge_coords(self) -> Union[List[np.ndarray], np.ndarray]:
        """Coordinate arrays for the top edge."""
        return extract_edge(self._per_channel_coords, 0)
    
    @property
    def bottom_edge_coords(self) -> Union[List[np.ndarray], np.ndarray]:
        """Coordinate arrays for the bottom edge."""
        return extract_edge(self._per_channel_coords, self.height - 1)
    
    # === Segment Access (for compatibility with corners interface) ===
    
    def get_top_segment_untransformed(self) -> np.ndarray:
        """Get the top line as a segment array.
        
        Returns:
            Array of shape (1, width, channels)
        """
        return self._top_line.reshape(1, self.line_width, -1)
    
    def get_bottom_segment_untransformed(self) -> np.ndarray:
        """Get the bottom line as a segment array.
        
        Returns:
            Array of shape (1, width, channels)
        """
        return self._bottom_line.reshape(1, self.line_width, -1)
    
    # === Edge Interpolation ===
    
    def simple_untransformed_interpolate_edge(
        self,
        horizontal_pos: float,
        is_top_edge: bool,
    ) -> np.ndarray:
        """Simple linear interpolation along edge without considering transforms.
        
        This is used for creating new sub-cell boundaries during partitioning.
        
        Args:
            horizontal_pos: Position along edge, 0.0 = left, 1.0 = right
            is_top_edge: If True, interpolate top edge; else bottom edge
            
        Returns:
            Interpolated color as numpy array
        """
        line = self._top_line if is_top_edge else self._bottom_line
        
        # Handle edge cases
        if horizontal_pos <= 0.0:
            return line[0].copy()
        if horizontal_pos >= 1.0:
            return line[-1].copy()
        
        # Linear interpolation between adjacent samples
        exact_idx = horizontal_pos * (len(line) - 1)
        left_idx = int(np.floor(exact_idx))
        right_idx = min(left_idx + 1, len(line) - 1)
        t = exact_idx - left_idx
        
        return (1 - t) * line[left_idx] + t * line[right_idx]
    
    # === Abstract Methods ===
    
    @abstractmethod
    def interpolate_edge(
        self,
        horizontal_pos: float,
        is_top_edge: bool,
    ) -> np.ndarray:
        """Interpolate at a specific position along top or bottom edge.
        
        Args:
            horizontal_pos: Position along edge, 0.0 = left, 1.0 = right
            is_top_edge: If True, interpolate top edge; else bottom edge
            
        Returns:
            Interpolated color as numpy array
        """
        ...
    
    @abstractmethod
    def convert_to_space(
        self,
        color_space: ColorSpaces,
        render_before: bool = False,
    ) -> LinesBase:
        """Convert to a different color space.
        
        Args:
            color_space: Target color space
            render_before: If True, render current value before converting
            
        Returns:
            New cell instance in target color space
        """
        ...
    
    # === Representation ===
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"line_width={self.line_width}, "
            f"color_space={self._color_space!r}, "
            f"line_method={self._line_method!r})"
        )