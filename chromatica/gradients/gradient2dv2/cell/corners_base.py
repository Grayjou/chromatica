# chromatica/gradients/gradient2dv2/cell/corners_base.py
"""Base class for corner-based 2D gradient cells."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

from boundednumbers import BoundType

from ....types.color_types import ColorModes
from .base import CellBase
from ._cell_coords import get_shape, extract_edge
from ._descriptors import CellPropertyDescriptor
from enum import IntEnum

class CornerIndex(IntEnum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3

class CornersBase(CellBase, ABC):
    """Abstract base class for corner-based 2D gradient cells.
    
    Provides common infrastructure for gradient cells defined by four corner colors.
    Subclasses implement specific interpolation strategies.
    
    Attributes:
        top_left: Color at top-left corner
        top_right: Color at top-right corner
        bottom_left: Color at bottom-left corner
        bottom_right: Color at bottom-right corner
        per_channel_coords: Coordinate arrays for each channel
        color_mode: Color space for interpolation
        hue_direction_x: Hue interpolation direction along X axis
        hue_direction_y: Hue interpolation direction along Y axis
        boundtypes: Boundary handling types per channel
        border_mode: OpenCV border mode for coordinate mapping
        border_value: Value used for constant border mode
    """
    
    # === Corner Properties (invalidate segments + cache) ===
    top_left: np.ndarray = CellPropertyDescriptor('top_left', invalidates_segments=True)
    top_right: np.ndarray = CellPropertyDescriptor('top_right', invalidates_segments=True)
    bottom_left: np.ndarray = CellPropertyDescriptor('bottom_left', invalidates_segments=True)
    bottom_right: np.ndarray = CellPropertyDescriptor('bottom_right', invalidates_segments=True)
    
    # === Coordinate Property (invalidates segments + cache) ===
    per_channel_coords: Union[List[np.ndarray], np.ndarray] = CellPropertyDescriptor(
        'per_channel_coords', invalidates_segments=True
    )
    
    # === Interpolation Properties ===
    hue_direction_x: Optional[str] = CellPropertyDescriptor('hue_direction_x')
    
    # === Read-only Properties ===
    color_mode: ColorModes = CellPropertyDescriptor('color_mode', readonly=True)
    hue_direction_y: Optional[str] = CellPropertyDescriptor('hue_direction_y')
    boundtypes: Union[List[BoundType], BoundType] = CellPropertyDescriptor('boundtypes', readonly=True)
    border_mode: Optional[int] = CellPropertyDescriptor('border_mode', readonly=True)
    border_value: Optional[float] = CellPropertyDescriptor('border_value', readonly=True)
    
    def __init__(
        self,
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        per_channel_coords: Union[List[np.ndarray], np.ndarray],
        color_mode: ColorModes,
        hue_direction_y: Optional[str] = None,
        hue_direction_x: Optional[str] = None,
        boundtypes: Union[List[BoundType], BoundType] = BoundType.CLAMP,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        *,
        value: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        
        # Corner colors
        self._top_left = top_left
        self._top_right = top_right
        self._bottom_left = bottom_left
        self._bottom_right = bottom_right
        
        # Coordinates and color space
        self._per_channel_coords = per_channel_coords
        self._color_mode = color_mode
        
        # Interpolation settings
        self._hue_direction_y = hue_direction_y
        self._hue_direction_x = hue_direction_x
        self._boundtypes = boundtypes
        self._border_mode = border_mode
        self._border_value = border_value
        
        # Cached values
        self._value = value
        self._top_segment: Optional[np.ndarray] = None
        self._bottom_segment: Optional[np.ndarray] = None
    
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
    
    # === Edge Coordinates ===
    
    @property
    def top_edge_coords(self) -> np.ndarray:
        """Coordinate array for the top edge."""
        return extract_edge(self._per_channel_coords, 0)
    
    @property
    def bottom_edge_coords(self) -> np.ndarray:
        """Coordinate array for the bottom edge."""
        return extract_edge(self._per_channel_coords, self.height - 1)
    
    # === Cache Management ===
    
    def _invalidate_segments(self) -> None:
        """Invalidate cached segment data."""
        self._top_segment = None
        self._bottom_segment = None
    
    # === Edge Interpolation ===
    
    def simple_untransformed_interpolate_edge(
        self,
        horizontal_pos: float,
        is_top_edge: bool,
    ) -> np.ndarray:
        """Linearly interpolate along an edge without transforms.
        
        Args:
            horizontal_pos: Position along edge, 0.0 = left, 1.0 = right
            is_top_edge: If True, interpolate top edge; else bottom edge
            
        Returns:
            Interpolated color as numpy array
        """
        # Handle edge cases for numerical stability
        if horizontal_pos < 1e-6:
            return self._top_left if is_top_edge else self._bottom_left
        if horizontal_pos > 1 - 1e-6:
            return self._top_right if is_top_edge else self._bottom_right
        
        start = self._top_left if is_top_edge else self._bottom_left
        end = self._top_right if is_top_edge else self._bottom_right
        return start + horizontal_pos * (end - start)
    
    # === Abstract Methods ===
    
    @abstractmethod
    def get_top_segment_untransformed(self) -> np.ndarray:
        """Get or create the top segment in uniform coordinates.
        
        Returns:
            Array of shape (1, width, channels)
        """
        ...
    
    @abstractmethod
    def get_bottom_segment_untransformed(self) -> np.ndarray:
        """Get or create the bottom segment in uniform coordinates.
        
        Returns:
            Array of shape (1, width, channels)
        """
        ...
    
    @abstractmethod
    def _interpolate_at_coords(self, coords_list: List[np.ndarray]) -> np.ndarray:
        """Core 2D interpolation at specific coordinates.
        
        Args:
            coords_list: List of coordinate arrays, one per channel
            
        Returns:
            Interpolated gradient array
        """
        ...
    
    @abstractmethod
    def convert_to_space(
        self,
        color_mode: ColorModes,
        render_before: bool = False,
    ) -> CornersBase:
        """Convert to a different color space.
        
        Args:
            color_mode: Target color space
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
            f"color_mode={self._color_mode!r})"
        )