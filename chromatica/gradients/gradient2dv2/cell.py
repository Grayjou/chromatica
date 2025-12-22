"""
GradientCell module for 2D gradient interpolation.

This module provides the GradientCell class which is similar to GradientSegment but for 2D operations.
"""

from __future__ import annotations
from typing import List, Optional, Union, Tuple
import numpy as np
from ...types.color_types import ColorSpace, is_hue_space
from abc import abstractmethod
from ..v2core.subgradient import SubGradient
from ..v2core import multival2d_lerp, lerp_between_lines, multival2d_lerp_uniform
from boundednumbers import BoundType
from ...conversions import np_convert
from ...types.format_type import FormatType
from ...colors.color_base import ColorBase
from ...types.array_types import ndarray_1d
from .helpers import HueMode, get_line_method, LineInterpMethods, CellMode


class CellBase(SubGradient):
    """Abstract base class for 2D gradient cells, extending SubGradient."""
    
    mode: CellMode
    
    def __init__(self):
        """Initialize with no cached value."""
        super().__init__()


class LinesCell(CellBase):
    mode: CellMode = CellMode.LINES
    """2D gradient cell defined by lines."""
    def __init__(self,
            top_line: np.ndarray,
            bottom_line: np.ndarray,
            per_channel_coords: List[np.ndarray] | np.ndarray,
            color_space: ColorSpace,
            hue_direction_y: HueMode,
            line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
            hue_direction_x: Optional[HueMode] = None,
            boundtypes: List[BoundType] | BoundType = BoundType.CLAMP, *, value: Optional[np.ndarray] = None) -> None:
        self.line_method = line_method
        self.color_space = color_space
        self.hue_direction_y = hue_direction_y
        self.hue_direction_x = hue_direction_x
        self.per_channel_coords = per_channel_coords
        self.boundtypes = boundtypes
        self.top_line = top_line
        self.bottom_line = bottom_line
        self._value = value

class CornersCell(CellBase):
    mode: CellMode = CellMode.CORNERS
    """2D gradient cell defined by corner colors."""
    def __init__(self,
            top_left: np.ndarray,
            top_right: np.ndarray,
            bottom_left: np.ndarray,
            bottom_right: np.ndarray,
            per_channel_coords: List[np.ndarray] | np.ndarray,
            color_space: ColorSpace,
            hue_direction_y: HueMode,
            hue_direction_x: HueMode,
            boundtypes: List[BoundType] | BoundType = BoundType.CLAMP, *, value: Optional[np.ndarray] = None) -> None:
        self.color_space = color_space
        self.hue_direction_y = hue_direction_y
        self.hue_direction_x = hue_direction_x
        self.per_channel_coords = per_channel_coords
        self.boundtypes = boundtypes
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self._value = value


"""

def get_transformed_segment(
    already_converted_start_color: Optional[np.ndarray] = None,
    already_converted_end_color: Optional[np.ndarray] = None,
    local_us: List[np.ndarray] = None,
    color_space: ColorSpace = None,
    hue_direction: Optional[str] = None,
    per_channel_transforms: Optional[dict] = None,
    bound_types: Optional[List[BoundType] | BoundType] = BoundType.CLAMP,
    *,
    value: Optional[np.ndarray] = None,
    # New parameters for conversion
    start_color: Optional[Union[ColorBase, Tuple, List, ndarray_1d]] = None,
    end_color: Optional[Union[ColorBase, Tuple, List, ndarray_1d]] = None,
    start_color_space: Optional[ColorSpace] = None,
    end_color_space: Optional[ColorSpace] = None,
    format_type: Optional[FormatType] = None,
) -> TransformedGradientSegment:
"""

#Supports conversion by default unlike segment
def get_transformed_lines_cell(
        top_line: np.ndarray,
        bottom_line: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,
        color_space: ColorSpace,
        top_line_color_space: ColorSpace,
        bottom_line_color_space: ColorSpace,
        hue_direction_y: HueMode,
        hue_direction_x: Optional[HueMode] = None,
        per_channel_transforms: Optional[dict] = None,
        line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        ):
    """Create a transformed LinesCell with proper color space conversion."""
    pass
