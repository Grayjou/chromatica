"""
GradientCell module for 2D gradient interpolation.

This module provides the GradientCell class which is similar to GradientSegment but for 2D operations.
"""

from __future__ import annotations
from typing import List, Optional, Union, Tuple
import numpy as np
from ...types.color_types import ColorSpace, is_hue_space
from abc import abstractmethod
from ...v2core.subgradient import SubGradient
from ...v2core import multival2d_lerp, lerp_between_lines, multival2d_lerp_uniform
from boundednumbers import BoundType
from ...conversions import np_convert
from ...types.format_type import FormatType
from ...colors.color_base import ColorBase
from ...types.array_types import ndarray_1d
from .helpers import (
    LineInterpMethods,
    interp_transformed_2d_from_corners,
    interp_transformed_2d_lines,
    CellMode,
    apply_per_channel_transforms_2d,
)
from ...utils.color_utils import convert_to_space_float
from ...types.color_types import HueMode
from ...conversions import np_convert


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
    def _render_value(self):
        print("Rendering LinesCell", self.line_method)
        return interp_transformed_2d_lines(
            line0=self.top_line,
            line1=self.bottom_line,
            transformed=self.per_channel_coords,
            color_space=self.color_space,
            huemode_y=self.hue_direction_y,
            huemode_x=self.hue_direction_x,
            line_method=self.line_method,
            bound_types=self.boundtypes,
        )
    def convert_to_space(self, color_space: ColorSpace) -> LinesCell:
        if self.color_space == color_space:
            return self
        converted_top = np_convert(self.top_line, self.color_space, color_space, fmt="float", output_type='float')
        converted_bottom = np_convert(self.bottom_line, self.color_space, color_space, fmt="float", output_type='float')
        converted_value = np_convert(self.get_value(), self.color_space, color_space, fmt="float", output_type='float') if self._value is not None else None
        return LinesCell(
            converted_top,
            converted_bottom,
            self.per_channel_coords,
            color_space,
            self.hue_direction_y,
            self.line_method,
            self.hue_direction_x,
            self.boundtypes,
            value=converted_value,
        )



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
    def _render_value(self):
        return interp_transformed_2d_from_corners(
            c_tl=self.top_left,
            c_tr=self.top_right,
            c_bl=self.bottom_left,
            c_br=self.bottom_right,
            transformed=self.per_channel_coords,
            color_space=self.color_space,
            huemode_x=self.hue_direction_x,
            huemode_y=self.hue_direction_y,
            bound_types=self.boundtypes,
        )
    def convert_to_space(self, color_space: ColorSpace) -> CornersCell:
        if self.color_space == color_space:
            return self
        converted_top_left = np_convert(self.top_left, self.color_space, color_space, fmt="float", output_type='float')
        converted_top_right = np_convert(self.top_right, self.color_space, color_space, fmt="float", output_type='float')
        converted_bottom_left = np_convert(self.bottom_left, self.color_space, color_space, fmt="float", output_type='float')
        converted_bottom_right = np_convert(self.bottom_right, self.color_space, color_space, fmt="float", output_type='float')
        converted_value = np_convert(self.get_value(), self.color_space, color_space, fmt="float", output_type='float') if self._value is not None else None
        return CornersCell(
            converted_top_left,
            converted_top_right,
            converted_bottom_left,
            converted_bottom_right,
            self.per_channel_coords,
            color_space,
            self.hue_direction_y,
            self.hue_direction_x,
            self.boundtypes,
            value=converted_value,
        )




#Supports conversion by default unlike segment
def get_transformed_lines_cell(
        top_line: np.ndarray,
        bottom_line: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,
        color_space: ColorSpace,
        top_line_color_space: ColorSpace,
        bottom_line_color_space: ColorSpace,

        hue_direction_y: Optional[HueMode] = None,
        input_format: FormatType = FormatType.INT,
        hue_direction_x: Optional[HueMode] = None,
        per_channel_transforms: Optional[dict] = None,
        line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        ):
    """Create a transformed LinesCell with proper color space conversion."""
    top_line_converted = convert_to_space_float(
        top_line, top_line_color_space, input_format, color_space
    ).value
    bottom_line_converted = convert_to_space_float(
        bottom_line, bottom_line_color_space, input_format, color_space
    ).value
    if per_channel_transforms is not None:
        per_channel_coords = apply_per_channel_transforms_2d(
            coords=per_channel_coords,
            per_channel_transforms=per_channel_transforms,
            num_channels=len(color_space),
        )
    return LinesCell(
        top_line=top_line_converted,
        bottom_line=bottom_line_converted,
        per_channel_coords=per_channel_coords,
        color_space=color_space,
        hue_direction_y=hue_direction_y,
        hue_direction_x=hue_direction_x,
        line_method=line_method,
        boundtypes=boundtypes,
    )
    

def get_transformed_corners_cell(
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,
        color_space: ColorSpace,
        top_left_color_space: ColorSpace,
        top_right_color_space: ColorSpace,
        bottom_left_color_space: ColorSpace,
        bottom_right_color_space: ColorSpace,
        hue_direction_y: HueMode,
        hue_direction_x: HueMode,
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[dict] = None,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        ):
    """Create a transformed CornersCell with proper color space conversion."""
    top_left_converted = convert_to_space_float(
        top_left, top_left_color_space, input_format, color_space
    ).value
    top_right_converted = convert_to_space_float(
        top_right, top_right_color_space, input_format, color_space
    ).value
    bottom_left_converted = convert_to_space_float(
        bottom_left, bottom_left_color_space, input_format, color_space
    ).value
    bottom_right_converted = convert_to_space_float(
        bottom_right, bottom_right_color_space, input_format, color_space
    ).value
    if per_channel_transforms is not None:
        per_channel_coords = apply_per_channel_transforms_2d(
            coords=per_channel_coords,
            per_channel_transforms=per_channel_transforms,
            num_channels=len(color_space),
        )
    return CornersCell(
        top_left=top_left_converted,
        top_right=top_right_converted,
        bottom_left=bottom_left_converted,
        bottom_right=bottom_right_converted,
        per_channel_coords=per_channel_coords,
        color_space=color_space,
        hue_direction_y=hue_direction_y,
        hue_direction_x=hue_direction_x,
        boundtypes=boundtypes,
    )