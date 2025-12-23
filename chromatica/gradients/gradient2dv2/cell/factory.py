"""Factory functions for creating gradient cells with proper color space conversion."""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from ....types.color_types import ColorSpace
from ....types.format_type import FormatType
from ....utils.color_utils import convert_to_space_float
from boundednumbers import BoundType
from ..helpers import LineInterpMethods, apply_per_channel_transforms_2d
from .lines import LinesCell
from .corners import CornersCell
from .corners_dual import CornersCellDual


def get_transformed_lines_cell(
        top_line: np.ndarray,
        bottom_line: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,
        color_space: ColorSpace,
        top_line_color_space: ColorSpace,
        bottom_line_color_space: ColorSpace,
        hue_direction_y: Optional[str] = None,
        input_format: FormatType = FormatType.INT,
        hue_direction_x: Optional[str] = None,
        per_channel_transforms: Optional[dict] = None,
        line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        ) -> LinesCell:
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
        hue_direction_y: Optional[str],
        hue_direction_x: Optional[str],
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[dict] = None,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        ) -> CornersCell:
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


def get_transformed_corners_cell_dual(
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,
        horizontal_color_space: ColorSpace,
        vertical_color_space: ColorSpace,
        top_left_color_space: ColorSpace,
        top_right_color_space: ColorSpace,
        bottom_left_color_space: ColorSpace,
        bottom_right_color_space: ColorSpace,
        hue_direction_y: Optional[str],
        hue_direction_x: Optional[str],
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[dict] = None,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        top_segment_hue_direction_x: Optional[str] = None,
        bottom_segment_hue_direction_x: Optional[str] = None,
        top_segment_color_space: Optional[ColorSpace] = None,
        bottom_segment_color_space: Optional[ColorSpace] = None
        ) -> CornersCellDual:
    """Create a transformed CornersCellDual with proper color space conversion."""

    top_left_converted = convert_to_space_float(
        top_left, top_left_color_space, input_format, horizontal_color_space
    ).value
    top_right_converted = convert_to_space_float(
        top_right, top_right_color_space, input_format, horizontal_color_space
    ).value
    bottom_left_converted = convert_to_space_float(
        bottom_left, bottom_left_color_space, input_format, horizontal_color_space
    ).value
    bottom_right_converted = convert_to_space_float(
        bottom_right, bottom_right_color_space, input_format, horizontal_color_space
    ).value
    if per_channel_transforms is not None:
        per_channel_coords = apply_per_channel_transforms_2d(
            coords=per_channel_coords,
            per_channel_transforms=per_channel_transforms,
            num_channels=len(horizontal_color_space),
        )
    return CornersCellDual(
        top_left=top_left_converted,
        top_right=top_right_converted,
        bottom_left=bottom_left_converted,
        bottom_right=bottom_right_converted,
        per_channel_coords=per_channel_coords,
        horizontal_color_space=horizontal_color_space,
        vertical_color_space=vertical_color_space,
        hue_direction_y=hue_direction_y,
        hue_direction_x=hue_direction_x,
        boundtypes=boundtypes,
        top_segment_hue_direction_x=top_segment_hue_direction_x,
        bottom_segment_hue_direction_x=bottom_segment_hue_direction_x,
        top_segment_color_space=top_segment_color_space,
        bottom_segment_color_space=bottom_segment_color_space,
    )
