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
        top_line_color_space: Optional[ColorSpace] = None,
        bottom_line_color_space: Optional[ColorSpace] = None,
        hue_direction_y: Optional[str] = None,
        input_format: FormatType = FormatType.INT,
        hue_direction_x: Optional[str] = None,
        per_channel_transforms: Optional[dict] = None,
        line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        ) -> LinesCell:
    """Create a transformed LinesCell with proper color space conversion.
    
    Args:
        top_line: Top line array
        bottom_line: Bottom line array
        per_channel_coords: Per-channel coordinate arrays
        color_space: Target color space for the cell
        top_line_color_space: Color space of top line. Defaults to color_space if not specified.
        bottom_line_color_space: Color space of bottom line. Defaults to color_space if not specified.
        hue_direction_y: Hue direction for vertical interpolation
        input_format: Format of input color data
        hue_direction_x: Hue direction for horizontal interpolation
        per_channel_transforms: Optional per-channel transformations
        line_method: Line interpolation method
        boundtypes: Boundary types for coordinate handling
        border_mode: Border handling mode (e.g., BORDER_CLAMP, BORDER_REPEAT)
        border_value: Border constant value for BORDER_CONSTANT mode
        
    Returns:
        LinesCell instance with converted colors
    """
    # Default line color spaces to the target color_space if not specified
    if top_line_color_space is None:
        top_line_color_space = color_space
    if bottom_line_color_space is None:
        bottom_line_color_space = color_space
    
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
        border_mode=border_mode,
        border_value=border_value,
    )
    

def _get_corners_color_spaces(
        color_space: ColorSpace,
        top_left_color_space: Optional[ColorSpace] = None,
        top_right_color_space: Optional[ColorSpace] = None,
        bottom_left_color_space: Optional[ColorSpace] = None,
        bottom_right_color_space: Optional[ColorSpace] = None,
    ) -> tuple[ColorSpace, ColorSpace, ColorSpace, ColorSpace]:
    """Determine the color spaces for each corner."""
    tl_space = top_left_color_space or color_space
    tr_space = top_right_color_space or color_space
    bl_space = bottom_left_color_space or color_space
    br_space = bottom_right_color_space or color_space
    return tl_space, tr_space, bl_space, br_space

def get_transformed_corners_cell(
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,
        color_space: ColorSpace,
        top_left_color_space: Optional[ColorSpace] = None,
        top_right_color_space: Optional[ColorSpace] = None,
        bottom_left_color_space: Optional[ColorSpace] = None,
        bottom_right_color_space: Optional[ColorSpace] = None,
        hue_direction_y: Optional[str] = None,
        hue_direction_x: Optional[str] = None,
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[dict] = None,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        ) -> CornersCell:
    """Create a transformed CornersCell with proper color space conversion."""
    top_left_color_space, top_right_color_space, bottom_left_color_space, bottom_right_color_space = _get_corners_color_spaces(
        color_space,
        top_left_color_space,
        top_right_color_space,
        bottom_left_color_space,
        bottom_right_color_space,
    )
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
        border_mode=border_mode,
        border_value=border_value,
    )


def get_transformed_corners_cell_dual(
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,
        horizontal_color_space: ColorSpace,
        vertical_color_space: ColorSpace,
        top_left_color_space: Optional[ColorSpace] = None,
        top_right_color_space: Optional[ColorSpace] = None,
        bottom_left_color_space: Optional[ColorSpace] = None,
        bottom_right_color_space: Optional[ColorSpace] = None,
        hue_direction_y: Optional[str] = None,
        hue_direction_x: Optional[str] = None,
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[dict] = None,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        top_segment_hue_direction_x: Optional[str] = None,
        bottom_segment_hue_direction_x: Optional[str] = None,
        top_segment_color_space: Optional[ColorSpace] = None,
        bottom_segment_color_space: Optional[ColorSpace] = None
        ) -> CornersCellDual:
    """Create a transformed CornersCellDual with proper color space conversion.
    
    Args:
        top_left: Top-left corner color
        top_right: Top-right corner color
        bottom_left: Bottom-left corner color
        bottom_right: Bottom-right corner color
        per_channel_coords: Per-channel coordinate arrays
        horizontal_color_space: Horizontal color space for the cell
        vertical_color_space: Vertical color space for the cell
        top_left_color_space: Color space of top-left corner. Defaults to horizontal_color_space if not specified.
        top_right_color_space: Color space of top-right corner. Defaults to horizontal_color_space if not specified.
        bottom_left_color_space: Color space of bottom-left corner. Defaults to horizontal_color_space if not specified.
        bottom_right_color_space: Color space of bottom-right corner. Defaults to horizontal_color_space if not specified.
        hue_direction_y: Hue direction for vertical interpolation
        hue_direction_x: Hue direction for horizontal interpolation
        input_format: Format of input color data
        per_channel_transforms: Optional per-channel transformations
        boundtypes: Boundary types for coordinate handling
        top_segment_hue_direction_x: Hue direction for top segment horizontal interpolation
        bottom_segment_hue_direction_x: Hue direction for bottom segment horizontal interpolation
        top_segment_color_space: Color space for top segment
        bottom_segment_color_space: Color space for bottom segment
        
    Returns:
        CornersCellDual instance with converted colors
    """
    # Default corner color spaces to the horizontal_color_space if not specified
    if top_left_color_space is None:
        top_left_color_space = horizontal_color_space
    if top_right_color_space is None:
        top_right_color_space = horizontal_color_space
    if bottom_left_color_space is None:
        bottom_left_color_space = horizontal_color_space
    if bottom_right_color_space is None:
        bottom_right_color_space = horizontal_color_space

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
