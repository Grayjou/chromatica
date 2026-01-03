"""Factory functions for creating gradient cells with proper color space conversion."""
#chromatica\gradients\gradient2dv2\cell\factory.py
from __future__ import annotations
from typing import List, Optional
import numpy as np
from ....types.color_types import ColorMode, HueDirection, is_hue_space
from ....types.format_type import FormatType
from ....types.transform_types import PerChannelCoords, TransformOutput
from ....utils.color_utils import convert_to_space_float, is_hue_color_grayscale
from ....utils.default import value_or_default
from boundednumbers import BoundType
from ..helpers import LineInterpMethods, apply_per_channel_transforms_2d
from .lines import LinesCell
from .corners import CornersCell
from .corners_dual import CornersCellDual


def get_transformed_lines_cell(
        top_line: np.ndarray,
        bottom_line: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,
        color_mode: ColorMode,
        top_line_color_mode: Optional[ColorMode] = None,
        bottom_line_color_mode: Optional[ColorMode] = None,
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
        color_mode: Target color space for the cell
        top_line_color_mode: Color space of top line. Defaults to color_mode if not specified.
        bottom_line_color_mode: Color space of bottom line. Defaults to color_mode if not specified.
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
    # Default line color spaces to the target color_mode if not specified
    if top_line_color_mode is None:
        top_line_color_mode = color_mode
    if bottom_line_color_mode is None:
        bottom_line_color_mode = color_mode
    
    top_line_converted = convert_to_space_float(
        top_line, top_line_color_mode, input_format, color_mode
    ).value
    bottom_line_converted = convert_to_space_float(
        bottom_line, bottom_line_color_mode, input_format, color_mode
    ).value
    if per_channel_transforms is not None:
        per_channel_coords = apply_per_channel_transforms_2d(
            coords=per_channel_coords,
            per_channel_transforms=per_channel_transforms,
            num_channels=len(color_mode),
            transform_output=TransformOutput.ARRAY3D, #Returns Array3D if no transforms applied
        )
    return LinesCell(
        top_line=top_line_converted,
        bottom_line=bottom_line_converted,
        per_channel_coords=per_channel_coords,
        color_mode=color_mode,
        hue_direction_y=hue_direction_y,
        hue_direction_x=hue_direction_x,
        line_method=line_method,
        boundtypes=boundtypes,
        border_mode=border_mode,
        border_value=border_value,
    )
    

def _get_corners_color_modes(
        color_mode: ColorMode,
        top_left_color_mode: Optional[ColorMode] = None,
        top_right_color_mode: Optional[ColorMode] = None,
        bottom_left_color_mode: Optional[ColorMode] = None,
        bottom_right_color_mode: Optional[ColorMode] = None,
    ) -> tuple[ColorMode, ColorMode, ColorMode, ColorMode]:
    """Determine the color spaces for each corner."""
    tl_space = top_left_color_mode or color_mode
    tr_space = top_right_color_mode or color_mode
    bl_space = bottom_left_color_mode or color_mode
    br_space = bottom_right_color_mode or color_mode
    return tl_space, tr_space, bl_space, br_space

def get_transformed_corners_cell(
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,
        color_mode: ColorMode,
        top_left_color_mode: Optional[ColorMode] = None,
        top_right_color_mode: Optional[ColorMode] = None,
        bottom_left_color_mode: Optional[ColorMode] = None,
        bottom_right_color_mode: Optional[ColorMode] = None,
        hue_direction_y: Optional[str] = None,
        hue_direction_x: Optional[str] = None,
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[dict] = None,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        ) -> CornersCell:
    """Create a transformed CornersCell with proper color space conversion."""
    top_left_color_mode, top_right_color_mode, bottom_left_color_mode, bottom_right_color_mode = _get_corners_color_modes(
        color_mode,
        top_left_color_mode,
        top_right_color_mode,
        bottom_left_color_mode,
        bottom_right_color_mode,
    )
    top_left_converted = convert_to_space_float(
        top_left, top_left_color_mode, input_format, color_mode
    ).value
    top_right_converted = convert_to_space_float(
        top_right, top_right_color_mode, input_format, color_mode
    ).value
    bottom_left_converted = convert_to_space_float(
        bottom_left, bottom_left_color_mode, input_format, color_mode
    ).value
    bottom_right_converted = convert_to_space_float(
        bottom_right, bottom_right_color_mode, input_format, color_mode
    ).value

    if per_channel_transforms is not None:
        per_channel_coords = apply_per_channel_transforms_2d(
            coords=per_channel_coords,
            per_channel_transforms=per_channel_transforms,
            num_channels=len(color_mode),
            transform_output=TransformOutput.ARRAY3D, #Returns Array3D if no transforms applied
        )
    return CornersCell(
        top_left=top_left_converted,
        top_right=top_right_converted,
        bottom_left=bottom_left_converted,
        bottom_right=bottom_right_converted,
        per_channel_coords=per_channel_coords,
        color_mode=color_mode,
        hue_direction_y=hue_direction_y,
        hue_direction_x=hue_direction_x,
        boundtypes=boundtypes,
        border_mode=border_mode,
        border_value=border_value,
    )

def _determine_grayscale_hue(
        color: np.ndarray,
        ColorMode: ColorMode,
    ) -> Optional[float]:
    if not is_hue_space(ColorMode):
        return None
    if is_hue_color_grayscale(color):
        return color[0]
    


    

def get_transformed_corners_cell_dual(
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        per_channel_coords: List[np.ndarray] | np.ndarray,

        vertical_color_mode: ColorMode,
        horizontal_color_mode: Optional[ColorMode] = None,
        top_left_color_mode: Optional[ColorMode] = None,
        top_right_color_mode: Optional[ColorMode] = None,
        bottom_left_color_mode: Optional[ColorMode] = None,
        bottom_right_color_mode: Optional[ColorMode] = None,
        hue_direction_y: Optional[HueDirection] = None,
        hue_direction_x: Optional[HueDirection] = None,
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[dict] = None,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        top_segment_hue_direction_x: Optional[str] = None,
        bottom_segment_hue_direction_x: Optional[str] = None,
        top_segment_color_mode: Optional[ColorMode] = None,
        bottom_segment_color_mode: Optional[ColorMode] = None,
        top_left_grayscale_hue: Optional[float] = None,
        top_right_grayscale_hue: Optional[float] = None,
        bottom_left_grayscale_hue: Optional[float] = None,
        bottom_right_grayscale_hue: Optional[float] = None,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        ) -> CornersCellDual:
    """Create a transformed CornersCellDual with proper color space conversion.
    
    Args:
        top_left: Top-left corner color
        top_right: Top-right corner color
        bottom_left: Bottom-left corner color
        bottom_right: Bottom-right corner color
        per_channel_coords: Per-channel coordinate arrays
        horizontal_color_mode: Horizontal color space for the cell
        vertical_color_mode: Vertical color space for the cell
        top_left_color_mode: Color space of top-left corner. Defaults to horizontal_color_mode if not specified.
        top_right_color_mode: Color space of top-right corner. Defaults to horizontal_color_mode if not specified.
        bottom_left_color_mode: Color space of bottom-left corner. Defaults to horizontal_color_mode if not specified.
        bottom_right_color_mode: Color space of bottom-right corner. Defaults to horizontal_color_mode if not specified.
        hue_direction_y: Hue direction for vertical interpolation
        hue_direction_x: Hue direction for horizontal interpolation
        input_format: Format of input color data
        per_channel_transforms: Optional per-channel transformations
        boundtypes: Boundary types for coordinate handling
        top_segment_hue_direction_x: Hue direction for top segment horizontal interpolation
        bottom_segment_hue_direction_x: Hue direction for bottom segment horizontal interpolation
        top_segment_color_mode: Color space for top segment
        bottom_segment_color_mode: Color space for bottom segment
        
    Returns:
        CornersCellDual instance with converted colors
    """

    if horizontal_color_mode is None:
        if any([space is None] for space in [top_segment_color_mode, bottom_segment_color_mode]):
            raise ValueError("Either horizontal_color_mode or both top_segment_color_mode and bottom_segment_color_mode must be provided.")
    top_segment_color_mode = value_or_default(top_segment_color_mode, horizontal_color_mode)
    bottom_segment_color_mode = value_or_default(bottom_segment_color_mode, horizontal_color_mode)
    if top_left_color_mode is None:
        if top_segment_color_mode is not None:
            top_left_color_mode = top_segment_color_mode
        else:
            top_left_color_mode = horizontal_color_mode
    if top_right_color_mode is None:
        if top_segment_color_mode is not None:
            top_right_color_mode = top_segment_color_mode
        else:
            top_right_color_mode = horizontal_color_mode
    if bottom_left_color_mode is None:
        if bottom_segment_color_mode is not None:
            bottom_left_color_mode = bottom_segment_color_mode
        else:
            bottom_left_color_mode = horizontal_color_mode
    if bottom_right_color_mode is None:
        if bottom_segment_color_mode is not None:
            bottom_right_color_mode = bottom_segment_color_mode
        else:
            bottom_right_color_mode = horizontal_color_mode

    top_left_converted = convert_to_space_float(
        top_left, top_left_color_mode, input_format, top_segment_color_mode
    ).value
    top_right_converted = convert_to_space_float(
        top_right, top_right_color_mode, input_format, top_segment_color_mode
    ).value
    bottom_left_converted = convert_to_space_float(
        bottom_left, bottom_left_color_mode, input_format, bottom_segment_color_mode
    ).value
    bottom_right_converted = convert_to_space_float(
        bottom_right, bottom_right_color_mode, input_format, bottom_segment_color_mode
    ).value
    if per_channel_transforms is not None:
        per_channel_coords = apply_per_channel_transforms_2d(
            coords=per_channel_coords,
            per_channel_transforms=per_channel_transforms,
            num_channels=len(horizontal_color_mode),
            transform_output=TransformOutput.ARRAY3D, #Returns Array3D if no transforms applied
        )
    top_left_grayscale_hue = value_or_default(top_left_grayscale_hue, _determine_grayscale_hue(top_left_converted, top_segment_color_mode))
    top_right_grayscale_hue = value_or_default(top_right_grayscale_hue, _determine_grayscale_hue(top_right_converted, top_segment_color_mode))
    bottom_left_grayscale_hue = value_or_default(bottom_left_grayscale_hue, _determine_grayscale_hue(bottom_left_converted, bottom_segment_color_mode))
    bottom_right_grayscale_hue = value_or_default(bottom_right_grayscale_hue, _determine_grayscale_hue(bottom_right_converted, bottom_segment_color_mode))
    return CornersCellDual(
        top_left=top_left_converted,
        top_right=top_right_converted,
        bottom_left=bottom_left_converted,
        bottom_right=bottom_right_converted,
        per_channel_coords=per_channel_coords,
        horizontal_color_mode=horizontal_color_mode,
        vertical_color_mode=vertical_color_mode,
        hue_direction_y=hue_direction_y,
        hue_direction_x=hue_direction_x,
        boundtypes=boundtypes,
        top_segment_hue_direction_x=top_segment_hue_direction_x,
        bottom_segment_hue_direction_x=bottom_segment_hue_direction_x,
        top_segment_color_mode=top_segment_color_mode,
        bottom_segment_color_mode=bottom_segment_color_mode,
        top_left_grayscale_hue=top_left_grayscale_hue,
        top_right_grayscale_hue=top_right_grayscale_hue,
        bottom_left_grayscale_hue=bottom_left_grayscale_hue,
        bottom_right_grayscale_hue=bottom_right_grayscale_hue,
        border_mode=border_mode,
        border_value=border_value,
    )

