#chromatica\gradients\gradient2dv2\cell.py
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
from ...types.color_types import HueMode, HueDirection
from ...conversions import np_convert
from .partitions import PerpendicularPartition, PartitionInterval, PerpendicularDualPartition, CellDualPartitionInterval

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
    def convert_to_space(self, color_space: ColorSpace, render_before: bool = False) -> LinesCell:
        if self.color_space == color_space:
            return self

        converted_top = np_convert(self.top_line, self.color_space, color_space, input_type="float", output_type='float')
        converted_bottom = np_convert(self.bottom_line, self.color_space, color_space, input_type="float", output_type='float')

        if render_before:
            self.get_value()
        converted_value = np_convert(self.get_value(), self.color_space, color_space, input_type="float", output_type='float') if self._value is not None else None
        return LinesCell(
            converted_top,
            converted_bottom,
            self.per_channel_coords.copy(),
            color_space,
            self.hue_direction_y,
            self.line_method,
            self.hue_direction_x,
            self.boundtypes,
            value=converted_value,
        )
    def partition_slice(self, partition: PerpendicularPartition, render_before: bool = False) -> List[LinesCell]:
        """Slice this cell along the width (perpendicular axis) using the partition."""
        slices: List[LinesCell] = []
        width = self.per_channel_coords[0].shape[1]
        color_width = self.top_line.shape[0]

        for start, end, partition_interval in partition.intervals():
            # Calculate discrete indices for the slice
            space = partition_interval.color_space
            hue_dir_y = partition_interval.hue_direction_y
            hue_dir_x = partition_interval.hue_direction_x
            start_idx = int(start * width + 0.5)
            end_idx = int(end * width + 0.5)

            # Skip empty slices (can happen due to rounding)
            if start_idx >= end_idx:
                continue
            cstart_idx = int(start * color_width + 0.5)
            cend_idx = int(end * color_width + 0.5)
            # Slice the lines (1D arrays of colors)
            sliced_top = self.top_line[cstart_idx:cend_idx].copy()
            sliced_bottom = self.bottom_line[cstart_idx:cend_idx].copy()
            
            # Slice per-channel coordinates and normalize to [0, 1]
            interval_length = end - start
            if isinstance(self.per_channel_coords, list):

                # Only the x coordinate needs to be adjusted
                sliced_coords = [pc.copy()[:, start_idx:end_idx] for pc in self.per_channel_coords]
                for sliced_pc in sliced_coords:
                    x_coords = sliced_pc[..., 0]
                    sliced_pc[..., 0] = (x_coords - start) / interval_length

            else:
                
                sliced_coords = self.per_channel_coords[:, start_idx:end_idx].copy()
                x_coords = sliced_coords[..., 0]
                sliced_coords[..., 0] = (x_coords - start) / interval_length
                
            
            # Try to reuse cached value if possible
            sliced_value = self._get_sliced_cached_value(
                start_idx, end_idx, space, hue_dir_y
            )

            # Create the sliced cell
            cell_slice = LinesCell(
                top_line=sliced_top,
                bottom_line=sliced_bottom,
                per_channel_coords=sliced_coords,
                color_space=self.color_space,
                hue_direction_y=hue_dir_y or self.hue_direction_y,  
                # If self.mode is hue, and space is non_hue, hue_dir_y will be None, therefore, 
                # conversion will call get_value() which uses self.hue_direction_y, which will raise if it's None.
                line_method=self.line_method,
                hue_direction_x=hue_dir_x or self.hue_direction_x,  # Keep original x direction
                boundtypes=self.boundtypes,
                value=sliced_value,
            )
            #cell_slice.get_value()
            # Convert to target color space if different
            if space != self.color_space:
                cell_slice = cell_slice.convert_to_space(space, render_before=render_before)

            slices.append(cell_slice)
        
        return slices

    def _get_sliced_cached_value(
        self, start_idx: int, end_idx: int, 
        target_space: ColorSpace, target_hue_dir_y: HueDirection
    ) -> Optional[np.ndarray]:
        """Get sliced portion of cached value if conditions allow reuse."""
        if self._value is None:
            return None
        
        # Check if we can reuse cached value
        spaces_match = self.color_space == target_space
        is_hue_space_ = is_hue_space(self.color_space)
        
        if is_hue_space_:
            # For hue spaces, need same hue direction for Y (vertical interpolation)
            hue_directions_match = self.hue_direction_y == target_hue_dir_y
            can_reuse = spaces_match and hue_directions_match
        else:
            # For non-hue spaces, only need same color space
            can_reuse = spaces_match
        
        if can_reuse:
            print("Reusing cached value slice", self.color_space, target_space, is_hue_space_, start_idx, end_idx)
            # Slice the cached value along width (second dimension)
            return self._value[:, start_idx:end_idx]
        
        return None

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
            hue_direction_y: HueMode | None = None,
            hue_direction_x: HueMode | None = None,
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
    def convert_to_space(self, color_space: ColorSpace, render_before: bool = False) -> CornersCell:
        if self.color_space == color_space:
            return self
        if render_before:
            self.get_value()
        converted_top_left = np_convert(self.top_left, self.color_space, color_space, input_type="float", output_type='float')
        converted_top_right = np_convert(self.top_right, self.color_space, color_space, input_type="float", output_type='float')
        converted_bottom_left = np_convert(self.bottom_left, self.color_space, color_space, input_type="float", output_type='float')
        converted_bottom_right = np_convert(self.bottom_right, self.color_space, color_space, input_type="float", output_type='float')
        converted_value = np_convert(self.get_value(), self.color_space, color_space, input_type="float", output_type='float') if self._value is not None else None
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
    def partition_slice(self, partition: PerpendicularPartition) -> List[CornersCell]:
        """Slice this cell along the width (perpendicular axis) using the partition."""
        from ..gradient1dv2.helpers.interpolation import (
            interpolate_transformed_hue_space, 
            interpolate_transformed_non_hue
        )
        
        slices: List[CornersCell] = []
        
        # Get width from per_channel_coords
        if isinstance(self.per_channel_coords, list):
            width = self.per_channel_coords[0].shape[1]
        else:
            width = self.per_channel_coords.shape[1]
        
        # Helper to interpolate along an edge (horizontal interpolation)
        def interpolate_edge(horizontal_index: int, vertical_index: int) -> np.ndarray:
            """Interpolate at a specific position along top or bottom edge."""
            # Determine if we're on top (0) or bottom (height-1) edge
            is_top_edge = vertical_index == 0
            
            if is_top_edge:
                start, end = self.top_left, self.top_right
            else:
                start, end = self.bottom_left, self.bottom_right
            
            # Get local coordinates at this position
            # Note: per_channel_coords is either list of arrays or single array
            if isinstance(self.per_channel_coords, list):
                # List of arrays: [coords_x, coords_y, ...]
                # We need the x-coordinate at this position
                x_coord = self.per_channel_coords[0][vertical_index, horizontal_index][0]

                # FAILED Chromatica\tests\gradients\test_cell.py::test_cell_corners_partition - ValueError: coeffs must have one array per channel
                per_channel_coords = [np.array([x_coord])]*len(start)  # Repeat for each channel
            else:
                # Single array with shape (height, width, 2)
                # Get x-coordinate (first channel)
                x_coord = self.per_channel_coords[vertical_index, horizontal_index, 0]
                per_channel_coords = [np.array([x_coord])]
            
            if is_hue_space(self.color_space):
                return interpolate_transformed_hue_space(
                    start=start,
                    end=end,
                    per_channel_coords=per_channel_coords,
                    hue_direction=self.hue_direction_x,  # Use original x-direction
                    bound_types=self.boundtypes,
                )[0]  # Return single color
            else:
                return interpolate_transformed_non_hue(
                    starts=[start],
                    ends=[end],
                    per_channel_coords=per_channel_coords,
                    bound_types=self.boundtypes,
                )[0]  # Return single color
        
        for start, end, partition_interval in partition.intervals():
            # Calculate slice indices
            space = partition_interval.color_space
            hue_dir_y = partition_interval.hue_direction_y
            hue_dir_x = partition_interval.hue_direction_x
            start_idx = int(start * width + 0.5)
            end_idx = int(end * width + 0.5)
            
            if start_idx >= end_idx:
                continue
            
            # Get new corners by interpolating at boundaries
            # Top left: at start index, top edge (vertical_index=0)
            tl = interpolate_edge(start_idx, vertical_index=0)
            # Top right: at end index-1, top edge
            tr = interpolate_edge(end_idx - 1, vertical_index=0)
            # Bottom left: at start index, bottom edge (vertical_index=height-1)
            bl = interpolate_edge(start_idx, vertical_index=-1)  # Last row
            # Bottom right: at end index-1, bottom edge
            br = interpolate_edge(end_idx - 1, vertical_index=-1)
            
            # Convert to target color space if needed
            converted_tl = np_convert(tl, self.color_space, space, input_type="float", output_type='float')
            converted_tr = np_convert(tr, self.color_space, space, input_type="float", output_type='float')
            converted_bl = np_convert(bl, self.color_space, space, input_type="float", output_type='float')
            converted_br = np_convert(br, self.color_space, space, input_type="float", output_type='float')
            
            # Slice per-channel coordinates and normalize to [0, 1]

            interval_length = end - start
            if isinstance(self.per_channel_coords, list):

                # Only the x coordinate needs to be adjusted
                sliced_coords = [pc.copy()[:, start_idx:end_idx] for pc in self.per_channel_coords]
                for sliced_pc in sliced_coords:
                    x_coords = sliced_pc[..., 0]
                    sliced_pc[..., 0] = (x_coords - start) / interval_length

            else:
                
                sliced_coords = self.per_channel_coords[:, start_idx:end_idx].copy()
                x_coords = sliced_coords[..., 0]
                sliced_coords[..., 0] = (x_coords - start) / interval_length
            
            # Try to reuse cached value (similar to LinesCell but for corners)
            sliced_value = None
            if self._value is not None:
                spaces_match = self.color_space == space
                if not is_hue_space(self.color_space):
                    if spaces_match:
                        sliced_value = self._value[:, start_idx:end_idx, :].copy()
                else:
                    hue_match = self.hue_direction_y == hue_dir_y
                    if spaces_match and hue_match:
                        sliced_value = self._value[:, start_idx:end_idx, :].copy()
            
            # Create the sliced cell
            cell_slice = CornersCell(
                top_left=converted_tl,
                top_right=converted_tr,
                bottom_left=converted_bl,
                bottom_right=converted_br,
                per_channel_coords=sliced_coords,
                color_space=space,
                hue_direction_y=hue_dir_y or self.hue_direction_y,  # Use partition's hue_dir_y
                hue_direction_x=hue_dir_x or self.hue_direction_x,  # Use partition's hue_dir_x
                boundtypes=self.boundtypes,
                value=sliced_value,
            )
            
            slices.append(cell_slice)
        
        return slices

from ..gradient1dv2.segment import get_transformed_segment, TransformedGradientSegment

class CornersCellDual(CellBase):
    mode: CellMode = CellMode.CORNERS_DUAL
    """2D gradient cell defined by dual corner colors."""
    def __init__(self,
            top_left: np.ndarray,
            top_right: np.ndarray,
            bottom_left: np.ndarray,
            bottom_right: np.ndarray,
            per_channel_coords: List[np.ndarray] | np.ndarray,
            horizontal_color_space: ColorSpace,
            vertical_color_space: ColorSpace,
            hue_direction_y: HueMode,
            hue_direction_x: HueMode,
            boundtypes: List[BoundType] | BoundType = BoundType.CLAMP, *, value: Optional[np.ndarray] = None, 
            top_segment_hue_direction_x: Optional[HueMode] = None,
            bottom_segment_hue_direction_x: Optional[HueMode] = None,
            top_segment_color_space: Optional[ColorSpace] = None,
            bottom_segment_color_space: Optional[ColorSpace] = None
            ) -> None:

        self.horizontal_color_space = horizontal_color_space
        self.vertical_color_space = vertical_color_space
        self.hue_direction_y = hue_direction_y
        self.hue_direction_x = hue_direction_x
        self._per_channel_coords = per_channel_coords
        self.boundtypes = boundtypes
        self._top_left = top_left
        self._top_right = top_right
        self._bottom_left = bottom_left
        self._bottom_right = bottom_right
        self._value = value
        self.top_segment: TransformedGradientSegment| None = None
        self.bottom_segment: TransformedGradientSegment| None = None
        self.top_segment_hue_direction_x = top_segment_hue_direction_x or hue_direction_x
        self.bottom_segment_hue_direction_x = bottom_segment_hue_direction_x or hue_direction_x
        self.top_segment_color_space = top_segment_color_space or horizontal_color_space
        self.bottom_segment_color_space = bottom_segment_color_space or horizontal_color_space
    def _invalidate_cached_segments(self):
        self.top_segment = None
        self.bottom_segment = None
    @property
    def top_left(self) -> np.ndarray:
        return self._top_left
    @top_left.setter
    def top_left(self, value: np.ndarray):
        self._top_left = value
        self._invalidate_cached_segments()
        self.invalidate_cache()
    @property
    def top_right(self) -> np.ndarray:
        return self._top_right
    @top_right.setter
    def top_right(self, value: np.ndarray):
        self._top_right = value
        self._invalidate_cached_segments()
        self.invalidate_cache()
    @property
    def bottom_left(self) -> np.ndarray:
        return self._bottom_left
    @bottom_left.setter
    def bottom_left(self, value: np.ndarray):
        self._bottom_left = value
        self._invalidate_cached_segments()
        self.invalidate_cache()
    @property
    def bottom_right(self) -> np.ndarray:
        return self._bottom_right
    @bottom_right.setter
    def bottom_right(self, value: np.ndarray):
        self._bottom_right = value
        self._invalidate_cached_segments()
        self.invalidate_cache()
    @property
    def top_per_channel_coords(self) -> List[np.ndarray] | np.ndarray:
        if isinstance(self.per_channel_coords, list):
            return [pc[0:1, :, :] for pc in self.per_channel_coords]
        else:
            return self.per_channel_coords[0:1, :, :]
    @property
    def bottom_per_channel_coords(self) -> List[np.ndarray] | np.ndarray:
        if isinstance(self.per_channel_coords, list):
            return [pc[-1:, :, :] for pc in self.per_channel_coords]
        else:
            return self.per_channel_coords[-1:, :, :]
    @property
    def per_channel_coords(self) -> List[np.ndarray] | np.ndarray:
        return self._per_channel_coords
    @per_channel_coords.setter
    def per_channel_coords(self, value: List[np.ndarray] | np.ndarray):
        self._per_channel_coords = value
        self._invalidate_cached_segments()
        self.invalidate_cache()
    def get_top_segment(self) -> TransformedGradientSegment:

        if self.top_segment is None:
            self.top_segment = get_transformed_segment(
                start_color=self.top_left,
                end_color=self.top_right,
                per_channel_coords=self.top_per_channel_coords,
                color_space=self.top_segment_color_space,
                hue_direction=self.top_segment_hue_direction_x,
                bound_types=self.boundtypes,
                format_type=FormatType.FLOAT,
                start_color_space=self.top_segment_color_space,
                end_color_space=self.top_segment_color_space,
            )
        return self.top_segment
    def get_bottom_segment(self) -> TransformedGradientSegment:

        if self.bottom_segment is None:
            self.bottom_segment = get_transformed_segment(
                start_color=self.bottom_left,
                end_color=self.bottom_right,
                per_channel_coords=self.bottom_per_channel_coords,
                color_space=self.bottom_segment_color_space,
                hue_direction=self.bottom_segment_hue_direction_x,
                bound_types=self.boundtypes,
                format_type=FormatType.FLOAT,
                start_color_space=self.bottom_segment_color_space,
                end_color_space=self.bottom_segment_color_space,
            )
        return self.bottom_segment
    def invalidate_cache(self):
        return super().invalidate_cache()
    def _render_value(self):
        top_segment = self.get_top_segment().get_value()
        bottom_segment = self.get_bottom_segment().get_value()
        # Construct a lines cell and interpolate
        lines_cell = get_transformed_lines_cell(
            top_line=top_segment,
            bottom_line=bottom_segment,
            per_channel_coords=self.per_channel_coords,
            color_space=self.vertical_color_space,
            top_line_color_space=self.top_segment_color_space,
            bottom_line_color_space=self.bottom_segment_color_space,
            hue_direction_y=self.hue_direction_y,
            hue_direction_x=self.hue_direction_x,
            line_method=LineInterpMethods.LINES_CONTINUOUS,
            boundtypes=self.boundtypes,
        )
        return lines_cell.get_value()
    @property
    def color_space(self) -> ColorSpace:
        """Return the output color space (vertical space, as that's the final render space)."""
        return self.vertical_color_space

    def convert_to_space(self, color_space: ColorSpace, render_before: bool = False) -> CornersCellDual:
        """Convert to a unified color space (both horizontal and vertical become the same)."""
        if (self.horizontal_color_space == color_space and 
            self.vertical_color_space == color_space and
            self.top_segment_color_space == color_space and
            self.bottom_segment_color_space == color_space):
            return self
        
        if render_before:
            self.get_value()
        
        # Convert corners from horizontal_color_space
        converted_top_left = np_convert(
            self.top_left, self.horizontal_color_space, color_space, 
            input_type="float", output_type='float'
        )
        converted_top_right = np_convert(
            self.top_right, self.horizontal_color_space, color_space, 
            input_type="float", output_type='float'
        )
        converted_bottom_left = np_convert(
            self.bottom_left, self.horizontal_color_space, color_space, 
            input_type="float", output_type='float'
        )
        converted_bottom_right = np_convert(
            self.bottom_right, self.horizontal_color_space, color_space, 
            input_type="float", output_type='float'
        )
        
        # Convert cached value from vertical_color_space if exists
        converted_value = None
        if self._value is not None:
            converted_value = np_convert(
                self._value, self.vertical_color_space, color_space, 
                input_type="float", output_type='float'
            )
        
        # Copy per_channel_coords
        if isinstance(self.per_channel_coords, list):
            copied_coords = [pc.copy() for pc in self.per_channel_coords]
        else:
            copied_coords = self.per_channel_coords.copy()
        
        return CornersCellDual(
            top_left=converted_top_left,
            top_right=converted_top_right,
            bottom_left=converted_bottom_left,
            bottom_right=converted_bottom_right,
            per_channel_coords=copied_coords,
            horizontal_color_space=color_space,
            vertical_color_space=color_space,
            hue_direction_y=self.hue_direction_y,
            hue_direction_x=self.hue_direction_x,
            boundtypes=self.boundtypes,
            value=converted_value,
            top_segment_hue_direction_x=self.top_segment_hue_direction_x,
            bottom_segment_hue_direction_x=self.bottom_segment_hue_direction_x,
            top_segment_color_space=color_space,
            bottom_segment_color_space=color_space,
        )

    def convert_to_spaces(
        self, 
        horizontal_color_space: ColorSpace, 
        vertical_color_space: ColorSpace,
        top_segment_color_space: Optional[ColorSpace] = None,
        bottom_segment_color_space: Optional[ColorSpace] = None,
        render_before: bool = False
    ) -> CornersCellDual:
        """Convert to specific horizontal and vertical color spaces."""
        top_seg_space = top_segment_color_space or horizontal_color_space
        bottom_seg_space = bottom_segment_color_space or horizontal_color_space
        
        # Check if already in target spaces
        if (self.horizontal_color_space == horizontal_color_space and 
            self.vertical_color_space == vertical_color_space and
            self.top_segment_color_space == top_seg_space and
            self.bottom_segment_color_space == bottom_seg_space):
            return self
        
        if render_before:
            self.get_value()
        
        # Convert corners from horizontal_color_space to new horizontal space
        converted_top_left = np_convert(
            self.top_left, self.horizontal_color_space, horizontal_color_space, 
            input_type="float", output_type='float'
        )
        converted_top_right = np_convert(
            self.top_right, self.horizontal_color_space, horizontal_color_space, 
            input_type="float", output_type='float'
        )
        converted_bottom_left = np_convert(
            self.bottom_left, self.horizontal_color_space, horizontal_color_space, 
            input_type="float", output_type='float'
        )
        converted_bottom_right = np_convert(
            self.bottom_right, self.horizontal_color_space, horizontal_color_space, 
            input_type="float", output_type='float'
        )
        
        # Convert cached value from vertical_color_space if exists
        converted_value = None
        if self._value is not None:
            converted_value = np_convert(
                self._value, self.vertical_color_space, vertical_color_space, 
                input_type="float", output_type='float'
            )
        
        # Copy per_channel_coords
        if isinstance(self.per_channel_coords, list):
            copied_coords = [pc.copy() for pc in self.per_channel_coords]
        else:
            copied_coords = self.per_channel_coords.copy()
        
        return CornersCellDual(
            top_left=converted_top_left,
            top_right=converted_top_right,
            bottom_left=converted_bottom_left,
            bottom_right=converted_bottom_right,
            per_channel_coords=copied_coords,
            horizontal_color_space=horizontal_color_space,
            vertical_color_space=vertical_color_space,
            hue_direction_y=self.hue_direction_y,
            hue_direction_x=self.hue_direction_x,
            boundtypes=self.boundtypes,
            value=converted_value,
            top_segment_hue_direction_x=self.top_segment_hue_direction_x,
            bottom_segment_hue_direction_x=self.bottom_segment_hue_direction_x,
            top_segment_color_space=top_seg_space,
            bottom_segment_color_space=bottom_seg_space,
        )

    def partition_slice(
        self, 
        partition: PerpendicularDualPartition, 
        render_before: bool = False
    ) -> List[CornersCellDual]:
        """Slice this cell along the width (perpendicular axis) using the dual partition."""
        from ..gradient1dv2.helpers.interpolation import (
            interpolate_transformed_hue_space, 
            interpolate_transformed_non_hue
        )
        
        slices: List[CornersCellDual] = []
        
        # Get width from per_channel_coords
        if isinstance(self.per_channel_coords, list):
            width = self.per_channel_coords[0].shape[1]
        else:
            width = self.per_channel_coords.shape[1]
        
        def interpolate_edge(horizontal_index: int, is_top_edge: bool) -> np.ndarray:
            """Interpolate at a specific position along top or bottom edge."""
            if is_top_edge:
                start, end = self.top_left, self.top_right
                segment_color_space = self.top_segment_color_space
                hue_dir = self.top_segment_hue_direction_x
            else:
                start, end = self.bottom_left, self.bottom_right
                segment_color_space = self.bottom_segment_color_space
                hue_dir = self.bottom_segment_hue_direction_x
            
            # Get local coordinates at this position
            vertical_index = 0 if is_top_edge else -1
            if isinstance(self.per_channel_coords, list):
                x_coord = self.per_channel_coords[0][vertical_index, horizontal_index, 0]
            else:
                x_coord = self.per_channel_coords[vertical_index, horizontal_index, 0]
            
            per_channel_coords = [np.array([x_coord])] * len(start)
            
            if is_hue_space(segment_color_space):
                return interpolate_transformed_hue_space(
                    start=start,
                    end=end,
                    per_channel_coords=per_channel_coords,
                    hue_direction=hue_dir,
                    bound_types=self.boundtypes,
                )[0]
            else:
                return interpolate_transformed_non_hue(
                    starts=[start],
                    ends=[end],
                    per_channel_coords=per_channel_coords,
                    bound_types=self.boundtypes,
                )[0]
        
        if render_before:
            self.get_value()
        
        for start, end, partition_interval in partition.intervals():
            start_idx = int(start * width + 0.5)
            end_idx = int(end * width + 0.5)
            
            if start_idx >= end_idx:
                continue
            
            # Extract partition settings
            h_space = partition_interval.horizontal_color_space
            v_space = partition_interval.vertical_color_space
            hue_dir_y = partition_interval.hue_direction_y
            hue_dir_x = partition_interval.hue_direction_x
            top_seg_space = partition_interval.top_segment_color_space
            bottom_seg_space = partition_interval.bottom_segment_color_space
            top_seg_hue_x = partition_interval.top_segment_hue_direction_x
            bottom_seg_hue_x = partition_interval.bottom_segment_hue_direction_x
            
            # Get new corners by interpolating at slice boundaries
            tl = interpolate_edge(start_idx, is_top_edge=True)
            tr = interpolate_edge(end_idx - 1, is_top_edge=True)
            bl = interpolate_edge(start_idx, is_top_edge=False)
            br = interpolate_edge(end_idx - 1, is_top_edge=False)
            
            # Convert corners to target horizontal color space
            converted_tl = np_convert(
                tl, self.horizontal_color_space, h_space, 
                input_type="float", output_type='float'
            )
            converted_tr = np_convert(
                tr, self.horizontal_color_space, h_space, 
                input_type="float", output_type='float'
            )
            converted_bl = np_convert(
                bl, self.horizontal_color_space, h_space, 
                input_type="float", output_type='float'
            )
            converted_br = np_convert(
                br, self.horizontal_color_space, h_space, 
                input_type="float", output_type='float'
            )
            
            # Slice and normalize per_channel_coords
            interval_length = end - start
            if isinstance(self.per_channel_coords, list):
                sliced_coords = [pc[:, start_idx:end_idx].copy() for pc in self.per_channel_coords]
                for sliced_pc in sliced_coords:
                    x_coords = sliced_pc[..., 0]
                    sliced_pc[..., 0] = (x_coords - start) / interval_length
            else:
                sliced_coords = self.per_channel_coords[:, start_idx:end_idx].copy()
                x_coords = sliced_coords[..., 0]
                sliced_coords[..., 0] = (x_coords - start) / interval_length
            
            # Try to reuse cached value
            sliced_value = self._get_sliced_cached_value(
                start_idx, end_idx, v_space, hue_dir_y
            )
            
            cell_slice = CornersCellDual(
                top_left=converted_tl,
                top_right=converted_tr,
                bottom_left=converted_bl,
                bottom_right=converted_br,
                per_channel_coords=sliced_coords,
                horizontal_color_space=h_space,
                vertical_color_space=v_space,
                hue_direction_y=hue_dir_y or self.hue_direction_y,
                hue_direction_x=hue_dir_x or self.hue_direction_x,
                boundtypes=self.boundtypes,
                value=sliced_value,
                top_segment_hue_direction_x=top_seg_hue_x or self.top_segment_hue_direction_x,
                bottom_segment_hue_direction_x=bottom_seg_hue_x or self.bottom_segment_hue_direction_x,
                top_segment_color_space=top_seg_space or self.top_segment_color_space,
                bottom_segment_color_space=bottom_seg_space or self.bottom_segment_color_space,
            )
            
            slices.append(cell_slice)
        
        return slices

    def _get_sliced_cached_value(
        self, 
        start_idx: int, 
        end_idx: int,
        target_space: ColorSpace, 
        target_hue_dir_y: HueDirection
    ) -> Optional[np.ndarray]:
        """Get sliced portion of cached value if conditions allow reuse."""
        if self._value is None:
            return None
        
        # Check against vertical_color_space since that's what _value is rendered in
        spaces_match = self.vertical_color_space == target_space
        
        if is_hue_space(self.vertical_color_space):
            hue_match = self.hue_direction_y == target_hue_dir_y
            can_reuse = spaces_match and hue_match
        else:
            can_reuse = spaces_match
        
        if can_reuse:
            return self._value[:, start_idx:end_idx].copy()
        
        return None


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
        hue_direction_y: HueMode,
        hue_direction_x: HueMode,
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[dict] = None,
        boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
        top_segment_hue_direction_x: HueMode | None = None,
        bottom_segment_hue_direction_x: HueMode | None = None,
        top_segment_color_space: ColorSpace | None = None,
        bottom_segment_color_space: ColorSpace | None = None
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