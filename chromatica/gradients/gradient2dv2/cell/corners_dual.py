"""CornersCellDual implementation for 2D gradient cells with dual color spaces."""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from ....types.color_types import ColorSpace, is_hue_space, HueDirection
from ....conversions import np_convert
from ....types.format_type import FormatType
from boundednumbers import BoundType
from ..partitions import PerpendicularDualPartition
from .base import CellBase, CellMode


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
            hue_direction_y: Optional[str],
            hue_direction_x: Optional[str],
            boundtypes: List[BoundType] | BoundType = BoundType.CLAMP, 
            *, 
            value: Optional[np.ndarray] = None, 
            top_segment_hue_direction_x: Optional[str] = None,
            bottom_segment_hue_direction_x: Optional[str] = None,
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
        self.top_segment = None
        self.bottom_segment = None
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
        """
        Extract x-coordinates from top row for horizontal segment interpolation.
        
        For a 2D cell with coordinates of shape (height, width, 2), where the last
        dimension contains [x, y] coordinates, this property extracts only the
        x-coordinates from the top row (index 0).
        
        Returns:
            If per_channel_coords is a list: List of 1D arrays, each of shape (width,)
            If per_channel_coords is an array: 1D array of shape (width,)
            
        Note:
            This method assumes per_channel_coords has the expected shape:
            - List[np.ndarray]: Each element has shape (height, width, 2)
            - np.ndarray: Shape is (height, width, 2)
            The extraction uses indexing [0, :, 0] which means:
            - [0, :, 0]: First row (top), all columns, first coordinate (x)
        """
        if isinstance(self.per_channel_coords, list):
            # Extract x-coordinates (first element of coordinate pair) from top row
            return [pc[0, :, 0] for pc in self.per_channel_coords]
        else:
            # Shape: (height, width, 2) -> extract top row, all widths, x-coord only
            return self.per_channel_coords[0, :, 0]
    
    @property
    def bottom_per_channel_coords(self) -> List[np.ndarray] | np.ndarray:
        """
        Extract x-coordinates from bottom row for horizontal segment interpolation.
        
        For a 2D cell with coordinates of shape (height, width, 2), where the last
        dimension contains [x, y] coordinates, this property extracts only the
        x-coordinates from the bottom row (index -1).
        
        Returns:
            If per_channel_coords is a list: List of 1D arrays, each of shape (width,)
            If per_channel_coords is an array: 1D array of shape (width,)
            
        Note:
            This method assumes per_channel_coords has the expected shape:
            - List[np.ndarray]: Each element has shape (height, width, 2)
            - np.ndarray: Shape is (height, width, 2)
            The extraction uses indexing [-1, :, 0] which means:
            - [-1, :, 0]: Last row (bottom), all columns, first coordinate (x)
        """
        if isinstance(self.per_channel_coords, list):
            # Extract x-coordinates (first element of coordinate pair) from bottom row
            return [pc[-1, :, 0] for pc in self.per_channel_coords]
        else:
            # Shape: (height, width, 2) -> extract bottom row, all widths, x-coord only
            return self.per_channel_coords[-1, :, 0]
    
    @property
    def per_channel_coords(self) -> List[np.ndarray] | np.ndarray:
        return self._per_channel_coords
    
    @per_channel_coords.setter
    def per_channel_coords(self, value: List[np.ndarray] | np.ndarray):
        self._per_channel_coords = value
        self._invalidate_cached_segments()
        self.invalidate_cache()
    
    def get_top_segment(self):
        """Get or create the top segment for horizontal interpolation."""
        from ...gradient1dv2.segment import get_transformed_segment
        
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
    
    def get_bottom_segment(self):
        """Get or create the bottom segment for horizontal interpolation."""
        from ...gradient1dv2.segment import get_transformed_segment
        
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
        from .lines import LinesCell
        from ..helpers import LineInterpMethods
        
        top_segment = self.get_top_segment().get_value()
        bottom_segment = self.get_bottom_segment().get_value()
        
        # Import factory function to create LinesCell
        from .factory import get_transformed_lines_cell
        
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
            input_format=FormatType.FLOAT,  # Segments return float values
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
        from ...gradient1dv2.helpers.interpolation import (
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
    
    @classmethod
    def get_top_lines(cls, cells: List[CornersCellDual]) -> np.ndarray:
        """
        Concatenate top line (top_left to top_right) from multiple cells for future stacking.
        
        Args:
            cells: List of CornersCellDual instances
            
        Returns:
            Concatenated top corners as array: [tl1, tr1, tl2, tr2, ...]
        """
        if not cells:
            raise ValueError("Cannot concatenate empty list of cells")
        result = []
        for cell in cells:
            result.extend([cell.top_left, cell.top_right])
        return np.array(result)
    
    @classmethod
    def get_bottom_lines(cls, cells: List[CornersCellDual]) -> np.ndarray:
        """
        Concatenate bottom line (bottom_left to bottom_right) from multiple cells for future stacking.
        
        Args:
            cells: List of CornersCellDual instances
            
        Returns:
            Concatenated bottom corners as array: [bl1, br1, bl2, br2, ...]
        """
        if not cells:
            raise ValueError("Cannot concatenate empty list of cells")
        result = []
        for cell in cells:
            result.extend([cell.bottom_left, cell.bottom_right])
        return np.array(result)
