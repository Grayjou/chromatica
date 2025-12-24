"""CornersCell implementation for 2D gradient cells defined by corner colors."""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from ....types.color_types import ColorSpace, is_hue_space
from ....conversions import np_convert
from boundednumbers import BoundType
from ..helpers import interp_transformed_2d_from_corners
from ..partitions import PerpendicularPartition
from .base import CellBase, CellMode


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
            hue_direction_y: Optional[str] = None,
            hue_direction_x: Optional[str] = None,
            boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
            border_mode: Optional[int] = None,
            border_value: Optional[float] = None,
            *, 
            value: Optional[np.ndarray] = None) -> None:
        self.color_space = color_space
        self.hue_direction_y = hue_direction_y
        self.hue_direction_x = hue_direction_x
        self.per_channel_coords = per_channel_coords
        self.boundtypes = boundtypes
        self.border_mode = border_mode
        self.border_value = border_value
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
            border_mode=self.border_mode,
            border_value=self.border_value,
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
            self.border_mode,
            self.border_value,
            value=converted_value,
        )
    
    def partition_slice(self, partition: PerpendicularPartition) -> List[CornersCell]:
        """Slice this cell along the width (perpendicular axis) using the partition."""
        from ...gradient1dv2.helpers.interpolation import (
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
                border_mode=self.border_mode,
                border_value=self.border_value,
                value=sliced_value,
            )
            
            slices.append(cell_slice)
        
        return slices
    
    @classmethod
    def get_top_lines(cls, cells: List[CornersCell]) -> np.ndarray:
        """
        Concatenate top line (top_left to top_right) from multiple cells for future stacking.
        
        Args:
            cells: List of CornersCell instances
            
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
    def get_bottom_lines(cls, cells: List[CornersCell]) -> np.ndarray:
        """
        Concatenate bottom line (bottom_left to bottom_right) from multiple cells for future stacking.
        
        Args:
            cells: List of CornersCell instances
            
        Returns:
            Concatenated bottom corners as array: [bl1, br1, bl2, br2, ...]
        """
        if not cells:
            raise ValueError("Cannot concatenate empty list of cells")
        result = []
        for cell in cells:
            result.extend([cell.bottom_left, cell.bottom_right])
        return np.array(result)
