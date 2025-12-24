"""LinesCell implementation for 2D gradient cells defined by lines."""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from ....types.color_types import ColorSpace, is_hue_space, HueDirection
from ....conversions import np_convert
from boundednumbers import BoundType
from ..helpers import LineInterpMethods, interp_transformed_2d_lines
from ..partitions import PerpendicularPartition
from .base import CellBase, CellMode


class LinesCell(CellBase):
    mode: CellMode = CellMode.LINES
    """2D gradient cell defined by lines."""
    
    def __init__(self,
            top_line: np.ndarray,
            bottom_line: np.ndarray,
            per_channel_coords: List[np.ndarray] | np.ndarray,
            color_space: ColorSpace,
            hue_direction_y: Optional[str] = None,
            line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
            hue_direction_x: Optional[str] = None,
            boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
            border_mode: Optional[int] = None,
            border_value: Optional[float] = None,
            *, 
            value: Optional[np.ndarray] = None) -> None:
        self.line_method = line_method
        self.color_space = color_space
        self.hue_direction_y = hue_direction_y
        self.hue_direction_x = hue_direction_x
        self.per_channel_coords = per_channel_coords
        self.boundtypes = boundtypes
        self.border_mode = border_mode
        self.border_value = border_value
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
            border_mode=self.border_mode,
            border_value=self.border_value,
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
            self.border_mode,
            self.border_value,
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
                border_mode=self.border_mode,
                border_value=self.border_value,
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
    
    @classmethod
    def get_top_lines(cls, cells: List[LinesCell]) -> np.ndarray:
        """
        Concatenate top lines from multiple cells for future stacking cell structures.
        
        Args:
            cells: List of LinesCell instances
            
        Returns:
            Concatenated top lines as a single array
        """
        if not cells:
            raise ValueError("Cannot concatenate empty list of cells")
        return np.concatenate([cell.top_line for cell in cells], axis=0)
    
    @classmethod
    def get_bottom_lines(cls, cells: List[LinesCell]) -> np.ndarray:
        """
        Concatenate bottom lines from multiple cells for future stacking cell structures.
        
        Args:
            cells: List of LinesCell instances
            
        Returns:
            Concatenated bottom lines as a single array
        """
        if not cells:
            raise ValueError("Cannot concatenate empty list of cells")
        return np.concatenate([cell.bottom_line for cell in cells], axis=0)
