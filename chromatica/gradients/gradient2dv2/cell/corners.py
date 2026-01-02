# chromatica\gradients\gradient2dv2\cell\corners.py
from __future__ import annotations
from typing import List, Optional
import numpy as np
from ....types.color_types import ColorSpaces
from ....conversions import np_convert
from boundednumbers import BoundType
from ..helpers import interp_transformed_2d_from_corners
from ..partitions import PerpendicularPartition
from .base import CellMode
from .corners_base import CornersBase
from ._cell_coords import get_shape, extract_edge, extract_point, lerp_point, slice_and_renormalize, slice_coords
from ._cell_cache import get_reusable_slice
from ....utils.num_utils import is_close_to_int
from ...gradient1dv2.segment import get_transformed_segment
from unitfield import flat_1d_upbm


class CornersCell(CornersBase):
    mode: CellMode = CellMode.CORNERS
    
    def __init__(self,
            top_left: np.ndarray,
            top_right: np.ndarray,
            bottom_left: np.ndarray,
            bottom_right: np.ndarray,
            per_channel_coords: List[np.ndarray] | np.ndarray,
            color_space: ColorSpaces,
            hue_direction_y: Optional[str] = None,
            hue_direction_x: Optional[str] = None,
            boundtypes: List[BoundType] | BoundType = BoundType.CLAMP,
            border_mode: Optional[int] = None,
            border_value: Optional[float] = None,
            *, 
            value: Optional[np.ndarray] = None) -> None:
        
        super().__init__(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
            per_channel_coords=per_channel_coords,
            color_space=color_space,
            hue_direction_y=hue_direction_y,
            hue_direction_x=hue_direction_x,
            boundtypes=boundtypes,
            border_mode=border_mode,
            border_value=border_value,
            value=value
        )
    
    # === Core interpolation ===
    def _interpolate_at_coords(self, coords_list: List[np.ndarray]) -> np.ndarray:
        """Core 2D interpolation at specific coordinates."""
        return interp_transformed_2d_from_corners(
            top_left=self.top_left,
            top_right=self.top_right,
            bottom_left=self.bottom_left,
            bottom_right=self.bottom_right,
            transformed=coords_list,
            color_space=self.color_space,
            huemode_x=self.hue_direction_x,
            huemode_y=self.hue_direction_y,
            bound_types=self.boundtypes,
            border_mode=self.border_mode,
            border_value=self.border_value,
        )[0, 0, :]
    
    # === Public edge interpolation methods ===
    def interpolate_edge_continuous(self, horizontal_pos: float, vertical_idx: int) -> np.ndarray:
        """Continuous interpolation along top (0) or bottom (height-1) edge."""
        # Fast path: exact pixel in cache
        if self._value is not None:
            exact_idx = horizontal_pos * (self.width - 1)
            if is_close_to_int(exact_idx):
                idx = int(round(exact_idx))
                return self._value[vertical_idx, idx, :].copy()
        
        # Normal path
        exact_idx = horizontal_pos * (self.width - 1)
        edge = self.top_edge_coords if vertical_idx == 0 else self.bottom_edge_coords
        
        if is_close_to_int(exact_idx):
            coords = extract_point(edge, int(round(exact_idx)))
        else:
            coords = lerp_point(edge, exact_idx)
        
        return self._interpolate_at_coords(coords)
    
    def index_interpolate_edge_discrete(self, horizontal_index: int, vertical_index: int) -> np.ndarray:
        """Discrete interpolation at specific pixel coordinates."""
        # Fast path: direct cache access
        if self._value is not None:
            if 0 <= horizontal_index < self.width and 0 <= vertical_index < self.height:
                return self._value[vertical_index, horizontal_index, :].copy()
        
        # Normalize negative indices
        if vertical_index < 0:
            vertical_index += self.height
        if horizontal_index < 0:
            horizontal_index += self.width
        
        edge = self.top_edge_coords if vertical_index == 0 else self.bottom_edge_coords
        coords = extract_point(edge, horizontal_index)
        return self._interpolate_at_coords(coords)
    
    def interpolate_edge(self, horizontal_pos: float, is_top_edge: bool) -> np.ndarray:
        """Convenience method matching LinesCell interface."""
        vertical_idx = 0 if is_top_edge else self.height - 1
        return self.interpolate_edge_continuous(horizontal_pos, vertical_idx)
    



    # === Segment Methods ===
    def get_top_segment_untransformed(self) -> np.ndarray:
        """Get or create the top segment in uniform coordinates."""
        # Fast path: extract from cached value if available
        if self._value is not None and self._top_segment is None:
            self._top_segment = self._value[0:1, :, :]
        
        if self._top_segment is not None:
            return self._top_segment
        
        # Create segment from corners using UNIFORM coordinates
        uniform_coords = [flat_1d_upbm(self.width)]
        
        segment = get_transformed_segment(
            already_converted_start_color=self.top_left,
            already_converted_end_color=self.top_right,
            per_channel_coords=uniform_coords,
            color_space=self.color_space,
            hue_direction=self.hue_direction_x,
            bound_types=self.boundtypes,
            homogeneous_per_channel_coords=True,
        )
        
        # Reshape from (width, channels) to (1, width, channels)
        self._top_segment = segment.get_value().reshape(1, self.width, -1)
        return self._top_segment
    
    def get_bottom_segment_untransformed(self) -> np.ndarray:
        """Get or create the bottom segment in uniform coordinates."""
        # Fast path: extract from cached value if available
        if self._value is not None and self._bottom_segment is None:
            self._bottom_segment = self._value[-1:, :, :]
        
        if self._bottom_segment is not None:
            return self._bottom_segment
        
        # Create segment from corners using UNIFORM coordinates
        uniform_coords = [flat_1d_upbm(self.width)]
        
        segment = get_transformed_segment(
            already_converted_start_color=self.bottom_left,
            already_converted_end_color=self.bottom_right,
            per_channel_coords=uniform_coords,
            color_space=self.color_space,
            hue_direction=self.hue_direction_x,
            bound_types=self.boundtypes,
            homogeneous_per_channel_coords=True,
        )
        
        # Reshape from (width, channels) to (1, width, channels)
        self._bottom_segment = segment.get_value().reshape(1, self.width, -1)
        return self._bottom_segment
    
    # === Rendering ===
    def _render_value(self):
        return interp_transformed_2d_from_corners(
            top_left=self.top_left,
            top_right=self.top_right,
            bottom_left=self.bottom_left,
            bottom_right=self.bottom_right,
            transformed=self.per_channel_coords,
            color_space=self.color_space,
            huemode_x=self.hue_direction_x,
            huemode_y=self.hue_direction_y,
            bound_types=self.boundtypes,
            border_mode=self.border_mode,
            border_value=self.border_value,
        )
    
    # === Color space conversion ===
    def convert_to_space(self, color_space: ColorSpaces, render_before: bool = False) -> CornersCell:
        if self.color_space == color_space:
            return self
        
        if render_before:
            self.get_value()
        
        # Convert all four corners
        converted_corners = [
            np_convert(self.top_left, self.color_space, color_space, input_type="float", output_type='float'),
            np_convert(self.top_right, self.color_space, color_space, input_type="float", output_type='float'),
            np_convert(self.bottom_left, self.color_space, color_space, input_type="float", output_type='float'),
            np_convert(self.bottom_right, self.color_space, color_space, input_type="float", output_type='float'),
        ]
        
        converted_value = None
        if self._value is not None:
            converted_value = np_convert(
                self._value, self.color_space, color_space,
                input_type="float", output_type='float'
            )
        
        return CornersCell(
            top_left=converted_corners[0],
            top_right=converted_corners[1],
            bottom_left=converted_corners[2],
            bottom_right=converted_corners[3],
            per_channel_coords=self.per_channel_coords,
            color_space=color_space,
            hue_direction_y=self.hue_direction_y,
            hue_direction_x=self.hue_direction_x,
            boundtypes=self.boundtypes,
            border_mode=self.border_mode,
            border_value=self.border_value,
            value=converted_value,
        )
    
    # === Utility methods ===
    def copy_with(self, **kwargs):
        """Create a copy with overridden values."""
        defaults = {
            'top_left': self.top_left,
            'top_right': self.top_right,
            'bottom_left': self.bottom_left,
            'bottom_right': self.bottom_right,
            'per_channel_coords': self.per_channel_coords,
            'color_space': self.color_space,
            'hue_direction_y': self.hue_direction_y,
            'hue_direction_x': self.hue_direction_x,
            'boundtypes': self.boundtypes,
            'border_mode': self.border_mode,
            'border_value': self.border_value,
        }
        defaults.update(kwargs)
        return CornersCell(**defaults)