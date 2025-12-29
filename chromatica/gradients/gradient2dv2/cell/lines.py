# chromatica/gradients/gradient2dv2/cell/lines.py
"""LinesCell implementation for 2D gradient cells defined by lines."""

from __future__ import annotations
from typing import List, Optional, Union
import numpy as np
from ....types.color_types import ColorSpaces
from ....conversions import np_convert
from boundednumbers import BoundType
from ..helpers import LineInterpMethods, interp_transformed_2d_lines
from .base import CellMode
from .lines_base import LinesBase
from ._cell_coords import get_shape, extract_point, lerp_point
from ....utils.num_utils import is_close_to_int


class LinesCell(LinesBase):
    """2D gradient cell defined by top and bottom color lines.
    
    Interpolates vertically between the two lines. Horizontal interpolation
    depends on the line_method parameter.
    """
    
    mode: CellMode = CellMode.LINES
    
    def __init__(
        self,
        top_line: np.ndarray,
        bottom_line: np.ndarray,
        per_channel_coords: Union[List[np.ndarray], np.ndarray],
        color_space: ColorSpaces,
        hue_direction_y: Optional[str] = None,
        hue_direction_x: Optional[str] = None,
        line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
        boundtypes: Union[List[BoundType], BoundType] = BoundType.CLAMP,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        *,
        value: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            top_line=top_line,
            bottom_line=bottom_line,
            per_channel_coords=per_channel_coords,
            color_space=color_space,
            hue_direction_y=hue_direction_y,
            hue_direction_x=hue_direction_x,
            line_method=line_method,
            boundtypes=boundtypes,
            border_mode=border_mode,
            border_value=border_value,
            value=value,
        )
    
    # === Core rendering ===
    
    def _render_value(self) -> np.ndarray:
        return interp_transformed_2d_lines(
            line0=self._top_line,
            line1=self._bottom_line,
            transformed=self._per_channel_coords,
            color_space=self._color_space,
            huemode_y=self._hue_direction_y,
            huemode_x=self._hue_direction_x,
            line_method=self._line_method,
            bound_types=self._boundtypes,
            border_mode=self._border_mode,
            border_value=self._border_value,
        )
    
    # === Edge interpolation ===
    
    def _interpolate_line_position(self, line: np.ndarray, horizontal_pos: float) -> np.ndarray:
        """Interpolate along a 1D line at a given horizontal position [0, 1]."""
        exact_idx = horizontal_pos * (len(line) - 1)
        
        # Discrete mode: round to nearest pixel
        if self._line_method == LineInterpMethods.LINES_DISCRETE:
            idx = int(exact_idx + 0.5)
            idx = max(0, min(idx, len(line) - 1))
            return line[idx].copy()
        
        # Continuous mode: linear interpolation
        if is_close_to_int(exact_idx):
            return line[int(round(exact_idx))].copy()
        
        left_idx = int(np.floor(exact_idx))
        right_idx = min(left_idx + 1, len(line) - 1)
        t = exact_idx - left_idx
        
        return (1 - t) * line[left_idx] + t * line[right_idx]
    
    def interpolate_edge(self, horizontal_pos: float, is_top_edge: bool) -> np.ndarray:
        """Interpolate at a specific position [0, 1] along top or bottom edge."""
        line = self._top_line if is_top_edge else self._bottom_line
        return self._interpolate_line_position(line, horizontal_pos)
    
    def interpolate_edge_continuous(self, horizontal_pos: float, vertical_idx: int) -> np.ndarray:
        """Continuous interpolation along top (0) or bottom (height-1) edge.
        
        This method matches the CornersCell interface for compatibility.
        """
        # Fast path: direct from cached value
        if self._value is not None:
            exact_idx = horizontal_pos * (self.width - 1)
            if is_close_to_int(exact_idx):
                idx = int(round(exact_idx))
                return self._value[vertical_idx, idx, :].copy()
        
        is_top = vertical_idx == 0
        return self.interpolate_edge(horizontal_pos, is_top_edge=is_top)
    
    def index_interpolate_edge_discrete(self, horizontal_index: int, vertical_index: int) -> np.ndarray:
        """Discrete interpolation at specific pixel coordinates."""
        # Fast path: direct cache access
        if self._value is not None:
            if 0 <= horizontal_index < self.width and 0 <= vertical_index < self.height:
                return self._value[vertical_index, horizontal_index, :].copy()
        
        # Normalize negative indices
        if horizontal_index < 0:
            horizontal_index += self.line_width
        
        is_top = vertical_index == 0 or (vertical_index < 0 and vertical_index == -self.height)
        line = self._top_line if is_top else self._bottom_line
        return line[horizontal_index].copy()
    
    # === Color space conversion ===
    
    def convert_to_space(self, color_space: ColorSpaces, render_before: bool = False) -> LinesCell:
        if self._color_space == color_space:
            return self
        
        if render_before:
            self.get_value()
        
        converted_top = np_convert(
            self._top_line, self._color_space, color_space,
            input_type="float", output_type='float'
        )
        converted_bottom = np_convert(
            self._bottom_line, self._color_space, color_space,
            input_type="float", output_type='float'
        )
        
        converted_value = None
        if self._value is not None:
            converted_value = np_convert(
                self._value, self._color_space, color_space,
                input_type="float", output_type='float'
            )
        
        return LinesCell(
            top_line=converted_top,
            bottom_line=converted_bottom,
            per_channel_coords=self._per_channel_coords,
            color_space=color_space,
            hue_direction_y=self._hue_direction_y,
            hue_direction_x=self._hue_direction_x,
            line_method=self._line_method,
            boundtypes=self._boundtypes,
            border_mode=self._border_mode,
            border_value=self._border_value,
            value=converted_value,
        )
    
    # === Slicing support ===
    
    def slice_lines(self, start_idx: int, end_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Slice both lines at the given pixel indices.
        
        Returns:
            Tuple of (sliced_top_line, sliced_bottom_line)
        """
        return (
            self._top_line[start_idx:end_idx].copy(),
            self._bottom_line[start_idx:end_idx].copy(),
        )
    
    # === Utility methods ===
    
    def copy_with(self, **kwargs) -> LinesCell:
        """Create a copy with overridden values."""
        defaults = {
            'top_line': self._top_line,
            'bottom_line': self._bottom_line,
            'per_channel_coords': self._per_channel_coords,
            'color_space': self._color_space,
            'hue_direction_y': self._hue_direction_y,
            'hue_direction_x': self._hue_direction_x,
            'line_method': self._line_method,
            'boundtypes': self._boundtypes,
            'border_mode': self._border_mode,
            'border_value': self._border_value,
        }
        defaults.update(kwargs)
        return LinesCell(**defaults)
    
    @classmethod
    def get_top_lines(cls, cells: List[LinesCell]) -> np.ndarray:
        """Concatenate top lines from multiple cells."""
        if not cells:
            raise ValueError("Cannot concatenate empty list of cells")
        return np.concatenate([cell.top_line for cell in cells], axis=0)
    
    @classmethod
    def get_bottom_lines(cls, cells: List[LinesCell]) -> np.ndarray:
        """Concatenate bottom lines from multiple cells."""
        if not cells:
            raise ValueError("Cannot concatenate empty list of cells")
        return np.concatenate([cell.bottom_line for cell in cells], axis=0)
    
    def __repr__(self) -> str:
        cached = "cached" if self._value is not None else "not cached"
        return (
            f"LinesCell(width={self.width}, height={self.height}, "
            f"line_width={self.line_width}, color_space={self._color_space!r}, "
            f"method={self._line_method.name}, {cached})"
        )