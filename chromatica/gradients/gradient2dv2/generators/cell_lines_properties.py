# chromatica/gradients/gradient2dv2/generators/cell_lines_properties.py
"""Factory properties for creating LinesCell instances."""

from __future__ import annotations
from .base_properties import BaseCellFactoryProperties  # NEW IMPORT
from typing import List, Optional, Callable, Dict, Union
import numpy as np

from ..cell.lines import LinesCell
from .descriptors import SyncedCellPropertyDescriptor
from ..helpers import LineInterpMethods
from ....types.format_type import FormatType
from ....types.color_types import ColorMode
from ....types.transform_types import PerChannelCoords
from boundednumbers import BoundType
from ....conversions import np_convert
from unitfield import upbm_2d


class LinesCellFactoryProperties(BaseCellFactoryProperties):  # INHERITANCE ADDED
    """Factory for creating LinesCell instances with configurable properties."""
    
    # === Line properties (sync with cell) ===
    top_line = SyncedCellPropertyDescriptor('top_line')
    bottom_line = SyncedCellPropertyDescriptor('bottom_line')
    
    # === Interpolation properties ===
    hue_direction_x = SyncedCellPropertyDescriptor('hue_direction_x')
    hue_direction_y = SyncedCellPropertyDescriptor('hue_direction_y')
    line_method = SyncedCellPropertyDescriptor('line_method')
    boundtypes = SyncedCellPropertyDescriptor('boundtypes')
    border_mode = SyncedCellPropertyDescriptor('border_mode')
    border_value = SyncedCellPropertyDescriptor('border_value')
    
    def __init__(
        self,
        width: int,
        height: int,
        top_line: np.ndarray,
        bottom_line: np.ndarray,
        color_mode: ColorMode,
        top_line_color_mode: Optional[ColorMode] = None,
        bottom_line_color_mode: Optional[ColorMode] = None,
        hue_direction_x: Optional[str] = None,
        hue_direction_y: Optional[str] = None,
        line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]] = None,
        boundtypes: Optional[Union[List[BoundType], BoundType]] = None,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        *,
        per_channel_coords: Optional[PerChannelCoords] = None,
    ):
        # Validate lines
        top_line = np.asarray(top_line)
        bottom_line = np.asarray(bottom_line)
        if top_line.shape[0] != bottom_line.shape[0]:
            raise ValueError("Top line and bottom line must have the same width.")
        
        # Initialize base class
        super().__init__(
            width=width,
            height=height,
            per_channel_coords=per_channel_coords,
        )
        
        self._color_mode = ColorMode(color_mode)
        
        # Convert lines to working color space
        self._top_line, self._bottom_line = self._convert_lines(
            lines=[top_line, bottom_line],
            line_spaces=[top_line_color_mode, bottom_line_color_mode],
            target_space=self._color_mode,
            input_format=input_format,
        )
        
        self._hue_direction_x = hue_direction_x
        self._hue_direction_y = hue_direction_y
        self._line_method = line_method
        self._per_channel_transforms = per_channel_transforms
        self._boundtypes = boundtypes if boundtypes is not None else BoundType.CLAMP
        self._border_mode = border_mode
        self._border_value = border_value
    
    @staticmethod
    def _convert_lines(
        lines: List[np.ndarray],
        line_spaces: List[Optional[ColorMode]],
        target_space: ColorMode,
        input_format: FormatType,
    ) -> List[np.ndarray]:
        """Convert line arrays to target color space."""
        result = []
        for line, source_space in zip(lines, line_spaces):
            if source_space is not None and source_space != target_space:
                converted = np_convert(
                    line, source_space, target_space,
                    input_type=input_format.value if hasattr(input_format, 'value') else str(input_format),
                    output_type='float'
                )
                result.append(converted)
            elif input_format != FormatType.FLOAT:
                # Convert format even if color space matches
                converted = np_convert(
                    line, target_space, target_space,
                    input_type=input_format.value if hasattr(input_format, 'value') else str(input_format),
                    output_type='float'
                )
                result.append(converted)
            else:
                result.append(line.copy() if isinstance(line, np.ndarray) else np.asarray(line))
        return result
    
    # === Abstract method implementations ===
    @property
    def num_channels(self) -> int:
        """Number of channels in the color space."""
        return len(self._color_mode)
    
    def _get_color_mode_for_repr(self) -> str:
        """Return color space info for __repr__."""
        return f"color_mode={self._color_mode!r}"
    

    
    # === Color space property (special handling) ===
    @property
    def color_mode(self) -> ColorMode:
        return self._color_mode
    
    @color_mode.setter
    def color_mode(self, value: ColorMode):
        if self._color_mode == value:
            return
        
        # Convert both lines to new color space
        old_space = self._color_mode
        lines = [self._top_line, self._bottom_line]
        
        converted = [
            np_convert(line, old_space, value, input_type='float', output_type='float')
            for line in lines
        ]
        
        self._top_line, self._bottom_line = converted
        self._color_mode = value
        
        # Sync with cell if it exists
        if self._cell is not None:
            self._cell = self._cell.convert_to_space(value, render_before=False)
    
    # === Per-channel transforms (special handling) ===
    @property
    def per_channel_transforms(self) -> Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]]:
        return self._per_channel_transforms
    
    @per_channel_transforms.setter
    def per_channel_transforms(self, value: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]]):
        self._per_channel_transforms = value
        # Transforms are applied during cell creation, so invalidate cell
        self._cell = None
        self._per_channel_coords = None
    
    # === Computed properties ===
    @property
    def line_width(self) -> int:
        """Width of the lines (number of color samples)."""
        return self._top_line.shape[0]
    
    @property
    def lines(self) -> Dict[str, np.ndarray]:
        """Both lines as a dictionary."""
        return {
            'top_line': self._top_line,
            'bottom_line': self._bottom_line,
        }
    
    # === Copy methods ===
    def copy(self) -> LinesCellFactoryProperties:
        """Create a shallow copy of this factory."""
        return self.copy_with()
    
    def copy_with(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        top_line: Optional[np.ndarray] = None,
        bottom_line: Optional[np.ndarray] = None,
        color_mode: Optional[ColorMode] = None,
        hue_direction_x: Optional[str] = ...,  # Use ... as sentinel for "not provided"
        hue_direction_y: Optional[str] = ...,
        line_method: Optional[LineInterpMethods] = None,
        per_channel_transforms: Optional[Dict[int, Callable]] = ...,
        boundtypes: Optional[Union[List[BoundType], BoundType]] = None,
        border_mode: Optional[int] = ...,
        border_value: Optional[float] = ...,
    ) -> LinesCellFactoryProperties:
        """Create a copy with optionally overridden values."""
        def resolve(new_val, current_val):
            return current_val if new_val is ... else new_val
        
        return LinesCellFactoryProperties(
            width=width if width is not None else self._width,
            height=height if height is not None else self._height,
            top_line=top_line if top_line is not None else self._top_line.copy(),
            bottom_line=bottom_line if bottom_line is not None else self._bottom_line.copy(),
            color_mode=color_mode if color_mode is not None else self._color_mode,
            hue_direction_x=resolve(hue_direction_x, self._hue_direction_x),
            hue_direction_y=resolve(hue_direction_y, self._hue_direction_y),
            line_method=line_method if line_method is not None else self._line_method,
            input_format=FormatType.FLOAT,
            per_channel_transforms=resolve(per_channel_transforms, self._per_channel_transforms),
            boundtypes=boundtypes if boundtypes is not None else self._boundtypes,
            border_mode=resolve(border_mode, self._border_mode),
            border_value=resolve(border_value, self._border_value),
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinesCellFactoryProperties):
            return NotImplemented
        return (
            self._width == other._width
            and self._height == other._height
            and self._color_mode == other._color_mode
            and self._line_method == other._line_method
            and np.array_equal(self._top_line, other._top_line)
            and np.array_equal(self._bottom_line, other._bottom_line)
            and self._hue_direction_x == other._hue_direction_x
            and self._hue_direction_y == other._hue_direction_y
            and self._boundtypes == other._boundtypes
            and self._border_mode == other._border_mode
            and self._border_value == other._border_value
        )