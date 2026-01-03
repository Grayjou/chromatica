# chromatica/gradients/gradient2dv2/generators/cell_corners_properties.py
from __future__ import annotations
from ..cell.corners import CornersCell
from .base_properties import BaseCellFactoryProperties  # NEW IMPORT
import numpy as np
from typing import List, Optional, Callable, Dict, Union
from ....types.format_type import FormatType
from ....types.color_types import ColorMode
from boundednumbers import BoundType
from ....types.transform_types import PerChannelCoords
from ....utils.color_utils import convert_to_space_float
from .descriptors import SyncedCellPropertyDescriptor
from unitfield import upbm_2d


class CornersCellFactoryProperties(BaseCellFactoryProperties):  
    """Factory for creating CornersCell instances with configurable properties.
    
    This class manages the configuration and lazy creation of CornersCell objects.
    Properties are synced with any existing cell instance when modified.
    """
    
    # === Dimension properties (inherit from BaseCellFactoryProperties) ===
    # width and height properties are inherited from BaseCellFactoryProperties
    
    # === Corner properties (sync with cell) ===
    top_left = SyncedCellPropertyDescriptor('top_left')
    top_right = SyncedCellPropertyDescriptor('top_right')
    bottom_left = SyncedCellPropertyDescriptor('bottom_left')
    bottom_right = SyncedCellPropertyDescriptor('bottom_right')
    
    # === Interpolation properties ===
    hue_direction_x = SyncedCellPropertyDescriptor('hue_direction_x')
    hue_direction_y = SyncedCellPropertyDescriptor('hue_direction_y')
    boundtypes = SyncedCellPropertyDescriptor('boundtypes')
    border_mode = SyncedCellPropertyDescriptor('border_mode')
    border_value = SyncedCellPropertyDescriptor('border_value')
    
    def __init__(
        self,
        width: int,
        height: int,
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        color_mode: ColorMode,
        top_left_color_mode: Optional[ColorMode] = None,
        top_right_color_mode: Optional[ColorMode] = None,
        bottom_left_color_mode: Optional[ColorMode] = None,
        bottom_right_color_mode: Optional[ColorMode] = None,
        hue_direction_x: Optional[str] = None,
        hue_direction_y: Optional[str] = None,
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]] = None,
        boundtypes: Optional[Union[List[BoundType], BoundType]] = None,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        *,
        per_channel_coords: Optional[PerChannelCoords] = None,
    ):
        # Initialize base class
        super().__init__(
            width=width,
            height=height,
            per_channel_coords=per_channel_coords,
        )
        
        self._color_mode = color_mode
        lens = {len(corner) for corner in [top_left, top_right, bottom_left, bottom_right]} | {len(color_mode)}
        if len(lens) != 1:
            raise ValueError("All corners must have the same number of channels as the color space.")
        # Convert corners to working color space
        self._top_left, self._top_right, self._bottom_left, self._bottom_right = (
            self._convert_corners(
                corners=[top_left, top_right, bottom_left, bottom_right],
                corner_spaces=[
                    top_left_color_mode, 
                    top_right_color_mode,
                    bottom_left_color_mode, 
                    bottom_right_color_mode
                ],
                target_space=color_mode,
                input_format=input_format,
            )
        )
        
        self._hue_direction_x = hue_direction_x
        self._hue_direction_y = hue_direction_y
        self._per_channel_transforms = per_channel_transforms
        self._boundtypes = boundtypes if boundtypes is not None else BoundType.CLAMP
        self._border_mode = border_mode
        self._border_value = border_value
    
    @staticmethod
    def _convert_corners(
        corners: List[np.ndarray],
        corner_spaces: List[Optional[ColorMode]],
        target_space: ColorMode,
        input_format: FormatType,
    ) -> List[np.ndarray]:
        """Convert corner colors to target color space."""
        result = []
        for color, source_space in zip(corners, corner_spaces):
            if source_space is not None and source_space != target_space:
                converted = convert_to_space_float(
                    color, source_space, input_format, target_space
                ).value
                result.append(converted)
            elif input_format != FormatType.FLOAT:
                # Convert format even if color space matches
                converted = convert_to_space_float(
                    color, target_space, input_format, target_space
                ).value
                result.append(converted)
            else:
                result.append(color.copy() if isinstance(color, np.ndarray) else np.asarray(color))
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
        
        # Convert all corners to new color space
        old_space = self._color_mode
        corners = [self._top_left, self._top_right, self._bottom_left, self._bottom_right]
        
        converted = [
            convert_to_space_float(c, old_space, FormatType.FLOAT, value).value
            for c in corners
        ]
        
        self._top_left, self._top_right, self._bottom_left, self._bottom_right = converted
        self._color_mode = value
        
        # Sync with cell if it exists
        if self._cell is not None:
            self._cell = self._cell.convert_to_space(value, render_before=False)
    

    
    # === Per-channel transforms (special handling) ===
    # Override to use property from base class with specific setter
    
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
    def corners(self) -> Dict[str, np.ndarray]:
        """All four corners as a dictionary."""
        return {
            'top_left': self._top_left,
            'top_right': self._top_right,
            'bottom_left': self._bottom_left,
            'bottom_right': self._bottom_right,
        }
    
    # === Copy methods ===
    
    def copy(self) -> CornersCellFactoryProperties:
        """Create a shallow copy of this factory."""
        return self.copy_with()
    
    def copy_with(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        top_left: Optional[np.ndarray] = None,
        top_right: Optional[np.ndarray] = None,
        bottom_left: Optional[np.ndarray] = None,
        bottom_right: Optional[np.ndarray] = None,
        color_mode: Optional[ColorMode] = None,
        hue_direction_x: Optional[str] = ...,  # Use ... as sentinel for "not provided"
        hue_direction_y: Optional[str] = ...,
        per_channel_transforms: Optional[Dict[int, Callable]] = ...,
        boundtypes: Optional[Union[List[BoundType], BoundType]] = None,
        border_mode: Optional[int] = ...,
        border_value: Optional[float] = ...,
    ) -> CornersCellFactoryProperties:
        """Create a copy with optionally overridden values."""
        def resolve(new_val, current_val):
            return current_val if new_val is ... else new_val
        
        return CornersCellFactoryProperties(
            width=width if width is not None else self._width,
            height=height if height is not None else self._height,
            top_left=top_left if top_left is not None else self._top_left.copy(),
            top_right=top_right if top_right is not None else self._top_right.copy(),
            bottom_left=bottom_left if bottom_left is not None else self._bottom_left.copy(),
            bottom_right=bottom_right if bottom_right is not None else self._bottom_right.copy(),
            color_mode=color_mode if color_mode is not None else self._color_mode,
            hue_direction_x=resolve(hue_direction_x, self._hue_direction_x),
            hue_direction_y=resolve(hue_direction_y, self._hue_direction_y),
            input_format=FormatType.FLOAT,
            per_channel_transforms=resolve(per_channel_transforms, self._per_channel_transforms),
            boundtypes=boundtypes if boundtypes is not None else self._boundtypes,
            border_mode=resolve(border_mode, self._border_mode),
            border_value=resolve(border_value, self._border_value),
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CornersCellFactoryProperties):
            return NotImplemented
        return (
            self._width == other._width
            and self._height == other._height
            and self._color_mode == other._color_mode
            and np.array_equal(self._top_left, other._top_left)
            and np.array_equal(self._top_right, other._top_right)
            and np.array_equal(self._bottom_left, other._bottom_left)
            and np.array_equal(self._bottom_right, other._bottom_right)
            and self._hue_direction_x == other._hue_direction_x
            and self._hue_direction_y == other._hue_direction_y
            and self._boundtypes == other._boundtypes
            and self._border_mode == other._border_mode
            and self._border_value == other._border_value
        )