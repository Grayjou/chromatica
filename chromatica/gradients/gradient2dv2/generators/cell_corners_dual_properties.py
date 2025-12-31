# chromatica/gradients/gradient2dv2/generators/cell_corners_dual_properties.py
"""Properties mixin for CornersCellDualFactory."""

from __future__ import annotations
from .base_properties import BaseCellFactoryProperties  # NEW IMPORT
from typing import Callable, Dict, List, Optional, Union
import numpy as np
from boundednumbers import BoundType

from ....types.color_types import ColorSpace, is_hue_space
from ....types.format_type import FormatType
from ....types.transform_types import PerChannelCoords
from ....utils.color_utils import convert_to_space_float
from ..cell.corners_dual import CornersCellDual
from unitfield import upbm_2d
from .descriptors import SyncedCellPropertyDescriptor
from ..cell.factory import _determine_grayscale_hue, value_or_default

class CornersCellDualFactoryProperties(BaseCellFactoryProperties):  # INHERITANCE ADDED
    """Factory properties for creating CornersCellDual instances."""
    
    # === Corner Properties (sync with cell) ===
    top_left: np.ndarray = SyncedCellPropertyDescriptor('top_left')
    top_right: np.ndarray = SyncedCellPropertyDescriptor('top_right')
    bottom_left: np.ndarray = SyncedCellPropertyDescriptor('bottom_left')
    bottom_right: np.ndarray = SyncedCellPropertyDescriptor('bottom_right')
    
    # === Interpolation Properties ===
    hue_direction_x: Optional[str] = SyncedCellPropertyDescriptor('hue_direction_x')
    hue_direction_y: Optional[str] = SyncedCellPropertyDescriptor('hue_direction_y')
    boundtypes: Union[List[BoundType], BoundType] = SyncedCellPropertyDescriptor('boundtypes')
    border_mode: Optional[int] = SyncedCellPropertyDescriptor('border_mode')
    border_value: Optional[float] = SyncedCellPropertyDescriptor('border_value')
    
    # === Segment Properties (sync with cell) ===
    top_segment_hue_direction_x: Optional[str] = SyncedCellPropertyDescriptor(
        'top_segment_hue_direction_x'
    )
    bottom_segment_hue_direction_x: Optional[str] = SyncedCellPropertyDescriptor(
        'bottom_segment_hue_direction_x'
    )
    top_segment_color_space: ColorSpace = SyncedCellPropertyDescriptor(
        'top_segment_color_space', invalidates_cell=True
    )
    bottom_segment_color_space: ColorSpace = SyncedCellPropertyDescriptor(
        'bottom_segment_color_space', invalidates_cell=True
    )
    # === Grayscale Hue Properties ===
    top_left_grayscale_hue: Optional[float] = SyncedCellPropertyDescriptor(
        'top_left_grayscale_hue', invalidates_cell=True
    )
    top_right_grayscale_hue: Optional[float] = SyncedCellPropertyDescriptor(
        'top_right_grayscale_hue', invalidates_cell=True
    )
    bottom_left_grayscale_hue: Optional[float] = SyncedCellPropertyDescriptor(
        'bottom_left_grayscale_hue', invalidates_cell=True
    )
    bottom_right_grayscale_hue: Optional[float] = SyncedCellPropertyDescriptor(
        'bottom_right_grayscale_hue', invalidates_cell=True
    )
    def __init__(
        self,
        width: int,
        height: int,
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        vertical_color_space: ColorSpace,
        horizontal_color_space: Optional[ColorSpace] = None,
        top_left_color_space: Optional[ColorSpace] = None,
        top_right_color_space: Optional[ColorSpace] = None,
        bottom_left_color_space: Optional[ColorSpace] = None,
        bottom_right_color_space: Optional[ColorSpace] = None,
        hue_direction_x: Optional[str] = None,
        hue_direction_y: Optional[str] = None,
        input_format: FormatType = FormatType.INT,
        per_channel_transforms: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]] = None,
        boundtypes: Optional[Union[List[BoundType], BoundType]] = None,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        *,
        top_segment_hue_direction_x: Optional[str] = None,
        bottom_segment_hue_direction_x: Optional[str] = None,
        top_segment_color_space: Optional[ColorSpace] = None,
        bottom_segment_color_space: Optional[ColorSpace] = None,
        per_channel_coords: Optional[PerChannelCoords] = None,
        top_left_grayscale_hue: Optional[float] = None,
        top_right_grayscale_hue: Optional[float] = None,
        bottom_left_grayscale_hue: Optional[float] = None,
        bottom_right_grayscale_hue: Optional[float] = None,
    ):
        lens = {len(corner) for corner in [top_left, top_right, bottom_left, bottom_right]} | {len(vertical_color_space)} | ({len(horizontal_color_space)} if horizontal_color_space else set())
        if len(lens) != 1:
            raise ValueError("All corners must have the same number of channels as the color space.")
        
        # Initialize base class
        super().__init__(
            width=width,
            height=height,
            per_channel_coords=per_channel_coords,
        )
        
        # Resolve color spaces
        self._vertical_color_space = vertical_color_space
        self._horizontal_color_space = horizontal_color_space or vertical_color_space
        self._top_segment_color_space = top_segment_color_space or self._horizontal_color_space
        self._bottom_segment_color_space = bottom_segment_color_space or self._horizontal_color_space
        
        # Resolve corner color spaces (the ORIGINAL space of each input color)
        tl_space = top_left_color_space or self._top_segment_color_space
        tr_space = top_right_color_space or self._top_segment_color_space
        bl_space = bottom_left_color_space or self._bottom_segment_color_space
        br_space = bottom_right_color_space or self._bottom_segment_color_space
        
        # === DETERMINE GRAYSCALE HUES BEFORE CONVERSION ===
        # This must happen BEFORE _convert_corner because conversion loses hue info
        self._top_left_grayscale_hue = value_or_default(
            self._detect_grayscale_hue(top_left, tl_space, input_format), top_left_grayscale_hue
        )
        self._top_right_grayscale_hue = value_or_default(
            self._detect_grayscale_hue(top_right, tr_space, input_format), top_right_grayscale_hue
        )
        self._bottom_left_grayscale_hue = value_or_default(
            self._detect_grayscale_hue(bottom_left, bl_space, input_format), bottom_left_grayscale_hue
        )
        self._bottom_right_grayscale_hue = value_or_default(
            self._detect_grayscale_hue(bottom_right, br_space, input_format), bottom_right_grayscale_hue
        )

        # Convert corners to their segment color spaces (AFTER grayscale hue detection)

        self._top_left = self._convert_corner(
            top_left, tl_space, self._top_segment_color_space, input_format
        )
        self._top_right = self._convert_corner(
            top_right, tr_space, self._top_segment_color_space, input_format
        )
        self._bottom_left = self._convert_corner(
            bottom_left, bl_space, self._bottom_segment_color_space, input_format
        )
        self._bottom_right = self._convert_corner(
            bottom_right, br_space, self._bottom_segment_color_space, input_format
        )

        # Hue directions
        self._hue_direction_x = hue_direction_x
        self._hue_direction_y = hue_direction_y
        self._top_segment_hue_direction_x = top_segment_hue_direction_x or hue_direction_x
        self._bottom_segment_hue_direction_x = bottom_segment_hue_direction_x or hue_direction_x
        
        # Other settings
        self._per_channel_transforms = per_channel_transforms
        self._boundtypes = boundtypes if boundtypes is not None else BoundType.CLAMP
        self._border_mode = border_mode
        self._border_value = border_value

    @staticmethod
    def _detect_grayscale_hue(
        color: np.ndarray,
        color_space: ColorSpace,
        input_format: FormatType,
    ) -> Optional[float]:
        """Detect grayscale hue from a color in its ORIGINAL space before conversion.
        
        This must be called BEFORE converting the color to a different space,
        because conversion (e.g., HSV white -> RGB white) loses the hue information.
        
        Args:
            color: The color array in its original space
            color_space: The color's ORIGINAL color space (not target)
            input_format: The format of the color values (INT or FLOAT)
            
        Returns:
            The hue value if the color is grayscale in a hue space, None otherwise
        """
        if not is_hue_space(color_space):
            return None
        
        # Convert to float in the SAME color space for consistent grayscale checking
        if input_format != FormatType.FLOAT:
            color_float = convert_to_space_float(
                color, color_space, input_format, color_space  # same space!
            ).value
        else:
            color_float = np.asarray(color, dtype=float)
        
        return _determine_grayscale_hue(color_float, color_space)

    @staticmethod
    def _convert_corner(
        color: np.ndarray,
        source_space: ColorSpace,
        target_space: ColorSpace,
        input_format: FormatType,
    ) -> np.ndarray:
        """Convert a corner color to target space."""
        if source_space != target_space or input_format != FormatType.FLOAT:
            return convert_to_space_float(
                color, source_space, input_format, target_space
            ).value
        return color.copy() if isinstance(color, np.ndarray) else np.asarray(color)
    
    # === Abstract method implementations ===
    @property
    def num_channels(self) -> int:
        """Number of channels in the horizontal color space."""
        return len(self._horizontal_color_space)
    
    def _get_color_space_for_repr(self) -> str:
        """Return color space info for __repr__."""
        return (
            f"vertical_space={self._vertical_color_space!r}, "
            f"horizontal_space={self._horizontal_color_space!r}"
        )
    

    
    # === Color Space Properties (special handling) ===
    @property
    def vertical_color_space(self) -> ColorSpace:
        return self._vertical_color_space
    
    @vertical_color_space.setter
    def vertical_color_space(self, value: ColorSpace):
        if self._vertical_color_space == value:
            return
        self._vertical_color_space = value
        self._cell = None  # Requires rebuild
    
    @property
    def horizontal_color_space(self) -> ColorSpace:
        return self._horizontal_color_space
    
    @horizontal_color_space.setter
    def horizontal_color_space(self, value: ColorSpace):
        if self._horizontal_color_space == value:
            return
        
        # Convert corners to new horizontal space
        old_space = self._horizontal_color_space
        self._top_left = convert_to_space_float(
            self._top_left, old_space, FormatType.FLOAT, value
        ).value
        self._top_right = convert_to_space_float(
            self._top_right, old_space, FormatType.FLOAT, value
        ).value
        self._bottom_left = convert_to_space_float(
            self._bottom_left, old_space, FormatType.FLOAT, value
        ).value
        self._bottom_right = convert_to_space_float(
            self._bottom_right, old_space, FormatType.FLOAT, value
        ).value
        
        self._horizontal_color_space = value
        self._cell = None
    
    # === Per-channel Transforms (special handling) ===
    @property
    def per_channel_transforms(self) -> Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]]:
        return self._per_channel_transforms
    
    @per_channel_transforms.setter
    def per_channel_transforms(self, value: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]]):
        self._per_channel_transforms = value
        self._cell = None
        self._per_channel_coords = None
    
    # === Computed Properties ===
    @property
    def corners(self) -> Dict[str, np.ndarray]:
        """All four corners as a dictionary."""
        return {
            'top_left': self._top_left,
            'top_right': self._top_right,
            'bottom_left': self._bottom_left,
            'bottom_right': self._bottom_right,
        }
    
    @property
    def color_spaces(self) -> Dict[str, ColorSpace]:
        """All color spaces as a dictionary."""
        return {
            'vertical': self._vertical_color_space,
            'horizontal': self._horizontal_color_space,
            'top_segment': self._top_segment_color_space,
            'bottom_segment': self._bottom_segment_color_space,
        }
    
    # === Copy Methods ===
    def copy(self) -> CornersCellDualFactoryProperties:
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
        vertical_color_space: Optional[ColorSpace] = None,
        horizontal_color_space: Optional[ColorSpace] = None,
        top_segment_color_space: Optional[ColorSpace] = None,
        bottom_segment_color_space: Optional[ColorSpace] = None,
        hue_direction_x: Optional[str] = ...,
        hue_direction_y: Optional[str] = ...,
        top_segment_hue_direction_x: Optional[str] = ...,
        bottom_segment_hue_direction_x: Optional[str] = ...,
        per_channel_transforms: Optional[Dict[int, Callable]] = ...,
        boundtypes: Optional[Union[List[BoundType], BoundType]] = None,
        border_mode: Optional[int] = ...,
        border_value: Optional[float] = ...,
    ) -> CornersCellDualFactoryProperties:
        """Create a copy with optionally overridden values."""
        def resolve(new_val, current_val):
            return current_val if new_val is ... else new_val
        
        return CornersCellDualFactoryProperties(
            width=width if width is not None else self._width,
            height=height if height is not None else self._height,
            top_left=top_left if top_left is not None else self._top_left.copy(),
            top_right=top_right if top_right is not None else self._top_right.copy(),
            bottom_left=bottom_left if bottom_left is not None else self._bottom_left.copy(),
            bottom_right=bottom_right if bottom_right is not None else self._bottom_right.copy(),
            vertical_color_space=vertical_color_space or self._vertical_color_space,
            horizontal_color_space=horizontal_color_space or self._horizontal_color_space,
            top_segment_color_space=top_segment_color_space or self._top_segment_color_space,
            bottom_segment_color_space=bottom_segment_color_space or self._bottom_segment_color_space,
            hue_direction_x=resolve(hue_direction_x, self._hue_direction_x),
            hue_direction_y=resolve(hue_direction_y, self._hue_direction_y),
            top_segment_hue_direction_x=resolve(top_segment_hue_direction_x, self._top_segment_hue_direction_x),
            bottom_segment_hue_direction_x=resolve(bottom_segment_hue_direction_x, self._bottom_segment_hue_direction_x),
            input_format=FormatType.FLOAT,
            per_channel_transforms=resolve(per_channel_transforms, self._per_channel_transforms),
            boundtypes=boundtypes if boundtypes is not None else self._boundtypes,
            border_mode=resolve(border_mode, self._border_mode),
            border_value=resolve(border_value, self._border_value),
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CornersCellDualFactoryProperties):
            return NotImplemented
        return (
            self._width == other._width
            and self._height == other._height
            and self._vertical_color_space == other._vertical_color_space
            and self._horizontal_color_space == other._horizontal_color_space
            and self._top_segment_color_space == other._top_segment_color_space
            and self._bottom_segment_color_space == other._bottom_segment_color_space
            and np.array_equal(self._top_left, other._top_left)
            and np.array_equal(self._top_right, other._top_right)
            and np.array_equal(self._bottom_left, other._bottom_left)
            and np.array_equal(self._bottom_right, other._bottom_right)
            and self._hue_direction_x == other._hue_direction_x
            and self._hue_direction_y == other._hue_direction_y
            and self._top_segment_hue_direction_x == other._top_segment_hue_direction_x
            and self._bottom_segment_hue_direction_x == other._bottom_segment_hue_direction_x
            and self._boundtypes == other._boundtypes
            and self._border_mode == other._border_mode
            and self._border_value == other._border_value
        )