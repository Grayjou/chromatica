# chromatica/gradients/gradient2dv2/cell/corners_dual.py
"""CornersCellDual implementation for 2D gradient cells with dual color spaces."""

from __future__ import annotations
from typing import List, Optional, Union
import numpy as np

from boundednumbers import BoundType, UnitFloat

from ....types.color_types import ColorSpace, HueMode, is_hue_space
from ....types.format_type import FormatType
from ....conversions import np_convert
from ....utils.color_utils import convert_to_space_float, is_hue_color_grayscale, is_hue_color_arr_grayscale
from ....utils.default import value_or_default
from ....utils.num_utils import is_close_to_int
from unitfield import flat_1d_upbm

from .base import CellMode
from .corners_base import CornersBase, CornerIndex
from ._descriptors import CellPropertyDescriptor
from ._cell_coords import extract_edge, extract_point, lerp_point
from ..helpers import LineInterpMethods, interp_transformed_2d_from_corners
from ..helpers.interpolation.corners import hue_lerp_from_corners
from ...gradient1dv2.segment import get_transformed_segment


class CornersCellDual(CornersBase):
    """2D gradient cell with separate horizontal and vertical color spaces.
    
    This cell type allows different color spaces for horizontal interpolation
    (along each edge) and vertical interpolation (between edges). Each edge
    can also have its own color space and hue direction.
    
    Attributes:
        vertical_color_space: Color space for vertical interpolation
        horizontal_color_space: Default color space for horizontal interpolation
        top_segment_color_space: Color space for top edge interpolation
        bottom_segment_color_space: Color space for bottom edge interpolation
        top_segment_hue_direction_x: Hue direction for top edge
        bottom_segment_hue_direction_x: Hue direction for bottom edge
    """
    
    mode: CellMode = CellMode.CORNERS_DUAL
    
    # === Segment Properties (invalidate segments + cache) ===
    top_segment_hue_direction_x: Optional[str] = CellPropertyDescriptor(
        'top_segment_hue_direction_x', invalidates_segments=True
    )
    bottom_segment_hue_direction_x: Optional[str] = CellPropertyDescriptor(
        'bottom_segment_hue_direction_x', invalidates_segments=True
    )
    top_segment_color_space: ColorSpace = CellPropertyDescriptor(
        'top_segment_color_space', invalidates_segments=True
    )
    bottom_segment_color_space: ColorSpace = CellPropertyDescriptor(
        'bottom_segment_color_space', invalidates_segments=True
    )
    
    # === Read-only Color Space Properties ===
    vertical_color_space: ColorSpace = CellPropertyDescriptor(
        'vertical_color_space', readonly=True
    )
    horizontal_color_space: ColorSpace = CellPropertyDescriptor(
        'horizontal_color_space', readonly=True
    )
    
    def __init__(
        self,
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        per_channel_coords: Union[List[np.ndarray], np.ndarray],
        vertical_color_space: ColorSpace,
        horizontal_color_space: Optional[ColorSpace] = None,
        hue_direction_y: Optional[str] = None,
        hue_direction_x: Optional[str] = None,
        boundtypes: Union[List[BoundType], BoundType] = BoundType.CLAMP,
        border_mode: Optional[int] = None,
        border_value: Optional[float] = None,
        *,
        value: Optional[np.ndarray] = None,
        top_segment_hue_direction_x: Optional[str] = None,
        bottom_segment_hue_direction_x: Optional[str] = None,
        top_segment_color_space: Optional[ColorSpace] = None,
        bottom_segment_color_space: Optional[ColorSpace] = None,
        top_left_grayscale_hue: Optional[float] = None,
        top_right_grayscale_hue: Optional[float] = None,
        bottom_left_grayscale_hue: Optional[float] = None,
        bottom_right_grayscale_hue: Optional[float] = None,
    ) -> None:
        # Validate color space requirements
        if horizontal_color_space is None:
            if top_segment_color_space is None or bottom_segment_color_space is None:
                raise ValueError(
                    "Either horizontal_color_space or both top_segment_color_space "
                    "and bottom_segment_color_space must be provided."
                )
        
        # Initialize base with vertical color space
        super().__init__(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
            per_channel_coords=per_channel_coords,
            color_space=vertical_color_space,
            hue_direction_y=hue_direction_y,
            hue_direction_x=hue_direction_x,
            boundtypes=boundtypes,
            border_mode=border_mode,
            border_value=border_value,
            value=value,
        )
        
        # Dual color space settings
        self._vertical_color_space = vertical_color_space
        self._horizontal_color_space = value_or_default(
            horizontal_color_space, vertical_color_space
        )
        
        # Segment-specific settings
        self._top_segment_color_space = value_or_default(
            top_segment_color_space, self._horizontal_color_space
        )
        self._bottom_segment_color_space = value_or_default(
            bottom_segment_color_space, self._horizontal_color_space
        )
        self._top_segment_hue_direction_x = value_or_default(
            top_segment_hue_direction_x, hue_direction_x
        )
        self._bottom_segment_hue_direction_x = value_or_default(
            bottom_segment_hue_direction_x, hue_direction_x
        )
        # Grayscale hue settings
        self._top_left_grayscale_hue = top_left_grayscale_hue
        self._top_right_grayscale_hue = top_right_grayscale_hue
        self._bottom_left_grayscale_hue = bottom_left_grayscale_hue
        self._bottom_right_grayscale_hue = bottom_right_grayscale_hue
    # === Color Space Properties ===
    
    @property
    def color_space(self) -> ColorSpace:
        """Primary color space (vertical)."""
        return self._vertical_color_space
    
    @property
    def top_left_color_space(self) -> ColorSpace:
        """Color space of top-left corner."""
        return self._top_segment_color_space
    
    @property
    def top_right_color_space(self) -> ColorSpace:
        """Color space of top-right corner."""
        return self._top_segment_color_space
    
    @property
    def bottom_left_color_space(self) -> ColorSpace:
        """Color space of bottom-left corner."""
        return self._bottom_segment_color_space
    
    @property
    def bottom_right_color_space(self) -> ColorSpace:
        """Color space of bottom-right corner."""
        return self._bottom_segment_color_space
    
    # === Segment Creation ===
    
    def get_top_segment_untransformed(self) -> np.ndarray:
        """Get or create the top segment in uniform coordinates.
        
        Returns:
            Array of shape (1, width, channels)
        """
        if self._value is not None and self._top_segment is None:
            self._top_segment = self._value[0:1, :, :]
        
        if self._top_segment is not None:
            return self._top_segment
        
        uniform_coords = [flat_1d_upbm(self.width)]
        
        segment = get_transformed_segment(
            already_converted_start_color=self._top_left,
            already_converted_end_color=self._top_right,
            per_channel_coords=uniform_coords,
            color_space=self._top_segment_color_space,
            hue_direction=self._top_segment_hue_direction_x,
            bound_types=self._boundtypes,
            homogeneous_per_channel_coords=True,
        )
        
        self._top_segment = segment.get_value().reshape(1, self.width, -1)
        return self._top_segment
    
    def get_bottom_segment_untransformed(self) -> np.ndarray:
        """Get or create the bottom segment in uniform coordinates.
        
        Returns:
            Array of shape (1, width, channels)
        """
        if self._value is not None and self._bottom_segment is None:
            self._bottom_segment = self._value[-1:, :, :]
        
        if self._bottom_segment is not None:
            return self._bottom_segment
        
        uniform_coords = [flat_1d_upbm(self.width)]
        
        segment = get_transformed_segment(
            already_converted_start_color=self._bottom_left,
            already_converted_end_color=self._bottom_right,
            per_channel_coords=uniform_coords,
            color_space=self._bottom_segment_color_space,
            hue_direction=self._bottom_segment_hue_direction_x,
            bound_types=self._boundtypes,
            homogeneous_per_channel_coords=True,
        )
        
        self._bottom_segment = segment.get_value().reshape(1, self.width, -1)
        return self._bottom_segment
    
    # === Core Interpolation ===
    
    def _interpolate_in_space(
        self,
        coords: List[np.ndarray],
        space: ColorSpace,
        hue_x: Optional[str],
    ) -> np.ndarray:
        """Interpolate corners in a specific color space."""
        c_tl = self.convert_corner(CornerIndex.TOP_LEFT, space)
        c_tr = self.convert_corner(CornerIndex.TOP_RIGHT, space)
        c_bl = self.convert_corner(CornerIndex.BOTTOM_LEFT, space)
        c_br = self.convert_corner(CornerIndex.BOTTOM_RIGHT, space)
        
        return interp_transformed_2d_from_corners(
            top_left=c_tl,
            top_right=c_tr,
            bottom_left=c_bl,
            bottom_right=c_br,
            transformed=coords,
            color_space=space,
            huemode_x=hue_x or HueMode.SHORTEST,
            huemode_y=self._hue_direction_y or HueMode.SHORTEST,
            bound_types=self._boundtypes,
            border_mode=self._border_mode,
            border_value=self._border_value,
        )
    
    def _interpolate_at_coords(self, coords_list: List[np.ndarray]) -> np.ndarray:
        """Core interpolation at specific coordinates."""
        return self._interpolate_in_space(
            coords_list,
            self._vertical_color_space,
            self._hue_direction_y,
        )[0, 0, :]
    
    # === Edge Interpolation ===
    
    def interpolate_edge_continuous(
        self,
        horizontal_pos: float,
        vertical_idx: int,
    ) -> np.ndarray:
        """Continuous interpolation along an edge.
        
        Args:
            horizontal_pos: Position along edge, 0.0 = left, 1.0 = right
            vertical_idx: Row index (0 = top edge, height-1 = bottom edge)
            
        Returns:
            Interpolated color as numpy array
        """
        # Fast path: exact pixel in cache
        if self._value is not None:
            exact_idx = horizontal_pos * (self.width - 1)
            if is_close_to_int(exact_idx):
                idx = int(round(exact_idx))
                return self._value[vertical_idx, idx, :].copy()
        
        exact_idx = horizontal_pos * (self.width - 1)
        is_top_edge = vertical_idx == 0
        edge_coords = extract_edge(self._per_channel_coords, vertical_idx)
        
        if is_close_to_int(exact_idx):
            coords = extract_point(edge_coords, int(round(exact_idx)))
        else:
            coords = lerp_point(edge_coords, exact_idx)
        
        if is_top_edge:
            return self._interpolate_in_space(
                coords, self._top_segment_color_space, self._top_segment_hue_direction_x
            )[0, 0, :]
        else:
            return self._interpolate_in_space(
                coords, self._bottom_segment_color_space, self._bottom_segment_hue_direction_x
            )[0, 0, :]
    
    def index_interpolate_edge_discrete(
        self,
        horizontal_index: int,
        vertical_index: int,
    ) -> np.ndarray:
        """Discrete interpolation at a specific pixel.
        
        Args:
            horizontal_index: Column index (supports negative indexing)
            vertical_index: Row index (supports negative indexing)
            
        Returns:
            Color at the specified pixel
        """
        # Fast path: direct cache access
        if self._value is not None:
            if 0 <= horizontal_index < self.width and 0 <= vertical_index < self.height:
                return self._value[vertical_index, horizontal_index, :].copy()
        
        # Normalize negative indices
        if vertical_index < 0:
            vertical_index += self.height
        if horizontal_index < 0:
            horizontal_index += self.width
        
        is_top_edge = vertical_index == 0
        coords = extract_point(
            extract_edge(self._per_channel_coords, vertical_index),
            horizontal_index,
        )
        
        if is_top_edge:
            return self._interpolate_in_space(
                coords, self._top_segment_color_space, self._top_segment_hue_direction_x
            )[0, 0, :]
        else:
            return self._interpolate_in_space(
                coords, self._bottom_segment_color_space, self._bottom_segment_hue_direction_x
            )[0, 0, :]
    def _resolve_hue(self, *fallback_order: Optional[float], default: float = 0.0) -> float:
        """Return first non-None value from fallback order, or default."""
        for hue in fallback_order:
            if hue is not None:
                return hue
        return default

    def _get_resolved_grayscale_hues(self) -> tuple[float, float, float, float]:
        """Resolve grayscale hues with fallback logic."""
        tl = self._top_left_grayscale_hue
        tr = self._top_right_grayscale_hue
        bl = self._bottom_left_grayscale_hue
        br = self._bottom_right_grayscale_hue
        
        return (
            self._resolve_hue(tl, tr, bl, br),  # top-left: horizontal, then vertical, then diagonal
            self._resolve_hue(tr, tl, br, bl),  # top-right
            self._resolve_hue(bl, br, tl, tr),  # bottom-left
            self._resolve_hue(br, bl, tr, tl),  # bottom-right
        )
    def interpolate_edge(
        self,
        horizontal_pos: UnitFloat,
        is_top_edge: bool,
    ) -> np.ndarray:
        """Public interface for edge interpolation.
        
        Args:
            horizontal_pos: Position along edge, 0.0 = left, 1.0 = right
            is_top_edge: If True, interpolate top edge; else bottom edge
            
        Returns:
            Interpolated color as numpy array
        """
        vertical_idx = 0 if is_top_edge else self.height - 1
        return self.interpolate_edge_continuous(float(horizontal_pos), vertical_idx)
    
    # === Rendering ===
    
    def _render_value(self) -> np.ndarray:
        """Render full 2D gradient."""
        from .factory import get_transformed_lines_cell

        # Get segments in their respective spaces
        top_segment = self.get_top_segment_untransformed()
        bottom_segment = self.get_bottom_segment_untransformed()
        if is_hue_space(self._vertical_color_space): 
            mask_and_replace_top_hue = self._top_segment_color_space == ColorSpace.RGB
            mask_and_replace_bottom_hue = self._bottom_segment_color_space == ColorSpace.RGB
        else:
            mask_and_replace_top_hue = False
            mask_and_replace_bottom_hue = False


        # Convert segments to vertical space if needed
        if self._top_segment_color_space != self._vertical_color_space:
            top_segment = np_convert(
                top_segment,
                self._top_segment_color_space,
                self._vertical_color_space,
                input_type="float",
                output_type="float",
            )
        
        if self._bottom_segment_color_space != self._vertical_color_space:
            bottom_segment = np_convert(
                bottom_segment,
                self._bottom_segment_color_space,
                self._vertical_color_space,
                input_type="float",
                output_type="float",
            )
        # Reshape for lines cell: (1, width, channels) -> (width, channels)
        top_line = top_segment.reshape(top_segment.shape[1], top_segment.shape[2])
        bottom_line = bottom_segment.reshape(bottom_segment.shape[1], bottom_segment.shape[2])

        if mask_and_replace_top_hue:
            where_grayscale_top = is_hue_color_arr_grayscale(top_line)
            if np.any(where_grayscale_top):
                tl_ghue, tr_ghue, bl_ghue, br_ghue = self._get_resolved_grayscale_hues()
                
                top_coords = self.top_edge_coords
                if isinstance(top_coords, list):
                    top_coords = top_coords[0]
                elif isinstance(top_coords, np.ndarray) and top_coords.ndim == 4:
                    top_coords = top_coords[0]
                # Replace grayscale hues Top_line is flattened (width, channels)
                top_line[where_grayscale_top, 0] = hue_lerp_from_corners(
                    tl_ghue, tr_ghue, bl_ghue, br_ghue,
                    top_coords[:, where_grayscale_top].astype(np.float64),
                    self._hue_direction_x or HueMode.SHORTEST,
                    self._hue_direction_y or HueMode.SHORTEST,
                )
        if mask_and_replace_bottom_hue:

            where_grayscale_bottom = is_hue_color_arr_grayscale(bottom_line)

            if np.any(where_grayscale_bottom):
                tl_ghue, tr_ghue, bl_ghue, br_ghue = self._get_resolved_grayscale_hues()
                
                bottom_coords = self.bottom_edge_coords
                if isinstance(bottom_coords, list):
                    bottom_coords = bottom_coords[0]
                elif isinstance(bottom_coords, np.ndarray) and bottom_coords.ndim == 4:
                    bottom_coords = bottom_coords[0]
                    
                bottom_line[where_grayscale_bottom, 0] = hue_lerp_from_corners(
                    corners=[tl_ghue, tr_ghue, bl_ghue, br_ghue],
                    coords=bottom_coords[:, where_grayscale_bottom].astype(np.float64),
                    mode_x=self._hue_direction_x or HueMode.SHORTEST,
                    mode_y=self._hue_direction_y or HueMode.SHORTEST,
                )
            

        lines_cell = get_transformed_lines_cell(
            top_line=top_line,
            bottom_line=bottom_line,
            per_channel_coords=self._per_channel_coords,
            color_space=self._vertical_color_space,
            top_line_color_space=self._vertical_color_space,
            bottom_line_color_space=self._vertical_color_space,
            hue_direction_y=self._hue_direction_y or HueMode.SHORTEST,
            hue_direction_x=self._hue_direction_x or HueMode.SHORTEST,
            line_method=LineInterpMethods.LINES_CONTINUOUS,
            boundtypes=self._boundtypes,
            border_mode=self._border_mode,
            border_value=self._border_value,
            input_format=FormatType.FLOAT,
        )
        
        return lines_cell.get_value()
    
    # === Color Space Conversion ===
    
    def convert_to_space(
        self,
        color_space: ColorSpace,
        render_before: bool = False,
    ) -> CornersCellDual:
        """Convert to a unified color space.
        
        Args:
            color_space: Target color space for all interpolation
            render_before: If True, render current value before converting
            
        Returns:
            New cell instance in target color space
        """
        # Check if already in target space
        if (
            self._horizontal_color_space == color_space
            and self._vertical_color_space == color_space
            and self._top_segment_color_space == color_space
            and self._bottom_segment_color_space == color_space
        ):
            return self
        
        if render_before:
            self.get_value()
        
        # Convert corners
        converted_corners = [
            np_convert(
                corner,
                self._horizontal_color_space,
                color_space,
                input_type="float",
                output_type="float",
            )
            for corner in [self._top_left, self._top_right, self._bottom_left, self._bottom_right]
        ]
        
        # Convert cached value if present
        converted_value = None
        if self._value is not None:
            converted_value = np_convert(
                self._value,
                self._vertical_color_space,
                color_space,
                input_type="float",
                output_type="float",
            )
        
        return CornersCellDual(
            top_left=converted_corners[0],
            top_right=converted_corners[1],
            bottom_left=converted_corners[2],
            bottom_right=converted_corners[3],
            per_channel_coords=self._per_channel_coords,
            horizontal_color_space=color_space,
            vertical_color_space=color_space,
            hue_direction_y=self._hue_direction_y,
            hue_direction_x=self._hue_direction_x,
            boundtypes=self._boundtypes,
            border_mode=self._border_mode,
            border_value=self._border_value,
            value=converted_value,
            top_segment_hue_direction_x=self._top_segment_hue_direction_x,
            bottom_segment_hue_direction_x=self._bottom_segment_hue_direction_x,
            top_segment_color_space=color_space,
            bottom_segment_color_space=color_space,
        )
    
    def convert_to_spaces(
        self,
        horizontal_color_space: ColorSpace,
        vertical_color_space: ColorSpace,
        top_segment_color_space: Optional[ColorSpace] = None,
        bottom_segment_color_space: Optional[ColorSpace] = None,
        render_before: bool = False,
    ) -> CornersCellDual:
        """Convert to specific horizontal and vertical spaces.
        
        Args:
            horizontal_color_space: Target horizontal color space
            vertical_color_space: Target vertical color space
            top_segment_color_space: Target top segment space (defaults to horizontal)
            bottom_segment_color_space: Target bottom segment space (defaults to horizontal)
            render_before: If True, render current value before converting
            
        Returns:
            New cell instance in target color spaces
        """
        top_seg_space = top_segment_color_space or horizontal_color_space
        bottom_seg_space = bottom_segment_color_space or horizontal_color_space
        
        # Check if already in target spaces
        if (
            self._horizontal_color_space == horizontal_color_space
            and self._vertical_color_space == vertical_color_space
            and self._top_segment_color_space == top_seg_space
            and self._bottom_segment_color_space == bottom_seg_space
        ):
            return self
        
        if render_before:
            self.get_value()
        
        # Convert corners to horizontal space
        converted_corners = [
            np_convert(
                corner,
                self._horizontal_color_space,
                horizontal_color_space,
                input_type="float",
                output_type="float",
            )
            for corner in [self._top_left, self._top_right, self._bottom_left, self._bottom_right]
        ]
        
        # Convert cached value if present
        converted_value = None
        if self._value is not None:
            converted_value = np_convert(
                self._value,
                self._vertical_color_space,
                vertical_color_space,
                input_type="float",
                output_type="float",
            )
        
        return CornersCellDual(
            top_left=converted_corners[0],
            top_right=converted_corners[1],
            bottom_left=converted_corners[2],
            bottom_right=converted_corners[3],
            per_channel_coords=self._per_channel_coords,
            horizontal_color_space=horizontal_color_space,
            vertical_color_space=vertical_color_space,
            hue_direction_y=self._hue_direction_y,
            hue_direction_x=self._hue_direction_x,
            boundtypes=self._boundtypes,
            border_mode=self._border_mode,
            border_value=self._border_value,
            value=converted_value,
            top_segment_hue_direction_x=self._top_segment_hue_direction_x,
            bottom_segment_hue_direction_x=self._bottom_segment_hue_direction_x,
            top_segment_color_space=top_seg_space,
            bottom_segment_color_space=bottom_seg_space,
        )
    
    def _get_corner_color_space(self, corner: CornerIndex) -> ColorSpace:
        if corner == CornerIndex.TOP_LEFT:
            return self._top_segment_color_space
        elif corner == CornerIndex.TOP_RIGHT:
            return self._top_segment_color_space
        elif corner == CornerIndex.BOTTOM_LEFT:
            return self._bottom_segment_color_space
        elif corner == CornerIndex.BOTTOM_RIGHT:
            return self._bottom_segment_color_space
        else:
            raise ValueError(f"Invalid corner index: {corner}")
    def _get_corner_color(self, corner: CornerIndex) -> np.ndarray:
        if corner == CornerIndex.TOP_LEFT:
            return self._top_left
        elif corner == CornerIndex.TOP_RIGHT:
            return self._top_right
        elif corner == CornerIndex.BOTTOM_LEFT:
            return self._bottom_left
        elif corner == CornerIndex.BOTTOM_RIGHT:
            return self._bottom_right
        else:
            raise ValueError(f"Invalid corner index: {corner}")
        
    def _get_corner_grayscale_hue(self, corner: CornerIndex) -> Optional[float]:
        if corner == CornerIndex.TOP_LEFT:
            return self._top_left_grayscale_hue
        elif corner == CornerIndex.TOP_RIGHT:
            return self._top_right_grayscale_hue
        elif corner == CornerIndex.BOTTOM_LEFT:
            return self._bottom_left_grayscale_hue
        elif corner == CornerIndex.BOTTOM_RIGHT:
            return self._bottom_right_grayscale_hue
        else:
            raise ValueError(f"Invalid corner index: {corner}")
    def convert_corner(self, corner: CornerIndex, to_space: ColorSpace, from_space: ColorSpace=None):
        if from_space is None:
            from_space = self._get_corner_color_space(corner)
        corner_color = self._get_corner_color(corner)
        converted = np_convert(
            corner_color,
            from_space,
            to_space,
            input_type="float",
            output_type="float",
        )
        if is_hue_space(to_space) and is_hue_color_grayscale(converted):
            grayscale_hue = self._get_corner_grayscale_hue(corner)
            if grayscale_hue is not None:
                converted[0] = grayscale_hue
        return converted
        
    # === Copy ===
    
    def copy_with(self, **kwargs) -> CornersCellDual:
        """Create a copy with overridden values."""
        defaults = {
            'top_left': self._top_left,
            'top_right': self._top_right,
            'bottom_left': self._bottom_left,
            'bottom_right': self._bottom_right,
            'per_channel_coords': self._per_channel_coords,
            'horizontal_color_space': self._horizontal_color_space,
            'vertical_color_space': self._vertical_color_space,
            'hue_direction_y': self._hue_direction_y,
            'hue_direction_x': self._hue_direction_x,
            'boundtypes': self._boundtypes,
            'border_mode': self._border_mode,
            'border_value': self._border_value,
            'top_segment_hue_direction_x': self._top_segment_hue_direction_x,
            'bottom_segment_hue_direction_x': self._bottom_segment_hue_direction_x,
            'top_segment_color_space': self._top_segment_color_space,
            'bottom_segment_color_space': self._bottom_segment_color_space,
            'value': self._value,
        }
        defaults.update(kwargs)
        return CornersCellDual(**defaults)
    
    # === Representation ===
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"vertical_space={self._vertical_color_space!r}, "
            f"horizontal_space={self._horizontal_color_space!r})"
        )