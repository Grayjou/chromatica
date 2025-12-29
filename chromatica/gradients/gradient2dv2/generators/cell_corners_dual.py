# chromatica/gradients/gradient2dv2/generators/cell_corners_dual.py
"""Factory for creating CornersCellDual instances."""

from __future__ import annotations
from typing import List, Optional, Union, cast

import numpy as np

from ....types.format_type import FormatType
from ....types.transform_types import PerChannelCoords
from ..cell.corners_dual import CornersCellDual
from ..cell.factory import get_transformed_corners_cell_dual
from ..partitions import PerpendicularDualPartition, PartitionInterval, IndexRoundingMode
from .partition_utils import compute_partition_slices
from .slice_utils import slice_pcc_with_padding
from .cell_corners_dual_properties import CornersCellDualFactoryProperties


class CornersCellDualFactory(CornersCellDualFactoryProperties):
    """Factory for creating and manipulating CornersCellDual instances.
    
    Provides lazy cell creation, partition slicing, and value rendering
    with support for dual color space interpolation.
    
    Example:
        factory = CornersCellDualFactory(
            width=100, height=100,
            top_left=np.array([1, 0, 0]),
            top_right=np.array([0, 1, 0]),
            bottom_left=np.array([0, 0, 1]),
            bottom_right=np.array([1, 1, 0]),
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
        )
        
        # Get rendered gradient
        image = factory.get_value()
        
        # Partition into segments
        partitions = factory.partition_slice(partition, padding=1)
    """
    
    def get_per_channel_coords(self) -> PerChannelCoords:
        """Get per-channel coordinates, creating base coords if needed."""
        if self._per_channel_coords is None:
            return self.base_coords()
        return self._per_channel_coords
    
    def get_cell(self) -> CornersCellDual:
        """Get or create the CornersCellDual instance."""
        if self._cell is None:
            base_coords = self.get_per_channel_coords()
            self._cell = get_transformed_corners_cell_dual(
                top_left=self._top_left,
                top_right=self._top_right,
                bottom_left=self._bottom_left,
                bottom_right=self._bottom_right,
                per_channel_coords=base_coords,
                vertical_color_space=self._vertical_color_space,
                horizontal_color_space=self._horizontal_color_space,
                hue_direction_y=self._hue_direction_y,
                hue_direction_x=self._hue_direction_x,
                input_format=FormatType.FLOAT,
                per_channel_transforms=self._per_channel_transforms,
                boundtypes=self._boundtypes,
                top_segment_hue_direction_x=self._top_segment_hue_direction_x,
                bottom_segment_hue_direction_x=self._bottom_segment_hue_direction_x,
                top_segment_color_space=self._top_segment_color_space,
                bottom_segment_color_space=self._bottom_segment_color_space,
                top_left_grayscale_hue=self._top_left_grayscale_hue,
                top_right_grayscale_hue=self._top_right_grayscale_hue,
                bottom_left_grayscale_hue=self._bottom_left_grayscale_hue,
                bottom_right_grayscale_hue=self._bottom_right_grayscale_hue,
                )
        return self._cell
    def _interpolate_grayscale_hue(
        self, 
        horizontal_pos: float, 
        is_top: bool,
        respect_hue_direction: bool = True,
    ) -> Optional[float]:
        """Interpolate grayscale hue along an edge with proper hue wrapping.
        
        Args:
            horizontal_pos: Position along edge, 0.0 = left, 1.0 = right
            is_top: If True, interpolate top edge; else bottom edge
            respect_hue_direction: If True, use hue_direction_x for interpolation
            
        Returns:
            Interpolated hue value, or None if both endpoints are None
        """
        if is_top:
            left_hue = self._top_left_grayscale_hue
            right_hue = self._top_right_grayscale_hue
        else:
            left_hue = self._bottom_left_grayscale_hue
            right_hue = self._bottom_right_grayscale_hue
        
        # Handle None cases
        if left_hue is None and right_hue is None:
            return None
        if left_hue is None:
            return right_hue
        if right_hue is None:
            return left_hue
        
        # If at endpoints, return exact value
        if horizontal_pos <= 0.0:
            return left_hue
        if horizontal_pos >= 1.0:
            return right_hue
        
        # Use hue-aware interpolation if available
        if respect_hue_direction and self._hue_direction_x is not None:
            return self._interpolate_hue_with_direction(
                left_hue, right_hue, horizontal_pos, self._hue_direction_x
            )
        
        # Default: shortest path hue interpolation (wrapping at 1.0 or 360.0)
        return self._interpolate_hue_shortest(left_hue, right_hue, horizontal_pos)

    def _interpolate_hue_shortest(
        self, 
        start_hue: float, 
        end_hue: float, 
        t: float,
        max_hue: float = 1.0,  # Use 360.0 if your hues are in degrees
    ) -> float:
        """Interpolate hue using shortest path around the color wheel."""
        diff = end_hue - start_hue
        
        # Wrap to find shortest path
        if abs(diff) > max_hue / 2:
            if diff > 0:
                diff -= max_hue
            else:
                diff += max_hue
        
        result = start_hue + t * diff
        
        # Normalize to [0, max_hue)
        return result % max_hue

    def _interpolate_hue_with_direction(
        self,
        start_hue: float,
        end_hue: float,
        t: float,
        direction: str,
        max_hue: float = 1.0,
    ) -> float:
        """Interpolate hue respecting the specified direction."""
        from ....types.color_types import HueMode
        
        if direction == HueMode.SHORTEST:
            return self._interpolate_hue_shortest(start_hue, end_hue, t, max_hue)
        elif direction == HueMode.LONGEST:
            # Take the long way around
            diff = end_hue - start_hue
            if abs(diff) <= max_hue / 2:
                if diff > 0:
                    diff -= max_hue
                else:
                    diff += max_hue
            return (start_hue + t * diff) % max_hue
        elif direction == HueMode.CW:
            diff = end_hue - start_hue
            if diff < 0:
                diff += max_hue
            return (start_hue + t * diff) % max_hue
        elif direction == HueMode.CCW:
            diff = end_hue - start_hue
            if diff > 0:
                diff -= max_hue
            return (start_hue + t * diff) % max_hue
        else:
            # Fallback to linear
            return start_hue + t * (end_hue - start_hue)
    def get_value(self, init_cell: bool = True) -> Optional[np.ndarray]:
        """Get the rendered cell value, creating the cell if needed.
        
        Args:
            init_cell: If True, create cell if it doesn't exist
            
        Returns:
            Rendered gradient as numpy array, or None if cell not created
        """
        if self._cell is None and init_cell:
            self.get_cell()
        if self._cell is not None:
            return self._cell.get_value()
        return None
    
    def partition_slice(
        self,
        partition: PerpendicularDualPartition,
        padding: int = 1,
        *,
        pure_partition: bool = False,
        index_rounding_mode: IndexRoundingMode = IndexRoundingMode.ROUND,
    ) -> List[CornersCellDualFactory]:
        """Create partitioned CornersCellDualFactory instances based on the given partition.
        
        Args:
            partition: The partition defining slice boundaries
            padding: Padding to apply at shared edges.
                0 = edge goes to left slice only
                1 = both sides get the edge
                2 = left gets two copies, right gets one
            pure_partition: If True, use transformed coordinates without renormalization
            index_rounding_mode: How to round fractional pixel indices
            
        Returns:
            List of partitioned CornersCellDualFactory instances
        """
        cell = self.get_cell()
        corner_function = (
            cell.simple_untransformed_interpolate_edge
            if not pure_partition
            else cell.interpolate_edge
        )
        
        factories: List[CornersCellDualFactory] = []
        specs = compute_partition_slices(partition, self._width, padding, index_rounding_mode)
        if len(specs) == 1:
            return [self.copy_with()]
        
        # Create factories for each interval
        for spec in specs:
            interval = cast(PartitionInterval, spec.interval)
            
            # Get corner colors for this slice
            top_left = corner_function(spec.start_frac, is_top_edge=True)
            top_right = corner_function(spec.end_frac, is_top_edge=True)
            bottom_left = corner_function(spec.start_frac, is_top_edge=False)
            bottom_right = corner_function(spec.end_frac, is_top_edge=False)
            # === INTERPOLATE GRAYSCALE HUES ===
            slice_tl_ghue = self._interpolate_grayscale_hue(spec.start_frac, is_top=True)
            slice_tr_ghue = self._interpolate_grayscale_hue(spec.end_frac, is_top=True)
            slice_bl_ghue = self._interpolate_grayscale_hue(spec.start_frac, is_top=False)
            slice_br_ghue = self._interpolate_grayscale_hue(spec.end_frac, is_top=False)
            # Resolve interval properties with fallbacks
            v_space = interval.vertical_color_space or self._vertical_color_space
            h_space = interval.horizontal_color_space or self._horizontal_color_space
            top_seg_space = interval.top_segment_color_space or self._top_segment_color_space
            bottom_seg_space = interval.bottom_segment_color_space or self._bottom_segment_color_space
            
            hue_x = interval.hue_direction_x if interval.hue_direction_x is not None else self._hue_direction_x
            hue_y = interval.hue_direction_y if interval.hue_direction_y is not None else self._hue_direction_y
            top_hue_x = interval.top_segment_hue_direction_x or self._top_segment_hue_direction_x
            bottom_hue_x = interval.bottom_segment_hue_direction_x or self._bottom_segment_hue_direction_x
            
            slice_width = int(spec.width)
            
            # Handle per_channel_coords for pure_partition mode
            sliced_pcc = None
            if pure_partition:
                sliced_pcc = slice_pcc_with_padding(
                    cell.per_channel_coords,
                    spec.px_start,
                    spec.px_end,
                    spec.pad_left,
                    spec.pad_right,
                )
            
            # Create the factory for this slice
            factory = self.copy_with(
                width=slice_width,
                top_left=top_left,
                top_right=top_right,
                bottom_left=bottom_left,
                bottom_right=bottom_right,
                vertical_color_space=v_space,
                horizontal_color_space=h_space,
                top_segment_color_space=top_seg_space,
                bottom_segment_color_space=bottom_seg_space,
                hue_direction_x=hue_x,
                hue_direction_y=hue_y,
                top_segment_hue_direction_x=top_hue_x,
                bottom_segment_hue_direction_x=bottom_hue_x,
                top_left_grayscale_hue=slice_tl_ghue,
                top_right_grayscale_hue=slice_tr_ghue,
                bottom_left_grayscale_hue=slice_bl_ghue,
                bottom_right_grayscale_hue=slice_br_ghue,
            )
            
            # Attach pre-sliced cell for pure_partition mode
            if pure_partition and sliced_pcc is not None:
                sliced_cell = get_transformed_corners_cell_dual(
                    top_left=top_left,
                    top_right=top_right,
                    bottom_left=bottom_left,
                    bottom_right=bottom_right,
                    per_channel_coords=sliced_pcc,
                    vertical_color_space=v_space,
                    horizontal_color_space=h_space,
                    hue_direction_y=hue_y,
                    hue_direction_x=hue_x,
                    input_format=FormatType.FLOAT,
                    per_channel_transforms=self._per_channel_transforms,
                    boundtypes=self._boundtypes,
                    top_segment_hue_direction_x=top_hue_x,
                    bottom_segment_hue_direction_x=bottom_hue_x,
                    top_segment_color_space=top_seg_space,
                    bottom_segment_color_space=bottom_seg_space,
                    top_left_grayscale_hue=slice_tl_ghue,
                    top_right_grayscale_hue=slice_tr_ghue,
                    bottom_left_grayscale_hue=slice_bl_ghue,
                    bottom_right_grayscale_hue=slice_br_ghue,
                )
                factory._cell = sliced_cell
                factory._per_channel_coords = sliced_pcc
            
            factories.append(factory)
        
        return factories
    
    # Local pcc slicing helper removed; using shared slice_pcc_with_padding
    
    def copy_with(self, **kwargs) -> CornersCellDualFactory:
        """Create a copy of this factory with modified properties.
        
        Args:
            **kwargs: Properties to override
            
        Returns:
            New CornersCellDualFactory with specified overrides
        """
        params = {
            'width': self._width,
            'height': self._height,
            'top_left': self._top_left,
            'top_right': self._top_right,
            'bottom_left': self._bottom_left,
            'bottom_right': self._bottom_right,
            'vertical_color_space': self._vertical_color_space,
            'horizontal_color_space': self._horizontal_color_space,
            'top_segment_color_space': self._top_segment_color_space,
            'bottom_segment_color_space': self._bottom_segment_color_space,
            'hue_direction_x': self._hue_direction_x,
            'hue_direction_y': self._hue_direction_y,
            'top_segment_hue_direction_x': self._top_segment_hue_direction_x,
            'bottom_segment_hue_direction_x': self._bottom_segment_hue_direction_x,
            'per_channel_coords': self._per_channel_coords,
            'per_channel_transforms': self._per_channel_transforms,
            'boundtypes': self._boundtypes,
            'border_mode': self._border_mode,
            'border_value': self._border_value,
            'input_format': FormatType.FLOAT,
            'top_left_grayscale_hue': self._top_left_grayscale_hue,
            'top_right_grayscale_hue': self._top_right_grayscale_hue,
            'bottom_left_grayscale_hue': self._bottom_left_grayscale_hue,
            'bottom_right_grayscale_hue': self._bottom_right_grayscale_hue,
        }
        params.update(kwargs)
        return CornersCellDualFactory(**params)