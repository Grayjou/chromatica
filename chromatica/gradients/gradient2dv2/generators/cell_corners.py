#chromatica\gradients\gradient2dv2\generators\cell_corners.py
from __future__ import annotations
from ..cell.corners import CornersCell
from ..cell.factory import get_transformed_corners_cell
import numpy as np
from typing import List, Optional, Callable, Dict, Union, cast
from ....types.color_types import ColorMode
from ....types.format_type import FormatType
from ..partitions import PerpendicularPartition, PartitionInterval, IndexRoundingMode
from .partition_utils import compute_partition_slices
from .slice_utils import slice_pcc_with_padding
from .cell_corners_properties import CornersCellFactoryProperties
from ....types.transform_types import PerChannelCoords
from ....conversions import np_convert


class CornersCellFactory(CornersCellFactoryProperties):
    
    def get_per_channel_coords(self) -> PerChannelCoords:
        if self.per_channel_coords is None:
            return self.base_coords()
        return self.per_channel_coords

    def get_cell(self) -> CornersCell:
        """Get or create the CornersCell instance."""
        if self._cell is None:
            base_coords = self.get_per_channel_coords()
            the_cell = get_transformed_corners_cell(
                top_left=self.top_left,
                top_right=self.top_right,
                bottom_left=self.bottom_left,
                bottom_right=self.bottom_right,
                per_channel_coords=base_coords,
                color_mode=self.color_mode,
                hue_direction_y=self.hue_direction_y,
                hue_direction_x=self.hue_direction_x,
                input_format=FormatType.FLOAT,
                per_channel_transforms=self.per_channel_transforms,
                boundtypes=self.boundtypes,
                border_mode=self.border_mode,
                border_value=self.border_value,
            )
            self._cell = the_cell
        return self._cell
    def partition_slice(self, partition: PerpendicularPartition, padding: int = 1, *, pure_partition: bool = False, index_rounding_mode: IndexRoundingMode = IndexRoundingMode.FLOOR) -> List[CornersCellFactory]:
        """Create partitioned CornersCellFactory instances based on the given partition.

        Args:
            partition (PerpendicularPartition): The partition to slice the cell.
            padding (int): Padding to apply to each partition slice.
                0 means the edge gets into the left side of the slice and the right side doesn't get it. 
                1 means both sides get the edge.
                2 means left gets two times the edge, right gets one.
                This happens because shared edges have no width mathematically, but in arrays, they do ¯\\_(ツ)_/¯
                So expect the resulting sum of widths to be width + ((partitions-1)*padding)//2     
            pure_partition (bool): If True, use pure partition slices without renormalization and new corners.

        Returns:
            List[CornersCellFactory]: List of partitioned CornersCellFactory instances.
        """
        cell = self.get_cell()
        corner_function = cell.simple_untransformed_interpolate_edge if not pure_partition else cell.interpolate_edge
        factories: List[CornersCellFactory] = []
        
        specs = compute_partition_slices(
            partition=partition,
            total_width=self.width,
            padding=padding,
            index_rounding_mode=index_rounding_mode,
        )
        if len(specs) == 1:
            return [self.copy_with()]

        # Create factories for each interval using computed specs
        for spec in specs:
            # Get corner colors for this slice
            if not pure_partition:
                top_left = corner_function(spec.start_frac, is_top_edge=True)
                top_right = corner_function(spec.end_frac, is_top_edge=True)
                bottom_left = corner_function(spec.start_frac, is_top_edge=False)
                bottom_right = corner_function(spec.end_frac, is_top_edge=False)
            else:
                top_left = self.top_left
                top_right = self.top_right
                bottom_left = self.bottom_left
                bottom_right = self.bottom_right
            
            # Get interval properties with fallbacks
            interval = cast(PartitionInterval, spec.interval)
            color_mode = interval.color_mode if interval.color_mode is not None else self.color_mode
            hue_dir_x = interval.hue_direction_x if interval.hue_direction_x is not None else self.hue_direction_x
            hue_dir_y = interval.hue_direction_y if interval.hue_direction_y is not None else self.hue_direction_y
            top_left = np_convert(top_left, self.color_mode, color_mode, FormatType.FLOAT, output_type=FormatType.FLOAT)
            top_right = np_convert(top_right, self.color_mode, color_mode, FormatType.FLOAT, output_type=FormatType.FLOAT)
            bottom_left = np_convert(bottom_left, self.color_mode, color_mode, FormatType.FLOAT, output_type=FormatType.FLOAT)
            bottom_right = np_convert(bottom_right, self.color_mode, color_mode, FormatType.FLOAT, output_type=FormatType.FLOAT)
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
                color_mode=color_mode,
                hue_direction_x=hue_dir_x,
                hue_direction_y=hue_dir_y,
            )
            factory._invalidate_for_dimension_change()
            
            # If pure_partition, attach the pre-sliced cell
            if pure_partition and sliced_pcc is not None:
                sliced_cell = get_transformed_corners_cell(
                    top_left=top_left,
                    top_right=top_right,
                    bottom_left=bottom_left,
                    bottom_right=bottom_right,
                    per_channel_coords=sliced_pcc,
                    color_mode=color_mode,
                    hue_direction_y=hue_dir_y,
                    hue_direction_x=hue_dir_x,
                    input_format=FormatType.FLOAT,
                    per_channel_transforms=self.per_channel_transforms,
                    boundtypes=self.boundtypes,
                    border_mode=self.border_mode,
                    border_value=self.border_value,
                )
                factory._cell = sliced_cell
                factory._per_channel_coords = sliced_pcc
            
            factories.append(factory)
        
        return factories


    # Local slicing helper removed; using shared slice_pcc_with_padding utility
    def get_value(self, init_cell:bool = True) -> Optional[np.ndarray]:
        """Get the rendered cell value, creating the cell if needed."""
        if self._cell is None and init_cell:
            self.get_cell()
        if self._cell is not None:
            self._cell: CornersCell = cast(CornersCell, self._cell)
            return self._cell.get_value()
            
    def copy_with(self, **kwargs) -> CornersCellFactory:
        """Create a copy of this factory with modified properties."""
        params = {
            'width': self.width,
            'height': self.height,
            'top_left': self.top_left,
            'top_right': self.top_right,
            'bottom_left': self.bottom_left,
            'bottom_right': self.bottom_right,
            'color_mode': self.color_mode,
            'hue_direction_x': self.hue_direction_x,
            'hue_direction_y': self.hue_direction_y,
            'per_channel_coords': self.per_channel_coords,
            'per_channel_transforms': self.per_channel_transforms,
            'boundtypes': self.boundtypes,
            'border_mode': self.border_mode,
            'border_value': self.border_value,
            'input_format': FormatType.FLOAT,
        }
        params.update(kwargs)
        return CornersCellFactory(**params)