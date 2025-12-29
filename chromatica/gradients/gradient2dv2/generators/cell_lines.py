# chromatica/gradients/gradient2dv2/generators/cell_lines.py
"""Factory for creating LinesCell instances."""

from __future__ import annotations
from typing import List, Optional, Callable, Dict, Union
import numpy as np

from ..cell.lines import LinesCell
from ..helpers import LineInterpMethods
from ..helpers.cell_utils import apply_per_channel_transforms_2d
from ..cell.factory import get_transformed_lines_cell
from ....types.format_type import FormatType
from ....types.color_types import ColorSpace
from ....types.transform_types import PerChannelCoords
from ....conversions import np_convert
from boundednumbers import BoundType
from unitfield import upbm_2d
from ..partitions import PerpendicularPartition, PartitionInterval, IndexRoundingMode
from .partition_utils import compute_partition_slices
from .slice_utils import slice_pcc_with_padding, slice_lines_with_padding
from typing import cast
from .cell_lines_properties import LinesCellFactoryProperties


class LinesCellFactory(LinesCellFactoryProperties):
    """Factory for creating and managing LinesCell instances.
    
    Extends LinesCellFactoryProperties with cell creation and partitioning methods.
    """
    def get_per_channel_coords(self) -> PerChannelCoords:
        if self.per_channel_coords is None:
            return self.base_coords()
        return self.per_channel_coords

    def get_cell(self) -> LinesCell:
        """Get or create the LinesCell instance."""
        if self._cell is None:
            # Create base coordinates
            base_coords = self.get_per_channel_coords()
            
            self._cell = get_transformed_lines_cell(
                top_line=self._top_line,
                bottom_line=self._bottom_line,
                per_channel_coords=base_coords,
                color_space=self.color_space,
                hue_direction_y=self.hue_direction_y,
                hue_direction_x=self.hue_direction_x,
                input_format=FormatType.FLOAT,
                per_channel_transforms=self.per_channel_transforms,
                line_method=self.line_method,
                boundtypes=self.boundtypes,
                border_mode=self.border_mode,
                border_value=self.border_value,
            )


        return self._cell
    
    def get_value(self) -> np.ndarray:
        """Get the rendered gradient value."""
        return self.get_cell().get_value()
    
# cell_lines.py - updated partition_slice method

    def partition_slice(
        self,
        partition: PerpendicularPartition,
        padding: int = 1,
        *,
        pure_partition: bool = False,
        index_rounding_mode: IndexRoundingMode = IndexRoundingMode.ROUND,
    ) -> List[LinesCellFactory]:
        """Create partitioned LinesCellFactory instances based on the given partition.

        Args:
            partition: The partition to slice the cell.
            padding: Padding to apply to each partition slice.
                0 means the edge gets into the left side of the slice and the right side doesn't get it.
                1 means both sides get the edge.
                2 means left gets two times the edge, right gets one.
                This happens because shared edges have no width mathematically, but in arrays, they do ¯\\_(ツ)_/¯
                So expect the resulting sum of widths to be width + ((partitions-1)*padding)//2
            pure_partition: If True, use pure partition slices without renormalization and new corners.
            index_rounding_mode: Rounding mode for converting breakpoints to pixel indices.

        Returns:
            List of partitioned LinesCellFactory instances.
        """
        cell = self.get_cell()
        factories: List[LinesCellFactory] = []
        
        specs = compute_partition_slices(partition, self.width, padding, index_rounding_mode)
        if len(specs) == 1:
            return [self.copy_with()]
        
        # Create factories for each interval
        for spec in specs:
            px_start = spec.px_start
            px_end = spec.px_end
            slice_width = int(spec.width)
            
            # Get interval properties with fallbacks
            interval = cast(PartitionInterval, spec.interval)
            color_space = interval.color_space if interval.color_space is not None else self._color_space
            hue_dir_x = interval.hue_direction_x if interval.hue_direction_x is not None else self._hue_direction_x
            hue_dir_y = interval.hue_direction_y if interval.hue_direction_y is not None else self._hue_direction_y
            
            # Slice lines for this interval
            top_slice, bottom_slice = slice_lines_with_padding(
                self._top_line, self._bottom_line, px_start, px_end, spec.pad_left, spec.pad_right
            )
            top_slice = np_convert(top_slice, from_space=self._color_space, to_space=color_space, input_type=FormatType.FLOAT, output_type=FormatType.FLOAT)
            bottom_slice = np_convert(bottom_slice, from_space=self._color_space, to_space=color_space, input_type=FormatType.FLOAT, output_type=FormatType.FLOAT)
            # Handle per_channel_coords for pure_partition mode
            sliced_pcc = None
            if pure_partition:
                sliced_pcc = slice_pcc_with_padding(
                    cell.per_channel_coords,
                    px_start,
                    px_end,
                    spec.pad_left,
                    spec.pad_right,
                )
            
            # Create factory
            factory = LinesCellFactory(
                width=slice_width,
                height=self._height,
                top_line=top_slice,
                bottom_line=bottom_slice,
                color_space=color_space,
                hue_direction_x=hue_dir_x,
                hue_direction_y=hue_dir_y,
                line_method=self._line_method,
                input_format=FormatType.FLOAT,
                per_channel_transforms=self._per_channel_transforms,
                boundtypes=self._boundtypes,
                border_mode=self._border_mode,
                border_value=self._border_value,
            )
            
            # Attach pre-sliced cell for pure_partition
            if pure_partition and sliced_pcc is not None:
                sliced_cell = LinesCell(
                    top_line=top_slice,
                    bottom_line=bottom_slice,
                    per_channel_coords=sliced_pcc,
                    color_space=color_space,
                    hue_direction_y=hue_dir_y,
                    hue_direction_x=hue_dir_x,
                    line_method=self._line_method,
                    boundtypes=self._boundtypes,
                    border_mode=self._border_mode,
                    border_value=self._border_value,
                )
                factory._cell = sliced_cell
            
            factories.append(factory)
        
        return factories
    
    # Local line slicing helper removed; using shared slice_lines_with_padding
    
    # Local pcc slicing helper removed; using shared slice_pcc_with_padding
    
    @classmethod
    def from_corners(
        cls,
        width: int,
        height: int,
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        color_space: ColorSpace,
        hue_direction_x: Optional[str] = None,
        hue_direction_y: Optional[str] = None,
        line_method: LineInterpMethods = LineInterpMethods.LINES_DISCRETE,
        **kwargs,
    ) -> LinesCellFactory:
        """Create a LinesCellFactory by generating lines from corner colors.
        
        The top and bottom lines are created as linear gradients between
        the respective corner colors.
        """
        from ...gradient1dv2.segment import get_transformed_segment
        from unitfield import flat_1d_upbm
        
        # Generate uniform coordinates for line width
        coords = [flat_1d_upbm(width)]
        
        # Create top line segment
        top_segment = get_transformed_segment(
            already_converted_start_color=top_left,
            already_converted_end_color=top_right,
            per_channel_coords=coords,
            color_space=color_space,
            hue_direction=hue_direction_x,
            homogeneous_per_channel_coords=True,
        )
        
        # Create bottom line segment
        bottom_segment = get_transformed_segment(
            already_converted_start_color=bottom_left,
            already_converted_end_color=bottom_right,
            per_channel_coords=coords,
            color_space=color_space,
            hue_direction=hue_direction_x,
            homogeneous_per_channel_coords=True,
        )
        
        return cls(
            width=width,
            height=height,
            top_line=top_segment.get_value(),
            bottom_line=bottom_segment.get_value(),
            color_space=color_space,
            hue_direction_x=hue_direction_x,
            hue_direction_y=hue_direction_y,
            line_method=line_method,
            input_format=FormatType.FLOAT,
            **kwargs,
        )