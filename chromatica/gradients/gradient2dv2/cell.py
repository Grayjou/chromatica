#chromatica\gradients\gradient2dv2\cell.py
"""
GradientCell module for 2D gradient interpolation.

This module provides backward compatibility by re-exporting from the new cell submodule.
The implementation has been split into:
- cell/base.py: CellBase abstract class
- cell/lines.py: LinesCell implementation
- cell/corners.py: CornersCell implementation
- cell/corners_dual.py: CornersCellDual implementation
- cell/factory.py: Factory functions for creating cells
"""

from __future__ import annotations

# Re-export everything from the new cell module for backward compatibility
from .cell import (
    CellBase,
    LinesCell,
    CornersCell,
    CornersCellDual,
    get_transformed_lines_cell,
    get_transformed_corners_cell,
    get_transformed_corners_cell_dual,
)

# Re-export dependencies for backward compatibility
from .helpers import CellMode, LineInterpMethods
from ...types.color_types import HueMode, ColorSpace
from .partitions import (
    PerpendicularPartition, 
    PartitionInterval, 
    PerpendicularDualPartition, 
    CellDualPartitionInterval
)

__all__ = [
    'CellBase',
    'LinesCell',
    'CornersCell',
    'CornersCellDual',
    'get_transformed_lines_cell',
    'get_transformed_corners_cell',
    'get_transformed_corners_cell_dual',
    'CellMode',
    'LineInterpMethods',
    'HueMode',
    'ColorSpace',
    'PerpendicularPartition',
    'PartitionInterval',
    'PerpendicularDualPartition',
    'CellDualPartitionInterval',
]
