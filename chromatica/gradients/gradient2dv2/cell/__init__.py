"""Cell module for 2D gradient cells.

This module provides cell classes for 2D gradient interpolation:
- CellBase: Abstract base class
- LinesCell: Cell defined by top and bottom lines
- CornersCell: Cell defined by corner colors
- CornersCellDual: Cell with dual color spaces (horizontal/vertical)

Factory functions for creating cells with automatic color conversion:
- get_transformed_lines_cell
- get_transformed_corners_cell
- get_transformed_corners_cell_dual
"""

from .base import CellBase
from .lines import LinesCell
from .corners import CornersCell
from .corners_dual import CornersCellDual
from .factory import (
    get_transformed_lines_cell,
    get_transformed_corners_cell,
    get_transformed_corners_cell_dual,
)

__all__ = [
    # Base class
    'CellBase',
    # Cell types
    'LinesCell',
    'CornersCell',
    'CornersCellDual',
    # Factory functions
    'get_transformed_lines_cell',
    'get_transformed_corners_cell',
    'get_transformed_corners_cell_dual',
]
