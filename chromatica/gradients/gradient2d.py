from __future__ import annotations

import numpy as np
from numpy import ndarray as NDArray
from typing import Callable, Optional, Tuple, Union

from ..color_arr import Color2DArr
from ..colors.color_base import ColorBase
from ..format_type import FormatType
from .color_utils import convert_color, get_color_class


class Gradient2D(Color2DArr):
    """
    Represents a 2D gradient from four corner colors with bilinear interpolation.

    Extends Color2DArr with gradient-specific creation methods.
    """

    @classmethod
    def from_colors(
        cls,
        color_tl: Union[ColorBase, Tuple, int],
        color_tr: Union[ColorBase, Tuple, int],
        color_bl: Union[ColorBase, Tuple, int],
        color_br: Union[ColorBase, Tuple, int],
        width: int,
        height: int,
        color_space: str = "rgb",
        format_type: FormatType = FormatType.FLOAT,
        unit_transform_x: Optional[Callable[[NDArray], NDArray]] = None,
        unit_transform_y: Optional[Callable[[NDArray], NDArray]] = None,
    ) -> "Gradient2D":
        """
        Create a 2D gradient from four corner colors with optional transforms.

        Args:
            color_tl: Top-left color
            color_tr: Top-right color
            color_bl: Bottom-left color
            color_br: Bottom-right color
            width: Number of columns
            height: Number of rows
            color_space: Target color space ('rgb', 'hsv', 'hsl', etc.)
            format_type: Format type (INT or FLOAT)
            unit_transform_x: Optional transformation of x interpolation factors
            unit_transform_y: Optional transformation of y interpolation factors

        Returns:
            Gradient2D instance with bilinearly interpolated colors
        """
        color_space = color_space.lower()

        color_class = get_color_class(color_space, format_type)

        corners = [
            convert_color(corner_color, color_space, format_type)
            for corner_color in [color_tl, color_tr, color_bl, color_br]
        ]

        tl, tr, bl, br = [np.array(c.value, dtype=float) for c in corners]

        x = np.linspace(0.0, 1.0, width, dtype=float)
        y = np.linspace(0.0, 1.0, height, dtype=float)

        if unit_transform_x is not None:
            x = unit_transform_x(x)
        if unit_transform_y is not None:
            y = unit_transform_y(y)

        xx, yy = np.meshgrid(x, y)

        colors = (
            (1 - xx)[:, :, None] * (1 - yy)[:, :, None] * tl
            + xx[:, :, None] * (1 - yy)[:, :, None] * tr
            + (1 - xx)[:, :, None] * yy[:, :, None] * bl
            + xx[:, :, None] * yy[:, :, None] * br
        )

        if format_type == FormatType.INT:
            colors = np.round(colors).astype(np.uint16)
        else:
            colors = colors.astype(np.float32)

        gradient_color = color_class(colors)
        return cls(gradient_color)
