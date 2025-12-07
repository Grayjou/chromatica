from __future__ import annotations

import numpy as np
from numpy import ndarray as NDArray
from typing import Callable, Optional, Tuple, Union

from ..color_arr import Color1DArr
from ..colors.color_base import ColorBase
from ..format_type import FormatType
from .color_utils import convert_color, get_color_class


class Gradient1D(Color1DArr):
    """
    Represents a 1D gradient of colors with advanced interpolation.

    Extends Color1DArr with gradient-specific creation methods including:
    - Hue direction control for HSV/HSL (cw/ccw)
    - Custom interpolation transforms
    - Automatic color space handling
    """

    @classmethod
    def from_colors(
        cls,
        color1: Union[ColorBase, Tuple, int],
        color2: Union[ColorBase, Tuple, int],
        steps: int,
        color_space: str = "rgb",
        format_type: FormatType = FormatType.FLOAT,
        unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
        direction: Optional[str] = None,
    ) -> "Gradient1D":
        """
        Create a 1D gradient from two colors with optional hue direction control.

        Args:
            color1: First color (ColorBase instance or tuple/int)
            color2: Second color (ColorBase instance or tuple/int)
            steps: Number of steps in the gradient
            color_space: Target color space ('rgb', 'hsv', 'hsl', etc.)
            format_type: Format type (INT or FLOAT)
            unit_transform: Optional function to transform interpolation parameter
            direction: Hue direction for HSV/HSL - 'cw' (clockwise), 'ccw' (counter-clockwise),
                      or None for shortest path

        Returns:
            Gradient1D instance with interpolated colors
        """
        color_space = color_space.lower()

        color_class = get_color_class(color_space, format_type)

        c1 = convert_color(color1, color_space, format_type)
        c2 = convert_color(color2, color_space, format_type)

        start = np.array(c1.value, dtype=float)
        end = np.array(c2.value, dtype=float)

        u = np.linspace(0.0, 1.0, steps, dtype=float)[:, None]
        if unit_transform is not None:
            u = unit_transform(u)

        if color_space in ("hsv", "hsl", "hsva", "hsla"):
            h0 = start[0] % 360.0
            h1 = end[0] % 360.0

            if direction == "cw":
                if h1 <= h0:
                    h1 += 360.0
            elif direction == "ccw":
                if h1 >= h0:
                    h1 -= 360.0
            else:
                delta = h1 - h0
                if delta > 180.0:
                    h1 -= 360.0
                elif delta < -180.0:
                    h1 += 360.0

            dh = h1 - h0
            hues = (h0 + u * dh) % 360.0
            rest = start[1:] * (1 - u) + end[1:] * u
            colors = np.concatenate([hues, rest], axis=1)
        else:
            colors = start * (1 - u) + end * u

        if format_type == FormatType.INT:
            colors = np.round(colors).astype(np.uint16)
        else:
            colors = colors.astype(np.float32)

        gradient_color = color_class(colors)
        return cls(gradient_color)
