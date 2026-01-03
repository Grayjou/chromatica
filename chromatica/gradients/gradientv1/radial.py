from __future__ import annotations

import numpy as np
from numpy import ndarray as NDArray
from typing import Callable, List, Optional, Tuple, Union

from ...colors.color_base import ColorBase
from ...types.format_type import FormatType
from ...colors.color import convert_color, get_color_class


def radial_gradient(
    color1: Union[ColorBase, Tuple, int],
    color2: Union[ColorBase, Tuple, int],
    height: int,
    width: int,
    center: Union[Tuple[int, int], List[int]] = (0, 0),
    radius: float = 1.0,
    color_mode: str = "rgb",
    format_type: FormatType = FormatType.FLOAT,
    unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
    outside_fill: Optional[Union[ColorBase, Tuple, int]] = None,
    start: float = 0.0,
    end: float = 1.0,
    offset: float = 0.0,
    base: Optional[NDArray] = None,
) -> NDArray:
    """
    Create a radial gradient that radiates from a center point.

    Args:
        color1: Inner color (ColorBase instance or tuple)
        color2: Outer color (ColorBase instance or tuple)
        height: Height of the output array
        width: Width of the output array
        center: (x, y) center position of the gradient
        radius: Radius of the gradient in pixels
        color_mode: Target color space ('rgb', 'hsv', 'hsl', etc.)
        format_type: Format type (INT or FLOAT)
        unit_transform: Optional transformation of normalized distances
        outside_fill: Optional color to fill areas outside the gradient
        start: Start position of gradient (0-1 range)
        end: End position of gradient (0-1 range)
        offset: Offset to apply to normalized distances
        base: Optional base array to blend with

    Returns:
        NDArray with shape (height, width, channels)

    Notes:
        If `base` is provided and `outside_fill` is None, the gradient overwrites
        `base` only within the gradient area, leaving the rest untouched.
        If `outside_fill` is provided, it fills areas outside the gradient.
    """
    color_mode = color_mode.lower()

    color_class = get_color_class(color_mode, format_type)

    col1 = convert_color(color1, color_mode, format_type)
    col2 = convert_color(color2, color_mode, format_type)

    if outside_fill is not None:
        if isinstance(outside_fill, ColorBase):
            outside_fill_color = outside_fill.convert(color_mode, format_type).value  # type: ignore
        else:
            outside_fill_color = color_class(outside_fill).value
    else:
        outside_fill_color = None

    c1 = np.array(col1.value, dtype=float)
    c2 = np.array(col2.value, dtype=float)

    y, x = np.indices((height, width), dtype=float)
    cx, cy = center
    dx = x - cx
    dy = y - cy
    distance = np.sqrt(dx**2 + dy**2)

    unit_array = (distance / radius) - offset

    if unit_transform is not None:
        unit_array = np.where(
            (unit_array > 1.0) | (unit_array < 0.0),
            unit_array,
            unit_transform(unit_array),
        )

    unit_array_clipped = np.clip(unit_array, 0.0, 1.0)

    gradient = (
        c1 * (1 - unit_array_clipped[..., None]) +
        c2 * unit_array_clipped[..., None]
    )

    if base is not None:
        if base.shape != gradient.shape:
            raise ValueError(f"`base` shape {base.shape} does not match gradient shape {gradient.shape}")
        result = base.copy()
    else:
        result = gradient.copy()

    mask_inside = (
        (unit_array >= start) & (unit_array <= end) &
        (unit_array >= 0.0) & (unit_array <= 1.0)
    )
    mask_outside = ~mask_inside

    if outside_fill_color is not None:
        result[mask_outside] = outside_fill_color
    else:
        if base is not None:
            result[mask_inside] = gradient[mask_inside]
        else:
            result = gradient

    if format_type == FormatType.INT:
        result = np.round(result).astype(np.uint16)
        # Ensure hue values are wrapped to [0, 360) for hue-based color spaces
        if color_mode in ("hsv", "hsl", "hsva", "hsla"):
            result[..., 0] = result[..., 0] % 360
    else:
        result = result.astype(np.float32)

    return result
