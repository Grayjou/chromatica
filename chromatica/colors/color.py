from __future__ import annotations
from .color_base import ColorBase, ColorValue
from .hsl import hsl_tuple_to_class
from .rgb import rgb_tuple_to_class
from .hsv import hsv_tuple_to_class
from ..types.format_type import FormatType, max_non_hue
from ..conversions import convert, np_convert
from ..types.color_types import  ColorMode
from typing import Optional
from numpy import ndarray
import numpy as np
unified_tuple_to_class: dict[tuple[ColorMode, FormatType], type[ColorBase]]  = {**rgb_tuple_to_class, **hsl_tuple_to_class, **hsv_tuple_to_class}

def color_convert(self: ColorBase, to_space: ColorMode | None = None, to_format: FormatType | None = None, *, use_css_algo:bool = False) -> ColorBase:
    """
    Convert this color to a different color space and/or format.
    
    Automatically detects whether the value is a scalar or array and uses
    the appropriate conversion function (convert for scalars, np_convert for arrays).
    
    Args:
        to_space: Target color space (e.g., "rgb", "hsv", "hsl")
        to_format: Target format type (INT, FLOAT, PERCENTAGE). Defaults to current format.
        use_css_algo: Whether to use CSS Color 4/Culori algorithms for conversions
        
    Returns:
        New ColorBase instance in the target space/format
    """
    to_space = to_space or self.mode
    to_space = to_space.lower() # type: ignore
    from_space = self.mode
    from_format = self.format_type
    to_format = to_format or from_format

    # Check if value is an array or scalar
    if isinstance(self.value, ndarray):
        # Use vectorized conversion for arrays
        result = np_convert(
            color=self.value,
            from_space=from_space,
            to_space=to_space,
            input_type=from_format,
            output_type=to_format,
            use_css_algo=use_css_algo
        )
    else:
        # Use scalar conversion for tuples/scalars
        result = convert(
            color=self.value,
            from_space=from_space,
            to_space=to_space,
            input_type=from_format,
            output_type=to_format,
            use_css_algo=use_css_algo
        )

    cls = unified_tuple_to_class[(to_space, to_format)]
    return cls(result)

ALPHA_SPACE = {
    ColorMode.RGB:  ColorMode.RGBA,
    ColorMode.HSL:  ColorMode.HSLA,
    ColorMode.HSV:  ColorMode.HSVA,
}

def with_alpha(self: ColorBase, alpha: Optional[ColorValue] = None) -> ColorBase:
    if self.has_alpha:
        return self

    if alpha is None:
        alpha = max_non_hue[self.format_type]

    if isinstance(self.value, ndarray):
        if isinstance(alpha, ndarray):
            expected_shape = self.value.shape[:-1]
            if alpha.shape != expected_shape:
                raise ValueError(
                    f"Alpha array shape {alpha.shape} doesn't match color shape {expected_shape}"
                )
            alpha_array = np.expand_dims(alpha, axis=-1)
        else:
            alpha_array = np.full(
                self.value.shape[:-1] + (1,),
                alpha,
                dtype=self.value.dtype,
            )
        new_value = np.concatenate([self.value, alpha_array], axis=-1)
    else:
        if isinstance(alpha, ndarray):
            raise TypeError("Cannot use array alpha with scalar color value")
        new_value = tuple(self.value) + (alpha,)

    new_mode = ALPHA_SPACE[self.mode]
    cls = unified_tuple_to_class[(new_mode, self.format_type)]
    return cls(new_value)

ColorBase.convert = color_convert
ColorBase.with_alpha = with_alpha


def get_color_class(color_mode: ColorMode, format_type: FormatType):
    color_mode = ColorMode(color_mode)
    color_class = unified_tuple_to_class.get((color_mode, format_type))
    if color_class is None:
        raise ValueError(
            f"Unsupported color space/format combination: {color_mode}/{format_type}"
        )
    return color_class


def convert_color(value, color_mode: ColorMode, format_type: FormatType):
    color_class = get_color_class(color_mode, format_type)
    if isinstance(value, ColorBase):
        return value.convert(color_mode, format_type)  # type: ignore
    return color_class(value)
