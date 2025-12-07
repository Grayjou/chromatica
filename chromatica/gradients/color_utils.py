from __future__ import annotations

from ..colors.color_base import ColorBase
from ..colors.hsl import UnitHSL, ColorHSLINT, UnitHSLA, ColorHSLAINT
from ..colors.hsv import UnitHSV, ColorHSVINT, UnitHSVA, ColorHSVAINT
from ..colors.rgb import ColorUnitRGB, ColorRGBINT, ColorUnitRGBA, ColorRGBAINT
from ..format_type import FormatType


# Color type mapping for convenience
COLOR_CLASSES = {
    ("rgb", FormatType.INT): ColorRGBINT,
    ("rgb", FormatType.FLOAT): ColorUnitRGB,
    ("rgba", FormatType.INT): ColorRGBAINT,
    ("rgba", FormatType.FLOAT): ColorUnitRGBA,
    ("hsv", FormatType.INT): ColorHSVINT,
    ("hsv", FormatType.FLOAT): UnitHSV,
    ("hsva", FormatType.INT): ColorHSVAINT,
    ("hsva", FormatType.FLOAT): UnitHSVA,
    ("hsl", FormatType.INT): ColorHSLINT,
    ("hsl", FormatType.FLOAT): UnitHSL,
    ("hsla", FormatType.INT): ColorHSLAINT,
    ("hsla", FormatType.FLOAT): UnitHSLA,
}


def get_color_class(color_space: str, format_type: FormatType):
    color_class = COLOR_CLASSES.get((color_space, format_type))
    if color_class is None:
        raise ValueError(
            f"Unsupported color space/format combination: {color_space}/{format_type}"
        )
    return color_class


def convert_color(value, color_space: str, format_type: FormatType):
    color_class = get_color_class(color_space, format_type)
    if isinstance(value, ColorBase):
        return value.convert(color_space, format_type)  # type: ignore
    return color_class(value)
