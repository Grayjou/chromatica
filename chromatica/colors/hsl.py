from typing import ClassVar, Tuple
from ..format_type import FormatType
from ..conversions import ColorSpace
from .color_base import ColorBase, WithAlpha

class ColorHSLINT(ColorBase):
    num_channels: ClassVar[int] = 3
    mode:       ClassVar[ColorSpace] = "hsl"
    _type:      ClassVar[type] = int
    maxima:     ClassVar[Tuple[int, int, int]] = (360, 255, 255)
    null_value: ClassVar[Tuple[int, int, int]] = (0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT

class ColorHSLAINT(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode:       ClassVar[ColorSpace] = "hsla"
    _type:      ClassVar[type] = int
    maxima:     ClassVar[Tuple[int, int, int, int]] = (360, 255, 255, 255)
    null_value: ClassVar[Tuple[int, int, int, int]] = (0, 0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT

class UnitHSL(ColorBase):
    num_channels: ClassVar[int] = 3
    mode:       ClassVar[ColorSpace] = "hsl"
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float]] = (360.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT

class UnitHSLA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode:       ClassVar[ColorSpace] = "hsla"
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float, float]] = (360.0, 1.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT

class PercentageHSL(ColorBase):
    num_channels: ClassVar[int] = 3
    mode:       ClassVar[ColorSpace] = "hsl"
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float]] = (360.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE

class PercentageHSLA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode:       ClassVar[ColorSpace] = "hsla"
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float, float]] = (360.0, 100.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE

HSL = ColorHSLINT
HSLA = ColorHSLAINT

hsl_tuple_to_class: dict[tuple[ColorSpace, FormatType], type[ColorBase]]  = {
    ("hsl", FormatType.INT): ColorHSLINT,
    ("hsla", FormatType.INT): ColorHSLAINT,
    ("hsl", FormatType.FLOAT): UnitHSL,
    ("hsla", FormatType.FLOAT): UnitHSLA,
    ("hsl", FormatType.PERCENTAGE): PercentageHSL,
    ("hsla", FormatType.PERCENTAGE): PercentageHSLA,
}