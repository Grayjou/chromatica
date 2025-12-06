from typing import ClassVar, Tuple
from ..conversions.format_type import FormatType
from ..conversions import ColorSpace
from .color_base import ColorBase, WithAlpha

class ColorHSVINT(ColorBase):
    num_channels: ClassVar[int] = 3
    mode:       ClassVar[ColorSpace] = "hsv"
    _type:      ClassVar[type] = int
    maxima:     ClassVar[Tuple[int, int, int]] = (360, 255, 255)
    null_value: ClassVar[Tuple[int, int, int]] = (0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT

class ColorHSVAINT(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode:       ClassVar[ColorSpace] = "hsva"
    _type:      ClassVar[type] = int
    maxima:     ClassVar[Tuple[int, int, int, int]] = (360, 255, 255, 255)
    null_value: ClassVar[Tuple[int, int, int, int]] = (0, 0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT

class UnitHSV(ColorBase):
    num_channels: ClassVar[int] = 3
    mode:       ClassVar[ColorSpace] = "hsv"
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float]] = (360.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT

class UnitHSVA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode:       ClassVar[ColorSpace] = "hsva"
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float, float]] = (360.0, 1.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT

class PercentageHSV(ColorBase):
    num_channels: ClassVar[int] = 3
    mode:       ClassVar[ColorSpace] = "hsv"
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float]] = (360.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE

class PercentageHSVA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode:       ClassVar[ColorSpace] = "hsva"
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float, float]] = (360.0, 100.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE

HSV = ColorHSVINT
HSVA = ColorHSVAINT

hsv_tuple_to_class: dict[tuple[ColorSpace, FormatType], type[ColorBase]]  = {
    ("hsv", FormatType.INT): ColorHSVINT,
    ("hsva", FormatType.INT): ColorHSVAINT,
    ("hsv", FormatType.FLOAT): UnitHSV,
    ("hsva", FormatType.FLOAT): UnitHSVA,
    ("hsv", FormatType.PERCENTAGE): PercentageHSV,
    ("hsva", FormatType.PERCENTAGE): PercentageHSVA,
}