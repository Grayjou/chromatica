from typing import ClassVar, Tuple
from ..types.format_type import FormatType
from ..types.color_types import ColorSpace
from .color_base import ColorBase, WithAlpha, build_registry

class ColorHSVINT(ColorBase):
    num_channels: ClassVar[int] = 3
    mode:       ClassVar[ColorSpace] = ColorSpace.HSV
    _type:      ClassVar[type] = int
    maxima:     ClassVar[Tuple[int, int, int]] = (360, 255, 255)
    null_value: ClassVar[Tuple[int, int, int]] = (0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT

class ColorHSVAINT(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode:       ClassVar[ColorSpace] = ColorSpace.HSVA
    _type:      ClassVar[type] = int
    maxima:     ClassVar[Tuple[int, int, int, int]] = (360, 255, 255, 255)
    null_value: ClassVar[Tuple[int, int, int, int]] = (0, 0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT

class UnitHSV(ColorBase):
    num_channels: ClassVar[int] = 3
    mode:       ClassVar[ColorSpace] = ColorSpace.HSV
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float]] = (360.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT

class UnitHSVA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode:       ClassVar[ColorSpace] = ColorSpace.HSVA
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float, float]] = (360.0, 1.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT

class PercentageHSV(ColorBase):
    num_channels: ClassVar[int] = 3
    mode:       ClassVar[ColorSpace] = ColorSpace.HSV
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float]] = (360.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE

class PercentageHSVA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode:       ClassVar[ColorSpace] = ColorSpace.HSVA
    _type:      ClassVar[type] = float
    maxima:     ClassVar[Tuple[float, float, float, float]] = (360.0, 100.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE

HSV = ColorHSVINT
HSVA = ColorHSVAINT



hsv_tuple_to_class = build_registry(
    ColorHSVINT,
    ColorHSVAINT,
    UnitHSV,
    UnitHSVA,
    PercentageHSV,
    PercentageHSVA,
)
