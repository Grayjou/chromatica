from typing import ClassVar, Tuple
from ..types.format_type import FormatType
from ..types.color_types import ColorMode
from .color_base import ColorBase, WithAlpha, build_registry


class ColorHSLINT(ColorBase):
    num_channels: ClassVar[int] = 3
    mode: ClassVar[ColorMode] = ColorMode.HSL
    _type: ClassVar[type] = int
    maxima: ClassVar[Tuple[int, int, int]] = (360, 255, 255)
    null_value: ClassVar[Tuple[int, int, int]] = (0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT


class ColorHSLAINT(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode: ClassVar[ColorMode] = ColorMode.HSLA
    _type: ClassVar[type] = int
    maxima: ClassVar[Tuple[int, int, int, int]] = (360, 255, 255, 255)
    null_value: ClassVar[Tuple[int, int, int, int]] = (0, 0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT


class UnitHSL(ColorBase):
    num_channels: ClassVar[int] = 3
    mode: ClassVar[ColorMode] = ColorMode.HSL
    _type: ClassVar[type] = float
    maxima: ClassVar[Tuple[float, float, float]] = (360.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT


class UnitHSLA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode: ClassVar[ColorMode] = ColorMode.HSLA
    _type: ClassVar[type] = float
    maxima: ClassVar[Tuple[float, float, float, float]] = (360.0, 1.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT


class PercentageHSL(ColorBase):
    num_channels: ClassVar[int] = 3
    mode: ClassVar[ColorMode] = ColorMode.HSL
    _type: ClassVar[type] = float
    maxima: ClassVar[Tuple[float, float, float]] = (360.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE


class PercentageHSLA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode: ClassVar[ColorMode] = ColorMode.HSLA
    _type: ClassVar[type] = float
    maxima: ClassVar[Tuple[float, float, float, float]] = (360.0, 100.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE


HSL = ColorHSLINT
HSLA = ColorHSLAINT


hsl_tuple_to_class = build_registry(
    ColorHSLINT,
    ColorHSLAINT,
    UnitHSL,
    UnitHSLA,
    PercentageHSL,
    PercentageHSLA,
)
