from typing import ClassVar, Tuple
from ..types.format_type import FormatType
from ..types.color_types import ColorSpace
from .color_base import ColorBase, WithAlpha, build_registry


class ColorRGBINT(ColorBase):
    num_channels: ClassVar[int] = 3
    mode: ClassVar[ColorSpace] = ColorSpace.RGB
    _type: ClassVar[type] = int
    maxima: ClassVar[Tuple[int, int, int]] = (255, 255, 255)
    null_value: ClassVar[Tuple[int, int, int]] = (0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT


class ColorRGBAINT(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode: ClassVar[ColorSpace] = ColorSpace.RGBA
    _type: ClassVar[type] = int
    maxima: ClassVar[Tuple[int, int, int, int]] = (255, 255, 255, 255)
    null_value: ClassVar[Tuple[int, int, int, int]] = (0, 0, 0, 0)
    format_type: ClassVar[FormatType] = FormatType.INT


class ColorUnitRGB(ColorBase):
    num_channels: ClassVar[int] = 3
    mode: ClassVar[ColorSpace] = ColorSpace.RGB
    _type: ClassVar[type] = float
    maxima: ClassVar[Tuple[float, float, float]] = (1.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT


class ColorUnitRGBA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode: ClassVar[ColorSpace] = ColorSpace.RGBA
    _type: ClassVar[type] = float
    maxima: ClassVar[Tuple[float, float, float, float]] = (1.0, 1.0, 1.0, 1.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.FLOAT


class ColorPercentageRGB(ColorBase):
    num_channels: ClassVar[int] = 3
    mode: ClassVar[ColorSpace] = ColorSpace.RGB
    _type: ClassVar[type] = float
    maxima: ClassVar[Tuple[float, float, float]] = (100.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE


class ColorPercentageRGBA(ColorBase, WithAlpha):
    num_channels: ClassVar[int] = 4
    mode: ClassVar[ColorSpace] = ColorSpace.RGBA
    _type: ClassVar[type] = float
    maxima: ClassVar[Tuple[float, float, float, float]] = (100.0, 100.0, 100.0, 100.0)
    null_value: ClassVar[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)
    format_type: ClassVar[FormatType] = FormatType.PERCENTAGE


RGB = ColorRGBINT
RGBA = ColorRGBAINT


rgb_tuple_to_class = build_registry(
    ColorRGBINT,
    ColorRGBAINT,
    ColorUnitRGB,
    ColorUnitRGBA,
    ColorPercentageRGB,
    ColorPercentageRGBA,
)
