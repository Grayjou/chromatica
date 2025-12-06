"""Basic Chromatica usage examples.

Run directly with:
    python examples/basic_usage.py
"""
from chromatica import (
    ColorRGB,
    ColorHSV,
    Gradient1D,
    Gradient2D,
    color_convert,
)
from chromatica.conversions.format_type import FormatType


def demonstrate_colors() -> None:
    # Construct typed colors and convert between spaces.
    accent = ColorRGB((255, 128, 64))
    print("RGB as floats:", accent.unit_values)

    converted = color_convert(accent.value, "rgb", "hsv", input_type="int", output_type="int")
    print("RGB -> HSV (int):", converted)

    hsv_color = ColorHSV(converted)
    print("HSV -> RGB (float):", hsv_color.to_rgb().unit_values)


def demonstrate_gradients() -> None:
    # Build gradients in RGB and HSV spaces.
    strip = Gradient1D.from_colors(
        (255, 0, 0),
        (0, 0, 255),
        steps=10,
        color_space="rgb",
        format_type=FormatType.INT,
    )
    print("1D RGB gradient sample:", strip.value[:3])

    # Hue-aware interpolation from green to purple in HSV with clockwise hue direction.
    ring = Gradient1D.from_colors(
        (120, 100, 100),
        (300, 100, 100),
        steps=10,
        color_space="hsv",
        format_type=FormatType.INT,
        direction="cw",
    )
    print("1D HSV gradient sample:", ring.value[:3])

    # Simple 2D gradient between four corners.
    grid = Gradient2D.from_colors(
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        width=4,
        height=4,
        format_type=FormatType.INT,
    )
    print("2D RGB gradient shape:", grid.value.shape)


if __name__ == "__main__":
    demonstrate_colors()
    demonstrate_gradients()
