"""Array-friendly color utilities in Chromatica.

Run with:
    python examples/array_operations.py
"""
import numpy as np

from chromatica import Color1DArr, Color2DArr
from chromatica.functions import clamp, cyclic_wrap_float
from chromatica.gradient import radial_gradient
from Chromatica.chromatica.format_type import FormatType


def demonstrate_arrays() -> None:
    # Build a 1D gradient and tile it to a small image.
    base = Color1DArr.from_colors(
        (255, 128, 64),
        (64, 128, 255),
        steps=8,
        color_space="rgb",
        format_type=FormatType.INT,
    )
    tiled = base.repeat(times=2, axis=1)
    print("Tiled gradient shape:", tiled.value.shape)

    # Construct a radial gradient and wrap hue values.
    radial = radial_gradient(
        center=(0.5, 0.5),
        radius=0.45,
        start=(0, 100, 100),
        end=(300, 100, 100),
        format_type=FormatType.INT,
        color_space="hsv",
        size=(32, 32),
    )
    normalized = cyclic_wrap_float(radial.value[..., 0], 0.0, 360.0)
    print("Wrapped hue min/max:", float(normalized.min()), float(normalized.max()))

    # Convert to Color2DArr for further manipulation.
    arr = Color2DArr(np.array(radial.value, dtype=np.uint8))
    resized = arr.repeat(times=2)
    clamped = clamp(resized.value, 0, 255)
    print("Repeated radial gradient shape:", clamped.shape)


def main() -> None:
    demonstrate_arrays()


if __name__ == "__main__":
    main()
