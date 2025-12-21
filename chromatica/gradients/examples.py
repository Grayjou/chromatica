from __future__ import annotations

import numpy as np
from numpy import ndarray as NDArray

from ..color_arr import Color1DArr
from ..colors.rgb import ColorUnitRGB
from ..types.format_type import FormatType
from .gradient1d import Gradient1D
from .gradient2d import Gradient2D
from .radial import radial_gradient


def example(output_path=None):
    """Simple 2D gradient example with basic colors."""
    from PIL import Image

    grad2d = Gradient2D.from_colors(
        color_tl=(255, 0, 255),
        color_tr=(255, 255, 0),
        color_bl=(255, 0, 128),
        color_br=(255, 128, 0),
        width=500,
        height=500,
        color_space="rgb",
        format_type=FormatType.INT,
    )
    img = Image.fromarray(np.array(grad2d._color.value, dtype=np.uint8), mode="RGB")
    if output_path:
        img.save(output_path)
    img.show()


def example_2d_gradient(output_path=None):
    """2D gradient with sinusoidal transformations on both axes."""
    from PIL import Image

    grad2d = Gradient2D.from_colors(
        color_tl=(255, 0, 255),
        color_tr=(255, 255, 0),
        color_bl=(255, 0, 128),
        color_br=(255, 128, 0),
        width=500,
        height=500,
        color_space="rgb",
        format_type=FormatType.INT,
        unit_transform_x=lambda x: (1 - np.cos(4 * x * np.pi)) / 2,
        unit_transform_y=lambda y: (1 - np.cos(4 * y * np.pi)) / 2,
    )
    img = Image.fromarray(np.array(grad2d._color.value, dtype=np.uint8), mode="RGB")
    if output_path:
        img.save(output_path)
    img.show()


def example_radial_gradient(output_path=None):
    """Radial gradient example with layered gradients and alpha blending."""
    from PIL import Image

    def extreme_ease_in(x: NDArray) -> NDArray:
        func = lambda x: 1 - np.sqrt(np.abs(1 - x**2))
        return func(func(x))

    gradient_base = np.full((500, 500, 4), (150, 255, 255, 255), dtype=np.uint16)

    radial_arr = radial_gradient(
        color1=(0, 0, 0, 255),
        color2=(255, 0, 180, 0),
        height=500,
        width=500,
        center=(250, 250),
        radius=125,
        color_space="rgba",
        format_type=FormatType.INT,
        unit_transform=extreme_ease_in,
        outside_fill=None,
        start=0.0,
        end=1.0,
        offset=1.0,
        base=gradient_base,
    )

    radial_arr = radial_gradient(
        color1=(0, 0, 255, 0),
        color2=(0, 0, 0, 255),
        height=500,
        width=500,
        center=(250, 250),
        radius=125,
        color_space="rgba",
        format_type=FormatType.INT,
        unit_transform=None,
        outside_fill=None,
        start=0.0,
        end=1.0,
        offset=0.0,
        base=radial_arr,
    )

    base = Image.new("RGBA", (500, 500), (150, 255, 255, 255))
    img = Image.fromarray(radial_arr.astype(np.uint8), mode="RGBA")
    base.paste(img, (0, 0), img)
    if output_path:
        base.save(output_path)
    base.show()


def example_gradient_rotate(output_path=None):
    """Angular gradient wrapping around a center point."""
    from PIL import Image

    grad1d = Gradient1D.from_colors(
        color1=(255, 0, 0),
        color2=(0, 0, 255),
        steps=125,
        color_space="rgb",
        format_type=FormatType.INT,
    )
    rotated_grad = grad1d.wrap_around(
        width=500,
        height=500,
        center=(250, 250),
        angle_start=0.0,
        angle_end=2 * np.pi,
        unit_transform=lambda x: (1 - np.cos(8 * x * np.pi)) / 2,
        outside_fill=(255, 255, 255),
        radius_offset=0,
    )
    img = Image.fromarray(rotated_grad.astype(np.uint8), mode="RGB")
    if output_path:
        img.save(output_path)
    img.show()


def example_arr_rotate(output_path=None):
    """Random color array wrapped around a center point."""
    from PIL import Image

    arr_1d = Color1DArr(
        ColorUnitRGB(np.random.uniform(0.5, 1.0, (125, 3)))
    )
    rotated_grad = arr_1d.wrap_around(
        width=500,
        height=500,
        center=(250, 250),
        angle_start=0.0,
        angle_end=2 * np.pi,
        unit_transform=lambda x: (1 - np.cos(8 * x * np.pi)) / 2,
        outside_fill=(255, 255, 255),
        radius_offset=0,
    )
    img = Image.fromarray(rotated_grad.astype(np.uint8), mode="RGB")
    if output_path:
        img.save(output_path)
    img.show()
