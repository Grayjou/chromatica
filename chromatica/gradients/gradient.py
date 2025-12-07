"""
Gradient Generation Module
===========================

This module provides gradient generation classes and functions for creating
1D and 2D color gradients with advanced interpolation options.

Features
--------
- 1D gradients with hue direction control (cw/ccw for HSV/HSL)
- 2D gradients from corner colors with bilinear interpolation
- Radial gradients with customizable easing
- Angular gradients (wrap_around)
- Support for all color spaces (RGB, HSV, HSL)
- Custom interpolation transforms
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray as NDArray
from typing import Tuple, Union, Callable, Optional, List
from ..color_arr import Color1DArr, Color2DArr
from ..colors.color_base import ColorBase
from ..colors.rgb import ColorUnitRGB, ColorRGBINT, ColorUnitRGBA, ColorRGBAINT
from ..colors.hsv import UnitHSV, ColorHSVINT, UnitHSVA, ColorHSVAINT
from ..colors.hsl import UnitHSL, ColorHSLINT, UnitHSLA, ColorHSLAINT
from ..format_type import FormatType


# Color type mapping for convenience
COLOR_CLASSES = {
    ('rgb', FormatType.INT): ColorRGBINT,
    ('rgb', FormatType.FLOAT): ColorUnitRGB,
    ('rgba', FormatType.INT): ColorRGBAINT,
    ('rgba', FormatType.FLOAT): ColorUnitRGBA,
    ('hsv', FormatType.INT): ColorHSVINT,
    ('hsv', FormatType.FLOAT): UnitHSV,
    ('hsva', FormatType.INT): ColorHSVAINT,
    ('hsva', FormatType.FLOAT): UnitHSVA,
    ('hsl', FormatType.INT): ColorHSLINT,
    ('hsl', FormatType.FLOAT): UnitHSL,
    ('hsla', FormatType.INT): ColorHSLAINT,
    ('hsla', FormatType.FLOAT): UnitHSLA,
}


class Gradient1D(Color1DArr):
    """
    Represents a 1D gradient of colors with advanced interpolation.
    
    Extends Color1DArr with gradient-specific creation methods including:
    - Hue direction control for HSV/HSL (cw/ccw)
    - Custom interpolation transforms
    - Automatic color space handling
    """

    @classmethod
    def from_colors(
        cls,
        color1: Union[ColorBase, Tuple, int],
        color2: Union[ColorBase, Tuple, int],
        steps: int,
        color_space: str = 'rgb',
        format_type: FormatType = FormatType.FLOAT,
        unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
        direction: Optional[str] = None  # 'cw', 'ccw', or None for shortest path
    ) -> 'Gradient1D':
        """
        Create a 1D gradient from two colors with optional hue direction control.

        Args:
            color1: First color (ColorBase instance or tuple/int)
            color2: Second color (ColorBase instance or tuple/int)
            steps: Number of steps in the gradient
            color_space: Target color space ('rgb', 'hsv', 'hsl', etc.)
            format_type: Format type (INT or FLOAT)
            unit_transform: Optional function to transform interpolation parameter
            direction: Hue direction for HSV/HSL - 'cw' (clockwise), 'ccw' (counter-clockwise),
                      or None for shortest path

        Returns:
            Gradient1D instance with interpolated colors
        """
        color_space = color_space.lower()
        
        # Get the appropriate color class
        color_class = COLOR_CLASSES.get((color_space, format_type))
        if color_class is None:
            raise ValueError(f"Unsupported color space/format combination: {color_space}/{format_type}")
        
        # Convert inputs to target space
        if isinstance(color1, ColorBase):
            c1 = color1.convert(color_space, format_type) # type: ignore
        else:
            c1 = color_class(color1)
        
        if isinstance(color2, ColorBase):
            c2 = color2.convert(color_space, format_type) # type: ignore
        else:
            c2 = color_class(color2)
        
        # Extract values as float arrays for interpolation
        start = np.array(c1.value, dtype=float)
        end = np.array(c2.value, dtype=float)
        
        # Build unit interpolation array
        u = np.linspace(0.0, 1.0, steps, dtype=float)[:, None]
        if unit_transform is not None:
            u = unit_transform(u)
        
        # HSV/HSL: special circular hue logic
        if color_space in ('hsv', 'hsl', 'hsva', 'hsla'):
            # Normalize hue into [0, 360)
            h0 = start[0] % 360.0
            h1 = end[0] % 360.0
            
            if direction == 'cw':
                # Always go clockwise (increasing hue)
                if h1 <= h0:
                    h1 += 360.0
            elif direction == 'ccw':
                # Always go counter-clockwise (decreasing hue)
                if h1 >= h0:
                    h1 -= 360.0
            else:
                # Shortest path
                delta = h1 - h0
                if delta > 180.0:
                    h1 -= 360.0
                elif delta < -180.0:
                    h1 += 360.0
            
            dh = h1 - h0
            hues = (h0 + u * dh) % 360.0
            
            # Interpolate remaining channels linearly
            rest = start[1:] * (1 - u) + end[1:] * u
            colors = np.concatenate([hues, rest], axis=1)
        else:
            # RGB or other linear spaces: simple linear interpolation
            colors = start * (1 - u) + end * u
        
        # Convert to appropriate dtype
        if format_type == FormatType.INT:
            colors = np.round(colors).astype(np.uint16)
        else:
            colors = colors.astype(np.float32)
        
        # Create ColorBase instance and wrap in Gradient1D
        gradient_color = color_class(colors)
        return cls(gradient_color)


class Gradient2D(Color2DArr):
    """
    Represents a 2D gradient from four corner colors with bilinear interpolation.
    
    Extends Color2DArr with gradient-specific creation methods.
    """

    @classmethod
    def from_colors(
        cls,
        color_tl: Union[ColorBase, Tuple, int],
        color_tr: Union[ColorBase, Tuple, int],
        color_bl: Union[ColorBase, Tuple, int],
        color_br: Union[ColorBase, Tuple, int],
        width: int,
        height: int,
        color_space: str = 'rgb',
        format_type: FormatType = FormatType.FLOAT,
        unit_transform_x: Optional[Callable[[NDArray], NDArray]] = None,
        unit_transform_y: Optional[Callable[[NDArray], NDArray]] = None
    ) -> 'Gradient2D':
        """
        Create a 2D gradient from four corner colors with optional transforms.

        Args:
            color_tl: Top-left color
            color_tr: Top-right color
            color_bl: Bottom-left color
            color_br: Bottom-right color
            width: Number of columns
            height: Number of rows
            color_space: Target color space ('rgb', 'hsv', 'hsl', etc.)
            format_type: Format type (INT or FLOAT)
            unit_transform_x: Optional transformation of x interpolation factors
            unit_transform_y: Optional transformation of y interpolation factors

        Returns:
            Gradient2D instance with bilinearly interpolated colors
        """
        color_space = color_space.lower()
        
        # Get the appropriate color class
        color_class = COLOR_CLASSES.get((color_space, format_type))
        if color_class is None:
            raise ValueError(f"Unsupported color space/format combination: {color_space}/{format_type}")
        
        # Convert all corner colors to the target space
        corners = []
        for corner_color in [color_tl, color_tr, color_bl, color_br]:
            if isinstance(corner_color, ColorBase):
                corners.append(corner_color.convert(color_space, format_type)) # type: ignore
            else:
                corners.append(color_class(corner_color))
        
        tl, tr, bl, br = [np.array(c.value, dtype=float) for c in corners]
        
        # Create interpolation factors
        x = np.linspace(0.0, 1.0, width, dtype=float)
        y = np.linspace(0.0, 1.0, height, dtype=float)
        
        if unit_transform_x is not None:
            x = unit_transform_x(x)
        if unit_transform_y is not None:
            y = unit_transform_y(y)
        
        xx, yy = np.meshgrid(x, y)
        
        # Bilinear interpolation
        colors = (
            (1 - xx)[:, :, None] * (1 - yy)[:, :, None] * tl +
            xx[:, :, None] * (1 - yy)[:, :, None] * tr +
            (1 - xx)[:, :, None] * yy[:, :, None] * bl +
            xx[:, :, None] * yy[:, :, None] * br
        )
        
        # Convert to appropriate dtype
        if format_type == FormatType.INT:
            colors = np.round(colors).astype(np.uint16)
        else:
            colors = colors.astype(np.float32)
        
        # Create ColorBase instance and wrap in Gradient2D
        gradient_color = color_class(colors)
        return cls(gradient_color)



def radial_gradient(
        color1: Union[ColorBase, Tuple, int],
        color2: Union[ColorBase, Tuple, int],
        height: int,
        width: int,
        center: Union[Tuple[int, int], List[int]] = (0, 0),
        radius: float = 1.0,
        color_space: str = 'rgb',
        format_type: FormatType = FormatType.FLOAT,
        unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
        outside_fill: Optional[Union[ColorBase, Tuple, int]] = None,
        start: float = 0.0,
        end: float = 1.0,
        offset: float = 0.0,
        base: Optional[NDArray] = None
    ) -> NDArray:
    """
    Create a radial gradient that radiates from a center point.
    
    Args:
        color1: Inner color (ColorBase instance or tuple)
        color2: Outer color (ColorBase instance or tuple)
        height: Height of the output array
        width: Width of the output array
        center: (x, y) center position of the gradient
        radius: Radius of the gradient in pixels
        color_space: Target color space ('rgb', 'hsv', 'hsl', etc.)
        format_type: Format type (INT or FLOAT)
        unit_transform: Optional transformation of normalized distances
        outside_fill: Optional color to fill areas outside the gradient
        start: Start position of gradient (0-1 range)
        end: End position of gradient (0-1 range)
        offset: Offset to apply to normalized distances
        base: Optional base array to blend with
    
    Returns:
        NDArray with shape (height, width, channels)
    
    Notes:
        If `base` is provided and `outside_fill` is None, the gradient overwrites 
        `base` only within the gradient area, leaving the rest untouched.
        If `outside_fill` is provided, it fills areas outside the gradient.
    """
    color_space = color_space.lower()
    
    # Get the appropriate color class
    color_class = COLOR_CLASSES.get((color_space, format_type))
    if color_class is None:
        raise ValueError(f"Unsupported color space/format combination: {color_space}/{format_type}")
    
    # Convert colors to target space
    if isinstance(color1, ColorBase):
        col1 = color1.convert(color_space, format_type) # type: ignore
    else:
        col1 = color_class(color1)
    
    if isinstance(color2, ColorBase):
        col2 = color2.convert(color_space, format_type) # type: ignore
    else:
        col2 = color_class(color2)
    
    if outside_fill is not None:
        if isinstance(outside_fill, ColorBase):
            outside_fill_color = outside_fill.convert(color_space, format_type).value # type: ignore
        else:
            outside_fill_color = color_class(outside_fill).value
    else:
        outside_fill_color = None
    
    # Convert to numpy arrays
    c1 = np.array(col1.value, dtype=float)
    c2 = np.array(col2.value, dtype=float)
    
    # Create distance field
    y, x = np.indices((height, width), dtype=float)
    cx, cy = center
    dx = x - cx
    dy = y - cy
    distance = np.sqrt(dx**2 + dy**2)
    
    # Normalize distances to [0, 1]
    unit_array = (distance / radius) - offset
    
    if unit_transform is not None:
        unit_array = np.where(
            (unit_array > 1.0) | (unit_array < 0.0),
            unit_array,
            unit_transform(unit_array)
        )
    
    # Clip to [0, 1] for interpolation
    unit_array_clipped = np.clip(unit_array, 0.0, 1.0)
    
    # Interpolate colors
    gradient = (
        c1 * (1 - unit_array_clipped[..., None]) +
        c2 * unit_array_clipped[..., None]
    )
    
    # Initialize result array
    if base is not None:
        if base.shape != gradient.shape:
            raise ValueError(f"`base` shape {base.shape} does not match gradient shape {gradient.shape}")
        result = base.copy()
    else:
        result = gradient.copy()
    
    # Compute masks
    mask_inside = (
        (unit_array >= start) & (unit_array <= end) & 
        (unit_array >= 0.0) & (unit_array <= 1.0)
    )
    mask_outside = ~mask_inside
    
    if outside_fill_color is not None:
        # Fill outside gradient with outside_fill_color
        result[mask_outside] = outside_fill_color
    else:
        if base is not None:
            # Only overwrite inside the gradient, keep base outside
            result[mask_inside] = gradient[mask_inside]
        else:
            # No base, no outside_fill → use gradient everywhere
            result = gradient
    
    # Convert to appropriate dtype
    if format_type == FormatType.INT:
        result = np.round(result).astype(np.uint16)
    else:
        result = result.astype(np.float32)
    
    return result


    
def example(output_path=None):
    """Simple 2D gradient example with basic colors."""
    from PIL import Image
    grad2d = Gradient2D.from_colors(
        color_tl=(255, 0, 255),       # top-left: pink
        color_tr=(255, 255, 0),       # top-right: yellow
        color_bl=(255, 0, 128),       # bottom-left: deep pink
        color_br=(255, 128, 0),       # bottom-right: orange
        width=500,
        height=500,
        color_space='rgb',
        format_type=FormatType.INT
    )
    img = Image.fromarray(np.array(grad2d._color.value, dtype=np.uint8), mode='RGB')
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
        color_space='rgb',
        format_type=FormatType.INT,
        unit_transform_x=lambda x: (1 - np.cos(4*x * np.pi)) / 2,
        unit_transform_y=lambda y: (1 - np.cos(4*y * np.pi)) / 2
    )
    img = Image.fromarray(np.array(grad2d._color.value, dtype=np.uint8), mode='RGB')
    if output_path:
        img.save(output_path)
    img.show()


def example_radial_gradient(output_path=None):
    """Radial gradient example with layered gradients and alpha blending."""
    from PIL import Image
    
    def extreme_ease_in(x: NDArray) -> NDArray:
        func = lambda x: 1 - np.sqrt(np.abs(1 - x**2))
        return func(func(x))
    
    # Create base gradient
    gradient_base = np.full((500, 500, 4), (150, 255, 255, 255), dtype=np.uint16)
    
    radial_arr = radial_gradient(
        color1=(0, 0, 0, 255),
        color2=(255, 0, 180, 0),
        height=500,
        width=500,
        center=(250, 250),
        radius=125,
        color_space='rgba',
        format_type=FormatType.INT,
        unit_transform=extreme_ease_in,
        outside_fill=None,
        start=0.0,
        end=1.0,
        offset=1.0,
        base=gradient_base
    )
    
    # Layer second radial gradient
    radial_arr = radial_gradient(
        color1=(0, 0, 255, 0),
        color2=(0, 0, 0, 255),
        height=500,
        width=500,
        center=(250, 250),
        radius=125,
        color_space='rgba',
        format_type=FormatType.INT,
        unit_transform=None,
        outside_fill=None,
        start=0.0,
        end=1.0,
        offset=0.0,
        base=radial_arr
    )
    
    base = Image.new('RGBA', (500, 500), (150, 255, 255, 255))
    img = Image.fromarray(radial_arr.astype(np.uint8), mode='RGBA')
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
        color_space='rgb',
        format_type=FormatType.INT
    )
    rotated_grad = grad1d.wrap_around(
        width=500,
        height=500,
        center=(250, 250),
        angle_start=0.0,
        angle_end=2 * np.pi,
        unit_transform=lambda x: (1 - np.cos(8*x * np.pi)) / 2,
        outside_fill=(255, 255, 255),
        radius_offset=0
    )
    img = Image.fromarray(rotated_grad.astype(np.uint8), mode='RGB')
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
        unit_transform=lambda x: (1 - np.cos(8*x * np.pi)) / 2,
        outside_fill=(255, 255, 255),
        radius_offset=0
    )
    img = Image.fromarray(rotated_grad.astype(np.uint8), mode='RGB')
    if output_path:
        img.save(output_path)
    img.show()

