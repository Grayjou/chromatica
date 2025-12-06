# Chromatica

Chromatica is a Python toolkit for color manipulation, conversion, and gradient generation. It ships typed color objects, robust conversion utilities (scalar and NumPy-based), and helpers for constructing complex gradients across RGB, HSV, and HSL color spaces.

## Features
- Typed color classes for common modes (RGB, HSV, HSL, CMYK, grayscale, palette, and alpha-enabled variants).
- High-level conversion helpers with CSS Color 4-compatible algorithms and NumPy acceleration.
- Gradient generators for 1D, 2D, and radial patterns with hue direction control, easing hooks, and alpha support.
- Utility helpers for clamping, cyclic wrapping, and working with array-backed colors.

## Installation
Chromatica targets Python 3.8+. To install from source:

```bash
pip install .
```

For development (with testing tools):

```bash
pip install -e .[dev]
```

## Quick start

### Color conversion
Use the conversion wrapper to move between color spaces. You can choose integer (0–255), float (0–1), or percentage formats and opt into the CSS algorithm when needed.

```python
from chromatica import convert

# Integer RGB to HSV (analytical algorithm)
rgb = (255, 128, 64)
h, s, v = convert(rgb, from_space="rgb", to_space="hsv", input_type="int", output_type="int")

# HSV back to RGB using CSS Color 4 math
converted_rgb = convert((h, s, v), from_space="hsv", to_space="rgb", use_css_algo=True)
```

### Working with color objects
Color classes clamp inputs to valid ranges and expose helpers for channel access and conversion.

```python
from chromatica import ColorRGB, ColorHSV

sunset = ColorRGB((255, 128, 64))
print(sunset.unit_values)  # normalized tuple

hsv = ColorHSV((30, 50, 100))
print(hsv.to_rgb())
```

### Building gradients
Create smooth gradients in the color space of your choice. Hue-based spaces support clockwise/counterclockwise interpolation.

```python
from chromatica import Gradient1D, Gradient2D
from chromatica.conversions.format_type import FormatType

# 1D HSV gradient, forcing clockwise hue rotation
strip = Gradient1D.from_colors((0, 100, 100), (300, 100, 100), steps=32, color_space="hsv", format_type=FormatType.INT, direction="cw")

# 2D RGB gradient from four corner colors
canvas = Gradient2D.from_colors((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), width=256, height=256)
array = canvas.value  # NumPy array of shape (H, W, channels)
```

### Utility helpers
Clamp, wrap, and bounce numeric values when preparing inputs or post-processing channels.

```python
from chromatica.functions import clamp, bounce, cyclic_wrap_float

clamped = clamp(300, 0, 255)
looped_hue = cyclic_wrap_float(390.0, 0.0, 360.0)
runtime_safe = bounce(-10, 0, 100)
```

## Running tests

Chromatica includes pytest-based coverage. From the repository root:

```bash
pytest
```

This exercises conversion math, utility functions, and gradient helpers.

## Examples

Runnable examples live in the `examples/` folder:

- `examples/basic_usage.py` demonstrates typed colors, conversions, and 1D/2D gradients.
- `examples/array_operations.py` shows array-backed gradients, hue wrapping, and simple tiling.

Run either script directly with Python, e.g.:

```bash
python examples/basic_usage.py
```
