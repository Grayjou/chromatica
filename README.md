# Chromatica
<p align="center">
  <img src="https://raw.githubusercontent.com/Grayjou/chromatica/main/images/filling_ring_combined.png"
       alt="Base File Highlight icon" width="160" />
</p>
**Chromatica** is a comprehensive Python toolkit for advanced color manipulation, conversion, and gradient generation. It provides typed color objects, robust conversion utilities (both scalar and NumPy-accelerated), and powerful tools for constructing complex gradients across multiple color spaces including RGB, HSV, HSL, and their alpha-enabled variants.

## Project Scope

Chromatica is designed for developers, designers, and data visualization specialists who need:

- **Precise color conversions** between multiple color spaces with support for different numeric formats (int, float, percentage)
- **Complex gradient generation** including 1D, 2D, radial, and angular-radial patterns with advanced interpolation controls
- **Hue-aware interpolation** for smooth color transitions in HSV/HSL spaces with clockwise/counterclockwise direction control
- **Array-based operations** for efficient batch processing of colors using NumPy
- **Type safety** with dedicated color classes for each color space and format
- **Flexible format handling** supporting integer (0-255), float (0.0-1.0), and percentage representations
- **Animation support** for creating smooth color transitions and dynamic gradient effects

## Key Features

### Color Spaces & Formats
- **Supported color spaces**: RGB, RGBA, HSV, HSVA, HSL, HSLA, CMYK, grayscale, and palette modes
- **Multiple format types**: Integer (0-255), Float (0.0-1.0), and percentage representations
- **Alpha channel support**: Full alpha transparency handling across all compatible color spaces
- **CSS Color 4 compatibility**: Optional CSS-compliant algorithms for web-standard conversions

### Gradient Generation
- **1D gradients**: Linear color interpolation with customizable steps and easing functions
- **2D gradients**: Bilinear interpolation from four corner colors
- **Cell-based gradients**: Flexible 2D gradient cells with advanced features:
  - **Corner cells**: Bilinear interpolation from four corner colors
  - **Line cells**: Interpolation between two horizontal lines with per-pixel color control
  - **Dual corner cells**: Advanced cells with independent top and bottom edge interpolation
  - **Cell generators**: Factory classes for creating and partitioning gradient cells
  - **Partitioning**: Split cells into multiple regions with different color spaces
  - **Per-channel transforms**: Apply different coordinate transformations to each color channel
  - **Border handling**: Configurable edge behavior (clamp, wrap, constant, etc.)
- **Radial gradients**: Circular gradients radiating from a center point with configurable radius
- **Angular-radial gradients**: Advanced polar coordinate gradients with separate angular and radial interpolation
  - Simple angular-radial: Two-color gradients with angular and radial variation
  - Full parametrical: Multi-ring gradients with per-ring color control and custom easing functions
- **Hue interpolation**: Intelligent hue path selection (shortest, clockwise, or counterclockwise) for smooth color transitions
- **Masking & bounds**: Configurable angular and radial masks for partial gradient coverage
- **Outside fill**: Custom colors for regions outside the gradient boundaries

### Color Manipulation
- **Typed color classes**: Dedicated classes for each color space with automatic value clamping
- **Arithmetic operations**: Add, subtract, multiply, and divide colors with automatic range handling
- **Format conversion**: Seamless conversion between integer, float, and percentage formats
- **Space conversion**: Convert between any supported color spaces with high precision
- **Array operations**: Batch processing of color arrays with NumPy efficiency
- **Channel access**: Direct manipulation of individual color channels (hue, saturation, brightness, etc.)

### Utility Functions
- **Value clamping**: Keep values within valid ranges
- **Cyclic wrapping**: Handle hue wraparound for seamless color wheel navigation
- **Bounce wrapping**: Reflect values that exceed boundaries
- **Easing functions**: Apply custom interpolation curves to gradients
- **Coordinate transforms**: Advanced spatial transformations for complex gradient patterns

## Use Cases

Chromatica is ideal for:

- **Data Visualization**: Create color scales and gradients for heatmaps, charts, and scientific visualizations
- **Image Processing**: Manipulate image colors, apply gradients, and perform color space conversions
- **UI/UX Design**: Generate color palettes, themes, and dynamic color schemes
- **Animation & Motion Graphics**: Create smooth color transitions and animated gradients
- **Game Development**: Generate procedural textures, environmental effects, and dynamic color themes
- **Web Development**: Produce CSS-compatible color values and gradient backgrounds
- **Art & Creative Coding**: Experiment with color theory and create generative art
- **Scientific Computing**: Perform precise color space transformations for research applications

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

## Advanced Features

### Cell-Based 2D Gradients

Chromatica provides powerful cell-based gradient generation with advanced control over interpolation and transformations:

```python
from chromatica.gradients.gradient2dv2.generators import CornersCellFactory
from chromatica.gradients.gradient2dv2.partitions import PerpendicularPartition, PartitionInterval
from chromatica.types.color_types import ColorSpace
from chromatica.types.format_type import FormatType
import numpy as np

# Create a corners cell with custom transformations
factory = CornersCellFactory(
    width=500,
    height=500,
    top_left=np.array([0.0, 1.0, 0.5]),      # HSV color
    top_right=np.array([120.0, 1.0, 0.5]),
    bottom_left=np.array([240.0, 1.0, 0.5]),
    bottom_right=np.array([360.0, 1.0, 0.5]),
    color_space=ColorSpace.HSV,
    hue_direction_x='cw',   # Clockwise hue interpolation horizontally
    hue_direction_y='ccw',  # Counter-clockwise vertically
)

# Apply per-channel transforms for complex effects
transforms = {
    0: lambda coords: coords ** 2,  # Square transform on hue channel
    1: lambda coords: coords ** 0.5,  # Sqrt transform on saturation
}
factory.per_channel_transforms = transforms

# Get the rendered gradient
cell = factory.get_cell()
gradient_array = cell.get_value()  # NumPy array (H, W, 3)

# Partition the cell for multiple color space regions
partition = PerpendicularPartition(
    breakpoints=[0.5],  # Split at 50%
    intervals=[
        PartitionInterval(color_space=ColorSpace.RGB),
        PartitionInterval(color_space=ColorSpace.HSV, hue_direction='cw'),
    ]
)
partitioned_cells = factory.partition_slice(partition)
```

**Cell Types:**
- `CornersCellFactory`: Creates cells from four corner colors with bilinear interpolation
- `LinesCellFactory`: Creates cells from two horizontal lines (top and bottom)
- `CornersCellDualFactory`: Advanced cells with independent edge interpolation

**Features:**
- Per-channel coordinate transformations
- Partitioning for multi-region gradients with different color spaces
- Border handling modes (clamp, wrap, constant value)
- Discrete and continuous sampling methods

### Angular-Radial Gradients

Create sophisticated polar coordinate gradients with independent control over angular and radial interpolation:

```python
from chromatica.gradients.simple_angular_radial import SimpleAngularRadialGradient
from chromatica.format_type import FormatType

# Create a gradient that varies both angularly and radially
gradient = SimpleAngularRadialGradient.generate(
    width=500,
    height=500,
    radius=200,
    inner_ring_colors=(
        (0, 255, 255),    # Cyan at start angle
        (120, 255, 255)   # Green at end angle
    ),
    outer_ring_colors=(
        (240, 255, 255),  # Blue at start angle
        (300, 255, 255)   # Magenta at end angle
    ),
    color_space='hsv',
    format_type=FormatType.INT,
    deg_start=0.0,
    deg_end=360.0,
    radius_start=0.3,
    radius_end=1.0,
    hue_direction_theta='cw',  # Clockwise hue interpolation
    outside_fill=(0, 0, 0)
)
```

### Hue Direction Control

When working with HSV/HSL color spaces, control how hues interpolate:

```python
from chromatica import Gradient1D
from chromatica.format_type import FormatType

# Shortest path (default) - goes through the closest colors
gradient_short = Gradient1D.from_colors(
    (350, 100, 100), (10, 100, 100),
    steps=20, color_space='hsv', direction=None
)

# Clockwise - forces interpolation through increasing hue values
gradient_cw = Gradient1D.from_colors(
    (350, 100, 100), (10, 100, 100),
    steps=20, color_space='hsv', direction='cw'
)

# Counter-clockwise - forces interpolation through decreasing hue values
gradient_ccw = Gradient1D.from_colors(
    (350, 100, 100), (10, 100, 100),
    steps=20, color_space='hsv', direction='ccw'
)
```

### Custom Easing Functions

Apply easing functions to control interpolation curves:

```python
import numpy as np
from chromatica import Gradient1D

# Ease-in-out cubic
def ease_in_out_cubic(t):
    return np.where(t < 0.5, 4 * t**3, 1 - (-2 * t + 2)**3 / 2)

gradient = Gradient1D.from_colors(
    (255, 0, 0), (0, 0, 255),
    steps=50,
    color_space='rgb',
    unit_transform=ease_in_out_cubic
)
```

### Array-Based Color Operations

Efficiently process multiple colors using NumPy arrays:

```python
from chromatica import ColorRGBArr
import numpy as np

# Create an array of colors
colors = ColorRGBArr(np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
]))

# Convert all colors to HSV
hsv_colors = colors.convert('hsv')

# Perform arithmetic operations
brightened = colors * 1.2  # Increase brightness
mixed = colors + ColorRGB((50, 50, 50))  # Add to each color
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features through the [GitHub issues page](https://github.com/Grayjou/chromatica/issues).

## License

Chromatica is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This library builds upon established color science principles and aims to provide a robust, performant, and user-friendly interface for color manipulation in Python.
