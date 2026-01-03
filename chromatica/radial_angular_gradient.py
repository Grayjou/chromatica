"""
Radial Angular Gradient Module
================================

This module provides advanced radial-angular gradient generation with:
- Angular color stops (theta-based color transitions)
- Radial variation (distance-based transitions)
- Custom radius functions r(θ) for non-circular gradients
- Per-channel easing functions for both angular and radial dimensions
- Hue direction control for HSV/HSL color spaces

Features
--------
- Define colors at specific angles with inner/outer color pairs
- Variable radius based on angle (elliptical, star-shaped, custom shapes)
- Independent easing per color channel
- Hue interpolation control (clockwise/counter-clockwise)
- Support for all color spaces and formats
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray as NDArray
from typing import Callable, List, Tuple, Dict, Optional, Union
from .colors.color_base import ColorBase
from .types.format_type import FormatType
from .colors.color import unified_tuple_to_class

def interpolate_hue_vector(
    c0: NDArray, 
    c1: NDArray, 
    u: NDArray, 
    direction: Optional[str] = None
) -> NDArray:
    """
    Interpolate between two HSV/HSL color vectors with hue wrapping.

    Args:
        c0: Starting color vector [H, S, V/L, ...] where H is in degrees
        c1: Ending color vector [H, S, V/L, ...]
        u: Interpolation parameter(s) in [0, 1]
        direction: 'cw' (clockwise), 'ccw' (counter-clockwise), or None (shortest path)

    Returns:
        Interpolated color vector(s)
    """
    h0 = c0[..., 0] % 360.0
    h1 = c1[..., 0] % 360.0

    if direction == 'cw':
        # Always go clockwise (increasing hue)
        mask = h1 <= h0
        h1 = np.where(mask, h1 + 360.0, h1)
    elif direction == 'ccw':
        # Always go counter-clockwise (decreasing hue)
        mask = h1 >= h0
        h1 = np.where(mask, h1 - 360.0, h1)
    else:
        # Shortest path
        delta = h1 - h0
        h1 = np.where(delta > 180.0, h1 - 360.0, h1)
        h1 = np.where(delta < -180.0, h1 + 360.0, h1)

    dh = h1 - h0
    h = (h0 + u * dh) % 360.0
    
    # Interpolate remaining channels linearly
    rest = c0[..., 1:] * (1 - u)[..., None] + c1[..., 1:] * u[..., None]
    return np.concatenate([h[..., None], rest], axis=-1)


def lerp(a: NDArray, b: NDArray, t: NDArray) -> NDArray:
    """Linear interpolation between a and b."""
    return a * (1 - t) + b * t


def apply_channel_easing(
    u: NDArray, 
    easing_dict: Optional[Dict[int, Callable[[NDArray], NDArray]]], 
    channel: int
) -> NDArray:
    """
    Apply per-channel easing function if provided.

    Args:
        u: Interpolation parameter array
        easing_dict: Dictionary mapping channel index to easing function
        channel: Channel index to apply easing to

    Returns:
        Eased interpolation parameter or original if no easing defined
    """
    if easing_dict is None:
        return u
    func = easing_dict.get(channel)
    if func is None:
        return u
    return func(u)


def build_r_theta_from_stops(radius_stops: List[Tuple[float, float]]) -> Callable[[Union[float, NDArray]], Union[float, NDArray]]:
    """
    Build a radius function r(θ) from angular stops.

    Args:
        radius_stops: List of (angle_degrees, radius_value) pairs

    Returns:
        Callable that takes angle in degrees and returns interpolated radius

    Example:
        >>> r_fn = build_r_theta_from_stops([(0, 100), (90, 150), (180, 100), (270, 50)])
        >>> r_fn(45)  # Returns interpolated radius at 45 degrees
    """
    # Normalize and sort stops
    stops = sorted([(t % 360.0, v) for (t, v) in radius_stops], key=lambda x: x[0])

    # Ensure we have a stop at 0 degrees
    if stops[0][0] != 0.0:
        stops.insert(0, (0.0, stops[0][1]))

    # Add wraparound stop at 360
    t_last = stops[-1][0]
    if t_last != 360.0:
        stops.append((360.0, stops[0][1]))  # Wrap to first radius

    # Separate into arrays for efficient lookup
    angles = np.array([s[0] for s in stops], dtype=float)
    values = np.array([s[1] for s in stops], dtype=float)

    def r_theta(theta_deg: Union[float, NDArray]) -> Union[float, NDArray]:
        """Interpolate radius at given angle(s)."""
        theta = np.asarray(theta_deg) % 360.0
        scalar_input = np.ndim(theta) == 0
        theta = np.atleast_1d(theta)
        
        # Find interpolation segments
        idx = np.searchsorted(angles, theta) - 1
        idx = np.clip(idx, 0, len(angles) - 2)

        t0 = angles[idx]
        t1 = angles[idx + 1]
        delta = t1 - t0
        u = np.divide(theta - t0, delta, out=np.zeros_like(theta), where=delta != 0)

        result = values[idx] * (1 - u) + values[idx + 1] * u
        
        return result.item() if scalar_input else result

    return r_theta


def build_theta_interpolators(
    theta_stops: List[Tuple[float, Union[Tuple, list, ColorBase], Union[Tuple, list, ColorBase]]],
    color_class: type,
    hue_direction: Optional[str] = None
) -> Tuple[Callable, Callable]:
    """
    Build interpolation functions for inner and outer colors based on angular stops.

    Args:
        theta_stops: List of (angle, inner_color, outer_color) tuples
        color_class: ColorBase subclass for the target color space
        hue_direction: Hue interpolation direction for HSV/HSL spaces

    Returns:
        Tuple of (inner_color_fn, outer_color_fn) callables that take angle and return color
    """
    # Normalize angles and convert colors
    stops = []
    for theta, ci, co in theta_stops:
        if isinstance(ci, ColorBase):
            ci_val = np.array(ci.value, dtype=float)
        else:
            ci_val = np.array(color_class(ci).value, dtype=float)

        if isinstance(co, ColorBase):
            co_val = np.array(co.value, dtype=float)
        else:
            co_val = np.array(color_class(co).value, dtype=float)

        stops.append((theta % 360.0, ci_val, co_val))

    stops.sort(key=lambda x: x[0])

    # Ensure wraparound at 360
    if stops[0][0] != 0.0:
        stops.insert(0, (0.0, stops[0][1], stops[0][2]))
    if stops[-1][0] != 360.0:
        t_last, ci_last, co_last = stops[-1]
        stops.append((360.0, stops[0][1], stops[0][2]))  # Wrap to first colors

    # Prepare arrays
    angles = np.array([s[0] for s in stops], dtype=float)
    inner_vals = np.array([s[1] for s in stops], dtype=float)
    outer_vals = np.array([s[2] for s in stops], dtype=float)

    # Determine if hue logic is needed
    is_hue_space = color_class.mode in ('hsv', 'hsl', 'hsva', 'hsla')

    def interp_color(theta_deg: Union[float, NDArray], table: NDArray) -> NDArray:
        """Interpolate color at given angle(s) from table."""
        theta = np.asarray(theta_deg) % 360.0
        scalar_input = np.ndim(theta) == 0
        theta = np.atleast_1d(theta)
        
        idx = np.searchsorted(angles, theta) - 1
        idx = np.clip(idx, 0, len(angles) - 2)

        t0 = angles[idx]
        t1 = angles[idx + 1]
        delta = t1 - t0
        u = np.divide(theta - t0, delta, out=np.zeros_like(theta), where=delta != 0)

        if is_hue_space:
            # Use hue interpolation
            result = np.zeros((*theta.shape, table.shape[1]), dtype=float)
            for i, (idx_i, u_i) in enumerate(zip(idx, u)):
                c0 = table[idx_i]
                c1 = table[idx_i + 1]
                result[i] = interpolate_hue_vector(c0, c1, u_i, direction=hue_direction)
        else:
            # Linear interpolation
            c0 = table[idx]
            c1 = table[idx + 1]
            result = c0 * (1 - u)[..., None] + c1 * u[..., None]
        
        return result[0] if scalar_input else result

    def inner_fn(theta_deg: Union[float, NDArray]) -> NDArray:
        return interp_color(theta_deg, inner_vals)

    def outer_fn(theta_deg: Union[float, NDArray]) -> NDArray:
        return interp_color(theta_deg, outer_vals)

    return inner_fn, outer_fn


class RadialAngularGradient:
    """
    Advanced radial-angular gradient generator.

    This class creates gradients that vary both radially (distance from center)
    and angularly (direction from center), with support for:
    - Custom radius functions (elliptical, star-shaped, etc.)
    - Per-channel easing in both dimensions
    - Hue direction control for HSV/HSL
    - Angular color stops
    """

    def __init__(
        self,
        theta_stops: List[Tuple[float, Union[Tuple, list, ColorBase], Union[Tuple, list, ColorBase]]],
        radius_stops: Optional[List[Tuple[float, float]]] = None,
        r_theta: Optional[Callable[[float], float]] = None,
        easing_theta: Optional[Dict[int, Callable[[NDArray], NDArray]]] = None,
        easing_r: Optional[Dict[int, Callable[[NDArray], NDArray]]] = None,
        hue_direction_theta: Optional[str] = None,
        hue_direction_r: Optional[str] = None,
        color_mode: str = 'rgb',
        format_type: FormatType = FormatType.FLOAT,
        bivariable_space_transforms=None,   # NEW FEATURE
        bivariable_channel_transforms=None   # NEW FEATURE
        # Goes from (r, thetha, channels) to new_channels
    ):
        """
        Initialize a radial-angular gradient.

        Args:
            theta_stops: List of (angle, inner_color, outer_color) tuples defining
                        color transitions at specific angles (in degrees)
            radius_stops: Optional list of (angle, radius) tuples for variable radius
            r_theta: Optional custom radius function taking angle and returning radius
            easing_theta: Optional dict mapping channel index to angular easing function
            easing_r: Optional dict mapping channel index to radial easing function
            hue_direction_theta: Hue interpolation direction for angular transitions
            hue_direction_r: Hue interpolation direction for radial transitions
            color_mode: Color space ('rgb', 'hsv', 'hsl', 'rgba', etc.)
            format_type: Format type (INT or FLOAT),
            bivariable_space_transforms: Optional dict mapping channel index to functions
                                   that transform (r, theta) to (new_r, new_theta)

        Example:
            >>> gradient = RadialAngularGradient(
            ...     theta_stops=[
            ...         (0, (255, 0, 0), (0, 0, 255)),      # Red to blue at 0°
            ...         (90, (0, 255, 0), (255, 255, 0)),   # Green to yellow at 90°
            ...         (180, (0, 0, 255), (255, 0, 0)),    # Blue to red at 180°
            ...         (270, (255, 255, 0), (0, 255, 0))   # Yellow to green at 270°
            ...     ],
            ...     color_mode='rgb',
            ...     format_type=FormatType.INT
            ... )
            >>> image = gradient.render(500, 500, center=(250, 250), base_radius=200)
        """
        self.color_mode = color_mode.lower()
        self.format_type = format_type
        self.color_class = unified_tuple_to_class.get((self.color_mode, format_type))
        
        if self.color_class is None:
            raise ValueError(f"Unsupported color space/format: {color_mode}/{format_type}")

        # Build angular interpolators for inner and outer colors
        self.inner_fn, self.outer_fn = build_theta_interpolators(
            theta_stops,
            self.color_class,
            hue_direction_theta
        )

        # Build radius function (priority: callable > stops > constant)
        if r_theta is not None:
            self.r_theta = r_theta
        elif radius_stops is not None:
            self.r_theta = build_r_theta_from_stops(radius_stops)
        else:
            self.r_theta = None  # Use constant base_radius

        self.easing_theta = easing_theta
        self.easing_r = easing_r
        self.hue_direction_r = hue_direction_r
        # NEW: per-channel (r,theta) transforms
        self.bivariable_space_transforms = bivariable_space_transforms or {}
        
    def render(
        self,
        height: int,
        width: int,
        center: Tuple[int, int] = (0, 0),
        base_radius: float = 1.0,
        outside_fill: Optional[Union[ColorBase, Tuple, int]] = None
    ) -> NDArray:
        """
        Render the gradient to an image array.

        Args:
            height: Image height in pixels
            width: Image width in pixels
            center: (x, y) center position of the gradient
            base_radius: Base radius value (scaled by r_theta if provided)
            outside_fill: Optional color to fill areas outside the gradient

        Returns:
            NDArray with shape (height, width, channels)
        """
        cx, cy = center

        # Prepare outside fill color
        if outside_fill is not None:
            if isinstance(outside_fill, ColorBase):
                fill_val = np.array(outside_fill.value, dtype=float)
            else:
                fill_val = np.array(self.color_class(outside_fill).value, dtype=float)
        else:
            fill_val = None

        # Create pixel coordinate grids
        y, x = np.indices((height, width), dtype=float)
        dx = x - cx
        dy = y - cy

        # Calculate angle and distance for each pixel
        theta = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        dist = np.sqrt(dx * dx + dy * dy)

        # Apply angular easing per channel if specified
        theta_eased = np.zeros((height, width, self.color_class.num_channels if hasattr(self.color_class, 'num_channels') else 4), dtype=float)
        channels = len(self.inner_fn(0))
        
        for ch in range(channels):
            theta_eased[..., ch] = apply_channel_easing(theta, self.easing_theta, ch)

        # Compute local radius based on angle
        if self.r_theta:
            local_radius = self.r_theta(theta)
        else:
            local_radius = base_radius

        # Normalized radial factor (0 at center, 1 at radius)
        u_r = dist / local_radius

        result = np.zeros((height, width, channels), dtype=float)
        # ---------------------------------------------------------------------
        # MAIN LOOP: per-channel interpolation + bivariable transforms
        # ---------------------------------------------------------------------
        for ch in range(channels):

            # 1) First apply optional (r,theta) → (new_r,new_theta)
            if ch in self.bivariable_space_transforms:
                r_ch, theta_ch = self.bivariable_space_transforms[ch](dist, theta)
            else:
                r_ch = dist
                theta_ch = theta_eased[..., ch]

            # 2) Get angular colors using transformed theta
            inner = self.inner_fn(theta_ch)
            outer = self.outer_fn(theta_ch)

            # 3) Compute u_r for this channel (using transformed radius)
            u_r_ch = r_ch / local_radius
            u_r_ch = apply_channel_easing(u_r_ch, self.easing_r, ch)
            u_r_ch = np.clip(u_r_ch, 0.0, 1.0)

            # 4) Interpolate
            result[..., ch] = lerp(inner[..., ch], outer[..., ch], u_r_ch)

        # ---------------------------------------------------------------------
        # OUTSIDE FILL
        # ---------------------------------------------------------------------
        if fill_val is not None:
            mask_outside = u_r > 1.0
            result[mask_outside] = fill_val

        # Format output
        if self.format_type == FormatType.INT:
            return np.round(result).astype(np.uint16)
        return result.astype(np.float32)


def example_radial_angular():
    """Example: Four-color radial angular gradient."""
    from PIL import Image
    
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (255, 0, 0), (0, 0, 255)),      # Red center to blue edge at 0°
            (90, (0, 255, 0), (255, 255, 0)),   # Green center to yellow edge at 90°
            (180, (0, 0, 255), (255, 0, 0)),    # Blue center to red edge at 180°
            (270, (255, 255, 0), (0, 255, 0))   # Yellow center to green edge at 270°
        ],
        color_mode='rgb',
        format_type=FormatType.INT
    )
    
    img_array = gradient.render(500, 500, center=(250, 250), base_radius=200, outside_fill=(0,0,0))
    img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    img.show()


def example_elliptical():
    """Example: Elliptical gradient with variable radius."""
    from PIL import Image, ImageDraw
    import numpy as np
    def r_tetha(theta_deg: float) -> float:
        """Elliptical radius function."""
        theta_rad = np.radians(theta_deg)
        a = 200  # Semi-major axis
        b = 100  # Semi-minor axis
        return (a * b) / np.sqrt((b * np.cos(theta_rad))**2 + (a * np.sin(theta_rad))**2)
    # Create ellipse with radius varying by angle
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (100, 0, 0), (255, 100, 100)),    # Dark red center, light red edge
            (180, (0, 0, 100), (100, 100, 255))   # Dark blue center, light blue edge
        ],
        radius_stops=[
            (0, 200),    # Major axis horizontal
            (90, 100),   # Minor axis vertical
            (180, 200),
            (270, 100)
        ],r_theta=r_tetha,
        color_mode='rgb',
        format_type=FormatType.INT
    )
    
    img_array = gradient.render(500, 500, center=(250, 250), base_radius=1.0, outside_fill=(0,0,0))
    img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    draw = ImageDraw.Draw(img)
    #draw a small circle at the stops for reference
    for angle, radius in [(0, 200), (90, 100), (180, 200), (270, 100)]:
        theta_rad = np.radians(angle)
        x = 250 + radius * np.cos(theta_rad)
        y = 250 + radius * np.sin(theta_rad)
        # Get the outer color at this angle from the gradient
        outer_color = gradient.outer_fn(angle)
        color_tuple = tuple(int(c) for c in outer_color[:3])
        draw.ellipse([x-3, y-3, x+3, y+3], fill=color_tuple)
    img.show()


def example_star_shaped():
    """Example: Star-shaped gradient with 5 points."""
    from PIL import Image
    
    # Create 5-pointed star shape
    points = 5
    radius_stops = []
    for i in range(points * 2):
        angle = i * (360 / (points * 2))
        radius = 200 if i % 2 == 0 else 80  # Alternate between outer and inner
        radius_stops.append((angle, radius))
    
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (255, 255, 0), (255, 0, 0)),
            (72, (255, 0, 255), (0, 0, 255)),
            (144, (0, 255, 255), (0, 255, 0)),
            (216, (255, 128, 0), (128, 0, 255)),
            (288, (0, 255, 128), (128, 255, 0))
        ],
        radius_stops=radius_stops,
        color_mode='rgb',
        format_type=FormatType.INT
    )
    
    img_array = gradient.render(500, 500, center=(250, 250), base_radius=1.0, outside_fill=(0,0,0))
    img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    img.show()


def example_spiral():
    """Example: Spiral radial-angular gradient."""
    from PIL import Image
    import numpy as np

    def r_theta_spiral(theta_deg: float) -> float:
        """Spiral radius function."""
        return 20 + (theta_deg / 360.0) * 180  # Increases radius with angle

    def quadratic_tetha_easing(tetha: NDArray) -> NDArray:
        """Quadratic easing function for angle."""
        unit_tetha = tetha/360.0
        return 360*(unit_tetha * unit_tetha)
    def sine_tight_fade_in_out(x, length=0.1):
        x = np.asarray(x)  # ensure array
        fade_in = (0 <= x) & (x <= length)
        fade_out = (1 - length <= x) & (x <= 1)
        out_of_bounds = (x < 0) | (x > 1)
        return np.where(out_of_bounds, 0, np.where(fade_in | fade_out, np.sin((np.pi/(2*length)) * x), 1))
    def tight_fade_in_out(x):
        return sine_tight_fade_in_out(x, length=0.1)
    def sine_fade_in_out(x):
        x = np.asarray(x)  # ensure array
        return (np.sin(np.pi * x))
    def fade_radius_on_tetha_start_and_end(radius: NDArray, theta: NDArray) -> tuple[NDArray, NDArray]:

        """Fade radius to zero at start and end of theta range."""
        unit_tetha = theta / 360.0
        fade = tight_fade_in_out(unit_tetha)
        return radius * fade, theta
    def zeros_easing(x: NDArray) -> NDArray:
        return np.zeros_like(x)
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (255, 0, 255, 0), (255, 0, 255, 255)),
            (180, (255, 255, 0, 0), (255, 255, 0, 255)),
            (360.0, (0, 0, 255, 0), (0, 0, 255, 255))
        ],
        r_theta=r_theta_spiral,
        color_mode='rgba',
        format_type=FormatType.INT,
        #easing_theta={0: quadratic_tetha_easing, 1: quadratic_tetha_easing, 2: quadratic_tetha_easing, 3: quadratic_tetha_easing},
        easing_r={0: sine_fade_in_out, 1: sine_fade_in_out, 2: sine_fade_in_out, 3: lambda x: sine_tight_fade_in_out(x, length=0.5)},
        bivariable_space_transforms={3: fade_radius_on_tetha_start_and_end}
    )
    
    img_array = gradient.render(2*500, 2*500, center=(2*250, 2*250), base_radius=2*1.0, 
                                outside_fill=(0,0,0,0)
                                )
    img = Image.fromarray(img_array.astype(np.uint8), mode='RGBA')
    img.show()



if __name__ == '__main__':
    # Run examples
    print("Generating radial-angular gradient examples...")
    example_radial_angular()
    example_elliptical()
    example_star_shaped()
