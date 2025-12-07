"""
Example demonstrations of FullParametricalAngularRadialGradient
================================================================

This module contains examples showcasing the advanced capabilities of the
FullParametricalAngularRadialGradient class.
"""

import numpy as np
from PIL import Image
from .full_parametrical_angular_radial import FullParametricalAngularRadialGradient
from ..format_type import FormatType


def example_basic_three_rings():
    """Basic example: Three concentric rings with different color stops."""
    print("Generating basic three-ring gradient...")
    
    gradient = FullParametricalAngularRadialGradient.generate(
        width=600,
        height=600,
        inner_r_theta=lambda theta: np.full_like(theta, 50),
        outer_r_theta=lambda theta: np.full_like(theta, 250),
        color_rings=[
            # Inner ring: red → green → blue → red
            ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0)),
            # Middle ring: cyan → magenta → yellow → cyan
            ((0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255)),
            # Outer ring: white → black → white → black
            ((255, 255, 255), (0, 0, 0), (255, 255, 255), (0, 0, 0))
        ],
        color_space='rgb',
        format_type=FormatType.INT,
        outside_fill=(32, 32, 32)
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGB')
    img.show()
    return gradient


def example_elliptical_gradient():
    """Elliptical gradient with variable radius."""
    print("Generating elliptical gradient...")
    
    def ellipse_r(theta):
        """Elliptical radius function."""
        theta_rad = np.radians(theta)
        a, b = 200, 100
        return (a * b) / np.sqrt((b * np.cos(theta_rad))**2 + (a * np.sin(theta_rad))**2)
    
    gradient = FullParametricalAngularRadialGradient.generate(
        width=600,
        height=600,
        inner_r_theta=lambda theta: ellipse_r(theta) * 0.3,
        outer_r_theta=ellipse_r,
        color_rings=[
            ((255, 0, 0), (255, 128, 0)),
            ((0, 0, 255), (128, 0, 255))
        ],
        color_space='rgb',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 0)
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGB')
    img.show()
    return gradient


def example_flower_shaped():
    """Flower-shaped gradient with 6 petals."""
    print("Generating flower-shaped gradient...")
    
    def flower_r(theta):
        """Flower-shaped radius function."""
        petals = 6
        return 150 + 80 * np.abs(np.sin(petals * np.radians(theta) / 2))
    
    gradient = FullParametricalAngularRadialGradient.generate(
        width=700,
        height=700,
        inner_r_theta=lambda theta: np.full_like(theta, 40),
        outer_r_theta=flower_r,
        color_rings=[
            # Center: yellow
            ((255, 255, 0),),
            # Petals transition through rainbow
            ((255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (128, 0, 255)),
            # Outer edge: dark
            ((64, 0, 64), (0, 64, 64), (64, 64, 0), (64, 0, 0), (0, 0, 64), (64, 32, 0))
        ],
        color_space='rgb',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 0)
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGB')
    img.show()
    return gradient


def example_spiral_with_transform():
    """Spiral gradient using bivariable space transform."""
    print("Generating spiral gradient with transform...")
    
    def spiral_transform(r, theta):
        """Transform that creates a spiral effect."""
        # Rotate theta based on radius
        spiral_rate = 0.5  # degrees per pixel
        theta_new = (theta + r * spiral_rate) % 360.0
        return r, theta_new
    
    gradient = FullParametricalAngularRadialGradient.generate(
        width=600,
        height=600,
        inner_r_theta=lambda theta: np.full_like(theta, 20),
        outer_r_theta=lambda theta: np.full_like(theta, 280),
        color_rings=[
            ((255, 0, 255), (255, 255, 0), (0, 255, 255)),
            ((128, 0, 0), (0, 128, 0), (0, 0, 128)),
        ],
        color_space='rgb',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 0),
        bivariable_space_transforms={0: spiral_transform, 1: spiral_transform, 2: spiral_transform}
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGB')
    img.show()
    return gradient


def example_hsv_with_hue_directions():
    """HSV gradient with controlled hue interpolation."""
    print("Generating HSV gradient with hue directions...")
    
    gradient = FullParametricalAngularRadialGradient.generate(
        width=600,
        height=600,
        inner_r_theta=lambda theta: np.full_like(theta, 50),
        outer_r_theta=lambda theta: np.full_like(theta, 250),
        color_rings=[
            # Inner ring: red → green (clockwise through rainbow)
            ((0, 100, 100), (120, 100, 100)),
            # Outer ring: green → red (counter-clockwise)
            ((120, 100, 80), (0, 100, 80))
        ],
        color_space='hsv',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 0),
        hue_directions_theta=[
            ['cw'],  # Inner ring: clockwise hue
            ['ccw']  # Outer ring: counter-clockwise hue
        ],
        hue_directions_r=['cw']  # Radial: clockwise hue between rings
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='HSV').convert('RGB')
    img.show()
    return gradient


def example_easing_functions():
    """Gradient with custom easing functions."""
    print("Generating gradient with easing functions...")
    
    def ease_in_out_cubic(x):
        """Cubic ease-in-out function."""
        return np.where(x < 0.5, 4 * x**3, 1 - (-2 * x + 2)**3 / 2)
    
    def pulse_easing(x):
        """Pulsing easing function."""
        return (np.sin(x * np.pi * 4) + 1) / 2
    
    gradient = FullParametricalAngularRadialGradient.generate(
        width=600,
        height=600,
        inner_r_theta=lambda theta: np.full_like(theta, 50),
        outer_r_theta=lambda theta: np.full_like(theta, 280),
        color_rings=[
            ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
            ((255, 255, 0), (255, 0, 255), (0, 255, 255))
        ],
        color_space='rgb',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 0),
        easing_theta={0: pulse_easing, 1: pulse_easing, 2: pulse_easing},
        easing_r={0: ease_in_out_cubic, 1: ease_in_out_cubic, 2: ease_in_out_cubic}
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGB')
    img.show()
    return gradient


def example_star_burst():
    """Star burst pattern with many rings."""
    print("Generating star burst pattern...")
    
    def star_r(theta):
        """Star-shaped radius function."""
        points = 8
        base = 200
        spike = 100
        return base + spike * np.abs(np.cos(points * np.radians(theta) / 2))
    
    # Create many rings for smooth gradation
    num_rings = 8
    color_rings = []
    for i in range(num_rings):
        # Vary hue across rings
        hue = (i * 360 / num_rings) % 360
        # Each ring has 8 color stops (matching star points)
        ring_colors = []
        for j in range(8):
            angle_hue = (hue + j * 45) % 360
            ring_colors.append((angle_hue, 100, 100))
        color_rings.append(tuple(ring_colors))
    
    gradient = FullParametricalAngularRadialGradient.generate(
        width=800,
        height=800,
        inner_r_theta=lambda theta: np.full_like(theta, 30),
        outer_r_theta=star_r,
        color_rings=color_rings,
        color_space='hsv',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 20)
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='HSV').convert('RGB')
    img.show()
    return gradient


def example_angular_segment():
    """Gradient constrained to an angular segment."""
    print("Generating angular segment gradient...")
    
    gradient = FullParametricalAngularRadialGradient.generate(
        width=600,
        height=600,
        inner_r_theta=lambda theta: np.full_like(theta, 80),
        outer_r_theta=lambda theta: np.full_like(theta, 280),
        color_rings=[
            ((255, 0, 0), (255, 128, 0), (255, 255, 0)),
            ((0, 255, 0), (0, 255, 128), (0, 255, 255)),
            ((0, 0, 255), (128, 0, 255), (255, 0, 255))
        ],
        color_space='rgb',
        format_type=FormatType.INT,
        deg_start=45.0,
        deg_end=315.0,
        normalize_theta=True,
        outside_fill=(32, 32, 32)
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGB')
    img.show()
    return gradient


def example_color_transform():
    """Gradient with post-processing color transform."""
    print("Generating gradient with color transform...")
    
    def fade_by_angle(r, theta, color):
        """Fade color based on angle."""
        fade = (np.sin(np.radians(theta * 3)) + 1) / 2
        return color * fade
    
    gradient = FullParametricalAngularRadialGradient.generate(
        width=600,
        height=600,
        inner_r_theta=lambda theta: np.full_like(theta, 50),
        outer_r_theta=lambda theta: np.full_like(theta, 280),
        color_rings=[
            ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
            ((255, 255, 0), (255, 0, 255), (0, 255, 255))
        ],
        color_space='rgb',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 0),
        bivariable_color_transforms={0: fade_by_angle, 1: fade_by_angle, 2: fade_by_angle}
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGB')
    img.show()
    return gradient


def run_all_examples():
    """Run all example functions."""
    examples = [
        example_basic_three_rings,
        example_elliptical_gradient,
        example_flower_shaped,
        example_spiral_with_transform,
        example_hsv_with_hue_directions,
        example_easing_functions,
        example_star_burst,
        example_angular_segment,
        example_color_transform
    ]
    
    print("Running all FullParametricalAngularRadialGradient examples...")
    print("=" * 70)
    
    for example_func in examples:
        try:
            example_func()
            print(f"✓ {example_func.__name__} completed successfully\n")
        except Exception as e:
            print(f"✗ {example_func.__name__} failed: {e}\n")
    
    print("=" * 70)
    print("All examples completed!")


if __name__ == '__main__':
    run_all_examples()
