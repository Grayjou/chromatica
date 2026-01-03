import pytest
import numpy as np
from ..chromatica.radial_angular_gradient import (
    RadialAngularGradient,
    build_r_theta_from_stops,
    interpolate_hue_vector,
    build_theta_interpolators
)
from ..chromatica.colors.rgb import ColorUnitRGB
from ..chromatica.types.format_type import FormatType


def test_radial_angular_gradient_basic():
    """Test basic radial angular gradient creation."""
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (255, 0, 0), (0, 0, 255)),
            (180, (0, 255, 0), (255, 255, 0))
        ],
        color_mode='rgb',
        format_type=FormatType.INT
    )
    
    result = gradient.render(100, 100, center=(50, 50), base_radius=40)
    
    # Check shape
    assert result.shape == (100, 100, 3)
    
    # Check center is close to inner color at 0 degrees (red)
    center_color = result[50, 50]
    assert center_color[0] > 200  # Should be reddish


def test_radial_angular_gradient_rgba():
    """Test radial angular gradient with alpha channel."""
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (255, 0, 0, 255), (0, 0, 255, 128)),
            (180, (0, 255, 0, 255), (255, 255, 0, 128))
        ],
        color_mode='rgba',
        format_type=FormatType.INT
    )
    
    result = gradient.render(100, 100, center=(50, 50), base_radius=40)
    
    # Check shape has 4 channels
    assert result.shape == (100, 100, 4)
    
    # Check center has high alpha
    center_alpha = result[50, 50, 3]
    assert center_alpha > 200


def test_build_r_theta_from_stops():
    """Test building radius function from angular stops."""
    r_fn = build_r_theta_from_stops([
        (0, 100),
        (90, 150),
        (180, 100),
        (270, 50)
    ])
    
    # Test specific angles
    assert r_fn(0) == 100
    assert r_fn(90) == 150
    assert r_fn(180) == 100
    assert r_fn(270) == 50
    
    # Test interpolation
    r_45 = r_fn(45)
    assert 100 < r_45 < 150
    
    # Test wraparound
    r_360 = r_fn(360)
    assert r_360 == r_fn(0)


def test_radial_angular_with_radius_stops():
    """Test radial angular gradient with variable radius (elliptical)."""
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (255, 0, 0), (0, 0, 255)),
            (180, (0, 255, 0), (255, 255, 0))
        ],
        radius_stops=[
            (0, 50),    # Horizontal major axis
            (90, 25),   # Vertical minor axis
            (180, 50),
            (270, 25)
        ],
        color_mode='rgb',
        format_type=FormatType.INT
    )
    
    result = gradient.render(100, 100, center=(50, 50), base_radius=1.0)
    
    # Check shape
    assert result.shape == (100, 100, 3)


def test_radial_angular_outside_fill():
    """Test outside fill functionality."""
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (255, 0, 0), (0, 0, 255)),
            (180, (0, 255, 0), (255, 255, 0))
        ],
        color_mode='rgb',
        format_type=FormatType.INT
    )
    
    result = gradient.render(
        100, 100, 
        center=(50, 50), 
        base_radius=20,
        outside_fill=(255, 255, 255)  # White outside
    )
    
    # Check corners are white (outside radius)
    corner = result[0, 0]
    assert np.allclose(corner, [255, 255, 255], atol=5)


def test_interpolate_hue_vector():
    """Test hue interpolation with direction control."""
    c0 = np.array([0.0, 1.0, 1.0])    # Red in HSV
    c1 = np.array([240.0, 1.0, 1.0])  # Blue in HSV
    
    # Test clockwise
    result_cw = interpolate_hue_vector(c0, c1, np.array(0.5), direction='cw')
    assert 0 < result_cw[0] < 240  # Should go through yellow/green
    
    # Test counter-clockwise
    result_ccw = interpolate_hue_vector(c0, c1, np.array(0.5), direction='ccw')
    assert result_ccw[0] > 240  # Should go through magenta (>240, wrapped)


def test_radial_angular_hsv_with_hue_direction():
    """Test HSV gradient with hue direction control."""
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (0.0, 1.0, 1.0), (120.0, 1.0, 1.0)),      # Red to green
            (180, (240.0, 1.0, 1.0), (0.0, 1.0, 1.0))     # Blue to red
        ],
        hue_direction_theta='cw',
        color_mode='hsv',
        format_type=FormatType.FLOAT
    )
    
    result = gradient.render(100, 100, center=(50, 50), base_radius=40)
    
    # Check shape
    assert result.shape == (100, 100, 3)
    
    # Check hue values are in valid range
    hues = result[..., 0]
    assert np.all((hues >= 0) & (hues <= 360))


def test_radial_angular_colorbase_input():
    """Test that radial angular gradient accepts ColorBase instances."""
    color_inner = ColorUnitRGB((1.0, 0.0, 0.0))
    color_outer = ColorUnitRGB((0.0, 0.0, 1.0))
    
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, color_inner, color_outer),
            (180, color_outer, color_inner)
        ],
        color_mode='rgb',
        format_type=FormatType.FLOAT
    )
    
    result = gradient.render(50, 50, center=(25, 25), base_radius=20)
    
    # Check shape
    assert result.shape == (50, 50, 3)
    
    # Check center is reddish
    center = result[25, 25]
    assert center[0] > 0.8  # High red component


def test_build_theta_interpolators():
    """Test building angular color interpolators."""
    from ..chromatica.colors.rgb import ColorRGBINT
    
    theta_stops = [
        (0, (255, 0, 0), (0, 0, 255)),
        (180, (0, 255, 0), (255, 255, 0))
    ]
    
    inner_fn, outer_fn = build_theta_interpolators(
        theta_stops,
        ColorRGBINT,
        hue_direction=None
    )
    
    # Test at 0 degrees
    inner_0 = inner_fn(0)
    outer_0 = outer_fn(0)
    assert np.allclose(inner_0, [255, 0, 0])
    assert np.allclose(outer_0, [0, 0, 255])
    
    # Test at 180 degrees
    inner_180 = inner_fn(180)
    outer_180 = outer_fn(180)
    assert np.allclose(inner_180, [0, 255, 0])
    assert np.allclose(outer_180, [255, 255, 0])
    
    # Test interpolation at 90 degrees
    inner_90 = inner_fn(90)
    # Should be interpolated between red and green
    assert inner_90[0] > 0  # Has red
    assert inner_90[1] > 0  # Has green


def test_radial_angular_four_quadrants():
    """Test radial angular gradient with different colors in four quadrants."""
    gradient = RadialAngularGradient(
        theta_stops=[
            (0, (255, 0, 0), (128, 0, 0)),      # Red quadrant
            (90, (0, 255, 0), (0, 128, 0)),     # Green quadrant
            (180, (0, 0, 255), (0, 0, 128)),    # Blue quadrant
            (270, (255, 255, 0), (128, 128, 0)) # Yellow quadrant
        ],
        color_mode='rgb',
        format_type=FormatType.INT
    )
    
    result = gradient.render(100, 100, center=(50, 50), base_radius=40)
    
    # Check shape
    assert result.shape == (100, 100, 3)
    
    # Check that different quadrants have different dominant colors
    # Right side (0°) should be reddish
    right = result[50, 90]
    assert right[0] > right[1] and right[0] > right[2]
    
    # Top side (90°) should be greenish
    top = result[10, 50]
    assert top[1] >= top[0] and top[1] >= top[2]  # Allow equal values due to interpolation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
