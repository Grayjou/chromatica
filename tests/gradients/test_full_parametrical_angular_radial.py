"""
Unit tests for FullParametricalAngularRadialGradient class.
Tests focus on individual helper methods and integration scenarios.
"""

import pytest
import numpy as np
from ...chromatica.gradients.full_parametrical_angular_radial import (
    FullParametricalAngularRadialGradient,
    build_angular_mask,
    normalize_theta_range,
    normalize_radial_distances,
    cached_normalize_color,
)
from ...chromatica.format_type import FormatType
from ..utils import get_point

class TestAngularMask:
    """Tests for angular mask generation."""
    
    def test_full_circle_no_normalization(self):
        """Test that without normalization, full mask is returned."""
        theta = np.array([0, 90, 180, 270])
        mask = build_angular_mask(theta, 0, 360, normalize_theta=False)
        assert np.all(mask)
    
    def test_full_circle_with_normalization(self):
        """Test full 360 degree range returns all True."""
        theta = np.array([0, 90, 180, 270])
        mask = build_angular_mask(theta, 0, 360, normalize_theta=True)
        assert np.all(mask)
    
    def test_partial_range_simple(self):
        """Test simple partial angular range."""
        theta = np.array([0, 90, 180, 270])
        mask = build_angular_mask(theta, 45, 135, normalize_theta=True)
        # Only 90 degrees should be in range
        assert mask[1]  # 90 is in range
        assert not mask[0]  # 0 is out of range
        assert not mask[2]  # 180 is out of range
        assert not mask[3]  # 270 is out of range
     
    def test_partial_range_gt360(self):
        """Test angular range greater than 360 degrees."""
        theta = np.array([0, 90, 180, 270])
        mask = build_angular_mask(theta, 360, 449, normalize_theta=True)
        # All should be in range since 450 > 360
        assert mask[0]
        assert not mask[1]
        assert not mask[2]
        assert not mask[3]

    def test_partial_range_lt0(self):
        """Test angular range less than 0 degrees."""
        theta = np.array([0, 90, 180, 270])
        mask = build_angular_mask(theta, -89, 45, normalize_theta=True)
        # Only 0 degrees should be in range
        assert mask[0]  # 0 is in range
        assert not mask[1]  # 90 is out of range
        assert not mask[2]  # 180 is out of range
        assert not mask[3]  # 270 is out of range

    def test_wrap_around_range(self):
        """Test angular range that wraps around 0."""
        theta = np.array([0, 90, 180, 270, 350])
        mask = build_angular_mask(theta, 315, 45, normalize_theta=True)
        # Should include 315-360 and 0-45
        assert mask[0]  # 0 is in range
        assert not mask[1]  # 90 is out of range
        assert not mask[2]  # 180 is out of range
        assert not mask[3]  # 270 is out of range
        assert mask[4]  # 350 is in range


class TestThetaNormalization:
    """Tests for theta normalization."""
    
    def test_no_normalization(self):
        """Test that without normalization, theta is scaled to [0, 1]."""
        theta = np.array([0, 90, 180, 270, 360])
        mask = np.ones_like(theta, dtype=bool)
        normalized, theta_range = normalize_theta_range(theta, 0, 360, mask, normalize_theta=False)
        
        assert theta_range == 360.0
        np.testing.assert_array_almost_equal(normalized, theta / 360.0)
    
    def test_simple_range(self):
        """Test normalization for a simple range."""
        theta = np.array([0, 45, 90, 135, 180])
        mask = (theta >= 45) & (theta <= 135)
        normalized, theta_range = normalize_theta_range(theta, 45, 135, mask, normalize_theta=True)
        
        assert theta_range == 90.0
        # 45 should map to 0, 135 should map to 1
        assert normalized[1] == pytest.approx(0.0, abs=0.01)
        assert normalized[3] == pytest.approx(1.0, abs=0.01)
    
    def test_wrap_around_range(self):
        """Test normalization for wrap-around range."""
        theta = np.array([0, 45, 315, 350])
        mask = (theta >= 315) | (theta <= 45)
        normalized, theta_range = normalize_theta_range(theta, 315, 45, mask, normalize_theta=True)
        
        # Range is 315-360 (45 degrees) + 0-45 (45 degrees) = 90 degrees
        assert theta_range == 90.0


class TestRadialNormalization:
    """Tests for radial distance normalization."""
    
    def test_simple_normalization(self):
        """Test basic radial normalization."""
        distances = np.array([0, 50, 100, 150, 200])
        inner = np.full_like(distances, 50.0)
        outer = np.full_like(distances, 150.0)
        
        normalized, mask = normalize_radial_distances(distances, inner, outer, normalize_radius=True)
        
        # Check bounds
        assert normalized[0] == 0.0  # Below inner radius
        assert normalized[1] == 0.0  # At inner radius
        assert normalized[2] == 0.5  # Midpoint
        assert normalized[3] == 1.0  # At outer radius
        assert normalized[4] == 1.0  # Beyond outer radius
        
        # Check mask
        assert not mask[0]  # Below inner
        assert mask[1]  # At inner
        assert mask[2]  # Between
        assert mask[3]  # At outer
        assert not mask[4]  # Beyond outer
    
    def test_varying_radii(self):
        """Test with varying inner and outer radii."""
        distances = np.array([50, 100, 150])
        inner = np.array([40, 80, 120])
        outer = np.array([60, 120, 180])
        
        normalized, mask = normalize_radial_distances(distances, inner, outer, normalize_radius=True)
        
        # Each should be at midpoint between its inner and outer
        assert normalized[0] == 0.5  # (50-40)/(60-40) = 0.5
        assert normalized[1] == 0.5  # (100-80)/(120-80) = 0.5
        assert normalized[2] == 0.5  # (150-120)/(180-120) = 0.5

    def test_parametric_radii(self):
        """Test with parametric inner and outer radius functions."""
        distances = np.array([0, 50, 100, 150, 200])
        inner = lambda t: 0.2 * t  # Inner radius grows with distance
        outer = lambda t: 0.8 * t  # Outer radius grows with distance
        
        inner_vals = inner(distances)
        outer_vals = outer(distances)
        
        normalized, mask = normalize_radial_distances(distances, inner_vals, outer_vals, normalize_radius=True)
        
        # Check some expected values
        assert normalized[1] == pytest.approx((50 - 10) / (40), abs=0.01)  # (50-10)/(40)
        assert normalized[3] == pytest.approx((150 - 30) / (120), abs=0.01)  # (150-30)/(120)

class TestColorNormalization:
    """Tests for color normalization."""
    
    def test_cached_normalize_color(self):
        """Test that color normalization caching works."""
        color1 = (255, 0, 0)
        color2 = (255, 0, 0)
        
        # These should return the same cached result
        result1 = cached_normalize_color(color1)
        result2 = cached_normalize_color(color2)
        
        assert result1 == result2
        assert isinstance(result1, tuple)


class TestInputValidation:
    """Tests for input validation methods."""
    
    def test_validate_inputs_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be positive"):
            FullParametricalAngularRadialGradient._validate_inputs(
                0, 100, lambda x: x, lambda x: x, [((255, 0, 0),)]
            )
    
    def test_validate_inputs_invalid_functions(self):
        """Test that non-callable radius functions raise TypeError."""
        with pytest.raises(TypeError, match="must be callable"):
            FullParametricalAngularRadialGradient._validate_inputs(
                100, 100, None, lambda x: x, [((255, 0, 0),)]
            )
    
    def test_validate_inputs_empty_color_rings(self):
        """Test that empty color rings raise ValueError."""
        with pytest.raises(ValueError, match="must contain at least one ring"):
            FullParametricalAngularRadialGradient._validate_inputs(
                100, 100, lambda x: x, lambda x: x, []
            )
    
    def test_validate_inputs_empty_ring(self):
        """Test that empty ring raises ValueError."""
        with pytest.raises(ValueError, match="must contain at least one color"):
            FullParametricalAngularRadialGradient._validate_inputs(
                100, 100, lambda x: x, lambda x: x, [()]
            )
    
    def test_validate_radius_functions_negative(self):
        """Test that negative radius values raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            FullParametricalAngularRadialGradient._validate_radius_functions(
                lambda x: np.full_like(x, -10),
                lambda x: np.full_like(x, 100)
            )
    
    def test_validate_hue_directions_wrong_count(self):
        """Test that wrong hue direction count raises ValueError."""
        with pytest.raises(ValueError, match="must have 2 entries"):
            FullParametricalAngularRadialGradient._validate_hue_directions(
                [['cw'], ['ccw'], ['cw']],  # 3 entries
                None,
                2  # Only 2 rings
            )


class TestHelperMethods:
    """Tests for static helper methods."""
    
    def test_normalize_color_rings(self):
        """Test color ring normalization."""
        rings = [
            ((255, 0, 0), (0, 255, 0)),
            ((0, 0, 255), (255, 255, 0))
        ]
        
        normalized = FullParametricalAngularRadialGradient._normalize_color_rings(rings)
        
        assert len(normalized) == 2
        assert len(normalized[0]) == 2
        assert len(normalized[1]) == 2
    
    def test_compute_radius_arrays(self):
        """Test radius array computation."""
        theta = np.array([0, 90, 180, 270])
        distances = np.array([50, 100, 150, 200])
        
        inner_r = lambda t: np.full_like(t, 50.0)
        outer_r = lambda t: np.full_like(t, 200.0)
        
        inner, outer = FullParametricalAngularRadialGradient._compute_radius_arrays(
            theta, inner_r, outer_r, distances, deg_start=0, 
            rotate_r_theta_with_theta_normalization=False
        )
        
        assert np.all(inner == 50.0)
        assert np.all(outer == 200.0)
    
    def test_apply_radial_easing(self):
        """Test radial easing application."""
        u_r = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        
        # Without easing
        result = FullParametricalAngularRadialGradient._apply_radial_easing(
            u_r, num_channels=3, easing_r=None
        )
        
        assert len(result) == 3
        for ch in range(3):
            np.testing.assert_array_equal(result[ch], u_r)
        
        # With easing (square function)
        easing = {0: lambda x: x ** 2}
        result = FullParametricalAngularRadialGradient._apply_radial_easing(
            u_r, num_channels=3, easing_r=easing
        )
        
        # Channel 0 should be squared
        np.testing.assert_array_almost_equal(result[0], u_r ** 2)
        # Channels 1 and 2 should be unchanged
        np.testing.assert_array_equal(result[1], u_r)
        np.testing.assert_array_equal(result[2], u_r)


class TestGradientGeneration:
    """Integration tests for gradient generation."""
    
    def test_simple_single_ring(self):
        """Test generating a simple single-ring gradient."""
        gradient = FullParametricalAngularRadialGradient.generate(
            width=100,
            height=100,
            inner_r_theta=lambda theta: np.full_like(theta, 10),
            outer_r_theta=lambda theta: np.full_like(theta, 40),
            color_rings=[((255, 0, 0), (0, 0, 255))],
            color_space='rgb',
            format_type=FormatType.INT
        )
        
        assert gradient.value.shape == (100, 100, 3)
        # Check it's an integer type
        assert np.issubdtype(gradient.value.dtype, np.integer)
        this_r, this_theta = 25, 180
        pixel = get_point(gradient, (50, 50), this_r, this_theta)
        # At angle 180, should be halfway between red and blue
        assert 120 < pixel[0] < 200  # Red channel
        assert 120 < pixel[2] < 200  # Blue channel
        assert pixel[1] == 0    # Green channel
    
    def test_multiple_rings(self):
        """Test generating a multi-ring gradient."""
        gradient = FullParametricalAngularRadialGradient.generate(
            width=100,
            height=100,
            inner_r_theta=lambda theta: np.full_like(theta, 5),
            outer_r_theta=lambda theta: np.full_like(theta, 45),
            color_rings=[
                ((255, 0, 0),(255, 255, 0)),
                ((0, 255, 0), (0, 255, 255)),
                ((0, 0, 255), (255, 0, 255))
            ],
            color_space='rgb',
            format_type=FormatType.INT
        )
        this_r, this_theta = 25, 180 
        pixel = get_point(gradient, (50, 50), this_r, this_theta)
        # Should be in the second ring (green to cyan) at halfway point
        assert pixel[0] < 5    # Red channel
        assert pixel[1] > 250  # Green channel
        assert 120 < pixel[2] < 200  # Blue channel
        assert gradient.value.shape == (100, 100, 3)
    
    def test_angular_range(self):
        """Test gradient with limited angular range."""
        gradient = FullParametricalAngularRadialGradient.generate(
            width=100,
            height=100,
            inner_r_theta=lambda theta: np.full_like(theta, 10),
            outer_r_theta=lambda theta: np.full_like(theta, 40),
            color_rings=[((255, 0, 0), (0, 0, 255))],
            color_space='rgb',
            format_type=FormatType.INT,
            deg_start=0,
            deg_end=180,
            normalize_theta=True
        )
        pixel = get_point(gradient, (50, 50), 25, 181)  # Outside angular range
        assert np.array_equal(pixel, [0, 0, 0])  # Should be black (outside fill)
        assert gradient.value.shape == (100, 100, 3)
    
    def test_with_outside_fill(self):
        """Test gradient with outside fill color."""
        gradient = FullParametricalAngularRadialGradient.generate(
            width=100,
            height=100,
            inner_r_theta=lambda theta: np.full_like(theta, 10),
            outer_r_theta=lambda theta: np.full_like(theta, 30),
            color_rings=[((255, 0, 0),)],
            color_space='rgb',
            format_type=FormatType.INT,
            outside_fill=(128, 128, 128)
        )
        
        assert gradient.value.shape == (100, 100, 3)
        # Check that corners (likely outside) have the fill color
        corner = gradient.value[0, 0]
        # Outside areas should have gray-ish color
        assert corner[0] == corner[1] == corner[2]
    
    def test_float_format(self):
        """Test generating gradient in float format."""
        gradient = FullParametricalAngularRadialGradient.generate(
            width=50,
            height=50,
            inner_r_theta=lambda theta: np.full_like(theta, 5),
            outer_r_theta=lambda theta: np.full_like(theta, 20),
            color_rings=[((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
            color_space='rgb',
            format_type=FormatType.FLOAT
        )
        
        assert gradient.value.shape == (50, 50, 3)
        assert gradient.value.dtype == np.float32
        assert np.all(gradient.value >= 0.0)
        assert np.all(gradient.value <= 1.0)


class TestShapeHelpers:
    """Tests for shape helper methods."""
    
    def test_create_elliptical(self):
        """Test elliptical gradient creation."""
        gradient = FullParametricalAngularRadialGradient.create_elliptical(
            width=100,
            height=100,
            colors=((255, 0, 0), (0, 0, 255)),
            eccentricity=0.5,
            color_space='rgb',
            format_type=FormatType.INT
        )
        
        assert gradient.value.shape == (100, 100, 3)
    
    def test_create_star(self):
        """Test star-shaped gradient creation."""
        gradient = FullParametricalAngularRadialGradient.create_star(
            width=100,
            height=100,
            colors=((255, 255, 0), (255, 0, 255)),
            points=5,
            inner_ratio=0.4,
            color_space='rgb',
            format_type=FormatType.INT
        )
        
        assert gradient.value.shape == (100, 100, 3)
    
    def test_create_star_with_different_points(self):
        """Test star with different number of points."""
        gradient_5 = FullParametricalAngularRadialGradient.create_star(
            width=100, height=100,
            colors=((255, 0, 0),),
            points=5,
            color_space='rgb',
            format_type=FormatType.INT
        )
        
        gradient_8 = FullParametricalAngularRadialGradient.create_star(
            width=100, height=100,
            colors=((255, 0, 0),),
            points=8,
            color_space='rgb',
            format_type=FormatType.INT
        )
        
        assert gradient_5.value.shape == (100, 100, 3)
        assert gradient_8.value.shape == (100, 100, 3)
        # Different star shapes should produce different results
        assert not np.array_equal(gradient_5.value, gradient_8.value)


class TestVariableRadiusFunctions:
    """Tests for gradients with variable radius functions."""
    
    def test_sinusoidal_radius(self):
        """Test gradient with sinusoidal radius function."""
        def wave_r(theta):
            return 30 + 10 * np.sin(np.radians(theta * 3))
        
        gradient = FullParametricalAngularRadialGradient.generate(
            width=100,
            height=100,
            inner_r_theta=lambda theta: np.full_like(theta, 10),
            outer_r_theta=wave_r,
            color_rings=[((255, 0, 0), (0, 0, 255))],
            color_space='rgb',
            format_type=FormatType.INT
        )

        pixel = get_point(gradient, (50, 50), 29, -1) 
        # Should be near outer color (blue)
        assert pixel[2] > 250  # Blue channel
        pixel = get_point(gradient, (50, 50), 40, 30) 
        assert pixel[0] > 250*(1-30/360)  # Red channel
        assert gradient.value.shape == (100, 100, 3)
    
    def test_flower_petals(self):
        """Test flower petal-like radius function."""
        def petal_r(theta):
            petals = 6
            return 25 + 15 * np.abs(np.sin(petals * np.radians(theta) / 2))
        
        gradient = FullParametricalAngularRadialGradient.generate(
            width=120,
            height=120,
            inner_r_theta=lambda theta: np.full_like(theta, 5),
            outer_r_theta=petal_r,
            color_rings=[((255, 255, 0), (255, 0, 255)),
                         ((255, 128, 255), (255, 255, 128))],
            color_space='rgb',
            format_type=FormatType.INT
        )

        pixel = get_point(gradient, (60, 60), 6, 30) 
        # Should be in first ring (yellow to magenta)
        assert pixel[0] > 250  # Red channel
        assert pixel[1] > 250*(1-30/360)  # Green channel
        assert pixel[2] > 255*(30/360)  # Blue channel
        outer_pixel = get_point(gradient, (60, 60), 39, 30)
        # Should be in second ring (pinkish)
        assert outer_pixel[0] > 250  # Red channel
        assert 100 < outer_pixel[1] < 200  # Green channel
        assert outer_pixel[2] > 200  # Blue channel
        assert gradient.value.shape == (120, 120, 3)


class TestHSVGradients:
    """Tests for HSV color space gradients."""
    
    def test_hsv_single_ring(self):
        """Test HSV gradient generation."""
        gradient = FullParametricalAngularRadialGradient.generate(
            width=100,
            height=100,
            inner_r_theta=lambda theta: np.full_like(theta, 10),
            outer_r_theta=lambda theta: np.full_like(theta, 40),
            color_rings=[((0.0, 1.0, 1.0), (120.0, 1.0, 1.0))],
            color_space='hsv',
            format_type=FormatType.FLOAT
        )
        gradient = gradient.convert('rgb', to_format=FormatType.INT)

        pixel = get_point(gradient, (50, 50), 25, -1)  # Midway hue between 0 and 120
        assert pixel[0] < 5  # Red channel
        assert pixel[1] > 250  # Green channel
        assert pixel[2] < 5   # Blue channel
        midpixel = get_point(gradient, (50, 50), 25, 180)  # yellow
        assert midpixel[0] > 250  # Red channel
        assert midpixel[1] > 250  # Green channel
        assert midpixel[2] < 5    # Blue channel
        assert gradient.value.shape == (100, 100, 3)
    
    def test_hsv_with_hue_direction(self):
        """Test HSV gradient with hue direction control."""
        gradient = FullParametricalAngularRadialGradient.generate(
            width=100,
            height=100,
            inner_r_theta=lambda theta: np.full_like(theta, 10),
            outer_r_theta=lambda theta: np.full_like(theta, 40),
            color_rings=[((0.0, 1.0, 1.0), (240.0, 1.0, 1.0))],
            color_space='hsv',
            format_type=FormatType.FLOAT,
            hue_directions_theta=[['cw']]
        )
        gradient = gradient.convert('rgb', to_format=FormatType.INT)

        royal_blue_pixel = get_point(gradient, (50, 50), 25, -1)  # Should be near royal blue
        assert royal_blue_pixel[0] < 5    # Red channel
        assert royal_blue_pixel[1] < 10    # Green channel
        assert royal_blue_pixel[2] > 250  # Blue channel
        midpoint_green = get_point(gradient, (50, 50), 25, 180)  # Should be near green
        assert midpoint_green[0] < 10    # Red channel
        assert midpoint_green[1] > 250  # Green channel
        assert midpoint_green[2] < 10    # Blue channel
        gradient = FullParametricalAngularRadialGradient.generate(
            width=100,
            height=100,
            inner_r_theta=lambda theta: np.full_like(theta, 10),
            outer_r_theta=lambda theta: np.full_like(theta, 40),
            color_rings=[((0.0, 1.0, 1.0), (240.0, 1.0, 1.0))],
            color_space='hsv',
            format_type=FormatType.FLOAT,
            hue_directions_theta=[['ccw']]
        )
        gradient = gradient.convert('rgb', to_format=FormatType.INT)
        midpoint_magenta = get_point(gradient, (50, 50), 25, 180)  # Should be near magenta
        assert midpoint_magenta[0] > 250  # Red channel
        assert midpoint_magenta[1] < 1    # Green channel
        assert midpoint_magenta[2] > 250  # Blue channel
        assert gradient.value.shape == (100, 100, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
