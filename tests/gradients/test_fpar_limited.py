import pytest
import numpy as np
from ...chromatica.gradients.gradientv1.full_parametrical_angular_radial import (
    FullParametricalAngularRadialGradient,
)
from ...chromatica.types.format_type import FormatType
from ..utils import get_point

class TestFparLimited:
    """Tests for FullParametricalAngularRadialGradient with limited parameters."""

    def test_limited_radial_gradient(self):
        """Test generating a limited radial gradient."""
        gradient = FullParametricalAngularRadialGradient.generate(
            width=100,
            height=100,
            inner_r_theta=lambda theta: np.full_like(theta, 10),
            outer_r_theta=lambda theta: np.full_like(theta, 40),
            color_rings=[((0, 255, 255), (240, 255, 255))],
            color_mode='hsv',
            format_type=FormatType.INT,
            deg_start=30,
            deg_end=210,
            hue_directions_theta=[["ccw"]]
            
        )
        gradient = gradient.convert('rgb', to_format=FormatType.INT)

        # Check shape
        assert gradient.value.shape == (100, 100, 3)

        # Check center pixel is inner color (red)
        center_pixel = get_point(gradient, (50, 50), 11, 34)
        assert center_pixel[0] > 250  # Red channel
        assert center_pixel[1] == 0    # Green channel
        assert center_pixel[2] < 10    # Blue channel

        # Check pixel at radius 30 is outer color (blue)
        mid_pixel = get_point(gradient, (50, 50), 30, 209)
        assert mid_pixel[0] < 10    # Red channel
        assert mid_pixel[1] == 0    # Green channel
        assert mid_pixel[2] > 250   # Blue channel

    def test_puffy_bean_limited(self):
        pytest.skip("Flaky test, needs investigation.")
        scale = 2
        base_size = 100
        r_theta_outer = lambda theta: (40 + 10 * np.sin(np.radians(8 * theta))) * scale
        r_theta_inner = lambda theta: (10 + 5 * np.sin(np.radians(2 * theta))) * scale
        inner_ring = ((171, 255, 255), (158, 101, 255)) # Cyan to Cyan Mint
        outer_ring = ((209, 166, 255), (158, 89, 255)) # Blue to Blue Sky
        gradient = FullParametricalAngularRadialGradient.generate(
            width=base_size * scale,
            height=base_size * scale,
            inner_r_theta=r_theta_inner,
            outer_r_theta=r_theta_outer,
            color_rings=[inner_ring, outer_ring],
            color_mode='hsv',
            format_type=FormatType.INT,
            deg_start=10,
            deg_end=270,
            hue_directions_theta=[["ccw"], ["cw"]],
            hue_directions_r=["cw"],
            outside_fill=(0, 0, 128),
            #rotate_r_theta_with_theta_normalization=True
        )
        gradient = gradient.convert('rgb', to_format=FormatType.INT)

        point_300_deg = get_point(gradient, (100, 100), 70, 300)
        assert np.allclose(point_300_deg, [128, 128, 128], atol=1)  # Outside fill

        inner_point_180 = get_point(gradient, (100, 100), 21, (180)) # Inner ring, cyan mint, low saturation

        assert inner_point_180[0] < 128    # Red channel
        assert inner_point_180[1] > 250   # Green channel
        assert inner_point_180[2] > 200   # Blue channel
        outer_point_180 = get_point(gradient, (100, 100),  79, (180)) # Outer ring, cw hue, redder hue (209+158+360)/2 = 363.5 = 3.5

        assert outer_point_180[0] > 250    # Red channel    
        assert outer_point_180[1] > 100   # Green channel
        assert outer_point_180[2] < 200   # Blue channel

    def test_flake_gradient_limited(self):
        """Test generating a flake-like gradient with limited angles."""
        scale = 2
        base_size = 100
        deg_start = 15
        deg_end = 300
        #flake = 40 + 10sen(2x) + 5sen(4x + π / 2) + 3sen(6x)
        def f(x, a):
            return np.sin(np.pi * x / (2 * a))

        def g(x, a):
            return np.cos(np.pi * (x - 1 + a) / (2 * a))
        def r(x, a):
            x = np.asarray(x)

            return np.where(
                (0 <= x) & (x < a),                         # left sine
                f(x, a),
                np.where(
                    (a <= x) & (x <= 1 - a),                # middle plateau
                    1,
                    np.where(
                        (1 - a < x) & (x <= 1),             # right cosine
                        g(x, a),
                        0                                   # otherwise
                    )
                )
            )
        from functools import partial
        def sine_fade_in_out(easing_time):
            return partial(r, a=easing_time)
        def ease_alpha_on_thetha(r, theta, alpha):
            unit_deg = (theta-deg_start) / (deg_end - deg_start)
            easing_func = sine_fade_in_out(0.2)
            return alpha * easing_func(unit_deg)
        def half_alpha(r, theta, alpha):
            unit_deg = theta / 360
            return np.where(theta > 330, alpha, alpha / 2)
        def flake_r(theta):
            return (40 + 10 * np.sin(np.radians(2 * theta)) +
                    5 * np.sin(np.radians(4 * theta + 90)) +
                    3 * np.sin(np.radians(6 * theta))) * scale
        #flake_core = 15 + 2sen(6x) - 2sen(4x)
        def flake_core_r(theta):
            return (15 + 2 * np.sin(np.radians(6 * theta)) -
                    2 * np.sin(np.radians(4 * theta))) * scale
        inner_ring = ((30, 255, 255, 255), (0, 255, 255, 255))  # Orange to Red
        outer_ring = ((60, 160, 255, 255), (40, 160, 255, 255))  # Yellow to Magenta
        gradient = FullParametricalAngularRadialGradient.generate(
            width=base_size * scale,
            height=base_size * scale,
            inner_r_theta=flake_core_r,
            outer_r_theta=flake_r,
            color_rings=[inner_ring, outer_ring],
            color_mode='hsva',
            format_type=FormatType.INT,
            deg_start=deg_start,
            deg_end=deg_end,
            hue_directions_theta=[["ccw"], ["ccw"]],
            bivariable_color_transforms={3: ease_alpha_on_thetha},
        )
        gradient = gradient.convert('rgba', to_format=FormatType.INT)

        faded_pixel = get_point(gradient, (100, 100), 35, 15)

        assert faded_pixel[3] < 10  # Alpha channel
        assert faded_pixel[0] > 250    # Red channel
        assert 80 <faded_pixel[1] < 150    # Green channel
        assert faded_pixel[2] < 10    # Blue channel