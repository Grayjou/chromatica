import pytest
import numpy as np
from ..chromatica.gradient import Gradient1D, Gradient2D, radial_gradient
from ..chromatica.colors.rgb import ColorUnitRGB, ColorUnitRGBA
from ..chromatica.format_type import FormatType


def test_gradient1d_from_colors_rgb():
    """Test creating a 1D gradient in RGB space."""
    grad = Gradient1D.from_colors(
        color1=(255, 0, 0),
        color2=(0, 0, 255),
        steps=10,
        color_space='rgb',
        format_type=FormatType.INT
    )
    
    # Check shape
    assert grad._color.value.shape == (10, 3)
    
    # Check endpoints
    assert np.allclose(grad._color.value[0], [255, 0, 0])
    assert np.allclose(grad._color.value[-1], [0, 0, 255])


def test_gradient1d_hsv_hue_direction():
    """Test hue direction control in HSV gradients."""
    # Test clockwise
    grad_cw = Gradient1D.from_colors(
        color1=(0.0, 1.0, 1.0),  # Red in HSV
        color2=(240.0, 1.0, 1.0),  # Blue in HSV
        steps=5,
        color_space='hsv',
        format_type=FormatType.FLOAT,
        direction='cw'
    )
    
    # Hue should increase from 0 to 240 going clockwise (0 -> 60 -> 120 -> 180 -> 240)
    hues = grad_cw._color.value[:, 0]
    assert hues[0] == 0.0
    assert hues[-1] == 240.0
    assert all(hues[i] < hues[i+1] for i in range(len(hues)-1))
    
    # Test counter-clockwise
    grad_ccw = Gradient1D.from_colors(
        color1=(240.0, 1.0, 1.0),  # Blue in HSV
        color2=(0.0, 1.0, 1.0),  # Red in HSV
        steps=5,
        color_space='hsv',
        format_type=FormatType.FLOAT,
        direction='ccw'
    )
    
    # Hue should decrease going ccw
    hues_ccw = grad_ccw._color.value[:, 0]
    assert hues_ccw[0] == 240.0
    assert hues_ccw[-1] == 0.0


def test_gradient2d_from_colors():
    """Test creating a 2D gradient from corner colors."""
    grad = Gradient2D.from_colors(
        color_tl=(255, 0, 0),
        color_tr=(0, 255, 0),
        color_bl=(0, 0, 255),
        color_br=(255, 255, 0),
        width=10,
        height=10,
        color_space='rgb',
        format_type=FormatType.INT
    )
    
    # Check shape
    assert grad._color.value.shape == (10, 10, 3)
    
    # Check corners (approximately, due to interpolation)
    assert np.allclose(grad._color.value[0, 0], [255, 0, 0], atol=2)  # Top-left
    assert np.allclose(grad._color.value[0, -1], [0, 255, 0], atol=2)  # Top-right
    assert np.allclose(grad._color.value[-1, 0], [0, 0, 255], atol=2)  # Bottom-left
    assert np.allclose(grad._color.value[-1, -1], [255, 255, 0], atol=2)  # Bottom-right


def test_radial_gradient_basic():
    """Test basic radial gradient creation."""
    result = radial_gradient(
        color1=(255, 0, 0),
        color2=(0, 0, 255),
        height=50,
        width=50,
        center=(25, 25),
        radius=25.0,
        color_space='rgb',
        format_type=FormatType.INT
    )
    
    # Check shape
    assert result.shape == (50, 50, 3)
    
    # Center should be close to color1 (red)
    assert np.allclose(result[25, 25], [255, 0, 0], atol=5)
    
    # Edge should be closer to color2 (blue)
    # But might not be exact due to interpolation
    edge_color = result[0, 25]
    assert edge_color[2] > edge_color[0]  # More blue than red


def test_gradient1d_with_colorbase():
    """Test that Gradient1D accepts ColorBase instances."""
    color1 = ColorUnitRGB((1.0, 0.0, 0.0))
    color2 = ColorUnitRGB((0.0, 0.0, 1.0))
    
    grad = Gradient1D.from_colors(
        color1=color1,
        color2=color2,
        steps=10,
        color_space='rgb',
        format_type=FormatType.FLOAT
    )
    
    assert grad._color.value.shape == (10, 3)
    assert np.allclose(grad._color.value[0], [1.0, 0.0, 0.0])
    assert np.allclose(grad._color.value[-1], [0.0, 0.0, 1.0])


def test_gradient1d_rgba_channels():
    """Test that RGBA gradients have 4 channels."""
    grad = Gradient1D.from_colors(
        color1=(255, 0, 0, 255),
        color2=(0, 0, 255, 128),
        steps=10,
        color_space='rgba',
        format_type=FormatType.INT
    )
    
    # Check shape has 4 channels
    assert grad._color.value.shape == (10, 4)
    assert grad._color.num_channels == 4
    assert grad._color.has_alpha is True
    
    # Check endpoints
    assert np.allclose(grad._color.value[0], [255, 0, 0, 255])
    assert np.allclose(grad._color.value[-1], [0, 0, 255, 128])
    
    # Check alpha interpolation
    alphas = grad._color.value[:, 3]
    assert alphas[0] == 255
    assert alphas[-1] == 128
    assert all(alphas[i] >= alphas[i+1] for i in range(len(alphas)-1))


def test_gradient1d_hsva_with_hue_direction():
    """Test HSVA gradient with hue direction and alpha interpolation."""
    grad = Gradient1D.from_colors(
        color1=(0.0, 1.0, 1.0, 1.0),  # Red with full opacity
        color2=(120.0, 1.0, 1.0, 0.0),  # Green with full transparency
        steps=6,
        color_space='hsva',
        format_type=FormatType.FLOAT,
        direction='cw'
    )
    
    # Check shape has 4 channels
    assert grad._color.value.shape == (6, 4)
    assert grad._color.has_alpha is True
    
    # Check hue interpolation (clockwise)
    hues = grad._color.value[:, 0]
    assert hues[0] == 0.0
    assert hues[-1] == 120.0
    assert all(hues[i] < hues[i+1] for i in range(len(hues)-1))
    
    # Check alpha interpolation
    alphas = grad._color.value[:, 3]
    assert alphas[0] == 1.0
    assert alphas[-1] == 0.0


def test_gradient2d_rgba_channels():
    """Test 2D gradient with alpha channels."""
    grad = Gradient2D.from_colors(
        color_tl=(255, 0, 0, 255),
        color_tr=(0, 255, 0, 200),
        color_bl=(0, 0, 255, 150),
        color_br=(255, 255, 0, 100),
        width=10,
        height=10,
        color_space='rgba',
        format_type=FormatType.INT
    )
    
    # Check shape has 4 channels
    assert grad._color.value.shape == (10, 10, 4)
    assert grad._color.num_channels == 4
    assert grad._color.has_alpha is True
    
    # Check corner alphas (approximately)
    assert grad._color.value[0, 0, 3] >= 250  # Top-left: ~255
    assert grad._color.value[0, -1, 3] >= 195  # Top-right: ~200
    assert grad._color.value[-1, 0, 3] >= 145  # Bottom-left: ~150
    assert grad._color.value[-1, -1, 3] >= 95  # Bottom-right: ~100


def test_radial_gradient_rgba():
    """Test radial gradient with alpha channel."""
    result = radial_gradient(
        color1=(255, 0, 0, 255),  # Opaque red
        color2=(0, 0, 255, 0),  # Transparent blue
        height=50,
        width=50,
        center=(25, 25),
        radius=25.0,
        color_space='rgba',
        format_type=FormatType.INT
    )
    
    # Check shape has 4 channels
    assert result.shape == (50, 50, 4)
    
    # Center should be opaque red
    assert np.allclose(result[25, 25, :3], [255, 0, 0], atol=5)
    assert result[25, 25, 3] >= 250  # Nearly full alpha
    
    # Edge should be more transparent
    edge_alpha = result[0, 25, 3]
    center_alpha = result[25, 25, 3]
    assert edge_alpha < center_alpha


def test_gradient1d_colorbase_rgba():
    """Test gradient with RGBA ColorBase instances."""
    color1 = ColorUnitRGBA((1.0, 0.0, 0.0, 1.0))
    color2 = ColorUnitRGBA((0.0, 0.0, 1.0, 0.5))
    
    grad = Gradient1D.from_colors(
        color1=color1,
        color2=color2,
        steps=10,
        color_space='rgba',
        format_type=FormatType.FLOAT
    )
    
    # Check shape has 4 channels
    assert grad._color.value.shape == (10, 4)
    assert grad._color.has_alpha is True
    
    # Check endpoints
    assert np.allclose(grad._color.value[0], [1.0, 0.0, 0.0, 1.0])
    assert np.allclose(grad._color.value[-1], [0.0, 0.0, 1.0, 0.5])


def test_gradient_channel_count_validation():
    """Test that channel counts are correct for different color spaces."""
    # RGB - 3 channels
    grad_rgb = Gradient1D.from_colors(
        (255, 0, 0), (0, 0, 255), steps=5,
        color_space='rgb', format_type=FormatType.INT
    )
    assert grad_rgb._color.num_channels == 3
    assert grad_rgb._color.has_alpha is False
    
    # RGBA - 4 channels
    grad_rgba = Gradient1D.from_colors(
        (255, 0, 0, 255), (0, 0, 255, 255), steps=5,
        color_space='rgba', format_type=FormatType.INT
    )
    assert grad_rgba._color.num_channels == 4
    assert grad_rgba._color.has_alpha is True
    
    # HSV - 3 channels
    grad_hsv = Gradient1D.from_colors(
        (0, 100, 100), (240, 100, 100), steps=5,
        color_space='hsv', format_type=FormatType.INT
    )
    assert grad_hsv._color.num_channels == 3
    assert grad_hsv._color.has_alpha is False
    
    # HSVA - 4 channels
    grad_hsva = Gradient1D.from_colors(
        (0, 100, 100, 255), (240, 100, 100, 255), steps=5,
        color_space='hsva', format_type=FormatType.INT
    )
    assert grad_hsva._color.num_channels == 4
    assert grad_hsva._color.has_alpha is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
