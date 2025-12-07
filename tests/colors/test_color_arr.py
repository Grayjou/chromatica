"""
Tests for ColorArr classes (Color1DArr and Color2DArr)
"""
import numpy as np
from ...chromatica.color_arr import Color1DArr, Color2DArr
from ...chromatica.colors.rgb import ColorUnitRGB
from ...chromatica.colors.hsv import UnitHSV
from ...chromatica.format_type import FormatType


def test_color1darr_creation():
    """Test creating a 1D color array"""
    # Create a simple gradient
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    rgb = ColorUnitRGB(colors)
    grad = Color1DArr(rgb)
    
    assert grad.is_array
    assert grad.shape == (3, 3)
    assert grad.num_channels == 3
    assert np.allclose(grad.value, colors)


def test_color1darr_conversion():
    """Test that Color1DArr inherits conversion from ColorBase"""
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    grad = Color1DArr(ColorUnitRGB(colors))
    
    # Convert to HSV
    hsv_grad = grad.convert("hsv", FormatType.FLOAT)
    
    assert isinstance(hsv_grad, UnitHSV)
    assert hsv_grad.is_array
    assert hsv_grad.shape == (3, 3)


def test_color1darr_arithmetic():
    """Test that Color1DArr wraps ColorBase which supports arithmetic"""
    colors = np.array([
        [0.5, 0.5, 0.5],
        [0.3, 0.3, 0.3]
    ], dtype=np.float32)
    
    # Import arithmetic to ensure operators are injected
    from ...chromatica.colors import arithmetic
    
    grad = Color1DArr(ColorUnitRGB(colors))
    
    # Test multiplication - ColorBase has arithmetic operators
    color_result = grad._color * 2.0
    # Result is ArithmeticProxy, access the underlying base
    assert np.allclose(color_result._base.value, [[1.0, 1.0, 1.0], [0.6, 0.6, 0.6]])
    
    # Can wrap result back into Color1DArr
    result_arr = Color1DArr(color_result._base)
    assert result_arr.shape == (2, 3)


def test_color1darr_repeat():
    """Test repeat method creates Color2DArr"""
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    grad = Color1DArr(ColorUnitRGB(colors))
    
    # Repeat to create 2D array
    result = grad.repeat(horizontally=2.0, vertically=3)
    
    assert isinstance(result, Color2DArr)
    assert result.value.ndim == 3
    assert result.value.shape[0] == 3  # vertical
    # horizontally: 3 colors * 2.0 = 6 colors
    assert result.value.shape[1] == 6


def test_color1darr_wrap_around():
    """Test wrap_around creates angular gradient"""
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    grad = Color1DArr(ColorUnitRGB(colors))
    
    # Create angular gradient
    result = grad.wrap_around(width=10, height=10, center=(5, 5))
    
    assert isinstance(result, Color2DArr)
    assert result.value.shape == (10, 10, 3)


def test_color1darr_rotate_around():
    """Test rotate_around creates radial gradient"""
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    grad = Color1DArr(ColorUnitRGB(colors))
    
    # Create radial gradient
    result = grad.rotate_around(width=10, height=10, center=(5, 5))
    
    assert isinstance(result, Color2DArr)
    assert result.value.shape == (10, 10, 3)


def test_color2darr_creation():
    """Test creating a 2D color array"""
    # Create a simple 2D array
    colors = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ], dtype=np.float32)
    
    arr = Color2DArr(ColorUnitRGB(colors))
    
    assert arr.is_array
    assert arr.shape == (2, 2, 3)
    assert arr.num_channels == 3


def test_color2darr_repeat():
    """Test 2D array repeat"""
    colors = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ], dtype=np.float32)
    
    arr = Color2DArr(ColorUnitRGB(colors))
    
    # Repeat 2x3
    result = arr.repeat(horizontally=2, vertically=3)
    
    assert isinstance(result, Color2DArr)
    assert result.value.shape == (6, 4, 3)  # 2*3 height, 2*2 width


def test_color2darr_conversion():
    """Test Color2DArr inherits conversion"""
    colors = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ], dtype=np.float32)
    
    arr = Color2DArr(ColorUnitRGB(colors))
    
    # Convert to HSV
    hsv_arr = arr.convert("hsv", FormatType.FLOAT)
    
    assert isinstance(hsv_arr, UnitHSV)
    assert hsv_arr.shape == (2, 2, 3)


def test_channels_property():
    """Test that channels property works for both 1D and 2D"""
    colors1d = np.array([
        [1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0]
    ], dtype=np.float32)
    
    grad = Color1DArr(ColorUnitRGB(colors1d))
    channels = grad.channels
    
    assert len(channels) == 3
    assert np.allclose(channels[0], [1.0, 0.0])
    assert np.allclose(channels[1], [0.5, 0.5])
    assert np.allclose(channels[2], [0.0, 1.0])
