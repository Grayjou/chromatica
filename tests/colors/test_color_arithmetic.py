from ...chromatica.colors.rgb import ColorUnitRGB
from ...chromatica.colors.hsv import UnitHSV
from ...chromatica.colors.hsl import UnitHSL
from ...chromatica.types.format_type import FormatType
from ...chromatica.colors.arithmetic import make_arithmetic, bounce
import numpy as np
from ..samples import samples_rgb_hsv, samples_rgb_hsl

def test_rgb_addition():
    rgb1 = ColorUnitRGB((0.2, 0.3, 0.4))
    rgb2 = ColorUnitRGB((0.5, 0.4, 0.3))
    result = rgb1 + rgb2
    assert isinstance(result._base, ColorUnitRGB)
    # tuples get transformed to numpy arrays internally
    assert np.allclose(result.value, (0.7, 0.7, 0.7))

def test_addition_clamps():
    rgb1 = ColorUnitRGB((0.8, 0.9, 1.0))
    rgb2 = ColorUnitRGB((0.5, 0.4, 0.3))
    result = rgb1 + rgb2
    assert isinstance(result._base, ColorUnitRGB)
    assert np.allclose(result.value, (1.0, 1.0, 1.0))  # Clamped to 1.0

    rgb3 = ColorUnitRGB((0.6, 0.7, 0.8))
    rgb4 = ColorUnitRGB((0.5, 0.6, 0.9))
    result = rgb3 - rgb4
    assert isinstance(result._base, ColorUnitRGB)
    assert np.allclose(result.value, (0.1, 0.1, 0.0))  # Clamped to 0.0

def test_addition_auto_converts():
    rgb = ColorUnitRGB((0.2, 0.3, 0.4))
    hsv = rgb.convert("hsv", FormatType.FLOAT)
    result = rgb + hsv  # Should auto-convert hsv to rgb
    assert isinstance(result._base, ColorUnitRGB)
    # We can compute expected value by converting hsv to rgb manually
    hsv_as_rgb = hsv.convert("rgb", FormatType.FLOAT)
    expected = np.clip(np.array(rgb.value) + np.array(hsv_as_rgb.value), 0.0, 1.0)
    assert np.allclose(result.value, expected)

def test_overflow_bounce():
    rgb1 = ColorUnitRGB((0.9, 0.8, 0.7))
    rgb2 = ColorUnitRGB((0.5, 0.4, 0.3))
    expected = np.array([0.6, 0.8, 1.0])  # Bounced values
    result = make_arithmetic(rgb1, overflow_function=bounce) + rgb2
    assert isinstance(result._base, ColorUnitRGB)
    # Bounced value
    expected = bounce(np.array(rgb1.value) + np.array(rgb2.value), 0.0, 1.0)
    assert np.allclose(result.value, expected)

def test_addition_with_samples_rgb_to_hsv():
    """Test that RGB + HSV converts HSV to RGB properly using sample data"""
    for rgb_tuple, hsv_tuple in samples_rgb_hsv.items():
        rgb = ColorUnitRGB(rgb_tuple)
        hsv = UnitHSV(hsv_tuple)
        
        # Add RGB + HSV (should convert HSV to RGB)
        result = rgb + hsv
        assert isinstance(result._base, ColorUnitRGB)
        
        # Manually convert hsv to rgb and add
        hsv_as_rgb = hsv.convert("rgb", FormatType.FLOAT)
        expected = np.clip(np.array(rgb.value) + np.array(hsv_as_rgb.value), 0.0, 1.0)
        assert np.allclose(result.value, expected, atol=1e-5)

def test_addition_with_samples_rgb_to_hsl():
    """Test that RGB + HSL converts HSL to RGB properly using sample data"""
    for rgb_tuple, hsl_tuple in samples_rgb_hsl.items():
        rgb = ColorUnitRGB(rgb_tuple)
        hsl = UnitHSL(hsl_tuple)
        
        # Add RGB + HSL (should convert HSL to RGB)
        result = rgb + hsl
        assert isinstance(result._base, ColorUnitRGB)
        
        # Manually convert hsl to rgb and add
        hsl_as_rgb = hsl.convert("rgb", FormatType.FLOAT)
        expected = np.clip(np.array(rgb.value) + np.array(hsl_as_rgb.value), 0.0, 1.0)
        assert np.allclose(result.value, expected, atol=1e-5)

def test_subtraction_with_samples():
    """Test that RGB - HSV converts properly using sample data"""
    for rgb_tuple, hsv_tuple in samples_rgb_hsv.items():
        rgb = ColorUnitRGB(rgb_tuple)
        hsv = UnitHSV(hsv_tuple)
        
        # Subtract RGB - HSV (should convert HSV to RGB)
        result = rgb - hsv
        assert isinstance(result._base, ColorUnitRGB)
        
        # Manually convert hsv to rgb and subtract
        hsv_as_rgb = hsv.convert("rgb", FormatType.FLOAT)
        expected = np.clip(np.array(rgb.value) - np.array(hsv_as_rgb.value), 0.0, 1.0)
        assert np.allclose(result.value, expected, atol=1e-5)

def test_multiplication():
    """Test multiplication with scalar values"""
    rgb = ColorUnitRGB((0.4, 0.6, 0.8))
    result = rgb * 0.5
    assert isinstance(result._base, ColorUnitRGB)
    assert np.allclose(result.value, (0.2, 0.3, 0.4))

def test_multiplication_clamps():
    """Test multiplication clamping"""
    rgb = ColorUnitRGB((0.6, 0.8, 1.0))
    result = rgb * 2.0
    assert isinstance(result._base, ColorUnitRGB)
    assert np.allclose(result.value, (1.0, 1.0, 1.0))  # Clamped to 1.0

def test_multiplication_with_samples():
    """Test multiplication maintains conversion accuracy"""
    for rgb_tuple, hsv_tuple in samples_rgb_hsv.items():
        rgb = ColorUnitRGB(rgb_tuple)
        hsv = UnitHSV(hsv_tuple)
        
        # Multiply RGB * 0.5, then add HSV
        rgb_scaled = rgb * 0.5
        result = rgb_scaled + hsv
        assert isinstance(result._base, ColorUnitRGB)
        
        # Verify conversion is accurate
        hsv_as_rgb = hsv.convert("rgb", FormatType.FLOAT)
        expected = np.clip(np.array(rgb.value) * 0.5 + np.array(hsv_as_rgb.value), 0.0, 1.0)
        assert np.allclose(result.value, expected, atol=1e-5)

def test_division():
    """Test division with scalar values"""
    rgb = ColorUnitRGB((0.4, 0.6, 0.8))
    result = rgb / 2.0
    assert isinstance(result._base, ColorUnitRGB)
    assert np.allclose(result.value, (0.2, 0.3, 0.4))

def test_division_clamps():
    """Test division clamping at minimum"""
    rgb = ColorUnitRGB((0.1, 0.05, 0.02))
    result = rgb / 10.0
    assert isinstance(result._base, ColorUnitRGB)
    assert np.allclose(result.value, (0.01, 0.005, 0.002))
    
    # Test that it doesn't go below 0
    rgb2 = ColorUnitRGB((0.0, 0.0, 0.0))
    result2 = rgb2 / 2.0
    assert np.allclose(result2.value, (0.0, 0.0, 0.0))

def test_division_with_samples():
    """Test division maintains conversion accuracy"""
    for rgb_tuple, hsl_tuple in samples_rgb_hsl.items():
        rgb = ColorUnitRGB(rgb_tuple)
        hsl = UnitHSL(hsl_tuple)
        
        # Divide RGB / 2.0, then add HSL
        rgb_scaled = rgb / 2.0
        result = rgb_scaled + hsl
        assert isinstance(result._base, ColorUnitRGB)
        
        # Verify conversion is accurate
        hsl_as_rgb = hsl.convert("rgb", FormatType.FLOAT)
        expected = np.clip(np.array(rgb.value) / 2.0 + np.array(hsl_as_rgb.value), 0.0, 1.0)
        assert np.allclose(result.value, expected, atol=1e-5)

def test_chained_arithmetic_with_conversion():
    """Test complex arithmetic chain with color space conversions"""
    for rgb_tuple, hsv_tuple in samples_rgb_hsv.items():
        rgb = ColorUnitRGB(rgb_tuple)
        hsv = UnitHSV(hsv_tuple)
        
        # Chain: (RGB * 0.5 + HSV) / 2.0
        result = (rgb * 0.5 + hsv) / 2.0
        assert isinstance(result._base, ColorUnitRGB)
        
        # Verify step by step
        hsv_as_rgb = hsv.convert("rgb", FormatType.FLOAT)
        step1 = np.array(rgb.value) * 0.5
        step2 = np.clip(step1 + np.array(hsv_as_rgb.value), 0.0, 1.0)
        expected = np.clip(step2 / 2.0, 0.0, 1.0)
        assert np.allclose(result.value, expected, atol=1e-5)