import numpy as np
from ...chromatica.colors.rgb import ColorRGBINT, ColorRGBAINT, ColorUnitRGB, ColorUnitRGBA
from ...chromatica.colors.hsv import ColorHSVAINT, ColorHSVINT, UnitHSV, UnitHSVA
from ...chromatica.colors.hsl import ColorHSLINT, ColorHSLAINT, UnitHSL, UnitHSLA
from ...chromatica.types.format_type import FormatType


def test_rgb_to_rgba_with_alpha():
    """Test adding alpha channel to RGB color."""
    rgb = ColorUnitRGB((0.5, 0.25, 0.75))
    rgba = rgb.with_alpha(0.8)
    assert isinstance(rgba, ColorUnitRGBA)
    assert rgba.value == (0.5, 0.25, 0.75, 0.8)
    
    # Test with default alpha (should use max value)
    other_rgba = rgb.with_alpha()
    assert isinstance(other_rgba, ColorUnitRGBA)
    assert other_rgba.value == (0.5, 0.25, 0.75, 1.0)


def test_rgba_to_rgba_with_alpha_returns_same():
    """Test that adding alpha to RGBA returns the same instance."""
    rgba = ColorUnitRGBA((0.5, 0.25, 0.75, 0.8))
    result = rgba.with_alpha(0.9)
    assert result is rgba
    assert result.value == (0.5, 0.25, 0.75, 0.8)  # Alpha unchanged


def test_hsv_to_hsva_with_alpha():
    """Test adding alpha channel to HSV color."""
    hsv = ColorHSVINT((180, 128, 200))
    hsva = hsv.with_alpha(255)
    assert isinstance(hsva, ColorHSVAINT)
    assert hsva.value == (180, 128, 200, 255)
    
    # Test with default alpha
    other_hsva = hsv.with_alpha()
    assert isinstance(other_hsva, ColorHSVAINT)
    assert other_hsva.value == (180, 128, 200, 255)


def test_hsl_to_hsla_with_alpha():
    """Test adding alpha channel to HSL color."""
    hsl = UnitHSL((120.0, 0.5, 0.6))
    hsla = hsl.with_alpha(0.75)
    assert isinstance(hsla, UnitHSLA)
    assert hsla.value == (120.0, 0.5, 0.6, 0.75)


def test_with_alpha_array():
    """Test adding alpha to array of colors."""
    rgb_array = np.array([
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.5],
        [0.5, 0.0, 1.0]
    ], dtype=np.float32)
    
    rgb = ColorUnitRGB(rgb_array)
    assert rgb.is_array
    
    rgba = rgb.with_alpha(0.8)
    assert isinstance(rgba, ColorUnitRGBA)
    assert rgba.is_array
    assert rgba.shape == (3, 4)
    np.testing.assert_allclose(rgba.alpha, 0.8)


def test_with_alpha_array_custom_alphas():
    """Test adding different alpha values to array elements."""
    rgb_array = np.array([
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.5]
    ], dtype=np.float32)
    
    rgb = ColorUnitRGB(rgb_array)
    alpha_array = np.array([0.3, 0.9], dtype=np.float32)
    
    rgba = rgb.with_alpha(alpha_array)
    assert isinstance(rgba, ColorUnitRGBA)
    assert rgba.is_array
    np.testing.assert_allclose(rgba.alpha, alpha_array)


def test_convert_to_rgba_adds_default_alpha():
    """Test that converting RGB to RGBA adds default alpha."""
    rgb = ColorUnitRGB((0.5, 0.25, 0.75))
    rgba = rgb.convert("rgba", FormatType.FLOAT)
    assert isinstance(rgba, ColorUnitRGBA)
    assert rgba.value == (0.5, 0.25, 0.75, 1.0)


def test_alpha_preserved_across_color_mode_conversion():
    """Test that alpha is preserved when converting between color spaces."""
    rgba = ColorUnitRGBA((0.5, 0.25, 0.75, 0.6))
    
    # Convert to HSVA
    hsva = rgba.convert("hsva", FormatType.FLOAT)
    assert isinstance(hsva, UnitHSVA)
    assert hsva.has_alpha
    assert abs(float(hsva.alpha) - 0.6) < 0.01
    
    # Convert to HSLA
    hsla = rgba.convert("hsla", FormatType.FLOAT)
    assert isinstance(hsla, UnitHSLA)
    assert hsla.has_alpha
    assert abs(float(hsla.alpha) - 0.6) < 0.01
    
    # Convert back to RGBA
    rgba_back = hsva.convert("rgba", FormatType.FLOAT)
    assert isinstance(rgba_back, ColorUnitRGBA)
    assert abs(float(rgba_back.alpha) - 0.6) < 0.01


def test_alpha_preserved_across_format_conversion():
    """Test that alpha is preserved when converting between formats."""
    rgba_int = ColorRGBAINT((255, 128, 64, 200))
    
    # Convert to float
    rgba_float = rgba_int.convert("rgba", FormatType.FLOAT)
    assert isinstance(rgba_float, ColorUnitRGBA)
    expected_alpha = 200 / 255
    assert abs(float(rgba_float.alpha) - expected_alpha) < 0.01
    
    # Convert back to int
    rgba_int_back = rgba_float.convert("rgba", FormatType.INT)
    assert isinstance(rgba_int_back, ColorRGBAINT)
    assert rgba_int_back.value[3] == 200


def test_has_alpha_property():
    """Test has_alpha property on various color types."""
    rgb = ColorUnitRGB((0.5, 0.25, 0.75))
    rgba = ColorUnitRGBA((0.5, 0.25, 0.75, 0.8))
    hsv = UnitHSV((180.0, 0.5, 0.8))
    hsva = UnitHSVA((180.0, 0.5, 0.8, 0.9))
    hsl = UnitHSL((180.0, 0.5, 0.6))
    hsla = UnitHSLA((180.0, 0.5, 0.6, 0.7))
    
    assert not rgb.has_alpha
    assert rgba.has_alpha
    assert not hsv.has_alpha
    assert hsva.has_alpha
    assert not hsl.has_alpha
    assert hsla.has_alpha


def test_with_alpha_different_formats():
    """Test with_alpha across different format types."""
    # INT format
    rgb_int = ColorRGBINT((255, 128, 64))
    rgba_int = rgb_int.with_alpha(200)
    assert isinstance(rgba_int, ColorRGBAINT)
    assert rgba_int.value == (255, 128, 64, 200)
    
    # FLOAT format (already tested above, but for completeness)
    rgb_float = ColorUnitRGB((1.0, 0.5, 0.25))
    rgba_float = rgb_float.with_alpha(0.8)
    assert isinstance(rgba_float, ColorUnitRGBA)
    assert rgba_float.value == (1.0, 0.5, 0.25, 0.8)

