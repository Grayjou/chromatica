import numpy as np
from ...chromatica.colors.rgb import ColorRGBINT, ColorRGBAINT, ColorUnitRGB, ColorPercentageRGBA, ColorUnitRGBA, ColorPercentageRGB
from ...chromatica.colors.hsv import ColorHSVAINT, ColorHSVINT, UnitHSV, PercentageHSV, PercentageHSVA, UnitHSVA
from ...chromatica.colors.hsl import ColorHSLINT, ColorHSLAINT, UnitHSL, UnitHSLA, PercentageHSL, PercentageHSLA
from ...chromatica.conversions.format_type import FormatType
from ..samples import samples_rgb_hsv, samples_rgb_hsl


def test_class_conversion_rgb_to_hsv_array():
    """Test RGB to HSV conversion with a single array containing all test samples."""
    # Build arrays from samples
    rgb_list = list(samples_rgb_hsv.keys())
    hsv_expected_list = list(samples_rgb_hsv.values())
    
    rgb_array = np.array(rgb_list, dtype=np.float32)
    hsv_expected_array = np.array(hsv_expected_list, dtype=np.float32)
    
    # Create single ColorUnitRGB instance with array
    rgb = ColorUnitRGB(rgb_array)
    assert rgb.is_array
    assert rgb.shape == (len(rgb_list), 3)
    
    # Convert to INT
    hsv = rgb.convert("hsv", FormatType.INT)
    assert isinstance(hsv, ColorHSVINT)
    assert hsv.is_array
    expected_int = np.round(hsv_expected_array * [1, 255, 255]).astype(np.uint16)
    np.testing.assert_array_equal(hsv.value, expected_int)
    
    # Convert to FLOAT
    hsv = rgb.convert("hsv", FormatType.FLOAT)
    assert isinstance(hsv, UnitHSV)
    assert hsv.is_array
    np.testing.assert_allclose(hsv.value[:, 0], hsv_expected_array[:, 0], atol=0.5)
    np.testing.assert_allclose(hsv.value[:, 1], hsv_expected_array[:, 1], atol=1/255)
    np.testing.assert_allclose(hsv.value[:, 2], hsv_expected_array[:, 2], atol=1/255)
    
    # Convert to PERCENTAGE
    hsv = rgb.convert("hsv", FormatType.PERCENTAGE)
    assert isinstance(hsv, PercentageHSV)
    assert hsv.is_array
    expected_pct = hsv_expected_array * [1, 100, 100]
    np.testing.assert_allclose(hsv.value[:, 0], expected_pct[:, 0], atol=0.5)
    np.testing.assert_allclose(hsv.value[:, 1], expected_pct[:, 1], atol=100/255)
    np.testing.assert_allclose(hsv.value[:, 2], expected_pct[:, 2], atol=100/255)


def test_class_conversion_rgb_to_hsl_array():
    """Test RGB to HSL conversion with a single array containing all test samples."""
    rgb_list = list(samples_rgb_hsl.keys())
    hsl_expected_list = list(samples_rgb_hsl.values())
    
    rgb_array = np.array(rgb_list, dtype=np.float32)
    hsl_expected_array = np.array(hsl_expected_list, dtype=np.float32)
    
    # Create single ColorUnitRGB instance
    rgb = ColorUnitRGB(rgb_array)
    rgb = rgb.convert("rgb", FormatType.INT)
    
    # Convert with CSS algorithm to INT
    hsl = rgb.convert("hsl", FormatType.FLOAT, use_css_algo=True)
    hsl = hsl.convert("hsl", FormatType.INT)
    assert isinstance(hsl, ColorHSLINT)
    assert hsl.is_array
    expected_int = np.round(hsl_expected_array * [1, 255, 255]).astype(np.uint16)
    np.testing.assert_array_equal(hsl.value, expected_int)
    
    # Convert to FLOAT
    hsl = rgb.convert("hsl", FormatType.FLOAT)
    assert isinstance(hsl, UnitHSL)
    assert hsl.is_array
    np.testing.assert_allclose(hsl.value[:, 0], hsl_expected_array[:, 0], atol=0.5)
    np.testing.assert_allclose(hsl.value[:, 1], hsl_expected_array[:, 1], atol=1/255)
    np.testing.assert_allclose(hsl.value[:, 2], hsl_expected_array[:, 2], atol=1/255)
    
    # Convert to PERCENTAGE
    hsl = rgb.convert("hsl", FormatType.PERCENTAGE)
    assert isinstance(hsl, PercentageHSL)
    assert hsl.is_array
    expected_pct = hsl_expected_array * [1, 100, 100]
    np.testing.assert_allclose(hsl.value[:, 0], expected_pct[:, 0], atol=0.5)
    np.testing.assert_allclose(hsl.value[:, 1], expected_pct[:, 1], atol=100/255)
    np.testing.assert_allclose(hsl.value[:, 2], expected_pct[:, 2], atol=100/255)


def test_class_conversion_rgba_to_hsva_array():
    """Test RGBA to HSVA conversion with alpha channel."""
    rgb_list = list(samples_rgb_hsv.keys())
    hsv_expected_list = list(samples_rgb_hsv.values())
    
    # Add alpha channel
    rgba_array = np.column_stack([np.array(rgb_list, dtype=np.float32), 
                                   np.full(len(rgb_list), 0.5)])
    hsv_expected_array = np.array(hsv_expected_list, dtype=np.float32)
    
    rgba = ColorUnitRGBA(rgba_array)
    assert rgba.is_array
    np.testing.assert_allclose(rgba.alpha, 0.5)
    
    # Convert to INT
    hsva = rgba.convert("hsva", FormatType.INT)
    assert isinstance(hsva, ColorHSVAINT)
    assert hsva.is_array
    expected_int = np.column_stack([
        np.round(hsv_expected_array * [1, 255, 255]),
        np.full(len(rgb_list), 128)
    ]).astype(np.uint16)
    np.testing.assert_array_equal(hsva.value, expected_int)
    
    # Convert to FLOAT
    hsva = rgba.convert("hsva", FormatType.FLOAT)
    assert isinstance(hsva, UnitHSVA)
    assert hsva.is_array
    np.testing.assert_allclose(hsva.value[:, 0], hsv_expected_array[:, 0], atol=0.5)
    np.testing.assert_allclose(hsva.value[:, 1], hsv_expected_array[:, 1], atol=1/255)
    np.testing.assert_allclose(hsva.value[:, 2], hsv_expected_array[:, 2], atol=1/255)
    np.testing.assert_allclose(hsva.value[:, 3], 0.5, atol=1/255)
    
    # Convert to PERCENTAGE
    hsva = rgba.convert("hsva", FormatType.PERCENTAGE)
    assert isinstance(hsva, PercentageHSVA)
    assert hsva.is_array
    expected_pct = hsv_expected_array * [1, 100, 100]
    np.testing.assert_allclose(hsva.value[:, 0], expected_pct[:, 0], atol=0.5)
    np.testing.assert_allclose(hsva.value[:, 1], expected_pct[:, 1], atol=100/255)
    np.testing.assert_allclose(hsva.value[:, 2], expected_pct[:, 2], atol=100/255)
    np.testing.assert_allclose(hsva.value[:, 3], 50.0, atol=100/255)


def test_class_conversion_rgba_to_hsla_array():
    """Test RGBA to HSLA conversion with alpha channel."""
    rgb_list = list(samples_rgb_hsl.keys())
    hsl_expected_list = list(samples_rgb_hsl.values())
    
    rgba_array = np.column_stack([np.array(rgb_list, dtype=np.float32),
                                   np.full(len(rgb_list), 0.5)])
    hsl_expected_array = np.array(hsl_expected_list, dtype=np.float32)
    
    rgba = ColorUnitRGBA(rgba_array)
    
    # Convert with CSS algorithm to INT
    hsla = rgba.convert("hsla", FormatType.FLOAT, use_css_algo=True)
    hsla = hsla.convert("hsla", FormatType.INT)
    assert isinstance(hsla, ColorHSLAINT)
    assert hsla.is_array
    expected_int = np.column_stack([
        np.round(hsl_expected_array * [1, 255, 255]),
        np.full(len(rgb_list), 128)
    ]).astype(np.uint16)
    np.testing.assert_array_equal(hsla.value, expected_int)
    
    # Convert to FLOAT
    hsla = rgba.convert("hsla", FormatType.FLOAT)
    assert isinstance(hsla, UnitHSLA)
    assert hsla.is_array
    np.testing.assert_allclose(hsla.value[:, 0], hsl_expected_array[:, 0], atol=0.5)
    np.testing.assert_allclose(hsla.value[:, 1], hsl_expected_array[:, 1], atol=1/255)
    np.testing.assert_allclose(hsla.value[:, 2], hsl_expected_array[:, 2], atol=1/255)
    np.testing.assert_allclose(hsla.value[:, 3], 0.5, atol=1/255)
    
    # Convert to PERCENTAGE
    hsla = rgba.convert("hsla", FormatType.PERCENTAGE)
    assert isinstance(hsla, PercentageHSLA)
    assert hsla.is_array
    expected_pct = hsl_expected_array * [1, 100, 100]
    np.testing.assert_allclose(hsla.value[:, 0], expected_pct[:, 0], atol=0.5)
    np.testing.assert_allclose(hsla.value[:, 1], expected_pct[:, 1], atol=100/255)
    np.testing.assert_allclose(hsla.value[:, 2], expected_pct[:, 2], atol=100/255)
    np.testing.assert_allclose(hsla.value[:, 3], 50.0, atol=100/255)


def test_class_conversion_hsv_to_rgb_array():
    """Test HSV to RGB conversion with array."""
    rgb_expected_list = list(samples_rgb_hsv.keys())
    hsv_list = list(samples_rgb_hsv.values())
    
    rgb_expected_array = np.array(rgb_expected_list, dtype=np.float32)
    hsv_array = np.round(np.array(hsv_list) * [1, 255, 255]).astype(np.uint16)
    
    hsv = ColorHSVINT(hsv_array)
    assert hsv.is_array
    
    # Convert to INT
    rgb = hsv.convert("rgb", FormatType.INT)
    assert isinstance(rgb, ColorRGBINT)
    assert rgb.is_array
    expected_int = np.round(rgb_expected_array * 255).astype(np.uint16)
    np.testing.assert_array_equal(rgb.value, expected_int)
    
    # Convert to FLOAT
    rgb = hsv.convert("rgb", FormatType.FLOAT)
    assert isinstance(rgb, ColorUnitRGB)
    assert rgb.is_array
    np.testing.assert_allclose(rgb.value, rgb_expected_array, atol=1/255)
    
    # Convert to PERCENTAGE
    rgb = hsv.convert("rgb", FormatType.PERCENTAGE)
    assert isinstance(rgb, ColorPercentageRGB)
    assert rgb.is_array
    expected_pct = rgb_expected_array * 100
    np.testing.assert_allclose(rgb.value, expected_pct, atol=100/255)
