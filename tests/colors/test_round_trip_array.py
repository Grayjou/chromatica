import numpy as np
from ...chromatica.colors.rgb import ColorRGBINT, ColorRGBAINT, ColorUnitRGB, ColorPercentageRGBA, ColorUnitRGBA, ColorPercentageRGB
from ...chromatica.colors.hsv import ColorHSVAINT, ColorHSVINT, UnitHSV, PercentageHSV, PercentageHSVA, UnitHSVA
from ...chromatica.colors.hsl import ColorHSLINT, ColorHSLAINT, UnitHSL, UnitHSLA, PercentageHSL, PercentageHSLA
from ...chromatica.format_type import FormatType
from ..samples import samples_rgb_hsv, samples_rgb_hsl


def test_round_trip_rgb_to_hsv_to_rgb_array():
    """Test RGB → HSV → RGB round-trip with array."""
    rgb_list = list(samples_rgb_hsv.keys())
    rgb_array = np.array(rgb_list, dtype=np.float32)
    
    rgb = ColorUnitRGB(rgb_array)
    hsv = rgb.convert("hsv", FormatType.FLOAT)
    rgb_out = hsv.convert("rgb", FormatType.FLOAT)
    
    assert rgb_out.is_array
    np.testing.assert_allclose(rgb.value, rgb_out.value, atol=1/255)


def test_round_trip_rgb_to_hsl_to_rgb_array():
    """Test RGB → HSL → RGB round-trip with array."""
    rgb_list = list(samples_rgb_hsl.keys())
    rgb_array = np.array(rgb_list, dtype=np.float32)
    
    rgb = ColorUnitRGB(rgb_array)
    hsl = rgb.convert("hsl", FormatType.FLOAT, use_css_algo=True)
    rgb_out = hsl.convert("rgb", FormatType.FLOAT)
    
    assert rgb_out.is_array
    np.testing.assert_allclose(rgb.value, rgb_out.value, atol=1/255)


def test_round_trip_hsv_to_rgb_to_hsv_array():
    """Test HSV → RGB → HSV round-trip with array."""
    hsv_expected_list = list(samples_rgb_hsv.values())
    hsv_expected_array = np.array(hsv_expected_list, dtype=np.float32)
    hsv_int_array = np.round(hsv_expected_array * [1, 255, 255]).astype(np.uint16)
    
    hsv = ColorHSVINT(hsv_int_array)
    rgb_out = hsv.convert("rgb", FormatType.FLOAT)
    hsv_final = rgb_out.convert("hsv", FormatType.FLOAT)
    
    assert hsv_final.is_array
    np.testing.assert_allclose(hsv_final.value[:, 0], hsv_expected_array[:, 0], atol=0.5)
    np.testing.assert_allclose(hsv_final.value[:, 1], hsv_expected_array[:, 1], atol=1/255)
    np.testing.assert_allclose(hsv_final.value[:, 2], hsv_expected_array[:, 2], atol=1/255)


def test_round_trip_hsl_to_rgb_to_hsl_array():
    """Test HSL → RGB → HSL round-trip with array."""
    hsl_expected_list = list(samples_rgb_hsl.values())
    hsl_expected_array = np.array(hsl_expected_list, dtype=np.float32)
    hsl_int_array = np.round(hsl_expected_array * [1, 255, 255]).astype(np.uint16)
    
    hsl = ColorHSLINT(hsl_int_array)
    rgb_out = hsl.convert("rgb", FormatType.FLOAT)
    hsl_final = rgb_out.convert("hsl", FormatType.FLOAT, use_css_algo=True)
    
    assert hsl_final.is_array
    np.testing.assert_allclose(hsl_final.value[:, 0], hsl_expected_array[:, 0], atol=0.5)
    np.testing.assert_allclose(hsl_final.value[:, 1], hsl_expected_array[:, 1], atol=1/255)
    np.testing.assert_allclose(hsl_final.value[:, 2], hsl_expected_array[:, 2], atol=1/255)
