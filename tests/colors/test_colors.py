from ...chromatica.colors.rgb import ColorRGBINT, ColorRGBAINT, ColorUnitRGB, ColorPercentageRGBA, ColorUnitRGBA, ColorPercentageRGB
from ...chromatica.colors.hsv import ColorHSVAINT, ColorHSVINT, UnitHSV, PercentageHSV, PercentageHSVA, UnitHSVA
from ...chromatica.colors.hsl import ColorHSLINT, ColorHSLAINT, UnitHSL, UnitHSLA, PercentageHSL, PercentageHSLA
from ...chromatica.conversions.format_type import FormatType
from ..samples import samples_rgb_hsv, samples_rgb_hsl
import math
def test_class_conversion_rgb_to_hsv():
    for rgb, hsv_expected in samples_rgb_hsv.items():
        r, g, b = rgb
        h_exp, s_exp, v_exp = hsv_expected

        rgb = ColorUnitRGB((r, g, b))
        hsv = rgb.convert("hsv", FormatType.INT)
        assert hsv.value == (round(h_exp), round(s_exp*255), round(v_exp*255))
        assert isinstance(hsv, ColorHSVINT)

        hsv = rgb.convert("hsv", FormatType.FLOAT)
        h, s, v = hsv.value
        assert abs(h - h_exp) < .5
        assert abs(float(s) - s_exp) < 1/255
        assert abs(float(v) - v_exp) < 1/255
        assert isinstance(hsv, UnitHSV)

        hsv = rgb.convert("hsv", FormatType.PERCENTAGE)
        h, s, v = hsv.value
        assert abs(h - h_exp) < .5
        assert abs(float(s) - s_exp*100) < 100/255
        assert abs(float(v) - v_exp*100) < 100/255
        assert isinstance(hsv, PercentageHSV)

def test_class_conversion_rgb_to_hsl():
    for rgb, hsl_expected in samples_rgb_hsl.items():
        r, g, b = rgb
        h_exp, s_exp, l_exp = hsl_expected

        rgb = ColorUnitRGB((r, g, b))
        rgb = rgb.convert("rgb", FormatType.INT)
        #Samples use CSS algorithm for HSL conversion
        hsl = rgb.convert("hsl", FormatType.FLOAT, use_css_algo = True)
        hsl = hsl.convert("hsl", FormatType.INT)
        assert hsl.value == (round(h_exp), round(s_exp*255), round(l_exp*255))
        assert isinstance(hsl, ColorHSLINT)

        hsl = rgb.convert("hsl", FormatType.FLOAT)
        h, s, l = hsl.value
        assert abs(h - h_exp) < .5
        assert abs(float(s) - s_exp) < 1/255
        assert abs(float(l) - l_exp) < 1/255
        assert isinstance(hsl, UnitHSL)

        hsl = rgb.convert("hsl", FormatType.PERCENTAGE)
        h, s, l = hsl.value
        assert abs(h - h_exp) < .5
        assert abs(float(s) - s_exp*100) < 100/255
        assert abs(float(l) - l_exp*100) < 100/255
        assert isinstance(hsl, PercentageHSL)

def test_class_conversion_rgba_to_hsva():
    for rgb, hsv_expected in samples_rgb_hsv.items():
        r, g, b = rgb
        h_exp, s_exp, v_exp = hsv_expected

        rgba = ColorUnitRGBA((r, g, b, 0.5))
        hsva = rgba.convert("hsva", FormatType.INT)
        assert hsva.value == (round(h_exp), round(s_exp*255), round(v_exp*255), 128)
        assert isinstance(hsva, ColorHSVAINT)

        hsva = rgba.convert("hsva", FormatType.FLOAT)
        h, s, v, a = hsva.value
        assert abs(h - h_exp) < .5
        assert abs(float(s) - s_exp) < 1/255
        assert abs(float(v) - v_exp) < 1/255
        assert abs(float(a) - 0.5) < 1/255
        assert isinstance(hsva, UnitHSVA)

        hsva = rgba.convert("hsva", FormatType.PERCENTAGE)
        h, s, v, a = hsva.value
        assert abs(h - h_exp) < .5
        assert abs(float(s) - s_exp*100) < 100/255
        assert abs(float(v) - v_exp*100) < 100/255
        assert abs(float(a) - 50.0) < 100/255
        assert isinstance(hsva, PercentageHSVA)

def test_class_conversion_rgba_to_hsla():
    for rgb, hsl_expected in samples_rgb_hsl.items():
        r, g, b = rgb
        h_exp, s_exp, l_exp = hsl_expected

        rgba = ColorUnitRGBA((r, g, b, 0.5))
        #Samples use CSS algorithm for HSL conversion
        hsla = rgba.convert("hsla", FormatType.FLOAT, use_css_algo = True)
        hsla = hsla.convert("hsla", FormatType.INT)
        assert hsla.value == (round(h_exp), round(s_exp*255), round(l_exp*255), 128)
        assert isinstance(hsla, ColorHSLAINT)

        hsla = rgba.convert("hsla", FormatType.FLOAT)
        h, s, l, a = hsla.value
        assert abs(h - h_exp) < .5
        assert abs(float(s) - s_exp) < 1/255
        assert abs(float(l) - l_exp) < 1/255
        assert abs(float(a) - 0.5) < 1/255
        assert isinstance(hsla, UnitHSLA)

        hsla = rgba.convert("hsla", FormatType.PERCENTAGE)
        h, s, l, a = hsla.value
        assert abs(h - h_exp) < .5
        assert abs(float(s) - s_exp*100) < 100/255
        assert abs(float(l) - l_exp*100) < 100/255
        assert abs(float(a) - 50.0) < 100/255
        assert isinstance(hsla, PercentageHSLA)

def test_class_conversion_hsv_to_rgb():
    for rgb_expected, hsv in samples_rgb_hsv.items():
        r_exp, g_exp, b_exp = rgb_expected

        h, s, v = hsv
        hsv = ColorHSVINT((round(h), round(s*255), round(v*255)))

        rgb = hsv.convert("rgb", FormatType.INT)
        assert rgb.value == (round(r_exp*255), round(g_exp*255), round(b_exp*255))
        assert isinstance(rgb, ColorRGBINT)

        rgb = hsv.convert("rgb", FormatType.FLOAT)
        r, g, b = rgb.value
        assert abs(float(r) - r_exp) < 1/255
        assert abs(float(g) - g_exp) < 1/255
        assert abs(float(b) - b_exp) < 1/255
        assert isinstance(rgb, ColorUnitRGB)

        rgb = hsv.convert("rgb", FormatType.PERCENTAGE)
        r, g, b = rgb.value
        assert abs(float(r) - r_exp*100) < 100/255
        assert abs(float(g) - g_exp*100) < 100/255
        assert abs(float(b) - b_exp*100) < 100/255
        assert isinstance(rgb, ColorPercentageRGB)

def test_has_hue_property():
    rgb = ColorUnitRGB((0.5, 0.25, 0.75))
    rgba = ColorUnitRGBA((0.5, 0.25, 0.75, 0.8))
    hsv = UnitHSV((180.0, 0.5, 0.8))
    hsva = UnitHSVA((180.0, 0.5, 0.8, 0.9))
    hsl = UnitHSL((180.0, 0.5, 0.6))
    hsla = UnitHSLA((180.0, 0.5, 0.6, 0.7))

    assert not rgb.has_hue
    assert not rgba.has_hue
    assert hsv.has_hue
    assert hsva.has_hue
    assert hsl.has_hue
    assert hsla.has_hue