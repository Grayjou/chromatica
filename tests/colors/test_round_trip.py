from ...chromatica.colors.rgb import ColorRGBINT, ColorRGBAINT, ColorUnitRGB, ColorPercentageRGBA, ColorUnitRGBA, ColorPercentageRGB
from ...chromatica.colors.hsv import ColorHSVAINT, ColorHSVINT, UnitHSV, PercentageHSV, PercentageHSVA, UnitHSVA
from ...chromatica.colors.hsl import ColorHSLINT, ColorHSLAINT, UnitHSL, UnitHSLA, PercentageHSL, PercentageHSLA
from ...chromatica.types.format_type import FormatType
from ..samples import samples_rgb_hsv, samples_rgb_hsl

def test_round_trip_rgb_to_hsv_to_rgb():
    for rgb, hsv_expected in samples_rgb_hsv.items():
        r, g, b = rgb
        h_exp, s_exp, v_exp = hsv_expected

        rgb = ColorUnitRGB((r, g, b))
        hsv = rgb.convert("hsv", FormatType.FLOAT)
        rgb_out = hsv.convert("rgb", FormatType.FLOAT)

        r_out, g_out, b_out = rgb_out.value

        assert abs(float(r) - float(r_out)) < 1/255
        assert abs(float(g) - float(g_out)) < 1/255
        assert abs(float(b) - float(b_out)) < 1/255

def test_round_trip_rgb_to_hsl_to_rgb():
    for rgb, hsl_expected in samples_rgb_hsl.items():
        r, g, b = rgb
        h_exp, s_exp, l_exp = hsl_expected

        rgb = ColorUnitRGB((r, g, b))
        hsl = rgb.convert("hsl", FormatType.FLOAT, use_css_algo = True)
        rgb_out = hsl.convert("rgb", FormatType.FLOAT)

        r_out, g_out, b_out = rgb_out.value

        assert abs(float(r) - float(r_out)) < 1/255
        assert abs(float(g) - float(g_out)) < 1/255
        assert abs(float(b) - float(b_out)) < 1/255

def test_round_trip_hsv_to_rgb_to_hsv():
    for rgb, hsv_expected in samples_rgb_hsv.items():
        r, g, b = rgb
        h_exp, s_exp, v_exp = hsv_expected

        hsv = ColorHSVINT((round(h_exp), round(s_exp*255), round(v_exp*255)))
        rgb_out = hsv.convert("rgb", FormatType.FLOAT)
        hsv_final = rgb_out.convert("hsv", FormatType.FLOAT)

        h_out, s_out, v_out = hsv_final.value

        assert abs(h_exp - h_out) < .5
        assert abs(float(s_exp) - float(s_out)) < 1/255
        assert abs(float(v_exp) - float(v_out)) < 1/255

def test_round_trip_hsl_to_rgb_to_hsl():
    for rgb, hsl_expected in samples_rgb_hsl.items():
        r, g, b = rgb
        h_exp, s_exp, l_exp = hsl_expected

        hsl = ColorHSLINT((round(h_exp), round(s_exp*255), round(l_exp*255)))
        rgb_out = hsl.convert("rgb", FormatType.FLOAT)
        hsl_final = rgb_out.convert("hsl", FormatType.FLOAT, use_css_algo = True)

        h_out, s_out, l_out = hsl_final.value

        assert abs(h_exp - h_out) < .5
        assert abs(float(s_exp) - float(s_out)) < 1/255
        assert abs(float(l_exp) - float(l_out)) < 1/255