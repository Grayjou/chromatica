from ...chromatica.conversions.to_rgb import hsl_to_unit_rgb, hsv_to_unit_rgb, np_hsl_to_unit_rgb, np_hsv_to_unit_rgb
import numpy as np
from ..samples import samples_hsv_rgb, samples_hsl_rgb

def test_hsv_to_unit_rgb():
    for (h, s, v), (r_exp, g_exp, b_exp) in samples_hsv_rgb.items():
        r, g, b = hsv_to_unit_rgb(h, s, v)

        assert abs(float(r) - r_exp) < 1/510
        assert abs(float(g) - g_exp) < 1/510
        assert abs(float(b) - b_exp) < 1/510



def test_hsv_to_unit_rgb_numpy():
    the_matrix = np.array(list(samples_hsv_rgb.keys()))
    expected = np.array(list(samples_hsv_rgb.values()))
    result = np_hsv_to_unit_rgb(the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2])
    assert np.allclose(result, expected, atol=1/510)

samples_hsl_str = {
    "hsl(137 69.7% 61.5%)": "rgb(88 225 127)",
    "hsl(0 10.5% 35.5%)": "rbg(100, 81, 81)",
    "hsl(130 67.1% 49.1%)": "rgb(41 209 69)",
    "hsl(283 67.2% 23%)": "rgb(76 19 98)"
}



def test_hsl_to_unit_rgb():
    for (h, s, l), (r_exp, g_exp, b_exp) in samples_hsl_rgb.items():
        r, g, b = hsl_to_unit_rgb(h, s, l, use_css_algo=True)

        assert abs(float(r) - r_exp) < 1/255
        assert abs(float(g) - g_exp) < 1/255
        assert abs(float(b) - b_exp) < 1/255

def test_hsl_to_unit_rgb_numpy():
    the_matrix = np.array(list(samples_hsl_rgb.keys()))
    expected = np.array(list(samples_hsl_rgb.values()))
    result = np_hsl_to_unit_rgb(the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2])
    assert np.allclose(result, expected, atol=1/255)