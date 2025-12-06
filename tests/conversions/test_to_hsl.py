from ...chromatica.conversions.to_hsl import hsv_to_hsl, np_hsv_to_hsl, unit_rgb_to_hsl, np_unit_rgb_to_hsl
import numpy as np
from ..samples import samples_hsv_hsl, samples_rgb_hsl

def test_hsv_to_hsl():
    for (h, s, v), (h_exp, s_exp, l_exp) in samples_hsv_hsl.items():
        h_out, s_out, l_out = hsv_to_hsl(h, s, v)

        assert abs(h_out - h_exp) < 1/360
        assert abs(float(s_out) - s_exp) < 1/255
        assert abs(float(l_out) - l_exp) < 1/255
    
def test_hsv_to_hsl_numpy():
    the_matrix = np.array(list(samples_hsv_hsl.keys()))
    expected = np.array(list(samples_hsv_hsl.values()))
    result = np_hsv_to_hsl(the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2])
    assert np.allclose(result, expected, atol=1/255)

def test_unit_rgb_to_hsl():
    for (r, g, b), (h_exp, s_exp, l_exp) in samples_rgb_hsl.items():
        h_out, s_out, l_out = unit_rgb_to_hsl(r, g, b)

        assert abs(h_out - h_exp) < 1/2
        assert abs(float(s_out) - s_exp) < 1/255
        assert abs(float(l_out) - l_exp) < 1/255

def test_unit_rgb_to_hsl_numpy():
    the_matrix = np.array(list(samples_rgb_hsl.keys()))
    expected = np.array(list(samples_rgb_hsl.values()))
    r, g, b = the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2]
    hsl = np_unit_rgb_to_hsl(r, g, b)
    h_out, s_out, l_out = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    expected_h = expected[..., 0]
    expected_s = expected[..., 1]
    expected_l = expected[..., 2]

    assert np.allclose(h_out, expected_h, atol=1/2)
    assert np.allclose(s_out, expected_s, atol=1/255)
    assert np.allclose(l_out, expected_l, atol=1/255)