from ...chromatica.conversions.to_hsv import hsl_to_hsv, np_hsl_to_hsv, unit_rgb_to_hsv_analytic, np_unit_rgb_to_hsv, unit_rgb_to_hsv_picker
import numpy as np
from ..samples import samples_hsl_hsv, samples_rgb_hsv
import pytest

def test_hsl_to_hsv():
    for (h, s, l), (h_exp, s_exp, v_exp) in samples_hsl_hsv.items():
        h_out, s_out, v_out = hsl_to_hsv(h, s, l)

        assert abs(h_out - h_exp) < 1/360
        assert abs(float(s_out) - s_exp) < 1/255
        assert abs(float(v_out) - v_exp) < 1/255

def test_hsl_to_hsv_numpy():
    the_matrix = np.array(list(samples_hsl_hsv.keys()))
    expected = np.array(list(samples_hsl_hsv.values()))
    result = np_hsl_to_hsv(the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2])
    assert np.allclose(result, expected, atol=1/360)

def test_unit_rgb_to_hsv_analytic():
    for (r, g, b), (h_exp, s_exp, v_exp) in samples_rgb_hsv.items():
        h_out, s_out, v_out = unit_rgb_to_hsv_analytic(r, g, b)

        assert abs(h_out - h_exp) < .5
        assert abs(float(s_out) - s_exp) < 1/255
        assert abs(float(v_out) - v_exp) < 1/255

def test_unit_rgb_to_hsv_numpy():
    the_matrix = np.array(list(samples_rgb_hsv.keys()))
    expected = np.array(list(samples_rgb_hsv.values()))
    r, g, b = the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2]
    hsv = np_unit_rgb_to_hsv(r, g, b)
    h_out, s_out, v_out = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    expected_h = expected[..., 0]
    expected_s = expected[..., 1]
    expected_v = expected[..., 2]

    assert np.allclose(h_out, expected_h, atol=0.5)
    assert np.allclose(s_out, expected_s, atol=1/255)
    assert np.allclose(v_out, expected_v, atol=1/255)

def test_rgb_to_hsv_picker_rounding():
    for (r, g, b), (h_exp, s_exp, v_exp) in samples_rgb_hsv.items():
        h_out, s_out, v_out = unit_rgb_to_hsv_picker(r, g, b)

        assert abs(h_out - h_exp) < 1.0
        assert abs(float(s_out) - s_exp) < 1/255
        assert abs(float(v_out) - v_exp) < 1/255