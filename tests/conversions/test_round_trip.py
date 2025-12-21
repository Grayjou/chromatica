from ..samples import samples_hsl_hsv, samples_rgb_hsv, samples_hsl_rgb, samples_hsv_hsl, samples_hsv_rgb, samples_rgb_hsl
from ...chromatica.conversions.to_hsv import unit_rgb_to_hsv_analytic, hsl_to_hsv, np_unit_rgb_to_hsv, np_hsl_to_hsv
from ...chromatica.conversions.to_rgb import hsv_to_unit_rgb, hsl_to_unit_rgb, np_hsv_to_unit_rgb, np_hsl_to_unit_rgb
from ...chromatica.conversions.to_hsl import hsv_to_hsl, np_hsv_to_hsl, np_unit_rgb_to_hsl, unit_rgb_to_hsl
import numpy as np
import pytest


rgb_tolerance = 1e-3
hsv_tolerance = 1e-3
hsl_tolerance = 1e-3

def test_round_trip_rgb_hsv():
    for (r, g, b), (h_exp, s_exp, v_exp) in samples_rgb_hsv.items():
        h, s, v = unit_rgb_to_hsv_analytic(r, g, b)
        r_out, g_out, b_out = hsv_to_unit_rgb(h, s, v)

        assert abs(float(r) - r_out) < rgb_tolerance
        assert abs(float(g) - g_out) < rgb_tolerance
        assert abs(float(b) - b_out) < rgb_tolerance

def test_round_trip_hsl_hsv():
    for (h, s, l), (h_exp, s_exp, v_exp) in samples_hsl_hsv.items():
        h_out, s_out, v_out = hsl_to_hsv(h, s, l)
        h_final, s_final, l_final = hsv_to_hsl(h_out, s_out, v_out)

        assert abs(h - h_final) < hsl_tolerance
        assert abs(float(s) - float(s_final)) < hsl_tolerance
        assert abs(float(l) - float(l_final)) < hsl_tolerance

def test_round_trip_hsv_hsl():
    for (h, s, v), (h_exp, s_exp, l_exp) in samples_hsv_hsl.items():
        h_out, s_out, l_out = hsv_to_hsl(h, s, v)
        h_final, s_final, v_final = hsl_to_hsv(h_out, s_out, l_out)

        assert abs(h - h_final) < hsv_tolerance
        assert abs(float(s) - float(s_final)) < hsv_tolerance
        assert abs(float(v) - float(v_final)) < hsv_tolerance

def test_round_trip_rgb_hsl():
    for (r, g, b), (h_exp, s_exp, l_exp) in samples_rgb_hsl.items():
        h, s, l = unit_rgb_to_hsl(r, g, b)
        r_out, g_out, b_out = hsl_to_unit_rgb(h, s, l)

        assert abs(float(r) - r_out) < rgb_tolerance
        assert abs(float(g) - g_out) < rgb_tolerance
        assert abs(float(b) - b_out) < rgb_tolerance

def test_round_trip_hsv_rgb_numpy():
    the_matrix = np.array(list(samples_hsv_rgb.keys()))
    expected = np.array(list(samples_hsv_rgb.keys()))
    h, s, v = the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2]
    rgb = np_hsv_to_unit_rgb(h, s, v)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hsv = np_unit_rgb_to_hsv(r, g, b)
    h_out, s_out, v_out = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    assert np.allclose(h, h_out, atol=hsv_tolerance)
    assert np.allclose(s, s_out, atol=hsv_tolerance)
    assert np.allclose(v, v_out, atol=hsv_tolerance)

def test_round_trip_hsl_rgb_numpy():
    the_matrix = np.array(list(samples_hsl_rgb.keys()))
    expected = np.array(list(samples_hsl_rgb.keys()))
    h, s, l = the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2]
    rgb = np_hsl_to_unit_rgb(h, s, l)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hsl = np_unit_rgb_to_hsl(r, g, b)
    h_out, s_out, l_out = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    assert np.allclose(h, h_out, atol=hsl_tolerance)
    assert np.allclose(s, s_out, atol=hsl_tolerance)
    assert np.allclose(l, l_out, atol=hsl_tolerance)

def test_round_trip_hsv_hsl_numpy():
    the_matrix = np.array(list(samples_hsv_hsl.keys()))
    expected = np.array(list(samples_hsv_hsl.keys()))
    h, s, v = the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2]
    hsl = np_hsv_to_hsl(h, s, v)
    h_out, s_out, l_out = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    hsv = np_hsl_to_hsv(h_out, s_out, l_out)
    h_final, s_final, v_final = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    assert np.allclose(h, h_final, atol=hsv_tolerance)
    assert np.allclose(s, s_final, atol=hsv_tolerance)
    assert np.allclose(v, v_final, atol=hsv_tolerance)

def test_round_trip_hsl_hsv_numpy():
    the_matrix = np.array(list(samples_hsl_hsv.keys()))
    expected = np.array(list(samples_hsl_hsv.keys()))
    h, s, l = the_matrix[..., 0], the_matrix[..., 1], the_matrix[..., 2]
    hsv = np_hsl_to_hsv(h, s, l)
    h_out, s_out, v_out = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hsl = np_hsv_to_hsl(h_out, s_out, v_out)
    h_final, s_final, l_final = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    assert np.allclose(h, h_final, atol=hsl_tolerance)
    assert np.allclose(s, s_final, atol=hsl_tolerance)
    assert np.allclose(l, l_final, atol=hsl_tolerance)

