from ...chromatica.conversions import convert, ColorMode, FormatType, np_convert
from ..samples import samples_rgb_hsv

def test_convert_returns_tuple():
    from_space = "rgb"
    to_space = "hsv"
    color = (255, 128, 64)

    result = convert(color, from_space, to_space, output_type=FormatType.FLOAT)
    assert isinstance(result, tuple)
    assert len(result) == 3

def test_adds_alpha():
    from_space = "rgb"
    to_space = "rgba"
    color = (255, 128, 64)

    result = convert(color, from_space, to_space)
    assert result == (255, 128, 64, 255)

    from_space = "hsv"
    to_space = "hsva"
    color = (360, 0, 100)

    result = convert(color, from_space, to_space)
    assert result == (360, 0, 100, 255)

    from_space = "hsl"
    to_space = "hsla"
    color = (360, 100, 50)

    result = convert(color, from_space, to_space)
    assert result == (360, 100, 50, 255)

def test_conversion_adds_alpha():
    for (r, g, b), (h_exp, s_exp, v_exp) in samples_rgb_hsv.items():
        from_space = "rgba"
        to_space = "hsva"
        color = (r, g, b, 128)

        h, s, v, a = convert(color, from_space, to_space, input_type=FormatType.FLOAT, output_type=FormatType.FLOAT)

        assert abs(float(h) - h_exp) < .5
        assert abs(float(s) - s_exp) < 1/510
        assert abs(float(v) - v_exp) < 1/510
        assert a == 128

def test_adds_proper_format_alpha():
    from_space = "rgb"
    to_space = "rgba"
    color = (1.0, 0.5, 0.25)

    result = convert(color, from_space, to_space, input_type=FormatType.FLOAT, output_type=FormatType.INT)
    assert result == (255, 128, 64, 255)
    result = convert(color, from_space, to_space, input_type=FormatType.FLOAT, output_type=FormatType.PERCENTAGE)
    assert result == (100.0, 50.0, 25.0, 100.0)

    from_space = "rgb"
    to_space = "hsva"
    color, expected = next(iter(samples_rgb_hsv.items()))
    expected += (1.0,)
    result = convert(color, from_space, to_space, input_type=FormatType.FLOAT, output_type=FormatType.FLOAT)
    h, s, v, a = result
    assert abs(float(h) - expected[0]) < .5
    assert abs(float(s) - expected[1]) < 1/510
    assert abs(float(v) - expected[2]) < 1/510

    h, s, v, a = expected
    expected = (
        int(round(h)),
        int(round(s * 255)),
        int(round(v * 255)),
        255,
    )

    result = convert(color, from_space, to_space, input_type=FormatType.FLOAT, output_type=FormatType.INT)
    assert result == expected

    from_space = "hsl"
    to_space = "hsla"
    color = (360.0, 1.0, 0.5)

    result = convert(color, from_space, to_space, input_type=FormatType.FLOAT, output_type=FormatType.INT)
    assert result == (360, 255, 128, 255)

def test_removes_alpha():
    from_space = "rgba"
    to_space = "rgb"
    color = (255, 128, 64, 128)

    result = convert(color, from_space, to_space)
    assert result == (255, 128, 64)

    from_space = "hsva"
    to_space = "hsv"
    color = (360, 0, 100, 128)

    result = convert(color, from_space, to_space)
    assert result == (360, 0, 100)

    from_space = "hsla"
    to_space = "hsl"
    color = (360, 100, 50, 128)

    result = convert(color, from_space, to_space)
    assert result == (360, 100, 50)

import numpy as np
def test_input_wrapping_np():
    red = np.array([[1.0, 0.0, 0.0]])
    unwrapped_red = np.array([1.0, 0.0, 0.0])
    #Convert to HSV
    red_hsv = np_convert(red, 'rgb', 'hsv', input_type=FormatType.FLOAT, output_type=FormatType.FLOAT)
    unwrapped_red_hsv = np_convert(unwrapped_red, 'rgb', 'hsv', input_type=FormatType.FLOAT, output_type=FormatType.FLOAT)
    assert np.allclose(red_hsv[0], unwrapped_red_hsv, atol=1e-5)