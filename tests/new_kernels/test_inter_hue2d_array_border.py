import numpy as np

from ...chromatica.v2core.interp_hue_.wrappers import (
    hue_lerp_from_corners_array_border_typed,
    hue_lerp_between_lines_typed,
    hue_lerp_between_lines_typed_x_discrete,
)


def _hue_lerp(h0: float, h1: float, t: float) -> float:
    diff = h1 - h0
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return (h0 + t * diff) % 360


def test_hue_lerp_between_lines_basic():
    """Test basic hue line interpolation."""
    line0 = np.array([0.0, 90.0], dtype=np.float64)
    line1 = np.array([180.0, 270.0], dtype=np.float64)
    coords = np.array([[[0.5, 0.5]]], dtype=np.float64)
    
    result = hue_lerp_between_lines_typed(line0, line1, coords)
    
    # At x=0.5: line0->45, line1->225; y=0.5: hue blend
    top = _hue_lerp(0.0, 90.0, 0.5)  # 45
    bottom = _hue_lerp(180.0, 270.0, 0.5)  # 225
    expected_val = _hue_lerp(top, bottom, 0.5)
    expected = np.array([[[expected_val]]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_hue_lerp_between_lines_x_discrete_basic():
    """Test x-discrete hue line interpolation."""
    line0 = np.array([0.0, 90.0], dtype=np.float64)
    line1 = np.array([180.0, 270.0], dtype=np.float64)
    coords = np.array([[[0.3, 0.5]]], dtype=np.float64)  # x=0.3 rounds to 0
    
    result = hue_lerp_between_lines_typed_x_discrete(line0, line1, coords)
    
    # x rounds to 0: line0[0]=0, line1[0]=180; y=0.5 blends
    expected_val = _hue_lerp(0.0, 180.0, 0.5)
    expected = np.array([[[expected_val]]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_hue_corners_array_border_inbounds_matches_constant():
    corners = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    coords = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    border_array = np.zeros((2, 2), dtype=np.float64)

    result = hue_lerp_from_corners_array_border_typed(
        corners, coords, border_array
    )

    expected = np.array([[0.0, 90.0], [180.0, 270.0]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_hue_corners_array_border_out_of_bounds_blends_border():
    corners = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    coords = np.array([[[ -0.2, -0.2 ]]], dtype=np.float64)
    border_array = np.array([[300.0]], dtype=np.float64)

    result = hue_lerp_from_corners_array_border_typed(
        corners, coords, border_array
    )

    interp_val = _hue_lerp(0.0, 0.0, 1.0)  # clamped to top-left -> 0
    expected = _hue_lerp(interp_val, 300.0, 0.5)  # kernel uses 0.5 blend for OOB

    assert np.allclose(result, expected)