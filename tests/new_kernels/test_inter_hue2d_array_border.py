import numpy as np
import pytest
from ...chromatica.v2core.interp_hue.wrappers import (
    hue_lerp_from_corners_array_border,
    hue_lerp_between_lines_array_border,
    hue_lerp_between_lines_array_border_x_discrete,
    DistanceMode
)
from unitfield import upbm_2d
from ...chromatica.types.color_types import HueMode

def _hue_lerp(h0: float, h1: float, t: float) -> float:
    diff = h1 - h0
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return (h0 + t * diff) % 360

def _simple_lerp(v0: float, v1: float, t: float) -> float:

    return v0 + t * (v1 - v0)

def test_hue_lerp_between_lines_basic():
    """Test basic hue line interpolation."""
    line0 = np.array([0.0, 90.0], dtype=np.float64)
    line1 = np.array([180.0, 270.0], dtype=np.float64)
    coords = np.array([[[0.5, 0.5]]], dtype=np.float64)
    border_array = np.zeros((1, 1), dtype=np.float64)  # 2D for grid coords
    result = hue_lerp_between_lines_array_border(line0, line1, coords, border_array=border_array)
    
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
    border_array = np.zeros((1, 1), dtype=np.float64)  # 2D for grid coords
    
    result = hue_lerp_between_lines_array_border_x_discrete(line0, line1, coords, border_array=border_array)
    
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

    result = hue_lerp_from_corners_array_border(
        corners, coords, border_array
    )

    expected = np.array([[0.0, 90.0], [180.0, 270.0]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_hue_corners_array_border_out_of_bounds_blends_border():
    corners = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    coords = np.array([[[ -0.5, -0.5 ]]], dtype=np.float64)
    border_array = np.array([[300.0]], dtype=np.float64)

    result = hue_lerp_from_corners_array_border(
        corners, coords, border_array
    )

    
    assert np.allclose(result, border_array)


def test_lerp_between_lines_blending():
    line0 = np.linspace(90.0, 359.0, 10, dtype=np.float64)
    line1 = np.linspace(180.0, 179.0, 10, dtype=np.float64)
    
    coords = upbm_2d(width=2, height=2)-0.5  # coords: [[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]
    #coords = np.clip(coords,0,1)
    border_value = 80.0
    array_border = np.zeros((2, 2), dtype=np.float64) + border_value


    border_values = np.linspace(0.0, 80.0, 9, dtype=np.float64)

    corrects = {}
    incorrects = {}
    ccw_results = []
    for border_v in border_values:
        result = hue_lerp_between_lines_array_border(
            line0, line1, coords, border_array=np.zeros((2, 2), dtype=np.float64) + border_v,
            distance_mode=DistanceMode.MANHATTAN,
            border_feathering=1,
            feather_hue_mode=HueMode.CCW
        )
        expected = np.array([
            [border_v, _simple_lerp(_simple_lerp(90, 359, 0.5), border_v, 0.5)],
            [_simple_lerp(_simple_lerp(90, 180, 0.5), border_v, 0.5), _simple_lerp(_simple_lerp(90, 359, 0.5), _simple_lerp(180, 179, 0.5), 0.5)]]
        )
        if np.allclose(result, expected):
            corrects[border_v] = (result, expected)
        else:
            incorrects[border_v] = (result, expected)
        ccw_results.append(result)
    shortest_results = []
    for border_v in border_values:
        result = hue_lerp_between_lines_array_border(
            line0, line1, coords, border_array=np.zeros((2, 2), dtype=np.float64) + border_v,
            distance_mode=DistanceMode.MANHATTAN,
            border_feathering=1,
            feather_hue_mode=HueMode.SHORTEST
        )
        shortest_results.append(result)
    assert any((not np.allclose(ccw, short) for ccw, short in zip(ccw_results, shortest_results))), "CCW and Shortest results should differ for some border values"

    assert len(incorrects) == 0, f"Some border values produced incorrect results: {incorrects}"
#Failing

LERP_200_CCW = lambda t: _simple_lerp(360.0, 200.0, t)

@pytest.mark.parametrize(
    "feathering",
    np.linspace(0.1, 0.5, 5),
)
def test_hue_corners_array_border_feathering(feathering):
    corners = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    coords = np.array([[[1.25, 0.0]]], dtype=np.float64)  # 0.25 past right edge
    border_array = np.array([[200.0]], dtype=np.float64)

    result = hue_lerp_from_corners_array_border(
        corners,
        coords,
        border_array,
        border_feathering=feathering,
        feather_hue_mode=HueMode.SHORTEST,
    )

    # Shortest path from 0 → 200 is 360 → 200
    expected = max(LERP_200_CCW(0.25 / feathering), 200.0)

    assert np.allclose(result, expected)




LERP = lambda t: _simple_lerp(270.0, 300.0, t)

@pytest.mark.parametrize(
    "distance_mode, expected",
    [
        (DistanceMode.EUCLIDEAN, LERP(2**0.5 / 2)),
        (DistanceMode.MANHATTAN, 300.0),
        (DistanceMode.MAX_NORM, LERP(0.5)),
        (DistanceMode.SCALED_MANHATTAN, LERP(0.7071)),
        (DistanceMode.ALPHA_MAX, LERP(0.7071)),
        (DistanceMode.ALPHA_MAX_SIMPLE, LERP(0.75)),
        (DistanceMode.TAYLOR, LERP(0.625)),
        (DistanceMode.WEIGHTED_MINMAX, LERP(0.67925)),
    ],
)
def test_hue_corners_different_distance_modes(distance_mode, expected):
    corners = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    coords = np.array([[[1.5, 1.5]]], dtype=np.float64)  # 0.5 past right/top edge
    border_array = np.array([[300.0]], dtype=np.float64)

    result = hue_lerp_from_corners_array_border(
        corners,
        coords,
        border_array,
        distance_mode=distance_mode,
        border_feathering=1.0,
        feather_hue_mode=HueMode.SHORTEST,
    )

    assert np.allclose(result, expected)
