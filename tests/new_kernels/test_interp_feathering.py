import numpy as np

from ...chromatica.v2core.interp_2d_.wrappers import (
    BorderMode,
    DistanceMode,
    lerp_between_lines,
    lerp_from_corners,
)


def test_lerp_between_lines_feathering_blends_partial():
    line0 = np.zeros(2, dtype=np.float64)
    line1 = np.zeros(2, dtype=np.float64)
    coords = np.array([[[1.5, 0.0]]], dtype=np.float64)  # 0.5 past the right edge

    result = lerp_between_lines(
        line0,
        line1,
        coords,
        border_mode=BorderMode.CONSTANT,
        border_constant=200.0,
        border_feathering=1.0,
        distance_mode=DistanceMode.MAX_NORM,
    )

    # extra distance is 0.5, feather=1.0 -> blend_factor=0.5
    expected = np.array([[100.0]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_lerp_between_lines_feathering_clamps_to_border():
    line0 = np.zeros(2, dtype=np.float64)
    line1 = np.zeros(2, dtype=np.float64)
    coords = np.array([[[1.5, 0.0]]], dtype=np.float64)

    result = lerp_between_lines(
        line0,
        line1,
        coords,
        border_mode=BorderMode.CONSTANT,
        border_constant=200.0,
        border_feathering=0.25,  # smaller feather so extra>=feather -> full border
        distance_mode=DistanceMode.MAX_NORM,
    )

    expected = np.array([[200.0]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_lerp_from_corners_feathering_blends_partial():
    corners = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    coords = np.array([[[1.25, 0.0]]], dtype=np.float64)  # 0.25 past right edge

    result = lerp_from_corners(
        corners,
        coords,
        border_mode=BorderMode.CONSTANT,
        border_constant=200.0,
        border_feathering=0.5,
    )

    # extra distance is 0.25, feather=0.5 -> blend_factor=0.5
    expected = np.array([[100.0]], dtype=np.float64)
    assert np.allclose(result, expected)