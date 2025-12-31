import numpy as np

from chromatica.v2core.interp_hue_.wrappers import (
    # Lines
    hue_lerp_between_lines_typed,
    hue_lerp_between_lines_typed_x_discrete,
    # Corners
    hue_lerp_from_corners_typed,
)


def _hue_lerp(h0: float, h1: float, t: float) -> float:
    """Shortest-path hue lerp helper for expectations."""
    diff = h1 - h0
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return (h0 + t * diff) % 360


def test_hue_corners_grid_matches_expected():
    corners = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    coords = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float64,
    )

    result = hue_lerp_from_corners_typed(corners, coords)
    expected = np.array([[0.0, 90.0], [180.0, 270.0]], dtype=np.float64)

    assert np.allclose(result, expected)


def test_hue_corners_center_shortest_path():
    corners = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    coords = np.array([[0.5, 0.5]], dtype=np.float64)

    result = hue_lerp_from_corners_typed(corners, coords)

    top = _hue_lerp(0.0, 90.0, 0.5)
    bottom = _hue_lerp(180.0, 270.0, 0.5)
    expected = _hue_lerp(top, bottom, 0.5)

    assert np.allclose(result, expected)


def test_hue_corners_wraparound_shortest_path():
    corners = np.array([350.0, 10.0, 20.0, 40.0], dtype=np.float64)
    coords = np.array([[0.5, 0.5]], dtype=np.float64)

    result = hue_lerp_from_corners_typed(corners, coords)

    top = _hue_lerp(350.0, 10.0, 0.5)
    bottom = _hue_lerp(20.0, 40.0, 0.5)
    expected = _hue_lerp(top, bottom, 0.5)

    assert np.allclose(result, expected)