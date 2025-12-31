import numpy as np

from ...chromatica.v2core.interp_2d_.wrappers import (
    lerp_from_corners,
    lerp_from_unpacked_corners,
)


def test_lerp_from_corners_basic():
    """Test bilinear interpolation from four corner values."""
    corners = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)  # TL, TR, BL, BR
    coords = np.array([[[0.5, 0.5]]], dtype=np.float64)
    
    result = lerp_from_corners(corners, coords)
    
    # top: 0 + 0.5*(100-0) = 50; bottom: 200 + 0.5*(300-200) = 250; blend: 150
    expected = np.array([[[150.0]]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_lerp_from_corners_at_corner_positions():
    """Test that corner positions return exact corner values."""
    corners = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    coords = np.array([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0]]
    ], dtype=np.float64)
    
    result = lerp_from_corners(corners, coords)
    
    expected = np.array([
        [10.0, 20.0],
        [30.0, 40.0]
    ], dtype=np.float64)
    assert np.allclose(result, expected)


def test_lerp_from_unpacked_corners_basic():
    """Test unpacked corner variant."""
    tl, tr, bl, br = 0.0, 100.0, 200.0, 300.0
    coords = np.array([[[0.5, 0.5]]], dtype=np.float64)
    
    result = lerp_from_unpacked_corners(tl, tr, bl, br, coords)
    
    expected = np.array([[[150.0]]], dtype=np.float64)
    assert np.allclose(result, expected)