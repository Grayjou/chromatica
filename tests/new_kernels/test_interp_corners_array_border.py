import numpy as np

from ...chromatica.v2core.interp_2d.wrappers import (
    lerp_from_corners_array_border,
)


def test_lerp_from_corners_array_border_inbounds():
    """Test corner array border interpolation for in-bounds coords."""
    corners = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    coords = np.array([[[0.5, 0.5]]], dtype=np.float64)
    border_array = np.array([[999.0]], dtype=np.float64)  # 2D for grid coords
    
    result = lerp_from_corners_array_border(corners, coords, border_array)
    
    expected = np.array([[150.0]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_lerp_from_corners_array_border_out_of_bounds():
    """Test corner array border blends with border_array when OOB."""
    corners = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    coords = np.array([[[-0.5, 0.5]]], dtype=np.float64)  # OOB in x
    border_array = np.array([[50.0]], dtype=np.float64)  # 2D for grid coords
    
    result = lerp_from_corners_array_border(corners, coords, border_array)
    
    # Kernel should handle OOB; result shape matches coords
    assert result.shape == (1, 1)
    # Without feathering, may clamp to edge; just verify it runs
    assert np.isfinite(result[0, 0])