import numpy as np

from ...chromatica.v2core.interp_2d.wrappers import (
    lerp_between_lines_onto_array,
    lerp_between_lines_inplace,
)


def test_lerp_between_lines_onto_array_basic():
    """Test line interpolation writing onto existing array."""
    line0 = np.array([0.0, 100.0], dtype=np.float64)
    line1 = np.array([200.0, 300.0], dtype=np.float64)
    coords = np.array([[[0.5, 0.5]]], dtype=np.float64)
    out = np.zeros((1, 1), dtype=np.float64)  # 2D for grid coords
    
    result = lerp_between_lines_onto_array(line0, line1, coords, out)
    
    expected = np.array([[150.0]], dtype=np.float64)
    assert np.allclose(result, expected)
    # Note: onto_array returns new array, doesn't modify input


def test_lerp_between_lines_inplace_basic():
    """Test in-place line interpolation."""
    line0 = np.array([0.0, 100.0], dtype=np.float64)
    line1 = np.array([200.0, 300.0], dtype=np.float64)
    coords = np.array([[[0.5, 0.5]]], dtype=np.float64)
    out = np.zeros((1, 1), dtype=np.float64)  # 2D for grid coords
    
    lerp_between_lines_inplace(line0, line1, coords, out)
    
    expected = np.array([[150.0]], dtype=np.float64)
    assert np.allclose(out, expected)