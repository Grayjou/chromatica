import numpy as np
import pytest

from ...chromatica.v2core.interp_2d.wrappers import (
    lerp_between_lines,
    lerp_between_lines_x_discrete,
)


def test_lerp_between_lines_basic():
    """Test basic bilinear interpolation between two lines."""
    line0 = np.array([0.0, 100.0], dtype=np.float64)
    line1 = np.array([200.0, 300.0], dtype=np.float64)
    coords = np.array([[[0.5, 0.5]]], dtype=np.float64)
    
    result = lerp_between_lines(line0, line1, coords)
    
    # At x=0.5: line0 gives 50, line1 gives 250; at y=0.5: blend to 150
    expected = np.array([[[150.0]]], dtype=np.float64)
    assert np.allclose(result, expected)


@pytest.mark.skip(reason="Single-channel x_discrete not implemented yet")
def test_lerp_between_lines_x_discrete_basic():
    """Test x-discrete mode uses nearest-neighbor in x."""
    line0 = np.array([0.0, 100.0], dtype=np.float64)
    line1 = np.array([200.0, 300.0], dtype=np.float64)
    coords = np.array([[[0.4, 0.5]]], dtype=np.float64)
    
    result = lerp_between_lines_x_discrete(line0, line1, coords)
    
    # x=0.4 rounds to index 0; line0[0]=0, line1[0]=200; y=0.5 blends to 100
    expected = np.array([[[100.0]]], dtype=np.float64)
    assert np.allclose(result, expected)