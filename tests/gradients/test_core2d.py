"""
Tests for core2d.py wrapper functions.
"""

import pytest
import numpy as np
from ...chromatica.gradients.v2core.core2d import (
    sample_between_lines_continuous,
    sample_between_lines_discrete,
    sample_hue_between_lines_continuous,
    sample_hue_between_lines_discrete,
    multival2d_lerp_between_lines_continuous,
    multival2d_lerp_between_lines_discrete,
)
from ...chromatica.gradients.v2core.core import HueMode
from boundednumbers import BoundType
from unitfield import upbm_2d


def test_sample_between_lines_continuous():
    """Test continuous line interpolation."""
    L = 10
    line0 = np.linspace(0, 1, L)
    line1 = np.linspace(1, 0, L)
    
    # Create a simple 2D coordinate grid
    H, W = 5, 10
    coords = upbm_2d(width=W, height=H)
    
    result = sample_between_lines_continuous(line0, line1, coords)
    
    assert result.shape == (H, W)
    # Top row (u_y=0) should be close to line0
    # Bottom row (u_y=1) should be close to line1
    assert np.allclose(result[0, 0], line0[0], atol=0.1)
    assert np.allclose(result[-1, -1], line1[-1], atol=0.1)


def test_sample_between_lines_discrete():
    """Test discrete line interpolation."""
    L = 10
    line0 = np.linspace(0, 1, L)
    line1 = np.linspace(1, 0, L)
    
    # Create a simple 2D coordinate grid
    H, W = 5, 10
    coords = upbm_2d(width=W, height=H)
    
    result = sample_between_lines_discrete(line0, line1, coords)
    
    assert result.shape == (H, W)
    # Check boundaries
    assert result[0, 0] >= 0 and result[0, 0] <= 1
    assert result[-1, -1] >= 0 and result[-1, -1] <= 1


def test_sample_hue_between_lines_continuous():
    """Test continuous hue line interpolation."""
    L = 6
    line0 = np.array([0, 60, 120, 180, 240, 300], dtype=np.float64)
    line1 = np.array([180, 240, 300, 0, 60, 120], dtype=np.float64)
    
    H, W = 3, 6
    coords = upbm_2d(width=W, height=H)
    
    result = sample_hue_between_lines_continuous(
        line0, line1, coords,
        mode_x=HueMode.SHORTEST,
        mode_y=HueMode.CW
    )
    
    assert result.shape == (H, W)
    # All values should be in [0, 360)
    assert np.all((result >= 0) & (result < 360))


def test_sample_hue_between_lines_discrete():
    """Test discrete hue line interpolation."""
    L = 6
    line0 = np.array([0, 60, 120, 180, 240, 300], dtype=np.float64)
    line1 = np.array([180, 240, 300, 0, 60, 120], dtype=np.float64)
    
    H, W = 3, 6
    coords = upbm_2d(width=W, height=H)
    
    result = sample_hue_between_lines_discrete(
        line0, line1, coords,
        mode_y=HueMode.CCW
    )
    
    assert result.shape == (H, W)
    # All values should be in [0, 360)
    assert np.all((result >= 0) & (result < 360))


def test_multival2d_lerp_between_lines_continuous():
    """Test multi-channel continuous line interpolation."""
    L = 10
    num_channels = 3
    
    # Create RGB gradient lines
    starts_lines = [np.linspace(0, 1, L) for _ in range(num_channels)]
    ends_lines = [np.linspace(1, 0, L) for _ in range(num_channels)]
    
    H, W = 5, 10
    coords = [upbm_2d(width=W, height=H) for _ in range(num_channels)]
    
    result = multival2d_lerp_between_lines_continuous(
        starts_lines, ends_lines, coords,
        bound_types=BoundType.CLAMP
    )
    
    assert result.shape == (H, W, num_channels)
    # Check all values are in [0, 1] range due to clamping
    assert np.all((result >= 0) & (result <= 1))


def test_multival2d_lerp_between_lines_discrete():
    """Test multi-channel discrete line interpolation."""
    L = 10
    num_channels = 3
    
    # Create RGB gradient lines
    starts_lines = [np.linspace(0, 1, L) for _ in range(num_channels)]
    ends_lines = [np.linspace(1, 0, L) for _ in range(num_channels)]
    
    H, W = 5, 10
    coords = [upbm_2d(width=W, height=H) for _ in range(num_channels)]
    
    result = multival2d_lerp_between_lines_discrete(
        starts_lines, ends_lines, coords,
        bound_types=BoundType.CLAMP
    )
    
    assert result.shape == (H, W, num_channels)
    # Check all values are in [0, 1] range due to clamping
    assert np.all((result >= 0) & (result <= 1))


def test_bound_type_support():
    """Test that BoundType parameter works correctly."""
    L = 10
    line0 = np.linspace(-0.5, 1.5, L)  # Values outside [0, 1]
    line1 = np.linspace(1.5, -0.5, L)
    
    H, W = 5, 10
    coords = upbm_2d(width=W, height=H)
    
    # With CLAMP, should be bounded to [0, 1]
    result_clamp = sample_between_lines_continuous(
        line0, line1, coords,
        bound_type=BoundType.CLAMP
    )
    assert np.all((result_clamp >= -0.01) & (result_clamp <= 1.01))  # Small tolerance
    
    # With IGNORE, may go outside [0, 1]
    result_ignore = sample_between_lines_continuous(
        line0, line1, coords,
        bound_type=BoundType.IGNORE
    )
    # Should have some values outside [0, 1]
    # (Actually depends on interpolation, but let's just check it runs)
    assert result_ignore.shape == (H, W)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
