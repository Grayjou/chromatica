"""
Exhaustive tests for feathering in the new kernel.

These tests validate the border feathering functionality in the new kernel 
(interp_2d_ module) for line and corner interpolation.
"""
import numpy as np
import sys
import os

# Add the project root to path before any chromatica imports
# This ensures the new kernel modules can be imported directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import directly from the new kernel to avoid triggering old kernel imports
from chromatica.v2core.interp_2d_.interp_2d_fast_ import (
    lerp_between_lines_full_feathered,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    DIST_ALPHA_MAX,
    DIST_MAX_NORM,
    DIST_EUCLIDEAN,
    DIST_MANHATTAN,
)
from chromatica.v2core.interp_2d_.corner_interp_2d_fast_ import (
    lerp_from_corners_full_feathered,
)


def test_lerp_between_lines_feathering_blends_partial():
    """Test that feathering creates a smooth blend when partially outside bounds."""
    line0 = np.zeros(2, dtype=np.float64)
    line1 = np.zeros(2, dtype=np.float64)
    coords = np.array([[[1.5, 0.0]]], dtype=np.float64)  # 0.5 past the right edge

    result = lerp_between_lines_full_feathered(
        line0,
        line1,
        coords,
        border_mode=BORDER_CONSTANT,
        border_constant=200.0,
        border_feathering=1.0,
        distance_mode=DIST_MAX_NORM,
    )

    # extra distance is 0.5, feather=1.0 -> blend_factor=0.5
    expected = np.array([[100.0]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_lerp_between_lines_feathering_clamps_to_border():
    """Test that when extra distance exceeds feather width, output is full border value."""
    line0 = np.zeros(2, dtype=np.float64)
    line1 = np.zeros(2, dtype=np.float64)
    coords = np.array([[[1.5, 0.0]]], dtype=np.float64)

    result = lerp_between_lines_full_feathered(
        line0,
        line1,
        coords,
        border_mode=BORDER_CONSTANT,
        border_constant=200.0,
        border_feathering=0.25,  # smaller feather so extra>=feather -> full border
        distance_mode=DIST_MAX_NORM,
    )

    expected = np.array([[200.0]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_lerp_from_corners_feathering_blends_partial():
    """Test that corner interpolation feathering blends correctly at partial distance."""
    corners = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    coords = np.array([[[1.25, 0.0]]], dtype=np.float64)  # 0.25 past right edge

    result = lerp_from_corners_full_feathered(
        corners,
        coords,
        border_mode=BORDER_CONSTANT,
        border_constant=200.0,
        border_feathering=0.5,
    )

    # extra distance is 0.25, feather=0.5 -> blend_factor=0.5
    expected = np.array([[100.0]], dtype=np.float64)
    assert np.allclose(result, expected)


# =============================================================================
# Exhaustive Feathering Tests for New Kernel
# =============================================================================

class TestFeatheringLineInterp:
    """Exhaustive tests for line interpolation feathering."""

    def test_no_feathering_returns_border_value(self):
        """Without feathering, OOB coords should return edge value (clamped)."""
        line0 = np.array([10.0, 20.0], dtype=np.float64)
        line1 = np.array([30.0, 40.0], dtype=np.float64)
        coords = np.array([[[1.5, 0.5]]], dtype=np.float64)  # OOB in x

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CLAMP,
            border_constant=0.0,
            border_feathering=0.0,  # No feathering
        )

        # With CLAMP, u_x=1.5 clamps to 1.0, so we get rightmost values blended
        # line0[1]=20, line1[1]=40, u_y=0.5 -> 30
        expected = np.array([[30.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_feathering_with_constant_border_mode(self):
        """CONSTANT border with feathering should blend to border value."""
        line0 = np.array([100.0, 100.0], dtype=np.float64)
        line1 = np.array([100.0, 100.0], dtype=np.float64)
        coords = np.array([[[1.2, 0.5]]], dtype=np.float64)  # 0.2 past edge

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,  # Border value is 0
            border_feathering=0.4,  # 0.2 is 50% of 0.4
        )

        # blend_factor = 0.2 / 0.4 = 0.5
        # edge_val = 100, border_val = 0
        # result = 100 + 0.5 * (0 - 100) = 50
        expected = np.array([[50.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_feathering_multiple_coords(self):
        """Test feathering with multiple coordinate points."""
        line0 = np.zeros(3, dtype=np.float64)
        line1 = np.zeros(3, dtype=np.float64)
        # Different distances from edge
        coords = np.array([
            [[1.1, 0.5], [1.2, 0.5], [1.5, 0.5]],  # 0.1, 0.2, 0.5 past edge
        ], dtype=np.float64)

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=100.0,
            border_feathering=0.5,
            distance_mode=DIST_MAX_NORM,
        )

        # blend factors: 0.1/0.5=0.2, 0.2/0.5=0.4, 0.5/0.5=1.0
        # Results: 0 + 0.2*(100-0)=20, 0 + 0.4*(100-0)=40, 0 + 1.0*(100-0)=100
        expected = np.array([[20.0, 40.0, 100.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_feathering_negative_coords(self):
        """Test feathering with coordinates past the left/top edge."""
        line0 = np.array([50.0, 50.0], dtype=np.float64)
        line1 = np.array([50.0, 50.0], dtype=np.float64)
        coords = np.array([[[-0.25, 0.5]]], dtype=np.float64)  # 0.25 before left edge

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=0.5,
            distance_mode=DIST_MAX_NORM,
        )

        # extra = 0.25, feather = 0.5, blend = 0.5
        # result = 50 + 0.5 * (0 - 50) = 25
        expected = np.array([[25.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_feathering_y_direction(self):
        """Test feathering when OOB in y direction only."""
        line0 = np.array([100.0, 100.0], dtype=np.float64)
        line1 = np.array([100.0, 100.0], dtype=np.float64)
        coords = np.array([[[0.5, 1.3]]], dtype=np.float64)  # 0.3 past bottom edge

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=0.6,
            distance_mode=DIST_MAX_NORM,
        )

        # extra = 0.3, feather = 0.6, blend = 0.5
        # result = 100 + 0.5 * (0 - 100) = 50
        expected = np.array([[50.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_feathering_both_directions(self):
        """Test feathering when OOB in both x and y."""
        line0 = np.array([100.0, 100.0], dtype=np.float64)
        line1 = np.array([100.0, 100.0], dtype=np.float64)
        coords = np.array([[[1.2, 1.2]]], dtype=np.float64)  # 0.2 past both edges

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=0.4,
            distance_mode=DIST_MAX_NORM,  # max(0.2, 0.2) = 0.2
        )

        # With MAX_NORM, extra = max(0.2, 0.2) = 0.2
        # blend = 0.2 / 0.4 = 0.5
        # result = 100 + 0.5 * (0 - 100) = 50
        expected = np.array([[50.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_multichannel_feathering(self):
        """Test feathering with multi-channel lines."""
        line0 = np.array([[100.0, 200.0], [100.0, 200.0]], dtype=np.float64)
        line1 = np.array([[100.0, 200.0], [100.0, 200.0]], dtype=np.float64)
        coords = np.array([[[1.25, 0.5]]], dtype=np.float64)  # 0.25 past edge

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,  # Applies to all channels
            border_feathering=0.5,
            distance_mode=DIST_MAX_NORM,
        )

        # blend = 0.25 / 0.5 = 0.5
        # ch0: 100 + 0.5 * (0 - 100) = 50
        # ch1: 200 + 0.5 * (0 - 200) = 100
        expected = np.array([[[50.0, 100.0]]], dtype=np.float64)
        assert np.allclose(result, expected)


class TestFeatheringCornerInterp:
    """Exhaustive tests for corner interpolation feathering."""

    def test_no_feathering_clamps_to_edge(self):
        """Without feathering, OOB coords clamp to edge values."""
        corners = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)  # TL, TR, BL, BR
        coords = np.array([[[1.5, 0.5]]], dtype=np.float64)

        result = lerp_from_corners_full_feathered(
            corners, coords,
            border_mode=BORDER_CLAMP,
            border_constant=0.0,
            border_feathering=0.0,
        )

        # Clamped to u_x=1.0, u_y=0.5: top_right=20, bottom_right=40, blend=30
        expected = np.array([[30.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_feathering_corner_gradient(self):
        """Test feathering with a non-uniform corner gradient."""
        corners = np.array([0.0, 100.0, 100.0, 200.0], dtype=np.float64)
        coords = np.array([[[0.5, 1.2]]], dtype=np.float64)  # 0.2 past bottom

        result = lerp_from_corners_full_feathered(
            corners, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=0.4,
        )

        # At (0.5, 1.0): top blend = 50, bottom blend = 150, vertical blend = 150
        # extra = 0.2, feather = 0.4, blend_factor = 0.5
        # result = 150 + 0.5 * (0 - 150) = 75
        expected = np.array([[75.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_feathering_all_corners_same(self):
        """Test with all corners having the same value."""
        corners = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float64)
        coords = np.array([[[1.1, 0.5]]], dtype=np.float64)

        result = lerp_from_corners_full_feathered(
            corners, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=100.0,
            border_feathering=0.2,
        )

        # extra = 0.1, feather = 0.2, blend = 0.5
        # result = 50 + 0.5 * (100 - 50) = 75
        expected = np.array([[75.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_feathering_corner_at_boundary(self):
        """Test feathering exactly at the boundary."""
        corners = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        coords = np.array([[[1.0, 0.5]]], dtype=np.float64)  # Exactly at edge

        result = lerp_from_corners_full_feathered(
            corners, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=0.5,
        )

        # At boundary, no extra distance, so result = interpolated value = 100
        expected = np.array([[100.0]], dtype=np.float64)
        assert np.allclose(result, expected)


class TestDistanceModes:
    """Tests for different distance calculation modes in feathering."""

    def test_euclidean_distance_mode(self):
        """Test EUCLIDEAN distance mode for 2D coordinates."""
        line0 = np.array([100.0, 100.0], dtype=np.float64)
        line1 = np.array([100.0, 100.0], dtype=np.float64)
        # 0.3 past x, 0.4 past y -> euclidean = sqrt(0.3^2 + 0.4^2) = 0.5
        coords = np.array([[[1.3, 1.4]]], dtype=np.float64)

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=1.0,
            distance_mode=DIST_EUCLIDEAN,
        )

        # euclidean distance = 0.5, feather = 1.0, blend = 0.5
        # result = 100 + 0.5 * (0 - 100) = 50
        expected = np.array([[50.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_manhattan_distance_mode(self):
        """Test MANHATTAN distance mode (sum of absolute distances)."""
        line0 = np.array([100.0, 100.0], dtype=np.float64)
        line1 = np.array([100.0, 100.0], dtype=np.float64)
        # 0.2 past x, 0.3 past y -> manhattan = 0.2 + 0.3 = 0.5
        coords = np.array([[[1.2, 1.3]]], dtype=np.float64)

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=1.0,
            distance_mode=DIST_MANHATTAN,
        )

        # manhattan distance = 0.5, feather = 1.0, blend = 0.5
        # result = 100 + 0.5 * (0 - 100) = 50
        expected = np.array([[50.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_alpha_max_distance_mode(self):
        """Test ALPHA_MAX distance mode (smooth approximation of max)."""
        line0 = np.array([100.0, 100.0], dtype=np.float64)
        line1 = np.array([100.0, 100.0], dtype=np.float64)
        # Asymmetric distances to see effect
        coords = np.array([[[1.1, 1.4]]], dtype=np.float64)  # 0.1, 0.4 past edges

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=0.8,
            distance_mode=DIST_ALPHA_MAX,
        )

        # ALPHA_MAX gives smooth approximation, result should be between 0 and 100
        assert 0 <= result[0, 0] <= 100
        # Should be closer to border value (0) since we're significantly OOB
        assert result[0, 0] < 60


class TestEdgeCases:
    """Edge case tests for feathering."""

    def test_zero_feathering_with_oob_coords(self):
        """Zero feathering should give hard edge transition."""
        line0 = np.array([100.0, 100.0], dtype=np.float64)
        line1 = np.array([100.0, 100.0], dtype=np.float64)
        coords = np.array([[[1.001, 0.5]]], dtype=np.float64)  # Just barely OOB

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=0.0,  # No feathering
        )

        # With zero feathering, any OOB should give full border
        expected = np.array([[0.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_very_large_feathering(self):
        """Large feathering value should still work correctly."""
        line0 = np.array([100.0, 100.0], dtype=np.float64)
        line1 = np.array([100.0, 100.0], dtype=np.float64)
        coords = np.array([[[1.1, 0.5]]], dtype=np.float64)  # 0.1 past edge

        result = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=0.0,
            border_feathering=10.0,  # Very large feather
        )

        # blend = 0.1 / 10.0 = 0.01
        # result = 100 + 0.01 * (0 - 100) = 99
        expected = np.array([[99.0]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_inbounds_coords_unaffected_by_feathering(self):
        """In-bounds coordinates should be unaffected by feathering setting."""
        line0 = np.array([0.0, 100.0], dtype=np.float64)
        line1 = np.array([200.0, 300.0], dtype=np.float64)
        coords = np.array([[[0.5, 0.5]]], dtype=np.float64)

        result_no_feather = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=999.0,
            border_feathering=0.0,
        )

        result_with_feather = lerp_between_lines_full_feathered(
            line0, line1, coords,
            border_mode=BORDER_CONSTANT,
            border_constant=999.0,
            border_feathering=1.0,
        )

        # Both should give the same result for in-bounds coords
        assert np.allclose(result_no_feather, result_with_feather)
        # At (0.5, 0.5): line0 blend = 50, line1 blend = 250, result = 150
        expected = np.array([[150.0]], dtype=np.float64)
        assert np.allclose(result_no_feather, expected)