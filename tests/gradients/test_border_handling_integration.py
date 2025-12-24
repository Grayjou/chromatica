"""
Tests for border handling propagation from gradient cell factories to interpolation core.

This test suite verifies that border_mode and border_value parameters are correctly
passed through the entire pipeline from cell factory functions down to the interpolation core.
"""

import pytest
import numpy as np

try:
    from ...chromatica.types.color_types import ColorSpace
    from ...chromatica.gradients.gradient2dv2.cell.factory import (
        get_transformed_lines_cell,
        get_transformed_corners_cell,
    )
    from ...chromatica.v2core import (
        BORDER_REPEAT,
        BORDER_MIRROR,
        BORDER_CONSTANT,
        BORDER_CLAMP,
        BORDER_OVERFLOW,
    )
    IMPORTS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE,
    reason=f"Required modules not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}"
)


class TestBorderHandlingPropagation:
    """Test that border parameters are correctly propagated through the API."""

    def test_lines_cell_factory_accepts_border_params(self):
        """Test that get_transformed_lines_cell accepts border_mode and border_value."""
        # Create simple test data
        top_line = np.array([[255, 0, 0], [255, 255, 0]]).astype(np.uint8)  # Red to Yellow
        bottom_line = np.array([[0, 0, 255], [0, 255, 0]]).astype(np.uint8)  # Blue to Green

        # Create simple coordinate grid
        H, W = 10, 10
        ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Create cell with border parameters
        cell = get_transformed_lines_cell(
            top_line=top_line,
            bottom_line=bottom_line,
            per_channel_coords=per_channel_coords,
            color_space='rgb',
            border_mode=BORDER_CLAMP,
            border_value=0.5,
        )
        
        # Verify parameters are stored
        assert cell.border_mode == BORDER_CLAMP
        assert cell.border_value == 0.5

    def test_corners_cell_factory_accepts_border_params(self):
        """Test that get_transformed_corners_cell accepts border_mode and border_value."""
        # Create corner colors
        top_left = np.array([255, 0, 0])  # Red
        top_right = np.array([255, 255, 0])  # Yellow
        bottom_left = np.array([0, 0, 255])  # Blue
        bottom_right = np.array([0, 255, 0])  # Green
        
        # Create simple coordinate grid
        H, W = 10, 10
        ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Create cell with border parameters
        cell = get_transformed_corners_cell(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
            per_channel_coords=per_channel_coords,
            color_space='rgb',
            border_mode=BORDER_REPEAT,
            border_value=0.0,
        )
        
        # Verify parameters are stored
        assert cell.border_mode == BORDER_REPEAT
        assert cell.border_value == 0.0

    def test_lines_cell_renders_with_border_params(self):
        """Test that LinesCell properly renders with border parameters."""
        # Create simple test data
        top_line = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])  # Red to Yellow
        bottom_line = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # Blue to Green
        
        # Create coordinate grid with some out-of-bounds values
        H, W = 5, 5
        ux, uy = np.meshgrid(np.linspace(-0.2, 1.2, W), np.linspace(-0.2, 1.2, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Create cell with CLAMP border mode
        cell = get_transformed_lines_cell(
            top_line=top_line,
            bottom_line=bottom_line,
            per_channel_coords=per_channel_coords,
            color_space='rgb',
            border_mode=BORDER_CLAMP,
            border_value=None,
            input_format='float'
        )
        
        # Render should not raise an error
        result = cell.get_value()
        
        # Result should have expected shape
        assert result.shape == (H, W, 3)
        
        # All values should be valid (no NaN or inf)
        assert np.all(np.isfinite(result))

    def test_corners_cell_renders_with_border_params(self):
        """Test that CornersCell properly renders with border parameters."""
        # Create corner colors (normalized floats)
        top_left = np.array([1.0, 0.0, 0.0])  # Red
        top_right = np.array([1.0, 1.0, 0.0])  # Yellow
        bottom_left = np.array([0.0, 0.0, 1.0])  # Blue
        bottom_right = np.array([0.0, 1.0, 0.0])  # Green
        
        # Create coordinate grid with some out-of-bounds values
        H, W = 5, 5
        ux, uy = np.meshgrid(np.linspace(-0.1, 1.1, W), np.linspace(-0.1, 1.1, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Create cell with CLAMP border mode
        cell = get_transformed_corners_cell(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
            per_channel_coords=per_channel_coords,
            color_space='rgb',
            border_mode=BORDER_CLAMP,
            border_value=None,
            input_format='float'
        )
        
        # Render should not raise an error
        result = cell.get_value()
        
        # Result should have expected shape
        assert result.shape == (H, W, 3)
        
        # All values should be valid (no NaN or inf)
        assert np.all(np.isfinite(result))

    def test_border_params_preserved_through_conversion(self):
        """Test that border parameters are preserved when converting color spaces."""
        # Create simple test data
        top_line = np.array([[255, 0, 0], [255, 255, 0]])
        bottom_line = np.array([[0, 0, 255], [0, 255, 0]])
        
        H, W = 5, 5
        ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Create cell with border parameters
        cell_rgb = get_transformed_lines_cell(
            top_line=top_line,
            bottom_line=bottom_line,
            per_channel_coords=per_channel_coords,
            color_space='rgb',
            border_mode=BORDER_REPEAT,
            border_value=0.75,
        )
        
        # Convert to HSV
        cell_hsv = cell_rgb.convert_to_space("hsv")
        
        # Border parameters should be preserved
        assert cell_hsv.border_mode == BORDER_REPEAT
        assert cell_hsv.border_value == 0.75

    def test_default_border_params_are_none(self):
        """Test that border parameters default to None when not specified."""
        # Create simple test data
        top_line = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        bottom_line = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        
        H, W = 5, 5
        ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Create cell without specifying border parameters
        cell = get_transformed_lines_cell(
            top_line=top_line,
            bottom_line=bottom_line,
            per_channel_coords=per_channel_coords,
            color_space='rgb',
            input_format='float'
        )
        
        # Border parameters should be None by default
        assert cell.border_mode is None
        assert cell.border_value is None

    def test_lines_cell_with_different_border_modes(self):
        """Test that different border modes can be specified for LinesCell."""
        top_line = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        bottom_line = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        
        H, W = 5, 5
        ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Test each border mode
        for mode in [BORDER_REPEAT, BORDER_MIRROR, BORDER_CONSTANT, BORDER_CLAMP, BORDER_OVERFLOW]:
            cell = get_transformed_lines_cell(
                top_line=top_line,
                bottom_line=bottom_line,
                per_channel_coords=per_channel_coords,
                color_space='rgb',
                border_mode=mode,
                border_value=0.5,
                input_format='float'
            )
            
            assert cell.border_mode == mode
            
            # Cell should render without error
            result = cell.get_value()
            assert result.shape == (H, W, 3)

    def test_color_space_inference_with_border_params(self):
        """Test that color space inference works together with border parameters."""
        # Create simple test data
        top_line = np.array([[255, 0, 0], [255, 255, 0]])
        bottom_line = np.array([[0, 0, 255], [0, 255, 0]])
        
        H, W = 5, 5
        ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Create cell without specifying line color spaces (should default to target)
        # Also specify border parameters
        cell = get_transformed_lines_cell(
            top_line=top_line,
            bottom_line=bottom_line,
            per_channel_coords=per_channel_coords,
            color_space='rgb',
            # Not specifying top_line_color_space or bottom_line_color_space
            border_mode=BORDER_CLAMP,
            border_value=0.0,
        )
        
        # Should work without error
        result = cell.get_value()
        assert result.shape == (H, W, 3)
        assert cell.border_mode == BORDER_CLAMP


class TestColorSpaceInference:
    """Test that color space inference works correctly in factory functions."""

    def test_lines_cell_defaults_to_target_color_space(self):
        """Test that line color spaces default to target color space."""
        top_line = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        bottom_line = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        
        H, W = 5, 5
        ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Create cell without specifying line color spaces
        cell = get_transformed_lines_cell(
            top_line=top_line,
            bottom_line=bottom_line,
            per_channel_coords=per_channel_coords,
            color_space='rgb',
            input_format='float',
            # top_line_color_space and bottom_line_color_space not specified
        )
        
        # Should render successfully
        result = cell.get_value()
        assert result.shape == (H, W, 3)
        assert cell.color_space == 'rgb'

    def test_corners_cell_defaults_to_target_color_space(self):
        """Test that corner color spaces default to target color space."""
        top_left = np.array([1.0, 0.0, 0.0])
        top_right = np.array([1.0, 1.0, 0.0])
        bottom_left = np.array([0.0, 0.0, 1.0])
        bottom_right = np.array([0.0, 1.0, 0.0])
        
        H, W = 5, 5
        ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        per_channel_coords = [np.stack([ux, uy], axis=-1)] * 3
        
        # Create cell without specifying corner color spaces
        cell = get_transformed_corners_cell(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
            per_channel_coords=per_channel_coords,
            color_space='rgb',
            input_format='float',
            # Corner color spaces not specified
        )
        
        # Should render successfully
        result = cell.get_value()
        assert result.shape == (H, W, 3)
        assert cell.color_space == 'rgb'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
