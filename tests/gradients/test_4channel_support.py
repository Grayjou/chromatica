"""
Tests for 4-channel color space support (RGBA, HSVA) in segments and cells.
"""
import numpy as np
from ...chromatica.gradients.gradient1dv2.segment import get_transformed_segment, UniformGradientSegment
from ...chromatica.gradients.gradient2dv2.cell import (
    get_transformed_corners_cell,
    get_transformed_lines_cell,

)
from ...chromatica.gradients.gradient2dv2 import LineInterpMethods
from ...chromatica.types.color_types import HueMode
from boundednumbers import BoundType
from unitfield import upbm_2d


def test_rgba_segment_uniform():
    """Test that RGBA segments work with uniform coordinates."""
    start_color = np.array([1.0, 0.0, 0.0, 1.0])  # Red, fully opaque
    end_color = np.array([0.0, 0.0, 1.0, 0.5])    # Blue, semi-transparent
    
    u = np.linspace(0, 1, 10)
    
    segment = get_transformed_segment(
        already_converted_start_color=start_color,
        already_converted_end_color=end_color,
        per_channel_coords=[u],
        color_space="rgba",
        homogeneous_per_channel_coords=True,
    )
    
    assert isinstance(segment, UniformGradientSegment)
    result = segment.get_value()
    
    assert result.shape == (10, 4)
    # Check first value is start color
    assert np.allclose(result[0], start_color, atol=1e-6)
    # Check last value is end color
    assert np.allclose(result[-1], end_color, atol=1e-6)
    # Check alpha channel interpolates
    assert np.allclose(result[5, 3], 0.75, atol=0.05)  # Mid-point alpha


def test_rgba_segment_per_channel():
    """Test that RGBA segments work with per-channel coordinates."""
    start_color = np.array([1.0, 0.0, 0.0, 1.0])
    end_color = np.array([0.0, 1.0, 1.0, 0.0])  # Changed: blue channel now ends at 1.0
    
    # Different u for each channel
    u_r = np.linspace(0, 1, 5)
    u_g = np.linspace(0, 0.5, 5)
    u_b = np.linspace(0.5, 1, 5)
    u_a = np.linspace(0, 1, 5)
    
    segment = get_transformed_segment(
        already_converted_start_color=start_color,
        already_converted_end_color=end_color,
        per_channel_coords=[u_r, u_g, u_b, u_a],
        color_space="rgba",
        homogeneous_per_channel_coords=False,
    )
    
    result = segment.get_value()
    
    assert result.shape == (5, 4)
    # Red channel should fully interpolate
    assert np.allclose(result[-1, 0], 0.0, atol=1e-6)
    # Green channel should half interpolate (from 0 to 1, at u=0.5 -> 0.5)
    assert np.allclose(result[-1, 1], 0.5, atol=0.05)
    # Blue channel should go from middle to end (u_b from 0.5 to 1.0 maps to value 0.5 to 1.0)
    assert result[-1, 2] > 0.4  # At u_b=1.0, should be near 1.0


def test_hsva_segment():
    """Test that HSVA segments work correctly."""
    start_color = np.array([0.0, 1.0, 1.0, 1.0])    # Red in HSV
    end_color = np.array([120.0, 1.0, 1.0, 0.5])    # Green in HSV
    
    u = np.linspace(0, 1, 10)
    
    segment = get_transformed_segment(
        already_converted_start_color=start_color,
        already_converted_end_color=end_color,
        per_channel_coords=[u],
        color_space="hsva",
        hue_direction="shortest",
        homogeneous_per_channel_coords=True,
    )
    
    result = segment.get_value()
    
    assert result.shape == (10, 4)
    # Check alpha interpolates
    assert np.allclose(result[0, 3], 1.0, atol=1e-6)
    assert np.allclose(result[-1, 3], 0.5, atol=1e-6)
    # Check hue interpolates (should go through 60 degrees for yellow)
    assert result[5, 0] > 50 and result[5, 0] < 70


def test_rgba_lines_cell():
    """Test that RGBA works with LinesCell."""
    top_line = np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.int32)
    bottom_line = np.array([[0, 0, 255, 128], [255, 255, 0, 128]], dtype=np.int32)
    
    H, W = 3, 4
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(4)]
    
    cell = get_transformed_lines_cell(
        top_line=top_line,
        bottom_line=bottom_line,
        per_channel_coords=per_channel_coords,
        top_line_color_space="rgba",
        bottom_line_color_space="rgba",
        color_space="rgba",
        input_format="int",
        hue_direction_y=None,
        line_method=LineInterpMethods.LINES_CONTINUOUS,
        boundtypes=BoundType.CLAMP
    )
    
    result = cell.get_value()
    
    assert result.shape == (H, W, 4)
    # Check alpha channel is present and reasonable
    assert np.all(result[:, :, 3] >= 0.0)
    assert np.all(result[:, :, 3] <= 1.0)


def test_rgba_corners_cell():
    """Test that RGBA works with CornersCell."""
    top_left = np.array([255, 0, 0, 255], dtype=np.int32)
    top_right = np.array([0, 255, 0, 255], dtype=np.int32)
    bottom_left = np.array([0, 0, 255, 128], dtype=np.int32)
    bottom_right = np.array([255, 255, 0, 128], dtype=np.int32)
    
    H, W = 4, 4
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(4)]
    
    cell = get_transformed_corners_cell(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        per_channel_coords=per_channel_coords,
        top_left_color_space="rgba",
        top_right_color_space="rgba",
        bottom_left_color_space="rgba",
        bottom_right_color_space="rgba",
        color_space="rgba",
        hue_direction_y=None,
        hue_direction_x=None,
        boundtypes=BoundType.CLAMP
    )
    
    result = cell.get_value()
    
    assert result.shape == (H, W, 4)
    # Check corners
    assert np.allclose(result[0, 0], top_left / 255, atol=0.01)
    assert np.allclose(result[0, -1], top_right / 255, atol=0.01)
    assert np.allclose(result[-1, 0], bottom_left / 255, atol=0.01)
    assert np.allclose(result[-1, -1], bottom_right / 255, atol=0.01)


def test_hsva_corners_cell():
    """Test that HSVA works with CornersCell and mixed hue modes."""
    # Red, Green, Blue, Yellow in HSV with varying alpha
    top_left = np.array([0, 255, 255, 255], dtype=np.int32)      # Red
    top_right = np.array([120, 255, 255, 255], dtype=np.int32)   # Green
    bottom_left = np.array([240, 255, 255, 128], dtype=np.int32) # Blue
    bottom_right = np.array([60, 255, 255, 128], dtype=np.int32) # Yellow
    
    H, W = 4, 4
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(4)]
    
    cell = get_transformed_corners_cell(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        per_channel_coords=per_channel_coords,
        top_left_color_space="hsva",
        top_right_color_space="hsva",
        bottom_left_color_space="hsva",
        bottom_right_color_space="hsva",
        color_space="hsva",
        hue_direction_y=HueMode.CW,
        hue_direction_x=HueMode.CCW,
        boundtypes=BoundType.CLAMP
    )
    
    result = cell.get_value()
    
    assert result.shape == (H, W, 4)
    # Check alpha channel interpolates correctly
    assert np.allclose(result[0, 0, 3], 1.0, atol=0.01)
    assert np.allclose(result[-1, -1, 3], 128/255, atol=0.01)
    # Check hue values are within valid range
    assert np.all(result[:, :, 0] >= 0.0)
    assert np.all(result[:, :, 0] <= 360.0)
