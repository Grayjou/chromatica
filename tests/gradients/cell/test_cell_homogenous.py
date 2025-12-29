from ....chromatica.gradients.gradient2dv2.cell import (
    get_transformed_corners_cell,
    get_transformed_lines_cell,





    get_transformed_corners_cell_dual,
    CornersCellDual

)
from ....chromatica.types.color_types import HueMode, ColorSpace
from ....chromatica.types.format_type import FormatType
from ....chromatica.gradients.gradient2dv2 import LineInterpMethods, PartitionInterval as PInt, PerpendicularDualPartition
import numpy as np
from unitfield import upbm_2d
from boundednumbers import BoundType
from ....chromatica.gradients.gradient2dv2.partitions import PerpendicularPartition, CellDualPartitionInterval
from ....chromatica.gradients.gradient1dv2.gradient_1dv2 import Gradient1D

def test_init_corners_cell():
    top_left = np.array([1.0, 0, 0])
    top_right = np.array([0, 1.0, 0])
    bottom_left = np.array([0, 0, 1.0])
    bottom_right = np.array([1.0, 1, 0])
    cell = get_transformed_corners_cell(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        color_space=ColorSpace.RGB,
        per_channel_coords=upbm_2d(width=2, height=2),
        input_format=FormatType.FLOAT,
    )

def test_homgoenous_cells_renders():
    """Test that homogeneous cells render correctly."""
    top_left = np.array([0.5, 0.5, 0.5])
    top_right = np.array([0.5, 0.5, 0.5])
    bottom_left = np.array([0.5, 0.5, 0.5])
    bottom_right = np.array([0.5, 0.5, 0.5])
    cell = get_transformed_corners_cell(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        color_space=ColorSpace.RGB,
        per_channel_coords=upbm_2d(width=3, height=3),
        input_format=FormatType.FLOAT,
    )
    value = cell.get_value()
    expected_value = np.full((3, 3, 3), 0.5)
    assert np.allclose(value, expected_value)

def test_homog_cell_equal_channels():
    top_left = np.array([0.2, 0.2, 0.2])
    top_right = np.array([0.8, 0.8, 0.8])
    bottom_left = np.array([0.4, 0.4, 0.4])
    bottom_right = np.array([0.6, 0.6, 0.6])
    cell = get_transformed_corners_cell(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        color_space=ColorSpace.RGB,
        per_channel_coords=upbm_2d(width=3, height=3),
        input_format=FormatType.FLOAT,
    )
    value = cell.get_value()
    red = value[..., 0]
    green = value[..., 1]
    blue = value[..., 2]
    assert np.allclose(red, green)
    assert np.allclose(red, blue)
    assert np.allclose(green, blue)


def test_init_corners_dual_cell():
    top_left = np.array([1.0, 0, 0])
    top_right = np.array([0, 1.0, 0])
    bottom_left = np.array([0, 0, 1.0])
    bottom_right = np.array([1.0, 1, 0])
    cell = get_transformed_corners_cell_dual(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        vertical_color_space=ColorSpace.RGB,
        horizontal_color_space=ColorSpace.HSV,
        per_channel_coords=upbm_2d(width=2, height=2),
        input_format=FormatType.FLOAT,
    )
    
def test_homogenous_dual_cell_renders():
    """Test that homogeneous dual cells render correctly."""
    top_left = np.array([0.5, 0.5, 0.5])
    top_right = np.array([0.5, 0.5, 0.5])
    bottom_left = np.array([0.5, 0.5, 0.5])
    bottom_right = np.array([0.5, 0.5, 0.5])
    cell = get_transformed_corners_cell_dual(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        vertical_color_space=ColorSpace.RGB,
        horizontal_color_space=ColorSpace.RGB,
        per_channel_coords=upbm_2d(width=3, height=3),
        input_format=FormatType.FLOAT,
    )
    value = cell.get_value()
    expected_value = np.full((3, 3, 3), 0.5)
    assert np.allclose(value, expected_value)

def test_homog_dual_cell_equal_channels():
    top_left = np.array([0.2, 0.2, 0.2])
    top_right = np.array([0.8, 0.8, 0.8])
    bottom_left = np.array([0.4, 0.4, 0.4])
    bottom_right = np.array([0.6, 0.6, 0.6])
    cell = get_transformed_corners_cell_dual(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        vertical_color_space=ColorSpace.HSV,
        horizontal_color_space=ColorSpace.HSV,
        hue_direction_y=HueMode.SHORTEST,
        #per_channel_coords=[upbm_2d(width=3, height=3)]*3,
        per_channel_coords=upbm_2d(width=3, height=3),
        input_format=FormatType.FLOAT,
    )
    value = cell.get_value()
    red = value[..., 0]
    green = value[..., 1]
    blue = value[..., 2]
    assert np.allclose(red, green)
    assert np.allclose(red, blue)
    assert np.allclose(green, blue)
    cell = get_transformed_corners_cell_dual(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        vertical_color_space=ColorSpace.HSV,
        horizontal_color_space=ColorSpace.HSV,
        hue_direction_y=HueMode.SHORTEST,
        hue_direction_x=HueMode.SHORTEST,
        #per_channel_coords=[upbm_2d(width=3, height=3)]*3,
        per_channel_coords=upbm_2d(width=3, height=3),
        input_format=FormatType.FLOAT,
    )
    value = cell.get_value()
    red = value[..., 0]
    green = value[..., 1]
    blue = value[..., 2]
    assert np.allclose(red, green)
    assert np.allclose(red, blue)
    assert np.allclose(green, blue)

def test_init_lines_cell():
    line_top = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    line_bottom = np.array([[0, 0, 1.0], [1.0, 1, 0]])
    cell = get_transformed_lines_cell(
        top_line=line_top,
        bottom_line=line_bottom,
        color_space=ColorSpace.RGB,
        per_channel_coords=upbm_2d(width=2, height=2),
        input_format=FormatType.FLOAT,
    )
def test_homogenous_lines_cell_renders():
    """Test that homogeneous lines cells render correctly."""
    line_top = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    line_bottom = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    cell = get_transformed_lines_cell(
        top_line=line_top,
        bottom_line=line_bottom,
        color_space=ColorSpace.RGB,
        per_channel_coords=upbm_2d(width=3, height=3),
        input_format=FormatType.FLOAT,
    )
    value = cell.get_value()
    expected_value = np.full((3, 3, 3), 0.5)
    assert np.allclose(value, expected_value)