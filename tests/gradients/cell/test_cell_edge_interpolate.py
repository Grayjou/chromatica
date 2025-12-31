
from ....chromatica.gradients.gradient2dv2.cell import (
    get_transformed_corners_cell,
    get_transformed_lines_cell,





    get_transformed_corners_cell_dual,

    CornersCellDual

)
from ....chromatica.types.color_types import HueMode
from ....chromatica.gradients.gradient2dv2 import LineInterpMethods
import numpy as np
from unitfield import upbm_2d
from boundednumbers import BoundType


def test_interpolate_edge_corners():
    total = 255
    top_left = np.array([0, 0, 0], dtype=np.int32)
    top_right = np.array([total, 0, 0], dtype=np.int32)
    bottom_left = np.array([0, total, 0], dtype=np.int32)
    bottom_right = np.array([0, 0, total], dtype=np.int32)
    
    H, W = 2, 3
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]
    cell = get_transformed_corners_cell(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        per_channel_coords=per_channel_coords,
        top_left_color_space="rgb",
        top_right_color_space="rgb",
        bottom_left_color_space="rgb",
        bottom_right_color_space="rgb",
        color_space="rgb",
        hue_direction_x=HueMode.CCW,
        hue_direction_y=HueMode.CW,
        boundtypes=BoundType.CLAMP
    )
    top_edge = cell.interpolate_edge(0.5, is_top_edge=True)
    bottom_edge = cell.interpolate_edge(0.5, is_top_edge=False)
    assert np.allclose(top_edge, np.array([0.5, 0, 0]))
    assert np.allclose(bottom_edge, np.array([0, 0.5, 0.5]))
    one_third_top_edge = cell.interpolate_edge(1/3, is_top_edge=True)
    two_third_bottom_edge = cell.interpolate_edge(2/3, is_top_edge=False)
    assert np.allclose(one_third_top_edge, np.array([1/3, 0, 0]))
    assert np.allclose(two_third_bottom_edge, np.array([0, 1/3, 2/3]))


def test_interpolate_edge_corners():
    total = 255
    top_left = np.array([0, 0, 0], dtype=np.int32)
    top_right = np.array([total, 0, 0], dtype=np.int32)
    bottom_left = np.array([0, total, 0], dtype=np.int32)
    bottom_right = np.array([0, 0, total], dtype=np.int32)
    
    H, W = 2, 3
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]
    cell = get_transformed_corners_cell(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        per_channel_coords=per_channel_coords,
        top_left_color_space="rgb",
        top_right_color_space="rgb",
        bottom_left_color_space="rgb",
        bottom_right_color_space="rgb",
        color_space="rgb",
        hue_direction_x=HueMode.CCW,
        hue_direction_y=HueMode.CW,
        boundtypes=BoundType.CLAMP
    )
    top_edge = cell.interpolate_edge(0.5, is_top_edge=True)
    bottom_edge = cell.interpolate_edge(0.5, is_top_edge=False)
    assert np.allclose(top_edge, np.array([0.5, 0, 0]))
    assert np.allclose(bottom_edge, np.array([0, 0.5, 0.5]))
    one_third_top_edge = cell.interpolate_edge(1/3, is_top_edge=True)
    two_third_bottom_edge = cell.interpolate_edge(2/3, is_top_edge=False)
    assert np.allclose(one_third_top_edge, np.array([1/3, 0, 0]))
    assert np.allclose(two_third_bottom_edge, np.array([0, 1/3, 2/3]))

def test_cell_lines_interpolation_discrete():
    """Test LinesCell interpolation with discrete method."""
    L = 6
    total = 100
    inc = int(total / L)
    top_line = np.array([[i * inc + j for j in range(3)] for i in range(L)], dtype=np.int32)
    bottom_line = np.array([[i * inc * 2 + j for j in range(3)] for i in range(L)], dtype=np.int32)
    
    H, W = 4, 12
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]
    cell = get_transformed_lines_cell(
        top_line=top_line,
        bottom_line=bottom_line,
        per_channel_coords=per_channel_coords,
        top_line_color_space="rgb",
        bottom_line_color_space="rgb",
        color_space="rgb",
        input_format="int",
        line_method=LineInterpMethods.LINES_DISCRETE,
        boundtypes=BoundType.CLAMP
    )
    
    result = cell.get_value()

    assert result.shape == (H, W, 3)
    # Check values are within expected range
    assert np.all(result >= np.min(top_line)) and np.all(result <= np.max(bottom_line))
    print("top_line:" , top_line)
    print("bottom_line:" , bottom_line)
    print(result*255)
    assert int(result[0][0][0]*255) == 0
    assert int(result[0][1][0]*255) == 0 # No interp between discrete steps
    assert int(result[0][2][0]*255) == inc 


def test_interpolate_edge_corners_dual():
    total = 255
    top_left = np.array([total, 0, total], dtype=np.int32) #Hue = 300
    top_right = np.array([total, 0, 0], dtype=np.int32) # Hue = 0
    bottom_left = np.array([0, total, 0], dtype=np.int32) # Hue = 120
    bottom_right = np.array([0, 0, total], dtype=np.int32) # Hue = 240 = -120

    W = 5
    H = 2
    pcc = [upbm_2d(width=W, height=H) for _ in range(3)]

    cell:CornersCellDual = get_transformed_corners_cell_dual(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        per_channel_coords=pcc,
        horizontal_color_space="hsv",
        vertical_color_space="hsv",
        top_left_color_space="rgb",
        top_right_color_space="rgb",
        bottom_left_color_space="rgb",
        bottom_right_color_space="rgb",

        hue_direction_x=HueMode.CCW,
        hue_direction_y=HueMode.CW,
    )
    top_edge = cell.interpolate_edge(0.5, is_top_edge=True)
    bottom_edge = cell.interpolate_edge(0.5, is_top_edge=False)

    assert np.allclose(top_edge, np.array([150, 1, 1]))
    assert np.allclose(bottom_edge, np.array([0, 1, 1]))



def test_interpolate_edge_corners_dual_diff_spaces():
    total = 255
    top_left = np.array([total, 0, total], dtype=np.int32) #Hue = 300
    top_right = np.array([total, 0, 0], dtype=np.int32) # Hue = 0
    bottom_left = np.array([0, total, 0], dtype=np.int32) # Hue = 120
    bottom_right = np.array([0, 0, total], dtype=np.int32) # Hue = 240 = -120

    W = 3
    H = 2
    pcc = [upbm_2d(width=W, height=H) for _ in range(3)]

    cell:CornersCellDual = get_transformed_corners_cell_dual(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        per_channel_coords=pcc,
        horizontal_color_space="hsv",
        vertical_color_space="rgb",
        top_left_color_space="rgb",
        top_right_color_space="rgb",
        bottom_left_color_space="rgb",
        bottom_right_color_space="rgb",
        bottom_segment_color_space="rgb",
        top_segment_color_space="hsv",
        #hue_direction_x=HueMode.CCW,
        hue_direction_y=None,
        top_segment_hue_direction_x=HueMode.CW,
        bottom_segment_hue_direction_x=HueMode.CCW,
    )

    top_edge = cell.interpolate_edge(0.5, is_top_edge=True)
    bottom_edge = cell.interpolate_edge(0.5, is_top_edge=False)

    assert np.allclose(top_edge, np.array([330, 1, 1]))
    assert np.allclose(bottom_edge, np.array([0, 0.5, 0.5]))

def test_interpolate_edge_corners_dual_transformed():
    total = 255
    top_left = np.array([total, 0, total], dtype=np.int32) #Hue = 300
    top_right = np.array([total, 0, 0], dtype=np.int32) # Hue = 0
    bottom_left = np.array([0, total, 0], dtype=np.int32) # Hue = 120
    bottom_right = np.array([0, 0, total], dtype=np.int32) # Hue = 240 = -120
    def reverse_x_and_y(coords):
        return np.stack([1 - coords[..., 0], 1 - coords[..., 1]], axis=-1)
    def reverse_x(coords):
        return np.stack([1 - coords[..., 0], coords[..., 1]], axis=-1)
    W = 3
    H = 2
    pcc = [upbm_2d(width=W, height=H) for _ in range(3)]
    pcc = [reverse_x_and_y(coords) for coords in pcc]
    #pcc = [reverse_x(coords) for coords in pcc]
    cell:CornersCellDual = get_transformed_corners_cell_dual(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        per_channel_coords=pcc,
        horizontal_color_space="hsv",
        vertical_color_space="hsv",
        top_left_color_space="rgb",
        top_right_color_space="rgb",
        bottom_left_color_space="rgb",
        bottom_right_color_space="rgb",

        hue_direction_x=HueMode.CCW,
        hue_direction_y=HueMode.CW,
    )
    result = cell.get_value()
    top_edge = cell.interpolate_edge(0.5, is_top_edge=True)
    bottom_edge = cell.interpolate_edge(0.5, is_top_edge=False)
    top_edge_from_result = result[0, W//2]
    bottom_edge_from_result = result[-1, W//2]


    assert np.allclose(top_edge, top_edge_from_result) 
    assert np.allclose(bottom_edge, bottom_edge_from_result)


    assert result.shape == (H, W, 3)

def test_interpolate_edge_lines():
    L = 3
    total = 90
    inc = int(total / L)
    top_line = np.array([[i * inc + j for j in range(3)] for i in range(L)], dtype=np.int32)
    bottom_line = np.array([[i * inc * 3 + j for j in range(3)] for i in range(L)], dtype=np.int32)
    H, W = 2, 3
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]
    cell = get_transformed_lines_cell(
        top_line=top_line,
        bottom_line=bottom_line,
        per_channel_coords=per_channel_coords,
        top_line_color_space="rgb",
        bottom_line_color_space="rgb",
        color_space="rgb",
        input_format="int",
        hue_direction_y=HueMode.CW,
        line_method=LineInterpMethods.LINES_CONTINUOUS,
        boundtypes=BoundType.CLAMP
    )

    first_color_top = np.array([0, 1, 2])
    second_color_top = np.array([30, 31, 32])
    first_color_bottom = np.array([0, 1, 2])
    second_color_bottom = np.array([90, 91, 92])
    test = np.linspace(0, 1, num=5)
    for t in test:
        value = t
        expected_top = first_color_top + value * (second_color_top - first_color_top)
        top_edge_value = cell.interpolate_edge(value/2, is_top_edge=True)
        expected_bottom = first_color_bottom + value * (second_color_bottom - first_color_bottom)
        bottom_edge_value = cell.interpolate_edge(value/2, is_top_edge=False)
        assert np.allclose(bottom_edge_value, expected_bottom / 255)
        assert np.allclose(top_edge_value, expected_top / 255)

def test_interpolate_edge_lines_non_uniform():
    top_line = np.array([
        [100, 50, 50],
        [50, 100, 100],
        [200, 0, 50]
    ])
    bottom_line = np.array([
        [0, 0, 100],
        [100, 200, 50],
        [50, 50, 50]
    ])
    H, W = 2, 3
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]
    cell = get_transformed_lines_cell(
        top_line=top_line,
        bottom_line=bottom_line,
        per_channel_coords=per_channel_coords,
        top_line_color_space="rgb",
        bottom_line_color_space="rgb",
        color_space="rgb",
        input_format="int",
        hue_direction_y=HueMode.CW,
        line_method=LineInterpMethods.LINES_CONTINUOUS,
        boundtypes=BoundType.CLAMP
    )

    top_edge = cell.interpolate_edge(0.25, is_top_edge=True)
    bottom_edge = cell.interpolate_edge(0.25, is_top_edge=False)
    expected_top = np.array([75, 75, 75])
    expected_bottom = np.array([50, 100, 75])
    assert np.allclose(top_edge, expected_top / 255)
    assert np.allclose(bottom_edge, expected_bottom / 255)
    top_edge_75 = cell.interpolate_edge(0.75, is_top_edge=True)
    bottom_edge_75 = cell.interpolate_edge(0.75, is_top_edge=False)
    expected_top_75 = np.array([125,50, 75])
    expected_bottom_75 = np.array([75, 125, 50])
    assert np.allclose(top_edge_75*255, expected_top_75)
    assert np.allclose(bottom_edge_75*255, expected_bottom_75)