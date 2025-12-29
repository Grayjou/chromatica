from ....chromatica.gradients.gradient2dv2.cell import (
    get_transformed_corners_cell,
    get_transformed_lines_cell,





    get_transformed_corners_cell_dual,

    CornersCellDual

)
from ....chromatica.types.color_types import HueMode
from ....chromatica.gradients.gradient2dv2 import LineInterpMethods, PartitionInterval as PInt, PerpendicularDualPartition
import numpy as np
from unitfield import upbm_2d
from boundednumbers import BoundType
from ....chromatica.gradients.gradient2dv2.partitions import PerpendicularPartition, CellDualPartitionInterval
from ....chromatica.gradients.gradient1dv2.gradient_1dv2 import Gradient1D
def remap(coords, H, W):
    ux = coords[..., 0]
    uy = coords[..., 1]
    cx, cy = 0.5, 0.5
    r = np.sqrt((ux - cx) ** 2 + ((uy - cy)*H/W) ** 2)
    r_max = 0.25
    dux = ux*(1+r_max)#*np.sin(np.pi*r/r_max))
    duy = uy*(1+r_max)#*np.sin(np.pi*r/r_max))
    dcoords = np.stack([dux, duy], axis=-1)
    mask = r <= r_max
    return np.where(mask[..., np.newaxis], dcoords, coords)

def remap_f(coords, H, W):
    ux = coords[..., 0]
    uy = coords[..., 1]
    cx, cy = 0.5, 0.5

    r = np.sqrt((ux - cx) ** 2 + ((uy - cy) * H / W) ** 2)
    r_max = 0.25
    feather = 0.15

    # warp (no feathering here)
    dux = ux * (1 + r_max)
    duy = uy * (1 + r_max)
    warped = np.stack([dux, duy], axis=-1)

    # feather weight
    alpha = np.clip((r_max - r) / feather, 0.0, 1.0)[..., None]

    return alpha * warped + (1 - alpha) * coords




def test_cell_lines_interpolation_continuous():
    """Test LinesCell interpolation."""
    L =6
    total = 90
    inc = int(total / L)
    top_line = np.array([[i * inc + j for j in range(3)] for i in range(L)], dtype=np.int32)
    bottom_line = np.array([[i * inc * 3 + j for j in range(3)] for i in range(L)], dtype=np.int32)

    H, W = 4, 12
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]

    per_channel_coords = [remap(coords, H, W) for coords in per_channel_coords[:-1]] + per_channel_coords[-1:]
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
    
    result = cell.get_value()

    assert result.shape == (H, W, 3)
    # Check values are within expected range
    assert np.all(result >= np.min(top_line)) and np.all(result <= np.max(bottom_line))
    assert int(result[0][0][0]*255) == 0
    assert 0 < int(result[0][1][0]*255) < inc
    assert inc <= int(result[2][1][0]*255) < inc*1.5
    # Center value is yellowish
    center_pixel = result[H//2][W//2]
    assert center_pixel[0] > center_pixel[2]
    assert center_pixel[1] > center_pixel[2]


def test_cell_lines_interpolation_discrete():
    """Test LinesCell interpolation with discrete method."""
    L = 6
    total = 100
    inc = int(total / L)
    top_line = np.array([[i * inc + j for j in range(3)] for i in range(L)], dtype=np.int32)
    bottom_line = np.array([[i * inc * 2 + j for j in range(3)] for i in range(L)], dtype=np.int32)
    
    H, W = 4, 12
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]
    per_channel_coords = [remap(coords, H, W) for coords in per_channel_coords]
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
    assert int(result[0][0][0]*255) == 0
    assert int(result[0][1][0]*255) == 0 # No interp between discrete steps
    assert int(result[0][2][0]*255) == inc 

def test_cell_corners_interpolation():
    """Test CornersCell interpolation."""
    total = 255
    top_left = np.array([0, 0, 0], dtype=np.int32)
    top_right = np.array([total, 0, 0], dtype=np.int32)
    bottom_left = np.array([0, total, 0], dtype=np.int32)
    bottom_right = np.array([0, 0, total], dtype=np.int32)
    
    H, W = 5, 5
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]
    per_channel_coords = [remap(coords, H, W) for coords in per_channel_coords]
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
        hue_direction_y=HueMode.CW,
        hue_direction_x=HueMode.CCW,
        boundtypes=BoundType.CLAMP
    )
    result = cell.get_value()
    assert result.shape == (H, W, 3)
    # Check corner values
    assert np.allclose(result[0, 0], top_left / 255)
    assert np.allclose(result[0, -1], top_right / 255)
    assert np.allclose(result[-1, 0], bottom_left / 255)
    assert np.allclose(result[-1, -1], bottom_right / 255)


    
def test_interpolate_edge_corners_dual():
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

    top_edge = cell.interpolate_edge(0.5, is_top_edge=True)
    bottom_edge = cell.interpolate_edge(0.5, is_top_edge=False)

    assert np.allclose(top_edge, np.array([30/255, 31/255, 32/255]))
    assert np.allclose(bottom_edge, np.array([90/255, 91/255, 92/255]))




def test_cell_lines_interpolation_continuous():
    """Test LinesCell interpolation."""
    L =6
    total = 90
    inc = int(total / L)
    top_line = np.array([[i * inc + j for j in range(3)] for i in range(L)], dtype=np.int32)
    bottom_line = np.array([[i * inc * 3 + j for j in range(3)] for i in range(L)], dtype=np.int32)

    H, W = 4, 12
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]

    per_channel_coords = [remap(coords, H, W) for coords in per_channel_coords[:-1]] + per_channel_coords[-1:]
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
    
    result = cell.get_value()

    assert result.shape == (H, W, 3)
    # Check values are within expected range
    assert np.all(result >= np.min(top_line)) and np.all(result <= np.max(bottom_line))
    assert int(result[0][0][0]*255) == 0
    assert 0 < int(result[0][1][0]*255) < inc
    assert inc <= int(result[2][1][0]*255) < inc*1.5
    # Center value is yellowish
    center_pixel = result[H//2][W//2]
    assert center_pixel[0] > center_pixel[2]
    assert center_pixel[1] > center_pixel[2]




def test_cell_corners_interpolation():
    """Test CornersCell interpolation."""
    total = 255
    top_left = np.array([0, 0, 0], dtype=np.int32)
    top_right = np.array([total, 0, 0], dtype=np.int32)
    bottom_left = np.array([0, total, 0], dtype=np.int32)
    bottom_right = np.array([0, 0, total], dtype=np.int32)
    
    H, W = 5, 5
    per_channel_coords = [upbm_2d(width=W, height=H) for _ in range(3)]
    per_channel_coords = [remap(coords, H, W) for coords in per_channel_coords]
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
        hue_direction_y=HueMode.CW,
        hue_direction_x=HueMode.CCW,
        boundtypes=BoundType.CLAMP
    )
    result = cell.get_value()
    assert result.shape == (H, W, 3)
    # Check corner values
    assert np.allclose(result[0, 0], top_left / 255)
    assert np.allclose(result[0, -1], top_right / 255)
    assert np.allclose(result[-1, 0], bottom_left / 255)
    assert np.allclose(result[-1, -1], bottom_right / 255)


import pytest
from ....chromatica.v2core.core2d import sample_hue_between_lines_continuous

def test_sample_hue_between_lines_continuous():
    L = 6
    total = 360
    inc = int(total / L)
    #Hline0 [150.       120.        19.999998   0.      ]
    #Hline1 [  0.  80. 160. 240.]
    hue_line_0 = np.array([150, 120, 20, 0], dtype=np.float32)
    hue_line_1 = np.array([0, 80, 160, 240], dtype=np.float32)
    H, W = 4, 5
    #per_channel_coords = upbm_2d(width=W, height=H)
    per_channel_coords = [[[0.25, 0.  ], [0.5,  0.  ], [0.75, 0.  ],[1.,   0.  ]], [[0.25, 1.  ], [0.5,  1.  ], [0.75, 1.  ],[1.,   1.  ]]]
    per_channel_coords = np.array(per_channel_coords)
    result = sample_hue_between_lines_continuous(
        line0=hue_line_0,
        line1=hue_line_1,
        coords=per_channel_coords,
        bound_type=BoundType.CLAMP,
        border_mode=None,
        border_constant=None,
        mode_y=HueMode.CW,
        mode_x = HueMode.CW,
    )


