from ...chromatica.v2core.interp_hue import (
    hue_lerp_simple, 
    hue_lerp_arrays, 
    hue_lerp_between_lines, 
    hue_lerp_between_lines_x_discrete, 
    hue_multidim_lerp,
    hue_lerp_2d_spatial,)
from ...chromatica.v2core.core import HueMode
import numpy as np
from unitfield import upbm_2d, unit_positional_basematrix_ndim
import pytest
def test_hue_lerp_1d_spatial():
    """
    Simple 1D hue interpolation between two values.
    
    Args:
        h0: Start hue (degrees)
        h1: End hue (degrees)
        coeffs: Interpolation coefficients, shape (N,)
        mode: Interpolation mode (0=CW, 1=CCW, 2=SHORTEST, 3=LONGEST)
    
    Returns:
        Interpolated hues, shape (N,)
    """


    h0 = 30.0
    h1 = 300.0
    coeffs = np.linspace(0, 1, 5)

    # Test shortest path
    result_shortest = hue_lerp_simple(
        h0,
        h1,
        coeffs,
        HueMode.SHORTEST
    )
    expected_shortest = np.array([30.0, 7.5, 345, 322.5, 300.0])
    np.testing.assert_allclose(result_shortest, expected_shortest, atol=1e-5)

    # Test clockwise
    result_cw = hue_lerp_simple(
        h0,
        h1,
        coeffs,
        HueMode.CW
    )
    expected_cw = np.array([30.0, 97.5, 165.0, 232.5, 300.0])
    np.testing.assert_allclose(result_cw, expected_cw, atol=1e-5)

    # Test counter-clockwise
    result_ccw = hue_lerp_simple(
        h0,
        h1,
        coeffs,
        HueMode.CCW
    )
    expected_ccw = np.array([30.0, 7.5, 345.0, 322.5, 300.0])
    np.testing.assert_allclose(result_ccw, expected_ccw, atol=1e-5)

    # Test longest path
    result_longest = hue_lerp_simple(
        h0,
        h1,
        coeffs,
        HueMode.LONGEST
    )
    expected_longest = np.array([30.0, 97.5, 165.0, 232.5, 300.0])
    np.testing.assert_allclose(result_longest, expected_longest, atol=1e-5)

def test_hue_lerp_arrays():
    """
    Vectorized 1D hue interpolation for arrays of hue pairs.
    
    Args:
        h0_arr: Start hues, shape (M,)
        h1_arr: End hues, shape (M,)
        coeffs: Interpolation coefficients, shape (N,)
        mode: Interpolation mode
    
    Returns:
        Interpolated hues, shape (N, M)
    """
    W = 5
    H = 4
    h0_arr = np.linspace(0, 360, W)
    h1_arr = (h0_arr + 150) % 360
    coeffs = np.linspace(0, 1, H)
    result = hue_lerp_arrays(
        h0_arr,
        h1_arr,
        coeffs,
        HueMode.SHORTEST
    )
    columns = [ (0, 50, 100, 150), (90, 140, 190, 240), (180, 230, 280, 330), (270, 320, 10, 60), (0, 50, 100, 150) ]
    expected = np.array(columns).T
    np.testing.assert_allclose(result, expected, atol=1e-5)

def test_hue_lerp_between_lines():
    """
    Interpolate hue between two 1D lines with modes for each axis.
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[..., 0] = u_x (position along lines)
                coords[..., 1] = u_y (blend between lines)
        mode_x: Interpolation mode for sampling within lines
        mode_y: Interpolation mode for blending between lines
    
    Returns:
        Interpolated hues, shape (H, W), values in [0, 360)
    """
    L = 6
    W = 6
    H = 3
    line0 = np.array([0, 60, 120, 180, 240, 300]).astype(np.float64)
    line1 = np.array([180, 240, 300, 0, 60, 120]).astype(np.float64)
    columns = [
        hue_lerp_simple(line0[i], line1[i], np.linspace(0, 1, H), HueMode.CW)
        for i in range(L)
    ]

    coords = upbm_2d(width=W, height=H).astype(np.float64)
    result = hue_lerp_between_lines(
        line0,
        line1,
        coords,
        mode_x=HueMode.SHORTEST,
        mode_y=HueMode.CW
    )
    expected = np.array(columns).T
    np.testing.assert_allclose(result, expected, atol=1e-5)

def test_hue_lerp_between_lines_x_discrete():
    """
    Interpolate hue between two 1D lines with discrete x sampling.
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[..., 0] = x index (discrete position along lines)
                coords[..., 1] = u_y (blend between lines)
        mode_y: Interpolation mode for blending between lines
    
    Returns:
        Interpolated hues, shape (H, W), values in [0, 360)
    """
    L = 6
    W = 6
    H = 3
    line0 = np.array([0, 60, 120, 180, 240, 300]).astype(np.float64)
    line1 = np.array([180, 240, 300, 0, 60, 120]).astype(np.float64)
    columns = [
        hue_lerp_simple(line0[i], line1[i], np.linspace(0, 1, H), HueMode.CCW)
        for i in range(L)
    ]

    coords = upbm_2d(width=W, height=H).astype(np.float64)
    result = hue_lerp_between_lines_x_discrete(
        line0,
        line1,
        coords,
        mode_y=HueMode.CCW
    )
    expected = np.array(columns).T
    np.testing.assert_allclose(result, expected, atol=1e-5)

def test_hue_multidim_lerp():
    starts = np.array([0, 60, 120, 180])
    ends = np.array([180, 240, 300, 360])
    coeffs = unit_positional_basematrix_ndim(4,4,4).astype(np.float64)
    with pytest.raises(NotImplementedError):
        hue_multidim_lerp(
            starts,
            ends,
            coeffs,
            modes= np.array([HueMode.SHORTEST, HueMode.CW, HueMode.CCW])
        )

def test_hue_multidim_lerp_2d():
    starts = np.array([0, 60])
    ends = np.array([180, 240])
    coeffs = unit_positional_basematrix_ndim(4,4).astype(np.float64)
    result = hue_multidim_lerp(
        starts,
        ends,
        coeffs,
        modes= np.array([HueMode.CW, HueMode.CCW]) #Y CW, X CCW 
    )
    expected = np.array([
        [0.0, 260.0, 160.0, 60.0],
        [60.0, 320.0, 220.0, 120.0],
        [120.0, 20.0, 280.0, 180.0],
        [180.0, 80.0, 340.0, 240.0],
    ])
    np.testing.assert_allclose(result, expected, atol=1e-5)
    
def test_hue_lerp_2d_spatial_transformed():
    starts = np.array([0, 120]).astype(np.float64)
    ends = np.array([180, 300]).astype(np.float64)
    coeffs = upbm_2d(4,4)
    def transform(coords):
        u_x = coords[..., 0] ** 2
        u_y = coords[..., 1] ** 0.5
        return np.stack([u_x, u_y], axis=-1)
    coeffs = transform(coeffs).astype(np.float64)
    result = hue_lerp_2d_spatial(
        starts,
        ends,
        coeffs,
        modes = np.array([HueMode.SHORTEST, HueMode.CW]).astype(np.int32),
    )
    expected_first_row = [0.0, 20.0, 80.0, 180.0]
    expected_first_col = [0.0, 120 * np.sqrt(1/3), 120 * np.sqrt(2/3), 120.0]
    expected_second_col = [20.0, 20 + 120 * np.sqrt(1/3), 20 + 120 * np.sqrt(2/3), 120 + 20]

    np.testing.assert_allclose(result[0, :], expected_first_row, atol=1e-5)
    np.testing.assert_allclose(result[:, 0], expected_first_col, atol=1e-5)
    np.testing.assert_allclose(result[:, 1], expected_second_col, atol=1e-5)