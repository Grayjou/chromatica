from ...chromatica.v2core.core2d import lerp_from_corners, BorderMode, lerp_between_lines, lerp_between_lines_x_discrete_multichannel
#from ...chromatica.v2core.interp_2d.wrappers import _lerp_between_lines_full
import numpy as np
from unitfield import upbm_2d
import pytest
def test_border_constant_corners():
    corners = np.array([
        (1.0, 0.0, 0.0),  # Top-left: Red
        (0.0, 1.0, 0.0),  # Top-right: Green
        (0.0, 0.0, 1.0),  # Bottom-left: Blue
        (1.0, 1.0, 0.0)   # Bottom-right: Yellow
    ])
    coords = upbm_2d(4,4) - 1
    border_constant = np.array((0.5, 0.5, 0))  # Gray for out-of-bounds
    result = lerp_from_corners(
        corners=corners,
        coords_list=[coords]*3,
        border_mode=BorderMode.CONSTANT,
        border_constant=border_constant
    )

    expected = np.full_like(result, border_constant)
    # all but bottom-right corner should be border constant
    assert np.sum(result == expected) >= 45, "Border constant handling failed"

def test_border_constant_lines():
    line_start = np.array([
        (1.0, 0.0, 0.0),  # Start of line: Red
        (0.0, 1.0, 0.0),  # Start of line: Green
    ])
    line_end = np.array([
        (0.0, 0.0, 1.0),  # End of line: Blue
        (1.0, 1.0, 0.0),  # End of line: Yellow
    ])
    coords = upbm_2d(4,4) - 1
    border_constant = np.array((0.5, 0.5, 0))  # Gray for out-of-bounds
    result = lerp_between_lines(
        line0=line_start,
        line1=line_end,
        coords=coords,
        border_mode=BorderMode.CONSTANT,
        border_constant=border_constant
    )

    expected = np.full_like(result, border_constant)
    # all but bottom-right corner should be border constant
    assert np.sum(result == expected) >= 45, "Border constant handling failed"

def test_border_constant_lines_automatic_border_constant():
    line_start = np.array([
        (1.0, 0.0, 0.0),  # Start of line: Red
        (0.0, 1.0, 0.0),  # Start of line: Green
    ])
    line_end = np.array([
        (0.0, 0.0, 1.0),  # End of line: Blue
        (1.0, 1.0, 0.0),  # End of line: Yellow
    ])
    coords = upbm_2d(4,4) - 1
    result = lerp_between_lines(
        line0=line_start,
        line1=line_end,
        coords=coords,
        border_mode=BorderMode.CONSTANT,
        border_constant=None  # Let the function compute automatic border constant
    )

    # Manually compute expected automatic border constant

    expected_border_constant = 0
    print(result.shape)
    expected = np.full_like(result, expected_border_constant)
    # all but bottom-right corner should be border constant
    assert np.sum(np.all(result == expected, axis=-1)) >= 15, "Automatic border constant handling failed"

def test_border_constant_corners_automatic_border_constant():
    corners = np.array([
        (1.0, 0.0, 0.0),  # Top-left: Red
        (0.0, 1.0, 0.0),  # Top-right: Green
        (0.0, 0.0, 1.0),  # Bottom-left: Blue
        (1.0, 1.0, 0.0)   # Bottom-right: Yellow
    ])
    coords = upbm_2d(4,4) - 1
    result = lerp_from_corners(
        corners=corners,
        coords_list=[coords]*3,
        border_mode=BorderMode.CONSTANT,
        border_constant=None  # Let the function compute automatic border constant
    )

    # Manually compute expected automatic border constant

    expected_border_constant = 0
    expected = np.full_like(result, expected_border_constant)
    # all but bottom-right corner should be border constant
    assert np.sum(np.all(result == expected, axis=-1)) >= 15, "Automatic border constant handling failed"

def test_border_constant_wrong_shape():
    corners = np.array([
        (1.0, 0.0, 0.0),  # Top-left: Red
        (0.0, 1.0, 0.0),  # Top-right: Green
        (0.0, 0.0, 1.0),  # Bottom-left: Blue
        (1.0, 1.0, 0.0)   # Bottom-right: Yellow
    ])
    coords = upbm_2d(4,4) - 1
    border_constant = np.array((0.5, 0.5))  # Incorrect shape

    with pytest.raises(ValueError):
        lerp_from_corners(
            corners=corners,
            coords_list=[coords]*3,
            border_mode=BorderMode.CONSTANT,
            border_constant=border_constant
        )

def test_border_constant_scalar_value():
    corners = np.array([
        (1.0, 0.0, 0.0),  # Top-left: Red
        (0.0, 1.0, 0.0),  # Top-right: Green
        (0.0, 0.0, 1.0),  # Bottom-left: Blue
        (1.0, 1.0, 0.0)   # Bottom-right: Yellow
    ])
    coords = upbm_2d(4,4) - 1
    border_constant = 0.5  # Scalar value

    result = lerp_from_corners(
        corners=corners,
        coords_list=[coords]*3,
        border_mode=BorderMode.CONSTANT,
        border_constant=border_constant
    )

    expected = np.full_like(result, border_constant)
    # all but bottom-right corner should be border constant
    assert np.sum(result == expected) >= 45, "Border constant handling with scalar value failed"

def test_border_constant_lines_full():
    line_start = np.array([
        (1.0, 0.0, 0.0),  # Start of line: Red
        (0.0, 1.0, 0.0),  # Start of line: Green
    ])
    line_end = np.array([
        (0.0, 0.0, 1.0),  # End of line: Blue
        (1.0, 1.0, 0.0),  # End of line: Yellow
    ])
    coords = upbm_2d(4,4) - 1
    border_constant = np.array((0.5, 0.5, 0))  # Gray for out-of-bounds
    result = lerp_between_lines(
        line0=line_start,
        line1=line_end,
        coords=([coords]*3),
        border_mode=BorderMode.CONSTANT,
        border_constant=border_constant
    )

    expected = np.full_like(result, border_constant)
    # all but bottom-right corner should be border constant
    assert np.sum(result == expected) >= 45, "Border constant handling failed"

def test_border_constant_lines_full_automatic_border_constant():
    line_start = np.array([
        (1.0, 0.0, 0.0),  # Start of line: Red
        (0.0, 1.0, 0.0),  # Start of line: Green
    ])
    line_end = np.array([
        (0.0, 0.0, 1.0),  # End of line: Blue
        (1.0, 1.0, 0.0),  # End of line: Yellow
    ])
    coords = upbm_2d(4,4) - 1
    result = lerp_between_lines_x_discrete_multichannel(
        line0=line_start,
        line1=line_end,
        coords=([coords]*3),
        border_mode=BorderMode.CONSTANT,
        border_constant=None  # Let the function compute automatic border constant
    )

    # Manually compute expected automatic border constant

    expected_border_constant = 0
    print(result.shape)
    expected = np.full_like(result, expected_border_constant)
    # all but bottom-right corner should be border constant
    assert np.sum(np.all(result == expected, axis=-1)) >= 15, "Automatic border constant handling failed"

def test_between_lines_multichannel_border_constant():
    line_start = np.array([
        (1.0, 0.0, 0.0),  # Start of line: Red
        (0.0, 1.0, 0.0),  # Start of line: Green
    ])
    line_end = np.array([
        (0.0, 0.0, 1.0),  # End of line: Blue
        (1.0, 1.0, 0.0),  # End of line: Yellow
    ])
    coords = upbm_2d(4,4) - 1
    border_constant = np.array((0.5, 0.5, 0))  # Gray for out-of-bounds
    result = lerp_between_lines(
        line0=line_start,
        line1=line_end,
        coords=(coords),
        border_mode=BorderMode.CONSTANT,
        border_constant=border_constant
    )

    expected = np.full_like(result, border_constant)
    # all but bottom-right corner should be border constant
    assert np.sum(result == expected) >= 45, "Border constant handling failed"