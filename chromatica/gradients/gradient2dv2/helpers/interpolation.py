"""
Interpolation utilities for 2D gradients.
"""

from typing import Optional
import numpy as np
from enum import Enum

from ...v2core.core import HueMode
from ...v2core.core2d import (
    sample_hue_between_lines_continuous,
    sample_hue_between_lines_discrete,
    sample_between_lines_continuous,
    sample_between_lines_discrete,
)
from ...v2core.interp_hue import (
    hue_lerp_simple,
    hue_lerp_arrays,
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
    hue_multidim_lerp,
    hue_lerp_2d_spatial,
)
from ...v2core.core import single_channel_multidim_lerp
from boundednumbers import BoundType


class LineInterpMethods(Enum):
    """Methods for interpolating between lines."""
    LINES_CONTINUOUS = 1
    LINES_DISCRETE = 0
    

def get_line_method(method: LineInterpMethods, huemode_x: Optional[HueMode] = None):
    """
    Determine the line interpolation method to use.
    
    Args:
        method: Requested interpolation method
        huemode_x: Hue mode for x-axis (if None, uses discrete)
        
    Returns:
        LineInterpMethods enum value
    """
    if huemode_x is None:
        return LineInterpMethods.LINES_DISCRETE
    return method


def interp_transformed_hue_2d_corners(
    h_tl: float,
    h_tr: float,
    h_bl: float,
    h_br: float,
    transformed: np.ndarray,
    huemode_x: HueMode,
    huemode_y: HueMode,
) -> np.ndarray:
    """
    Interpolate hue values from 4 corner hues using transformed coordinates.
    
    Args:
        h_tl: Top-left hue
        h_tr: Top-right hue
        h_bl: Bottom-left hue
        h_br: Bottom-right hue
        transformed: Transformed coordinates, shape (H, W, 2)
        huemode_x: Hue interpolation mode for x-axis
        huemode_y: Hue interpolation mode for y-axis
        
    Returns:
        Interpolated hues, shape (H, W)
    """
    starts = np.array([h_tl, h_bl], dtype=np.float64)
    ends = np.array([h_tr, h_br], dtype=np.float64)
    modes = np.array([int(huemode_y), int(huemode_x)], dtype=np.int32)
    
    result = hue_lerp_2d_spatial(
        starts,
        ends,
        transformed,
        modes
    )
    return result


def interp_transformed_hue_2d_lines_continuous(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
    huemode_x: HueMode,
    huemode_y: HueMode,
) -> np.ndarray:
    """
    Interpolate hue values between two lines using continuous sampling.
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        transformed: Transformed coordinates, shape (H, W, 2)
        huemode_x: Hue interpolation mode for x-axis (along lines)
        huemode_y: Hue interpolation mode for y-axis (between lines)
        
    Returns:
        Interpolated hues, shape (H, W)
    """
    result = sample_hue_between_lines_continuous(
        line0,
        line1,
        transformed,
        mode_x=huemode_x,
        mode_y=huemode_y,
    )
    return result


def interp_transformed_hue_2d_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
    huemode_y: HueMode,
) -> np.ndarray:
    """
    Interpolate hue values between two lines using discrete x-sampling.
    
    Args:
        line0: First hue line, shape (L,)
        line1: Second hue line, shape (L,)
        transformed: Transformed coordinates, shape (H, W, 2)
        huemode_y: Hue interpolation mode for y-axis (between lines)
        
    Returns:
        Interpolated hues, shape (H, W)
    """
    result = sample_hue_between_lines_discrete(
        line0,
        line1,
        transformed,
        mode_y=huemode_y,
    )
    return result


def interp_transformed_non_hue_2d_corners(
    c_tl: float,
    c_tr: float,
    c_bl: float,
    c_br: float,
    transformed: np.ndarray,
) -> np.ndarray:
    """
    Interpolate non-hue values from 4 corner values using transformed coordinates.
    
    Args:
        c_tl: Top-left value
        c_tr: Top-right value
        c_bl: Bottom-left value
        c_br: Bottom-right value
        transformed: Transformed coordinates, shape (H, W, 2)
        
    Returns:
        Interpolated values, shape (H, W)
    """
    # Use bilinear interpolation from corners
    starts = np.array([c_tl, c_tr], dtype=np.float64)
    ends = np.array([c_bl, c_br], dtype=np.float64)
    
    # transformed has shape (H, W, 2) with u_x and u_y
    result = single_channel_multidim_lerp(
        starts=starts,
        ends=ends,
        coeffs=transformed,
        bound_type=BoundType.CLAMP,
    )
    return result


def interp_transformed_non_hue_2d_lines_continuous(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
) -> np.ndarray:
    """
    Interpolate non-hue values between two lines using continuous sampling.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        transformed: Transformed coordinates, shape (H, W, 2)
        
    Returns:
        Interpolated values, shape (H, W)
    """
    result = sample_between_lines_continuous(
        line0,
        line1,
        transformed,
        bound_type=BoundType.CLAMP,
    )
    return result


def interp_transformed_non_hue_2d_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
) -> np.ndarray:
    """
    Interpolate non-hue values between two lines using discrete x-sampling.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        transformed: Transformed coordinates, shape (H, W, 2)
        
    Returns:
        Interpolated values, shape (H, W)
    """
    result = sample_between_lines_discrete(
        line0,
        line1,
        transformed,
        bound_type=BoundType.CLAMP,
    )
    return result


__all__ = [
    'LineInterpMethods',
    'get_line_method',
    'interp_transformed_hue_2d_corners',
    'interp_transformed_hue_2d_lines_continuous',
    'interp_transformed_hue_2d_lines_discrete',
    'interp_transformed_non_hue_2d_corners',
    'interp_transformed_non_hue_2d_lines_continuous',
    'interp_transformed_non_hue_2d_lines_discrete',
]
