from ..v2core.interp_hue import (
    hue_lerp_simple,
    hue_lerp_arrays,
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
    hue_multidim_lerp,
    hue_lerp_2d_spatial,

)
from ..v2core.core import HueMode, single_channel_multidim_lerp
import numpy as np
from enum import Enum
from typing import Optional
from boundednumbers import BoundType


class LineInterpMethods(Enum):
    LINES_CONTINOUS = 1
    LINES_DISCRETE = 0
    

def get_line_method(method: LineInterpMethods, huemode_x: Optional[HueMode] = None):
    if huemode_x is None:
        return LineInterpMethods.LINES_DISCRETE
    return method


def interp_transformed_hue_2d_corners(
    h_tl: float,
    h_tr: float,
    h_bl: float,
    h_br: float,
    transformed: np.ndarray,
    huemode_y: HueMode,
    huemode_x: HueMode,
) -> np.ndarray:

    starts = np.array([h_tl, h_bl], dtype=np.float64)
    ends = np.array([h_tr, h_br], dtype=np.float64)
    modes = np.array([huemode_y, huemode_x], dtype=np.int32)
    result = hue_lerp_2d_spatial(
        starts,
        ends,
        transformed,
        modes
    )
    return result

def interp_transformed_hue_2d_lines_continous(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
    huemode_y: HueMode,
    huemode_x: HueMode,
) -> np.ndarray:
    
    result = hue_lerp_between_lines(
    line0,
    line1,
    transformed,
    mode_y=huemode_y,
    mode_x=huemode_x
    )
    return result

def interp_transformed_hue_2d_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
    huemode_y: HueMode,
) -> np.ndarray:
    
    result = hue_lerp_between_lines_x_discrete(
    line0,
    line1,
    transformed,
    mode_y=huemode_y
    )
    return result

def interp_transformed_non_hue_2d_corners(
    c_tl: float,
    c_tr: float,
    c_bl: float,
    c_br: float,
    transformed: np.ndarray,
) -> np.ndarray:

    # Should use lerp_between_corners_multichannel but that function is not implemented yet
    pass

def interp_transformed_non_hue_2d_lines_continous(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
    huemode_y: HueMode,
    huemode_x: HueMode,
) -> np.ndarray:
    
    # Should use lerp_between_lines_multichannel. Although the function exists, it is not wrapped in core2d yet
    pass

def interp_transformed_non_hue_2d_lines_discrete(
    line0: np.ndarray,
    line1: np.ndarray,
    transformed: np.ndarray,
    huemode_y: HueMode,
) -> np.ndarray:
    
    # Should use lerp_between_lines_multichannel_discrete but that function is not implemented yet
    pass