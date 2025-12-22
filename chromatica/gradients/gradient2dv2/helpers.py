from ..v2core.interp_hue import (
    hue_lerp_simple,
    hue_lerp_arrays,
    hue_lerp_between_lines,
    hue_lerp_between_lines_x_discrete,
    hue_multidim_lerp,
    hue_lerp_2d_spatial,
)
from ..v2core.core import HueMode
import numpy as np
from enum import Enum

class LineInterpMethods(Enum):
    LINES_CONTINOUS = 1
    LINES_DISCRETE = 0
    
def interp_transformeded_hue_2d_corners(
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

def interp_transformeded_hue_2d_lines_continous(
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

def interp_transformeded_hue_2d_lines_discrete(
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