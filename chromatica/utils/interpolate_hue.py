"""
Hue interpolation utilities (deprecated, uses Cython backend).

This module is deprecated. Use chromatica.v2core.core directly instead.
"""

import warnings
import numpy as np
from numpy import ndarray as NDArray
from typing import Optional

# Import from the new Cython backend

from ..v2core.core import hue_lerp, HueMode



def _convert_direction_to_mode(direction: Optional[str]) -> HueMode:
    """Convert old string direction to new HueMode enum."""
    if direction is None or direction == 'shortest' or direction == HueMode.SHORTEST:
        return HueMode.SHORTEST
    elif direction == 'cw' or direction == 'clockwise' or direction == HueMode.CW:
        return HueMode.CW
    elif direction == 'ccw' or direction == 'counterclockwise' or direction == HueMode.CCW:
        return HueMode.CCW
    elif direction == 'longest' or direction == HueMode.LONGEST:
        return HueMode.LONGEST
    else:
        raise ValueError(f"Invalid hue direction: {direction}")


def interpolate_hue(
    h0: np.ndarray,
    h1: np.ndarray,
    u: np.ndarray,
    direction: Optional[str] = None,
) -> np.ndarray:
    """
    Interpolate hue values with wrapping support.
    
    Deprecated: Use chromatica.v2core.core.hue_lerp instead.
    
    Args:
        h0: Start hue(s) in degrees [0, 360)
        h1: End hue(s) in degrees [0, 360)
        u: Interpolation coefficients
        direction: Interpolation direction ('cw', 'ccw', 'shortest', 'longest')
    
    Returns:
        Interpolated hue values in [0, 360)
    """
    warnings.warn(
        "interpolate_hue is deprecated. Use chromatica.v2core.core.hue_lerp instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    mode = _convert_direction_to_mode(direction)
    
    # Convert to float64 for Cython
    h0 = np.asarray(h0, dtype=np.float64)
    h1 = np.asarray(h1, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    
    # Handle scalar or array h0/h1
    if h0.shape == ():
        h0 = float(h0)
    if h1.shape == ():
        h1 = float(h1)
    
    # Call the Cython backend
    if isinstance(h0, float) and isinstance(h1, float):
        return hue_lerp(h0, h1, u, mode)
    else:
        # For array inputs, need to handle differently
        # The old code seems to expect element-wise operation
        # We'll apply the simple case
        h0_scalar = float(np.mean(h0)) if not isinstance(h0, float) else h0
        h1_scalar = float(np.mean(h1)) if not isinstance(h1, float) else h1
        return hue_lerp(h0_scalar, h1_scalar, u, mode)


def interpolate_hue_line(
    start: float,
    end: float,
    t: NDArray,
    direction: str
) -> NDArray:
    """
    Interpolate a hue line between two endpoints using a parameter array.
    
    Deprecated: Use chromatica.v2core.core.hue_lerp instead.

    Args:
        start: Start hue value
        end: End hue value
        t: Interpolation factors (same shape as output)
        direction: Hue direction ('cw', 'ccw', 'shortest')

    Returns:
        NDArray of interpolated hue values
    """
    warnings.warn(
        "interpolate_hue_line is deprecated. Use chromatica.v2core.core.hue_lerp instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    mode = _convert_direction_to_mode(direction)
    t = np.asarray(t, dtype=np.float64)
    return hue_lerp(start, end, t, mode)

