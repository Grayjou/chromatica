import numpy as np
from numpy import ndarray as NDArray
from typing import Optional
def interpolate_hue(
    h0: np.ndarray,
    h1: np.ndarray,
    u: np.ndarray,
    direction: Optional[str] = None,
) -> np.ndarray:
    """Interpolate hue values with wrapping support."""
    direction = direction or 'shortest'
    h0 = h0 % 360.0
    h1 = h1 % 360.0

    if direction == 'cw':
        mask = h1 <= h0
        h1 = np.where(mask, h1 + 360.0, h1)
    elif direction == 'ccw':
        mask = h1 >= h0
        h1 = np.where(mask, h1 - 360.0, h1)
    elif direction == "longest":

        delta = (h1 - h0 + 180.0) % 360.0 - 180.0
        delta = np.where(delta > 0.0, delta - 360.0, delta + 360.0)
        h1 = h0 + delta

    elif direction == "shortest":
        delta = h1 - h0
        h1 = np.where(delta > 180.0, h1 - 360.0, h1)
        h1 = np.where(delta < -180.0, h1 + 360.0, h1)
    else:
        raise ValueError(f"Invalid hue direction: {direction}")
    dh = h1 - h0

    return (h0 + u * dh) % 360.0

def interpolate_hue_line(
    start: float,
    end: float,
    t: NDArray,
    direction: str
) -> NDArray:
    """
    Interpolate a hue line between two endpoints using a parameter array.

    Args:
        start: Start hue value
        end: End hue value
        t: Interpolation factors (same shape as output)
        direction: Hue direction ('cw', 'ccw', 'shortest')

    Returns:
        NDArray of interpolated hue values
    """

    height, width = t.shape
    result = np.zeros((height, width), dtype=float)

    for i in range(height):
        for j in range(width):
            result[i, j] = interpolate_hue(start, end, t[i, j], direction)

    return result
