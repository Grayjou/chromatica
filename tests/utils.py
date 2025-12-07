
import numpy as np

def get_point(map, center, r, theta):
    """Helper to get pixel value at given polar coordinates."""
    x = int(center[0] + r * np.cos(np.radians(theta)))
    y = int(center[1] + r * np.sin(np.radians(theta)))
    return map.value[y, x]