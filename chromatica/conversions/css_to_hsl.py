import math
import numpy as np
from numpy import ndarray as NDArray
from .numbers import UnitFloat

def normalize_hue(h: float) -> float:
    """Normalize hue to [0, 360) range."""
    return h % 360

## HSL to RGB conversions

def css_hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    """
    Convert HSL to RGB using CSS Color 4 / Culori algorithm.
    Based on: https://en.wikipedia.org/wiki/HSL_and_HSV#Converting_to_RGB

    Args:
        h: Hue in degrees [0, 360)
        s: Saturation in [0, 1]
        l: Lightness in [0, 1]

    Returns:
        Tuple[float, float, float]: (r, g, b) in [0, 1]
    """
    h = normalize_hue(h if h is not None else 0)
    s = s if s is not None else 0
    l = l if l is not None else 0

    m1 = l + s * (l if l < 0.5 else 1 - l)
    m2 = m1 - (m1 - l) * 2 * abs(((h / 60) % 2) - 1)

    hue_section = int(math.floor(h / 60))

    if hue_section == 0:
        r, g, b = m1, m2, 2 * l - m1
    elif hue_section == 1:
        r, g, b = m2, m1, 2 * l - m1
    elif hue_section == 2:
        r, g, b = 2 * l - m1, m1, m2
    elif hue_section == 3:
        r, g, b = 2 * l - m1, m2, m1
    elif hue_section == 4:
        r, g, b = m2, 2 * l - m1, m1
    elif hue_section == 5:
        r, g, b = m1, 2 * l - m1, m2
    else:
        r, g, b = 2 * l - m1, 2 * l - m1, 2 * l - m1

    return r, g, b

def np_css_hsl_to_rgb(h: NDArray, s: NDArray, l: NDArray) -> NDArray:
    """
    Vectorized: Convert HSL to RGB using CSS Color 4 / Culori algorithm.

    Args:
        h: array-like or scalar, hue in degrees [0, 360)
        s: array-like or scalar, saturation in [0, 1]
        l: array-like or scalar, lightness in [0, 1]

    Returns:
        rgb: array of shape (..., 3): (r, g, b) in [0, 1]
    """
    h = np.asarray(h, dtype=float) % 360
    s = np.asarray(s, dtype=float)
    l = np.asarray(l, dtype=float)

    out_shape = np.broadcast(h, s, l).shape
    h = np.broadcast_to(h, out_shape)
    s = np.broadcast_to(s, out_shape)
    l = np.broadcast_to(l, out_shape)

    # Calculate m1 and m2
    m1 = l + s * np.where(l < 0.5, l, 1 - l)
    m2 = m1 - (m1 - l) * 2 * np.abs(((h / 60) % 2) - 1)

    # Initialize output arrays
    r = np.zeros(out_shape)
    g = np.zeros(out_shape)
    b = np.zeros(out_shape)

    # Calculate hue section
    hue_section = np.floor(h / 60).astype(int)

    # Apply logic for each hue section
    mask0 = (hue_section == 0)
    mask1 = (hue_section == 1)
    mask2 = (hue_section == 2)
    mask3 = (hue_section == 3)
    mask4 = (hue_section == 4)
    mask5 = (hue_section == 5)

    r[mask0] = m1[mask0]
    g[mask0] = m2[mask0]
    b[mask0] = 2 * l[mask0] - m1[mask0]

    r[mask1] = m2[mask1]
    g[mask1] = m1[mask1]
    b[mask1] = 2 * l[mask1] - m1[mask1]

    r[mask2] = 2 * l[mask2] - m1[mask2]
    g[mask2] = m1[mask2]
    b[mask2] = m2[mask2]

    r[mask3] = 2 * l[mask3] - m1[mask3]
    g[mask3] = m2[mask3]
    b[mask3] = m1[mask3]

    r[mask4] = m2[mask4]
    g[mask4] = 2 * l[mask4] - m1[mask4]
    b[mask4] = m1[mask4]

    r[mask5] = m1[mask5]
    g[mask5] = 2 * l[mask5] - m1[mask5]
    b[mask5] = m2[mask5]

    # Default case (should rarely happen, but handle edge cases)
    mask_other = ~(mask0 | mask1 | mask2 | mask3 | mask4 | mask5)
    r[mask_other] = 2 * l[mask_other] - m1[mask_other]
    g[mask_other] = 2 * l[mask_other] - m1[mask_other]
    b[mask_other] = 2 * l[mask_other] - m1[mask_other]

    return np.stack([r, g, b], axis=-1)

## RGB to HSL conversions

def css_rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, UnitFloat, UnitFloat]:
    """
    Convert RGB to HSL using CSS Color 4 / Culori algorithm.
    
    Args:
        r: Red component in [0, 1]
        g: Green component in [0, 1]
        b: Blue component in [0, 1]
    
    Returns:
        Tuple[float, UnitFloat, UnitFloat]: (hue [0,360), saturation [0,1], lightness [0,1])
    """
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    
    # Lightness
    lightness = (max_c + min_c) / 2.0
    
    # Saturation
    if delta == 0:
        saturation = 0.0
    else:
        saturation = delta / (1 - abs(2 * lightness - 1))
    
    # Hue
    if delta == 0:
        hue = 0.0
    elif max_c == r:
        hue = (60 * ((g - b) / delta) + 360) % 360
    elif max_c == g:
        hue = (60 * ((b - r) / delta) + 120) % 360
    else:
        hue = (60 * ((r - g) / delta) + 240) % 360
    
    return hue, UnitFloat(saturation), UnitFloat(lightness)

def np_css_rgb_to_hsl(r: NDArray, g: NDArray, b: NDArray) -> NDArray:
    """
    Vectorized: Convert RGB to HSL using CSS Color 4 / Culori algorithm.
    
    Args:
        r, g, b: array-like or scalar, [0,1]
    
    Returns:
        hsl: array of shape (..., 3): (hue [0,360), saturation [0,1], lightness [0,1])
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    b = np.asarray(b, dtype=float)
    
    out_shape = np.broadcast(r, g, b).shape
    r = np.broadcast_to(r, out_shape)
    g = np.broadcast_to(g, out_shape)
    b = np.broadcast_to(b, out_shape)
    
    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    delta = max_c - min_c
    
    # Lightness
    lightness = (max_c + min_c) / 2.0
    
    # Saturation
    saturation = np.zeros_like(lightness)
    mask_delta = delta > 0
    saturation[mask_delta] = delta[mask_delta] / (1 - np.abs(2 * lightness[mask_delta] - 1))
    
    # Hue
    hue = np.zeros_like(max_c)
    mask = delta > 0
    mask_r = mask & (max_c == r)
    mask_g = mask & (max_c == g)
    mask_b = mask & (max_c == b)
    
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360
    
    return np.stack([hue, saturation, lightness], axis=-1)

