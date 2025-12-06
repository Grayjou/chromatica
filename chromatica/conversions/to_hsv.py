from .numbers import UnitFloat
import numpy as np
from numpy import ndarray as NDArray
from .css_to_hsv import css_rgb_to_hsv, np_css_rgb_to_hsv

## HSL to HSV conversions

def hsl_to_hsv(hue_deg: float, saturation: UnitFloat, lightness: UnitFloat) -> tuple[float, UnitFloat, UnitFloat]:
    """
    Convert HSL (Hue, Saturation, Lightness) to HSV (Hue, Saturation, Value).

    Args:
        hue_deg (float): Hue in degrees [0.0, 360.0).
        saturation (UnitFloat): HSL saturation in [0.0, 1.0].
        lightness (UnitFloat): HSL lightness in [0.0, 1.0].

    Returns:
        Tuple[float, UnitFloat, UnitFloat]: (Hue in degrees, HSV saturation, value), all in [0.0, 1.0].
    """
    l = float(lightness)
    s = float(saturation)
    denom = 1 - abs(2 * l - 1)

    value = (2 * l + s * denom) / 2
    if value == 0:
        hsv_saturation = UnitFloat(0.0)
    else:
        hsv_saturation = UnitFloat(2 * (value - l) / value)

    return hue_deg, hsv_saturation, UnitFloat(value)

def np_hsl_to_hsv(hue_deg: NDArray, saturation: NDArray, lightness: NDArray) -> NDArray:
    """
    Vectorized HSL → HSV.

    Args:
        hue_deg: array-like or scalar, degrees [0,360)
        saturation: array-like or scalar, [0,1]
        lightness: array-like or scalar, [0,1]

    Returns:
        hsv: array of shape (..., 3): (hue_deg, hsv_saturation, value)
    """
    hue_deg = np.asarray(hue_deg)
    s_hsl = np.asarray(saturation)
    l = np.asarray(lightness)

    out_shape = np.broadcast(hue_deg, s_hsl, l).shape

    hue_deg = np.broadcast_to(hue_deg, out_shape)
    s_hsl = np.broadcast_to(s_hsl, out_shape)
    l = np.broadcast_to(l, out_shape)

    denom = 1 - np.abs(2*l - 1)
    value = (2*l + s_hsl*denom) / 2

    hsv_s = np.zeros_like(value)
    mask = value > 0
    hsv_s[mask] = 2 * (value[mask] - l[mask]) / value[mask]

    hsv = np.stack([hue_deg, hsv_s, value], axis=-1)
    return hsv

## RGB to HSV conversions

def _np_unit_rgb_to_hsv(r: NDArray, g: NDArray, b: NDArray) -> NDArray:
    """
    Vectorized RGB → HSV.

    Args:
        r, g, b: array-like or scalar, [0,1]

    Returns:
        hsv: array of shape (..., 3): (hue_deg, saturation, value)
    """
    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)

    out_shape = np.broadcast(r, g, b).shape

    r = np.broadcast_to(r, out_shape)
    g = np.broadcast_to(g, out_shape)
    b = np.broadcast_to(b, out_shape)

    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    delta = max_c - min_c

    hue = np.zeros_like(max_c)

    mask = delta > 0
    mask_r = mask & (max_c == r)
    mask_g = mask & (max_c == g)
    mask_b = mask & (max_c == b)

    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

    saturation = np.zeros_like(max_c)
    mask_v = max_c > 0
    saturation[mask_v] = delta[mask_v] / max_c[mask_v]

    value = max_c

    hsv = np.stack([hue, saturation, value], axis=-1)
    return hsv

def unit_rgb_to_hsv_analytic(r: UnitFloat, g: UnitFloat, b: UnitFloat) -> tuple[float, UnitFloat, UnitFloat]:
    """
    Convert RGB components to HSV.

    Args:
        r (UnitFloat): Red component in [0.0, 1.0].
        g (UnitFloat): Green component in [0.0, 1.0].
        b (UnitFloat): Blue component in [0.0, 1.0].

    Returns:
        Tuple[float, UnitFloat, UnitFloat]: (Hue in degrees, Saturation, Value).
    """
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c

    if delta == 0:
        hue = 0.0
    elif max_c == r:
        hue = 60 * (((g - b) / delta) % 6)
    elif max_c == g:
        hue = 60 * (((b - r) / delta) + 2)
    else:
        hue = 60 * (((r - g) / delta) + 4)

    saturation = UnitFloat(0 if max_c == 0 else delta / max_c)
    value = max_c

    return hue, saturation, value

def unit_rgb_to_hsv_picker(r: UnitFloat, g: UnitFloat, b: UnitFloat) -> tuple[float, UnitFloat, UnitFloat]:
    """Match color picker's rounding behavior."""
    h, s, v = unit_rgb_to_hsv_analytic(r, g, b)
    
    # Apply color picker rounding rules
    h_rounded = round(h)  # Integer degrees
    s_rounded = round(float(s) * 100, 1) / 100.0  # Percentage rounded to 1 decimal
    v_rounded = round(float(v) * 100, 1) / 100.0  # Percentage rounded to 1 decimal
    
    return h_rounded, UnitFloat(s_rounded), UnitFloat(v_rounded)

# RGB to HSV main functions

def unit_rgb_to_hsv(r: UnitFloat, g: UnitFloat, b: UnitFloat, *, use_css_algo: bool = False) -> tuple[float, UnitFloat, UnitFloat]:
    """
    Convert RGB components to HSV.

    Args:
        r (UnitFloat): Red component in [0.0, 1.0].
        g (UnitFloat): Green component in [0.0, 1.0].
        b (UnitFloat): Blue component in [0.0, 1.0].
        use_css_algo (bool): Whether to use CSS Color 4 / Culori-compatible algorithm.

    Returns:
        Tuple[float, UnitFloat, UnitFloat]: (Hue in degrees, Saturation, Value).
    """
    if use_css_algo:
        return css_rgb_to_hsv(r, g, b)  # type: ignore
    else:
        return unit_rgb_to_hsv_analytic(r, g, b)

def np_unit_rgb_to_hsv(r: NDArray, g: NDArray, b: NDArray, *, use_css_algo: bool = False) -> NDArray:
    """
    Vectorized RGB → HSV.

    Args:
        r, g, b: array-like or scalar, [0,1]
        use_css_algo (bool): Whether to use CSS Color 4 / Culori-compatible algorithm.

    Returns:
        hsv: array of shape (..., 3): (hue_deg, saturation, value)
    """
    if use_css_algo:
        return np_css_rgb_to_hsv(r, g, b)
    else:
        return _np_unit_rgb_to_hsv(r, g, b)
    
