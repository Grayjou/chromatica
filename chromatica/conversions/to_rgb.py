import numpy as np
from numpy import ndarray as NDArray
from .utils import hue_to_rgb_components
from .numbers import UnitFloat
from .css_to_hsv import css_hsv_to_rgb, np_css_hsv_to_rgb
from .css_to_hsl import css_hsl_to_rgb, np_css_hsl_to_rgb
## HSV to RGB conversions

def _np_hsv_to_unit_rgb(hue_deg: NDArray, saturation: NDArray, value: NDArray) -> NDArray:
    """
    Vectorized HSV → RGB. All inputs can be scalars or same-shaped arrays.
    Returns array of shape (..., 3)
    """
    hue_deg = np.asarray(hue_deg)
    saturation = np.asarray(saturation)
    value = np.asarray(value)

    out_shape = np.broadcast(hue_deg, saturation, value).shape

    hue_deg = np.broadcast_to(hue_deg, out_shape)
    saturation = np.broadcast_to(saturation, out_shape)
    value = np.broadcast_to(value, out_shape)

    chroma = value * saturation
    hue_section = hue_deg / 60.0
    x = chroma * (1 - np.abs(hue_section % 2 - 1))

    r_ = np.zeros(out_shape)
    g_ = np.zeros(out_shape)
    b_ = np.zeros(out_shape)

    condlist = [
        (0 <= hue_section) & (hue_section < 1),
        (1 <= hue_section) & (hue_section < 2),
        (2 <= hue_section) & (hue_section < 3),
        (3 <= hue_section) & (hue_section < 4),
        (4 <= hue_section) & (hue_section < 5),
        (5 <= hue_section) & (hue_section < 6),
    ]
    choicelist = [
        (chroma, x, 0),
        (x, chroma, 0),
        (0, chroma, x),
        (0, x, chroma),
        (x, 0, chroma),
        (chroma, 0, x),
    ]

    for cond, (r_c, g_c, b_c) in zip(condlist, choicelist):
        r_[cond] = r_c if np.isscalar(r_c) else r_c[cond]
        g_[cond] = g_c if np.isscalar(g_c) else g_c[cond]
        b_[cond] = b_c if np.isscalar(b_c) else b_c[cond]

    m = value - chroma
    r_, g_, b_ = r_ + m, g_ + m, b_ + m

    return np.stack([r_, g_, b_], axis=-1)

def hsv_to_unit_rgb_analytical(hue_deg: float, saturation: UnitFloat, value: UnitFloat) -> tuple[UnitFloat, UnitFloat, UnitFloat]:
    """
    Convert HSV (Hue, Saturation, Value) color to RGB with unit float components.

    Args:
        hue_deg (float): Hue angle in degrees (0 <= hue < 360).
        saturation (UnitFloat): Saturation value in the range [0.0, 1.0].
        value (UnitFloat): Value (brightness) in the range [0.0, 1.0].

    Returns:
        Tuple[UnitFloat, UnitFloat, UnitFloat]: RGB color with each component in [0.0, 1.0].
    """
    if saturation == 0.0:
        return value, value, value  # achromatic (gray)

    chroma = value * saturation
    hue_section = hue_deg / 60.0
    x = chroma * (1.0 - abs(hue_section % 2 - 1.0))

    if 0 <= hue_section < 1:
        r_, g_, b_ = chroma, x, 0.0
    elif 1 <= hue_section < 2:
        r_, g_, b_ = x, chroma, 0.0
    elif 2 <= hue_section < 3:
        r_, g_, b_ = 0.0, chroma, x
    elif 3 <= hue_section < 4:
        r_, g_, b_ = 0.0, x, chroma
    elif 4 <= hue_section < 5:
        r_, g_, b_ = x, 0.0, chroma
    else:  # 5 <= hue_section < 6
        r_, g_, b_ = chroma, 0.0, x

    m = value - chroma
    return UnitFloat(r_ + m), UnitFloat(g_ + m), UnitFloat(b_ + m)

# HSV to RGB main functions

def hsv_to_unit_rgb(hue_deg: float, saturation: UnitFloat, value: UnitFloat, *, use_css_algo:bool = False):
    """
    Convert HSV (Hue, Saturation, Value) color to RGB with unit float components.

    Args:
        hue_deg (float): Hue angle in degrees (0 <= hue < 360).
        saturation (UnitFloat): Saturation value in the range [0.0, 1.0].
        value (UnitFloat): Value (brightness) in the range [0.0, 1.0].
        use_css_algo (bool): Whether to use the CSS Color 4 algorithm for conversion.

    Returns:
        Tuple[UnitFloat, UnitFloat, UnitFloat]: RGB color with each component in [0.0, 1.0].
    """
    if use_css_algo:
        return css_hsv_to_rgb(hue_deg, saturation, value)  # type: ignore
    else:
        return hsv_to_unit_rgb_analytical(hue_deg, saturation, value)
    
def np_hsv_to_unit_rgb(hue_deg: NDArray, saturation: NDArray, value: NDArray, *, use_css_algo: bool = False) -> NDArray:
    """
    Vectorized HSV → RGB.

    Args:
        hue_deg: array-like or scalar, degrees [0.0,360.0)
        saturation: array-like or scalar, [0.0,1.0]
        value: array-like or scalar, [0.0,1.0]
        use_css_algo (bool): Whether to use CSS Color 4 / Culori-compatible algorithm.
    Returns:
        rgb: array of shape (...,3): (r, g, b)
    """
    if use_css_algo:
        return np_css_hsv_to_rgb(hue_deg, saturation, value)
    else:
        return _np_hsv_to_unit_rgb(hue_deg, saturation, value)

# HSL to RGB conversions

def _np_hsl_to_unit_rgb(hue_deg: NDArray, saturation: NDArray, lightness: NDArray) -> NDArray:
    """
    Vectorized HSL → RGB. All inputs can be scalars or same-shaped arrays.
    Returns array of shape (..., 3)
    """
    hue_deg = np.asarray(hue_deg)
    saturation = np.asarray(saturation)
    lightness = np.asarray(lightness)

    out_shape = np.broadcast(hue_deg, saturation, lightness).shape

    hue_deg = np.broadcast_to(hue_deg, out_shape)
    saturation = np.broadcast_to(saturation, out_shape)
    lightness = np.broadcast_to(lightness, out_shape)

    chroma = (1 - np.abs(2 * lightness - 1)) * saturation
    hue_section = hue_deg / 60.0
    x = chroma * (1 - np.abs(hue_section % 2 - 1))

    r_ = np.zeros(out_shape)
    g_ = np.zeros(out_shape)
    b_ = np.zeros(out_shape)

    condlist = [
        (0 <= hue_section) & (hue_section < 1),
        (1 <= hue_section) & (hue_section < 2),
        (2 <= hue_section) & (hue_section < 3),
        (3 <= hue_section) & (hue_section < 4),
        (4 <= hue_section) & (hue_section < 5),
        (5 <= hue_section) & (hue_section < 6),
    ]
    choicelist = [
        (chroma, x, 0),
        (x, chroma, 0),
        (0, chroma, x),
        (0, x, chroma),
        (x, 0, chroma),
        (chroma, 0, x),
    ]

    for cond, (r_c, g_c, b_c) in zip(condlist, choicelist):
        r_[cond] = r_c if np.isscalar(r_c) else r_c[cond]
        g_[cond] = g_c if np.isscalar(g_c) else g_c[cond]
        b_[cond] = b_c if np.isscalar(b_c) else b_c[cond]

    m = lightness - chroma / 2
    r_, g_, b_ = r_ + m, g_ + m, b_ + m

    return np.stack([r_, g_, b_], axis=-1)

def hsl_to_unit_rgb_analytical(hue_deg: float, saturation: UnitFloat, lightness: UnitFloat) -> tuple[UnitFloat, UnitFloat, UnitFloat]:
    """
    Convert HSL (Hue, Saturation, Lightness) color to RGB with unit float components.

    Args:
        hue_deg (float): Hue angle in degrees (0 <= hue < 360).
        saturation (UnitFloat): Saturation value in the range [0.0, 1.0].
        lightness (UnitFloat): Lightness value in the range [0.0, 1.0].

    Returns:
        Tuple[UnitFloat, UnitFloat, UnitFloat]: RGB color with each component in [0.0, 1.0].
    """
    if saturation == 0.0:
        return lightness, lightness, lightness  # achromatic (gray)

    chroma = (1.0 - abs(2.0 * lightness - 1.0)) * saturation
    hue_section = hue_deg / 60.0
    x = chroma * (1.0 - abs(hue_section % 2 - 1.0))

    if 0 <= hue_section < 1:
        r_, g_, b_ = chroma, x, 0.0
    elif 1 <= hue_section < 2:
        r_, g_, b_ = x, chroma, 0.0
    elif 2 <= hue_section < 3:
        r_, g_, b_ = 0.0, chroma, x
    elif 3 <= hue_section < 4:
        r_, g_, b_ = 0.0, x, chroma
    elif 4 <= hue_section < 5:
        r_, g_, b_ = x, 0.0, chroma
    else:  # 5 <= hue_section < 6
        r_, g_, b_ = chroma, 0.0, x

    m = lightness - chroma / 2.0
    return UnitFloat(r_ + m), UnitFloat(g_ + m), UnitFloat(b_ + m)

def hsl_to_unit_rgb_fast(hue: float, saturation: UnitFloat, lightness: UnitFloat) -> tuple[UnitFloat, UnitFloat, UnitFloat]:
    """
    Fast (approximate) HSL to RGB conversion using analytical functions.
    Recommended for performance-critical cases (e.g., shaders or previews).

    Args:
        hue (float): Hue in degrees [0.0, 360.0).
        saturation (UnitFloat): Saturation in [0.0, 1.0].
        lightness (UnitFloat): Lightness in [0.0, 1.0].

    Returns:
        Tuple[UnitFloat, UnitFloat, UnitFloat]: RGB components in [0.0, 1.0].
    """
    hue_normalized = (hue % 360.0) / 360.0  # Normalize hue to [0.0, 1.0)
    r_base, g_base, b_base = hue_to_rgb_components(hue_normalized)

    chroma = (1.0 - abs(2.0 * float(lightness) - 1.0)) * float(saturation)
    r = (float(r_base) - 0.5) * chroma + float(lightness)
    g = (float(g_base) - 0.5) * chroma + float(lightness)
    b = (float(b_base) - 0.5) * chroma + float(lightness)

    return UnitFloat(r), UnitFloat(g), UnitFloat(b)

# HSL to RGB main functions

def hsl_to_unit_rgb(hue_deg: float, saturation: UnitFloat, lightness: UnitFloat, *, use_css_algo: bool = False) -> tuple[UnitFloat, UnitFloat, UnitFloat]:
    """
    Convert HSL (Hue, Saturation, Lightness) color to RGB with unit float components.

    Args:
        hue_deg (float): Hue angle in degrees (0 <= hue < 360).
        saturation (UnitFloat): Saturation value in the range [0.0, 1.0].
        lightness (UnitFloat): Lightness value in the range [0.0, 1.0].
        use_css_algo (bool): Whether to use CSS Color 4 algorithm for conversion.

    Returns:
        Tuple[UnitFloat, UnitFloat, UnitFloat]: RGB color with each component in [0.0, 1.0].
    """
    if use_css_algo:
        return css_hsl_to_rgb(hue_deg, saturation, lightness)  # type: ignore
    else:
        return hsl_to_unit_rgb_analytical(hue_deg, saturation, lightness)

def np_hsl_to_unit_rgb(hue_deg: NDArray, saturation: NDArray, lightness: NDArray, *, use_css_algo: bool = False) -> NDArray:
    """
    Vectorized HSL → RGB.

    Args:
        hue_deg: array-like or scalar, degrees [0.0,360.0)
        saturation: array-like or scalar, [0.0,1.0]
        lightness: array-like or scalar, [0.0,1.0]
        use_css_algo (bool): Whether to use CSS Color 4 algorithm for conversion.
    Returns:
        rgb: array of shape (...,3): (r, g, b)
    """
    if use_css_algo:
        return np_css_hsl_to_rgb(hue_deg, saturation, lightness)
    else:
        return _np_hsl_to_unit_rgb(hue_deg, saturation, lightness)

def np_hsl_to_unit_rgb_fast(hue_deg: NDArray, saturation: NDArray, lightness: NDArray) -> NDArray:
    """
    Vectorized approximate HSL → RGB. All inputs can be scalars or same-shaped arrays.
    Returns array of shape (..., 3)
    """
    hue_deg = np.asarray(hue_deg) % 360.0
    saturation = np.asarray(saturation)
    lightness = np.asarray(lightness)

    hue_norm = hue_deg / 360.0

    # base RGB triangle wave
    r_base = np.clip(np.abs(hue_norm * 6 - 3) - 1, 0, 1)
    g_base = np.clip(2 - np.abs(hue_norm * 6 - 2), 0, 1)
    b_base = np.clip(2 - np.abs(hue_norm * 6 - 4), 0, 1)

    chroma = (1 - np.abs(2 * lightness - 1)) * saturation

    r = (r_base - 0.5) * chroma + lightness
    g = (g_base - 0.5) * chroma + lightness
    b = (b_base - 0.5) * chroma + lightness

    return np.stack([r, g, b], axis=-1)
