import math
import numpy as np
from numpy import ndarray as NDArray
# No dependencies 


def srgb_to_linear(c: float) -> float:
    """Convert nonlinear sRGB (0..1) to linear-light RGB."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def linear_to_srgb(c: float) -> float:
    """Convert linear-light RGB (0..1) to nonlinear sRGB."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1/2.4)) - 0.055

def np_srgb_to_linear(c: NDArray) -> NDArray:
    """Vectorized: Convert nonlinear sRGB (0..1) to linear-light RGB."""
    c = np.asarray(c, dtype=float)
    result = np.where(
        c <= 0.04045,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4
    )
    return result

def np_linear_to_srgb(c: NDArray) -> NDArray:
    """Vectorized: Convert linear-light RGB (0..1) to nonlinear sRGB."""
    c = np.asarray(c, dtype=float)
    result = np.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * (c ** (1/2.4)) - 0.055
    )
    return result


def css_rgb_to_hsv(r: float, g: float, b: float):
    """
    CSS Color 4 / Culori-compatible HSV from nonlinear sRGB (0..1).

    Input:
        r, g, b ∈ [0, 1]   nonlinear sRGB

    Output:
        h ∈ [0, 360)
        s ∈ [0, 1]   (linear-light saturation)
        v ∈ [0, 1]   (linear-light brightness)
    """

    # 1. Convert to linear RGB
    rl = srgb_to_linear(r)
    gl = srgb_to_linear(g)
    bl = srgb_to_linear(b)

    # 2. Value = maximum in linear RGB
    V = max(rl, gl, bl)
    m = min(rl, gl, bl)
    delta = V - m

    # 3. Hue = CSS hue angle (atan2)
    if delta == 0:
        h = 0.0
    else:
        h = math.degrees(math.atan2(
            math.sqrt(3) * (gl - bl),
            2 * rl - gl - bl
        ))
        h = (h % 360 + 360) % 360

    # 4. Saturation
    S = 0.0 if V == 0 else delta / V

    return h, S, V

def np_css_rgb_to_hsv(r: NDArray, g: NDArray, b: NDArray) -> NDArray:
    """
    Vectorized CSS Color 4 / Culori-compatible HSV from nonlinear sRGB (0..1).

    Args:
        r, g, b: array-like or scalar, [0,1] nonlinear sRGB

    Returns:
        hsv: array of shape (..., 3): (hue [0,360), saturation [0,1], value [0,1])
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    b = np.asarray(b, dtype=float)

    out_shape = np.broadcast(r, g, b).shape
    r = np.broadcast_to(r, out_shape)
    g = np.broadcast_to(g, out_shape)
    b = np.broadcast_to(b, out_shape)

    # 1. Convert to linear RGB
    rl = np_srgb_to_linear(r)
    gl = np_srgb_to_linear(g)
    bl = np_srgb_to_linear(b)

    # 2. Value = maximum in linear RGB
    V = np.maximum.reduce([rl, gl, bl])
    m = np.minimum.reduce([rl, gl, bl])
    delta = V - m

    # 3. Hue = CSS hue angle (atan2)
    sqrt3 = np.sqrt(3)
    h = np.degrees(np.arctan2(
        sqrt3 * (gl - bl),
        2 * rl - gl - bl
    ))
    h = (h % 360 + 360) % 360

    # 4. Saturation
    S = np.zeros_like(V)
    mask = V > 0
    S[mask] = delta[mask] / V[mask]

    return np.stack([h, S, V], axis=-1)


def css_hsv_to_rgb(h: float, S: float, V: float):
    """
    Convert CSS/Culori HSV back into nonlinear sRGB (0..1).
    """

    # Convert hue to radians
    h_rad = math.radians(h)

    # Reconstruct linear RGB from hue circle
    # (CSS Color 4 defines hue as rotation in linear RGB)
    x = math.cos(h_rad)
    y = math.sin(h_rad) / math.sqrt(3)

    rl = V * (2/3 + x/3 + y)
    gl = V * (2/3 - x/3 + y)
    bl = V * (2/3 - 2*y)

    # Clamp linear channels
    rl = max(0, min(1, rl))
    gl = max(0, min(1, gl))
    bl = max(0, min(1, bl))

    # Convert back to sRGB
    return (
        linear_to_srgb(rl),
        linear_to_srgb(gl),
        linear_to_srgb(bl),
    )

def np_css_hsv_to_rgb(h: NDArray, S: NDArray, V: NDArray) -> NDArray:
    """
    Vectorized: Convert CSS/Culori HSV back into nonlinear sRGB (0..1).

    Args:
        h: array-like or scalar, [0,360) hue
        S: array-like or scalar, [0,1] saturation
        V: array-like or scalar, [0,1] value

    Returns:
        rgb: array of shape (..., 3): (r, g, b) nonlinear sRGB [0,1]
    """
    h = np.asarray(h, dtype=float)
    S = np.asarray(S, dtype=float)
    V = np.asarray(V, dtype=float)

    out_shape = np.broadcast(h, S, V).shape
    h = np.broadcast_to(h, out_shape)
    S = np.broadcast_to(S, out_shape)
    V = np.broadcast_to(V, out_shape)

    # Convert hue to radians
    h_rad = np.radians(h)

    # Reconstruct linear RGB from hue circle
    sqrt3 = np.sqrt(3)
    x = np.cos(h_rad)
    y = np.sin(h_rad) / sqrt3

    rl = V * (2/3 + x/3 + y)
    gl = V * (2/3 - x/3 + y)
    bl = V * (2/3 - 2*y)

    # Clamp linear channels
    rl = np.clip(rl, 0, 1)
    gl = np.clip(gl, 0, 1)
    bl = np.clip(bl, 0, 1)

    # Convert back to sRGB
    r = np_linear_to_srgb(rl)
    g = np_linear_to_srgb(gl)
    b = np_linear_to_srgb(bl)

    return np.stack([r, g, b], axis=-1)
