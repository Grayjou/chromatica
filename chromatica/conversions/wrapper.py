import numpy as np
from typing import Literal, Tuple, cast, Dict, Callable

from ..types.format_type import FormatType, max_non_hue

from .to_rgb import np_hsv_to_unit_rgb, np_hsl_to_unit_rgb
from .to_hsv import np_unit_rgb_to_hsv, np_hsl_to_hsv
from .to_hsl import np_unit_rgb_to_hsl, np_hsv_to_hsl

from ..types.color_types import ColorElement, element_to_array, ColorSpace

# Functions that accept use_css_algo parameter
CONVERT_NUMPY_CSS: dict[tuple[str, str], Callable[[np.ndarray, np.ndarray, np.ndarray, bool], np.ndarray]] = {
    ("rgb", "hsv"): lambda r, g, b, use_css: np_unit_rgb_to_hsv(r, g, b, use_css_algo=use_css),
    ("hsv", "rgb"): lambda h, s, v, use_css: np_hsv_to_unit_rgb(h, s, v, use_css_algo=use_css),
    ("rgb", "hsl"): lambda r, g, b, use_css: np_unit_rgb_to_hsl(r, g, b, use_css_algo=use_css),
    ("hsl", "rgb"): lambda h, s, l, use_css: np_hsl_to_unit_rgb(h, s, l, use_css_algo=use_css),
}

# Functions without css algorithm (direct conversions between HSV and HSL)
CONVERT_NUMPY_DIRECT: dict[tuple[str, str], Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    ("hsv", "hsl"): np_hsv_to_hsl,
    ("hsl", "hsv"): np_hsl_to_hsv,
}

def normalize(color: np.ndarray, space: str, fmt: FormatType) -> np.ndarray:
    maxval = max_non_hue[fmt]

    if space == "rgb":
        return color / maxval

    if space in ("hsv", "hsl"):
        h = color[..., 0]
        a = color[..., 1] / maxval
        b = color[..., 2] / maxval
        return np.stack([h, a, b], axis=-1)

    raise ValueError(f"Unknown space: {space}")

def scale(color: np.ndarray, space: str, fmt: FormatType) -> np.ndarray:
    maxval = max_non_hue[fmt]

    if space == "rgb":
        scaled = color * maxval
        return np.round(scaled).astype(int) if fmt == FormatType.INT else scaled

    if space in ("hsv", "hsl"):
        h = color[..., 0]
        a = color[..., 1] * maxval
        b = color[..., 2] * maxval

        if fmt == FormatType.INT:
            return np.stack([np.round(h), np.round(a), np.round(b)], axis=-1).astype(int)

        return np.stack([h, a, b], axis=-1)

    raise ValueError(f"Unknown space: {space}")

def convert_alpha(alpha: np.ndarray | None, input_fmt: FormatType, output_fmt: FormatType) -> np.ndarray | None:
    if alpha is None:
        return None

    max_in  = max_non_hue[input_fmt]
    max_out = max_non_hue[output_fmt]

    result = alpha / max_in * max_out
    return np.round(result).astype(int) if output_fmt == FormatType.INT else result

def _convert_core(
    color: np.ndarray,
    from_space: str,
    to_space: str,
    input_fmt: FormatType,
    output_fmt: FormatType,
    use_css_algo: bool = False,
) -> np.ndarray:
    has_alpha_in  = from_space.endswith("a")
    has_alpha_out = to_space.endswith("a")

    if has_alpha_in:
        base = color[..., :3]
        alpha = color[..., 3]
    else:
        base = color
        alpha = None

    fs, ts = from_space[:3], to_space[:3]

    # normalize → convert → scale
    base_norm = normalize(base, fs, input_fmt)

    if fs == ts:
        converted = base_norm
    else:
        key = (fs, ts)
        if key in CONVERT_NUMPY_CSS:
            # Conversion involves RGB, use css algorithm parameter
            converted = CONVERT_NUMPY_CSS[key](
                base_norm[..., 0],
                base_norm[..., 1],
                base_norm[..., 2],
                use_css_algo,
            )
        else:
            # Direct HSV <-> HSL conversion, no css algorithm
            converted = CONVERT_NUMPY_DIRECT[key](
                base_norm[..., 0],
                base_norm[..., 1],
                base_norm[..., 2],
            )

    out = scale(converted, ts, output_fmt)

    # alpha output
    if has_alpha_out:
        new_alpha = convert_alpha(alpha, input_fmt, output_fmt)
        if new_alpha is None:
            # Default alpha value when no alpha in input
            default_alpha = max_non_hue[output_fmt]
            alpha_array = np.full(out.shape[:-1] + (1,), default_alpha)
            return np.concatenate([out, alpha_array], axis=-1)
        return np.concatenate([out, new_alpha[..., None]], axis=-1)

    return out


def convert(
    color: ColorElement,
    from_space: ColorSpace,
    to_space:   ColorSpace,
    input_type:  FormatType=FormatType.INT,
    output_type: FormatType=FormatType.INT,
    use_css_algo: bool = False,
 ) -> ColorElement:
    if from_space.lower() == to_space.lower() and input_type == output_type:
        return color  # No conversion needed
    color_array = element_to_array(color)
    result = _convert_core(
        color_array,
        from_space.lower(),
        to_space.lower(),
        FormatType(input_type),
        FormatType(output_type),
        use_css_algo,
    )
    # Convert back to tuple for scalar output
    return tuple(result.flat) if result.ndim == 1 else cast(ColorElement, result)

def np_convert(
    color: np.ndarray,
    from_space: ColorSpace,
    to_space:   ColorSpace,
    input_type:  Literal["int","float","percentage"]="int",
    output_type: Literal["int","float","percentage"]="int",
    use_css_algo: bool = False,
) -> np.ndarray:
    if from_space.lower() == to_space.lower() and input_type == output_type:
        return color  # No conversion needed
    return _convert_core(
        np.asarray(color, dtype=float),
        from_space.lower(),
        to_space.lower(),
        FormatType(input_type),
        FormatType(output_type),
        use_css_algo,
    )