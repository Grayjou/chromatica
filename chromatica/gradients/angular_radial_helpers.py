from __future__ import annotations
from typing import Optional, Tuple, Dict

import numpy as np
from numpy.typing import NDArray

from ..colors import ColorBase
from ..colors.color import unified_tuple_to_class
from ..normalizers.color_normalizer import normalize_color_input, ColorInput
from ..format_type import FormatType


class CoordinateGridCache:
    """Lightweight coordinate grid cache to avoid recomputation."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, Tuple[int, int]], Tuple[NDArray, NDArray]] = {}

    def get_grid(self, width: int, height: int, center: Tuple[int, int]) -> Tuple[NDArray, NDArray]:
        key = (width, height, center)
        if key not in self._cache:
            indices_matrix = np.indices((height, width), dtype=np.float32)
            y_indices = indices_matrix[0] - center[1]
            x_indices = indices_matrix[1] - center[0]
            distances = np.sqrt(x_indices**2 + y_indices**2)
            theta = (np.degrees(np.arctan2(y_indices, x_indices)) + 360.0) % 360.0
            self._cache[key] = (distances, theta)

            if len(self._cache) > 10:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

        return self._cache[key]


def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 360) range."""
    return angle % 360.0


def interpolate_hue(
    h0: np.ndarray,
    h1: np.ndarray,
    u: np.ndarray,
    direction: Optional[str] = None,
) -> np.ndarray:
    """Interpolate hue values with wrapping support."""
    h0 = h0 % 360.0
    h1 = h1 % 360.0

    if direction == 'cw':
        mask = h1 <= h0
        h1 = np.where(mask, h1 + 360.0, h1)
    elif direction == 'ccw':
        mask = h1 >= h0
        h1 = np.where(mask, h1 - 360.0, h1)
    else:
        delta = h1 - h0
        h1 = np.where(delta > 180.0, h1 - 360.0, h1)
        h1 = np.where(delta < -180.0, h1 + 360.0, h1)

    dh = h1 - h0
    return (h0 + u * dh) % 360.0


def linear_transform(x: NDArray, min_input, max_input, min_output, max_output) -> NDArray:
    """Linearly transform x from [min_input, max_input] to [min_output, max_output]."""
    x_clipped = np.clip(x, min_input, max_input)
    x_scaled = (x_clipped - min_input) / (max_input - min_input)
    return min_output + x_scaled * (max_output - min_output)


def compute_center(
    width: int, height: int, center: Optional[Tuple[int, int]], relative_center: Optional[Tuple[float, float]]
) -> Tuple[int, int]:
    """Resolve center tuple from absolute or relative inputs."""
    if center is not None:
        return center
    if relative_center is not None:
        center_x = int(relative_center[0] * width)
        center_y = int(relative_center[1] * height)
        return center_x, center_y
    return width // 2, height // 2


def validate_and_return_outside_fill_array(arr: np.ndarray, width: int, height: int, num_channels: int) -> np.ndarray | Tuple:
    if arr.ndim == 1:
        return tuple(arr.tolist())

    expected_shape = (height, width, num_channels)
    if arr.shape != expected_shape:
        raise ValueError("outside_fill array shape does not match the expected image shape.")
    return arr


def process_outside_fill(
    outside_fill: Optional[ColorInput], width: int, height: int, format_type: FormatType, color_space: str
) -> np.ndarray:
    respective_class = unified_tuple_to_class[(color_space, format_type)]
    num_channels = respective_class.num_channels

    if outside_fill is None:
        return np.zeros((height, width, num_channels))
    if isinstance(outside_fill, np.ndarray):
        return validate_and_return_outside_fill_array(outside_fill, width, height, num_channels)
    if isinstance(outside_fill, ColorBase):
        if isinstance(outside_fill.value, np.ndarray):
            return validate_and_return_outside_fill_array(outside_fill.value, width, height, num_channels)
        value = normalize_color_input(outside_fill)
        return np.full((height, width, num_channels), value)

    value = normalize_color_input(outside_fill)
    return np.array(value) if not isinstance(value, np.ndarray) else value

