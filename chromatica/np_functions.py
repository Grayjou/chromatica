"""NumPy-compatible utility functions for constraining numeric values."""

from __future__ import annotations
from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray

ArrayOrScalar = Union[float, int, NDArray[np.floating], NDArray[np.integer]]


def clamp(value: ArrayOrScalar, min_value: ArrayOrScalar, max_value: ArrayOrScalar) -> ArrayOrScalar:
    """Clamp a value between ``min_value`` and ``max_value``.
    
    Works with scalars and NumPy arrays.
    """
    return np.clip(value, min_value, max_value)


def clamp01(value: ArrayOrScalar) -> ArrayOrScalar:
    """Clamp a value to the inclusive range ``[0, 1]``.
    
    Works with scalars and NumPy arrays.
    """
    return clamp(value, 0.0, 1.0)


def extract_excess(value: ArrayOrScalar, min_value: ArrayOrScalar, max_value: ArrayOrScalar) -> Tuple[ArrayOrScalar, ArrayOrScalar]:
    """Extract the excess amount outside the [min_value, max_value] range.
    
    Works with scalars and NumPy arrays.
    """
    below_min = value < min_value
    above_max = value > max_value
    
    clamped = np.where(below_min, min_value, np.where(above_max, max_value, value))
    excess = np.where(below_min, value - min_value, np.where(above_max, value - max_value, 0.0))
    
    return clamped, excess


def bounce(value: ArrayOrScalar, min_value: ArrayOrScalar, max_value: ArrayOrScalar) -> ArrayOrScalar:
    """Bounce a value within a specified range.
    
    Works with scalars and NumPy arrays.
    """
    range_size = max_value - min_value
    if np.any(range_size <= 0):
        raise ValueError("max_value must be greater than min_value")

    # Use extract_excess to determine position relative to bounds
    clamped, excess = extract_excess(value, min_value, max_value)
    
    # If within bounds, return as-is
    within_bounds = excess == 0
    
    # Handle bouncing for values outside bounds
    mod_excess = np.abs(excess) % (2 * range_size)
    bounced = np.where(
        mod_excess > range_size,
        max_value - (mod_excess - range_size),
        np.where(excess < 0, min_value + mod_excess, max_value - mod_excess)
    )
    
    return np.where(within_bounds, clamped, bounced)


def cyclic_wrap(value: ArrayOrScalar, min_value: ArrayOrScalar, max_value: ArrayOrScalar) -> ArrayOrScalar:
    """Wrap inside ``[min_value, max_value]`` like a cyclic range.
    
    Works with scalars and NumPy arrays. For integer-like wrapping.
    """
    size = max_value - min_value + 1
    return (value - min_value) % size + min_value


def cyclic_wrap_float(value: ArrayOrScalar, min_value: ArrayOrScalar, max_value: ArrayOrScalar) -> ArrayOrScalar:
    """Wrap inside ``[min_value, max_value]`` like a cyclic range.
    
    Works with scalars and NumPy arrays. For continuous floating-point wrapping.
    """
    size = max_value - min_value
    return (value - min_value) % size + min_value
