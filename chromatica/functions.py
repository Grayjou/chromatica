"""Utility functions for constraining numeric values."""

from __future__ import annotations
from typing import Tuple, cast
RealNumber = float | int


def clamp(value: RealNumber, min_value: RealNumber, max_value: RealNumber) -> RealNumber:
    """Clamp a value between ``min_value`` and ``max_value``."""

    return max(min(value, max_value), min_value)


def clamp01(value: RealNumber) -> RealNumber:
    """Clamp a value to the inclusive range ``[0, 1]``."""

    return clamp(value, 0.0, 1.0)


def extract_excess(value: RealNumber, min_value: RealNumber, max_value: RealNumber) -> Tuple[RealNumber, RealNumber]:
    """Extract the excess amount outside the [min_value, max_value] range."""
    if value < min_value:
        return min_value, value - min_value
    elif value > max_value:
        return max_value, value - max_value
    else:
        zero = cast(RealNumber, 0.0)
        return value, zero

def bounce(value: RealNumber, min_value: RealNumber, max_value: RealNumber) -> RealNumber:
    """Bounce a value within a specified range."""

    range_size = max_value - min_value
    if range_size <= 0:
        raise ValueError("max_value must be greater than min_value")

    # Use extract_excess to determine position relative to bounds
    clamped, excess = extract_excess(value, min_value, max_value)
    
    # If within bounds, return as-is
    if excess == 0:
        return clamped
    
    # Handle bouncing for values outside bounds
    mod_excess = abs(excess) % (2 * range_size)
    if mod_excess > range_size:
        return max_value - (mod_excess - range_size)
    
    return min_value + mod_excess if excess < 0 else max_value - mod_excess


def cyclic_wrap(value: RealNumber, min_value: RealNumber, max_value: RealNumber) -> RealNumber:
    """Wrap inside ``[min_value, max_value]`` like a cyclic range."""

    size = max_value - min_value + 1
    return (value - min_value) % size + min_value

def cyclic_wrap_float(value: RealNumber, min_value: RealNumber, max_value: RealNumber) -> RealNumber:
    """Wrap inside ``[min_value, max_value]`` like a cyclic range."""

    size = max_value - min_value
    return (value - min_value) % size + min_value

