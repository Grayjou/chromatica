# __init__.py or main module file
from typing import Union, List
from enum import IntEnum
from ..types.array_types import ndarray_1d, ndarray_2d
# Define constants once at module level
class BorderMode(IntEnum):
    """Border mode constants."""
    REPEAT = 0
    MIRROR = 1
    CONSTANT = 2
    CLAMP = 3
    OVERFLOW = 4
BORDER_REPEAT = 0
BORDER_MIRROR = 1
BORDER_CONSTANT = 2
BORDER_CLAMP = 3
BORDER_OVERFLOW = 4

BorderModeInput = Union[BorderMode, List[BorderMode]]
BorderConstant = Union[float, List[float], ndarray_1d]
BorderArrayInput = Union[ndarray_2d, None]
DistanceModeInput = Union[str, int]
class DistanceMode(IntEnum):
    """Distance metrics for 2D border computation."""
    MAX_NORM = 1
    MANHATTAN = 2
    SCALED_MANHATTAN = 3
    ALPHA_MAX = 4
    ALPHA_MAX_SIMPLE = 5
    TAYLOR = 6
    EUCLIDEAN = 7
    WEIGHTED_MINMAX = 8

