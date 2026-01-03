#chromatica\gradients\gradient2dv2\partitions.py
from bisect import bisect_left
from typing import Generic, List, TypeVar, Literal, Generator
from boundednumbers import UnitFloat
from ...types.color_types import ColorModes
from enum import StrEnum
import numpy as np
T = TypeVar("T")


class Partition(Generic[T]):
    """
    Partitions the interval [0, 1] into regions defined by breakpoints.

    Breakpoints must be sorted and lie in (0, 1).
    Number of values must be len(breakpoints) + 1.

    Intervals are:
        [0, bp0]
        (bp0, bp1]
        ...
        (bpN, 1]
    """

    def __init__(self, breakpoints: List[UnitFloat], values: List[T]) -> None:
        if len(values) != len(breakpoints) + 1:
            raise ValueError(
                "Number of values must be exactly one more than number of breakpoints."
            )

        if any(bp <= 0.0 or bp >= 1.0 for bp in breakpoints):
            raise ValueError("Breakpoints must be strictly between 0 and 1.")

        if breakpoints != sorted(breakpoints):
            raise ValueError("Breakpoints must be sorted in increasing order.")

        self.breakpoints = breakpoints
        self.values = values

    def get_value(self, position: UnitFloat) -> T:
        if not 0.0 <= position <= 1.0:
            raise ValueError("Position must be within [0, 1].")

        index = bisect_left(self.breakpoints, position)
        return self.values[index]
    def __len__(self) -> int:
        return len(self.values)
    
HueDirection = Literal["shortest", "ccw", "cw"]


class HuePartition(Partition[HueDirection]):
    def __init__(
        self,
        breakpoints: List[UnitFloat],
        hue_directions: List[HueDirection],
    ) -> None:
        super().__init__(breakpoints, hue_directions)
    def get_hue_direction(self, position: UnitFloat) -> HueDirection:
        return self.get_value(position)

class ColorModePartition(Partition[str]):
    def __init__(
        self,
        breakpoints: List[UnitFloat],
        color_modes: List[str],
    ) -> None:
        super().__init__(breakpoints, color_modes)
    def get_color_mode(self, position: UnitFloat) -> str:
        return self.get_value(position)
    
class PartitionInterval:
    def __init__(self, color_mode: ColorModes, hue_direction_y: HueDirection | None = None, hue_direction_x: HueDirection | None = None) -> None:
        self.color_mode = color_mode
        self.hue_direction_y = hue_direction_y
        self.hue_direction_x = hue_direction_x


"""
Check the CellDual init T_T
    def __init__(self,
            top_left: np.ndarray,
            top_right: np.ndarray,
            bottom_left: np.ndarray,
            bottom_right: np.ndarray,
            per_channel_coords: List[np.ndarray] | np.ndarray,
            horizontal_color_mode: ColorModes,
            vertical_color_mode: ColorModes,
            hue_direction_y: HueDirection,
            hue_direction_x: HueDirection,
            boundtypes: List[BoundType] | BoundType = BoundType.CLAMP, *, value: Optional[np.ndarray] = None, 
            top_segment_hue_direction_x: Optional[HueDirection] = None,
            bottom_segment_hue_direction_x: Optional[HueDirection] = None,
            top_segment_color_mode: Optional[ColorModes] = None,
            bottom_segment_color_mode: Optional[ColorModes] = None
            ) -> None:
"""

class IndexRoundingMode(StrEnum):
    ROUND = "round"
    FLOOR = "floor"
    CEIL = "ceil"

index_rounding_mode_functions = {
    IndexRoundingMode.ROUND: np.round,
    IndexRoundingMode.FLOOR: np.floor,
    IndexRoundingMode.CEIL: np.ceil,
}

class PerpendicularPartition(Partition[tuple[PartitionInterval, ...]]):
    def __init__(
        self,
        breakpoints: List[UnitFloat],
        values: List[tuple[PartitionInterval, ...]],
    ) -> None:
        super().__init__(breakpoints, values)
    def get_color_mode(self, position: UnitFloat) -> ColorModes:
        return self.get_value(position)[0].color_mode
    def get_hue_direction(self, position: UnitFloat) -> HueDirection:
        if self.get_value(position)[1].hue_direction_y is None:
            return "shortest"
        return self.get_value(position)[1].hue_direction_y
    def __iter__(self):
        return iter(self.values)
    def __len__(self) -> int:
        return len(self.values)
    def intervals(self) -> Generator[tuple[UnitFloat, UnitFloat, PartitionInterval], None, None]:
        """Generate intervals as (start, end, value) tuples."""
        prev_bp = 0.0
        for i, bp in enumerate(self.breakpoints):
            yield (prev_bp, bp, self.values[i])
            prev_bp = bp
        yield (prev_bp, 1.0, self.values[-1])

class CellDualPartitionInterval:
    def __init__(self, horizontal_color_mode: ColorModes, 
                 vertical_color_mode: ColorModes,
                 hue_direction_y: HueDirection | None = None,
                 hue_direction_x: HueDirection | None = None,
                 top_segment_color_mode: ColorModes | None = None,
                 bottom_segment_color_mode: ColorModes | None = None,
                 top_segment_hue_direction_x: HueDirection | None = None,
                 bottom_segment_hue_direction_x: HueDirection | None = None,) -> None:
        self.horizontal_color_mode = horizontal_color_mode
        self.vertical_color_mode = vertical_color_mode
        self.hue_direction_y = hue_direction_y
        self.hue_direction_x = hue_direction_x
        self.top_segment_color_mode = top_segment_color_mode
        self.bottom_segment_color_mode = bottom_segment_color_mode
        self.top_segment_hue_direction_x = top_segment_hue_direction_x
        self.bottom_segment_hue_direction_x = bottom_segment_hue_direction_x

# Brace yourself.

class PerpendicularDualPartition(Partition[tuple[CellDualPartitionInterval, ...]]):
    def __init__(
        self,
        breakpoints: List[UnitFloat],
        values: List[tuple[CellDualPartitionInterval, ...]],
    ) -> None:
        super().__init__(breakpoints, values)
    def get_horizontal_color_mode(self, position: UnitFloat) -> ColorModes:
        return self.get_value(position)[0].horizontal_color_mode
    def get_vertical_color_mode(self, position: UnitFloat) -> ColorModes:
        return self.get_value(position)[0].vertical_color_mode
    def get_hue_direction_y(self, position: UnitFloat) -> HueDirection:
        if self.get_value(position)[0].hue_direction_y is None:
            return "shortest"
        return self.get_value(position)[0].hue_direction_y
    def __iter__(self):
        return iter(self.values)
    def __len__(self) -> int:
        return len(self.values)
    def intervals(self) -> Generator[tuple[UnitFloat, UnitFloat, CellDualPartitionInterval], None, None]:
        """Generate intervals as (start, end, value) tuples."""
        prev_bp = 0.0
        for i, bp in enumerate(self.breakpoints):
            yield (prev_bp, bp, self.values[i])
            prev_bp = bp
        yield (prev_bp, 1.0, self.values[-1])