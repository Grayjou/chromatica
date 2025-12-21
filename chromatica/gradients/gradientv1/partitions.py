from bisect import bisect_left
from typing import Generic, List, TypeVar, Literal
from boundednumbers import UnitFloat
from ...types.color_types import ColorSpace
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

class ColorSpacePartition(Partition[str]):
    def __init__(
        self,
        breakpoints: List[UnitFloat],
        color_spaces: List[str],
    ) -> None:
        super().__init__(breakpoints, color_spaces)
    def get_color_space(self, position: UnitFloat) -> str:
        return self.get_value(position)
    
class PerpendicularPartition(Partition[tuple[ColorSpace, HueDirection | None]]):
    def __init__(
        self,
        breakpoints: List[UnitFloat],
        values: List[tuple[ColorSpace, HueDirection | None]],
    ) -> None:
        super().__init__(breakpoints, values)
    def get_color_space(self, position: UnitFloat) -> ColorSpace:
        return self.get_value(position)[0]
    def get_hue_direction(self, position: UnitFloat) -> HueDirection:
        if self.get_value(position)[1] is None:
            return "shortest"
        return self.get_value(position)[1]
    def get_space_and_hue_direction(self, position: UnitFloat) -> tuple[ColorSpace, HueDirection | None]:
        return self.get_value(position)