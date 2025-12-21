from ...color_arr import Color2DArr, Color1DArr
from ...utils.list_mismatch import handle_list_size_mismatch
from ...types.format_type import FormatType
from ...types.color_types import ColorSpace, HueDirection, is_hue_space
from ...types.transform_types import PerChannelTransform, UnitTransform
from typing import List, Optional, Tuple
from ..partitions import PerpendicularPartition
class Gradient2D(Color2DArr):
    """
    Gradient2D class representing a 2D gradient of colors.
    Inherits from Color2DArr to handle 2D arrays of colors.
    """
    @classmethod
    def from_1d_arrays(cls, 
        top_array: Color1DArr, 
        bottom_array: Color1DArr,
        height: int,
        color_space: str = "rgb",
        format_type: FormatType = FormatType.FLOAT,
        unit_transform_y: Optional[UnitTransform] = None,
        hue_direction: HueDirection = "shortest"
        perpendicular_
        )