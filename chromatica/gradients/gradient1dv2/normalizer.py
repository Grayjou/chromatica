
#Chromatica\chromatica\gradients\gradient_1dv2.py
from __future__ import annotations
from typing import Optional, Tuple, Union, List
from ...colors import unified_tuple_to_class
from ...color_arr import Color1DArr
from ...colors.color_base import ColorBase
from ...types.format_type import FormatType
from ...types.transform_types import PerChannelTransform
from ...types.array_types import ndarray_1d
from ...types.color_types import ColorModes, HueDirection
from ...utils.list_mismatch import handle_list_size_mismatch


class _Gradient1DNormalizer(Color1DArr):
    ...
    @staticmethod
    def _normalize_list(
        lst: Optional[Union[List, str]],
        length: int,
        default,
    ) -> List:
        """Normalize a list to exact length, filling with defaults or truncating."""
        if lst is None:
            return [default] * length
        if isinstance(lst, str):
            return [lst] * length
        result = list(lst)
        return handle_list_size_mismatch(input_list=result, target_size=length, fill_value=default)


    @staticmethod
    def _normalize_input_spaces(
        spaces: Optional[Union[ColorModes, List[ColorModes]]],
        num_colors: int,
    ) -> List[ColorModes]:
        """Normalize input color spaces to match number of colors."""
        if spaces is None:
            return ["rgb"] * num_colors
        if isinstance(spaces, str):
            return [spaces] * num_colors
        result = list(spaces)
        return handle_list_size_mismatch(input_list=result, target_size=num_colors, fill_value="rgb")
    
    @classmethod
    def _normalize_gradient_sequence_settings(
        cls,
        colors: List[Union[ColorBase, Tuple, List, ndarray_1d]],
        color_modes: Optional[Union[ColorModes, List[ColorModes]]] = None,
        input_color_modes: Optional[Union[ColorModes, List[ColorModes]]] = None,
        hue_directions: Optional[List[HueDirection]] = None,
        per_channel_transforms: Optional[List[PerChannelTransform]] = None,
        num_segments: Optional[int] = None,
        ) -> Tuple[List[ColorModes], List[ColorModes], List[HueDirection], List[Optional[PerChannelTransform]]]:

        # Normalize all list parameters
        input_color_modes = cls._normalize_input_spaces(input_color_modes, len(colors))
        # I'd rather keep it this way, handle_list_size_mismatch call is more chunky and tests are passing
        color_modes = cls._normalize_list(color_modes, num_segments, "rgb")
        hue_directions = cls._normalize_list(hue_directions, num_segments, "shortest")
        per_channel_transforms = cls._normalize_list(per_channel_transforms, num_segments, None)
        return input_color_modes, color_modes, hue_directions, per_channel_transforms

    # Use the utility function from color_conversion_utils
    from ...utils.color_utils import convert_to_space_float as _convert_to_space_float