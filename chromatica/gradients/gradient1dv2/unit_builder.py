from __future__ import annotations
import numpy as np
from numpy import ndarray as NDArray
from typing import Optional, Tuple, List
from ...types.transform_types import UnitTransform
from unitfield import flat_1d_upbm
from .normalizer import _Gradient1DNormalizer
from .helpers import get_segment_lengths, construct_scaled_u, get_per_channel_coords, get_uniform_per_channel_coords


class _Gradient1DUnitBuilder(_Gradient1DNormalizer):
    """Handles unit parameter construction for gradient interpolation."""
    
    @classmethod
    def _construct_scaled_u_and_steps(
        cls,
        total_steps: Optional[int],
        segment_lengths: Optional[List[int]],
        num_segments: int,
        offset: int = 1,
        global_unit_transform: Optional[UnitTransform] = None,
    ) -> Tuple[NDArray, int]:
        """Construct scaled u values and calculate total steps."""
        if segment_lengths is not None:
            seg_lengths = get_segment_lengths(total_steps, segment_lengths, num_segments)
            actual_total_steps = int(np.sum(seg_lengths) + (num_segments - 1) * (offset - 1))
            u_scaled = construct_scaled_u(seg_lengths, offset=offset)
            is_unit_input = False
        else:
            if total_steps is None:
                raise ValueError("Either total_steps or segment_lengths must be provided")
            
            actual_total_steps = total_steps
            u_scaled = flat_1d_upbm(total_steps)
            is_unit_input = True
        
        u_scaled = cls._apply_global_transform(
            u_scaled, num_segments, global_unit_transform, is_unit_input
        )
        
        return u_scaled, actual_total_steps
    
    @classmethod
    def _apply_global_transform(
        cls,
        u_scaled: NDArray,
        num_segments: int,
        global_unit_transform: Optional[UnitTransform],
        is_unit_input: bool = False,
    ) -> NDArray:
        """Apply a global unit transform to scaled u values."""
        if global_unit_transform is None:
            if is_unit_input:
                return u_scaled * num_segments
            return u_scaled
        
        u_global = u_scaled / num_segments if not is_unit_input else u_scaled
        u_global = global_unit_transform(u_global)
        return u_global * num_segments
    
    @classmethod
    def _construct_per_channel_coords_no_transform(
        cls,
        total_steps: Optional[int],
        segment_lengths: Optional[List[int]],
        num_segments: int,
        offset: int = 1,
    ) -> List[np.ndarray]:
        """Construct local u arrays without global transform."""
        if segment_lengths is not None:
            seg_lengths = get_segment_lengths(total_steps, segment_lengths, num_segments)
            return get_per_channel_coords(seg_lengths, offset=offset)
        else:
            if total_steps is None:
                raise ValueError("Either total_steps or segment_lengths must be provided")
            return get_uniform_per_channel_coords(total_steps, num_segments)