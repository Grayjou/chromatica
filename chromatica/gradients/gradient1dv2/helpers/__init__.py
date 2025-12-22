"""
Helper utilities for gradient1dv2 operations.
"""

from .interpolation import (
    interpolate_transformed_non_hue,
    interpolate_transformed_hue_space,
    transform_1dchannels,

)

from .segment_utils import (
    get_segment_lengths,
    get_segment_indices,
    merge_endpoint_scaled_u,
    get_per_channel_coords_merged_endpoints,
    get_per_channel_coords,
    get_uniform_per_channel_coords,
    construct_scaled_u,
    get_segments_from_scaled_u,
)

__all__ = [
    # Interpolation
    'interpolate_transformed_non_hue',
    'interpolate_transformed_hue_space',
    'transform_1dchannels',

    
    # Segment utilities
    'get_segment_lengths',
    'get_segment_indices',
    'merge_endpoint_scaled_u',
    'get_per_channel_coords_merged_endpoints',
    'get_per_channel_coords',
    'get_uniform_per_channel_coords',
    'construct_scaled_u',
    'get_segments_from_scaled_u',
]
