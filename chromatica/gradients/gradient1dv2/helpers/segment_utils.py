"""
Segment utilities for 1D gradients.
"""

from typing import List, Optional
import numpy as np
from ....utils.list_mismatch import split_and_distribute_remainder
from unitfield import flat_1d_upbm


# ===================== Segment Length Helpers =====================

def get_segment_lengths(
    total_steps: Optional[int],
    segment_lengths: Optional[List[int]],
    segment_count: int,
) -> np.ndarray:
    """
    Determine segment lengths for gradient interpolation.

    Args:
        total_steps: Total number of steps in the gradient.
        segment_lengths: Optional explicit list of segment lengths.
        segment_count: Number of segments.

    Returns:
        Array of segment lengths.

    Raises:
        ValueError: If neither total_steps nor valid segment_lengths provided.
    """
    if segment_lengths:
        lengths = np.array(segment_lengths)
        if np.sum(lengths) > 0:
            return lengths
    
    if not total_steps:
        raise ValueError("Either total_steps or non-zero segment_lengths must be provided")
    
    return split_and_distribute_remainder(total_steps, segment_count)


def get_segment_indices(segment_lengths: np.ndarray) -> np.ndarray:
    """
    Compute segment index for each step in the gradient.

    For segments [4, 5], produces indices where:
    - First 3 steps (0-2) belong to segment 0 (length 4 minus shared endpoint)
    - Remaining 5 steps (3-7) belong to segment 1

    Args:
        segment_lengths: Length of each segment.

    Returns:
        Array mapping each step to its segment index.
    """
    segment_indices = []
    
    for seg_idx, length in enumerate(segment_lengths[:-1]):
        segment_indices.extend([seg_idx] * (length - 1))
    
    segment_indices.extend([len(segment_lengths) - 1] * segment_lengths[-1])
    return np.array(segment_indices)


# ===================== Unit Parameter Helpers =====================

def merge_endpoint_scaled_u(segment_lengths: np.ndarray) -> np.ndarray:
    """
    Construct scaled u values with merged endpoints between segments.

    Args:
        segment_lengths: Length of each segment.

    Returns:
        Array of scaled interpolation parameters with merged endpoints.
    """
    u_local = flat_1d_upbm(segment_lengths[0])
    current_index = 1.0
    
    for seg_len in segment_lengths[1:]:
        u_local = u_local[:-1]
        next_u = flat_1d_upbm(seg_len)
        u_local = np.concatenate((u_local, next_u + current_index))
        current_index += 1.0
    
    return u_local


def get_per_channel_coords_merged_endpoints(segment_lengths: np.ndarray) -> List[np.ndarray]:
    """
    Generate local u arrays for each segment with merged endpoints.

    Args:
        segment_lengths: Length of each segment.

    Returns:
        List of local u arrays for each segment.
    """
    u_local = flat_1d_upbm(segment_lengths[0])
    if len(segment_lengths) == 1:
        return [u_local]
    
    per_channel_coords = [u_local[:-1]]
    for seg_len in segment_lengths[1:-1]:
        u_local = flat_1d_upbm(seg_len)
        per_channel_coords.append(u_local[:-1])
    
    u_local = flat_1d_upbm(segment_lengths[-1])
    per_channel_coords.append(u_local)
    return per_channel_coords


def get_per_channel_coords(segment_lengths: np.ndarray, offset: int = 1) -> List[np.ndarray]:
    """
    Generate local u arrays for each segment.

    Args:
        segment_lengths: Length of each segment.
        offset: Points between segments:
            - 0: merge endpoints
            - 1: adjacent (no overlap, no gap)
            - >1: add (offset-1) repeated points between segments
    Returns:
        List of local u arrays for each segment.
    """
    if offset < 0:
        raise ValueError("Offset must be non-negative")
    elif offset == 0:
        return get_per_channel_coords_merged_endpoints(segment_lengths)
    
    per_channel_coords = []
    for seg_len in segment_lengths:
        u_local = flat_1d_upbm(seg_len)
        if offset > 1:
            padding = np.ones(offset - 1, dtype=u_local.dtype)
            u_local = np.concatenate((padding, u_local))
        per_channel_coords.append(u_local)
    
    return per_channel_coords


def get_uniform_per_channel_coords(total_steps: int, num_segments: int) -> List[np.ndarray]:
    """
    Generate uniformly distributed local u arrays for each segment.

    Args:
        total_steps: Total number of steps in the gradient.
        num_segments: Number of segments.

    Returns:
        List of local u arrays for each segment.
    """
    if total_steps < 2:
        raise ValueError("total_steps must be >= 2")
    
    indices = np.arange(total_steps, dtype=float)
    scale = num_segments / (total_steps - 1)
    per_channel_coords: List[np.ndarray] = []
    start = 0
    
    for seg_idx in range(num_segments):
        if seg_idx < num_segments - 1:
            stop = int(np.floor((total_steps - 1) * (seg_idx + 1) / num_segments)) + 1
        else:
            stop = total_steps
        
        this_slice = indices[start:stop]
        local_u = this_slice * scale - seg_idx
        per_channel_coords.append(local_u)
        start = stop
    
    return per_channel_coords


def construct_scaled_u(segment_lengths: np.ndarray, offset: int = 1) -> np.ndarray:
    """
    Construct scaled u values for all segments combined.

    Args:
        segment_lengths: Length of each segment.
        offset: Points between segments:
            - 0: merge endpoints
            - 1: adjacent (no overlap, no gap)
            - >1: add (offset-1) repeated points between segments
    Returns:
        Array of scaled interpolation parameters.
    """
    if offset < 0:
        raise ValueError("Offset must be non-negative")
    
    if offset == 0:
        return merge_endpoint_scaled_u(segment_lengths)
    
    u_local = flat_1d_upbm(segment_lengths[0])
    current_index = 1
    
    for seg_len in segment_lengths[1:]:
        next_u = flat_1d_upbm(seg_len)
        if offset > 1:
            padding = np.ones(offset - 1, dtype=next_u.dtype)
            next_u = np.concatenate((padding, next_u))
        u_local = np.concatenate((u_local, next_u + current_index))
        current_index += 1
    
    return u_local


def get_segments_from_scaled_u(arr: np.ndarray, max_value: float) -> List[tuple[int, np.ndarray]]:
    """
    Extract segments from scaled u array.
    
    Args:
        arr: Array of scaled u values
        max_value: Maximum segment index
        
    Returns:
        List of (segment_index, local_u_values) tuples
    """
    if not arr.size:
        raise ValueError("Input array must not be empty.")
    
    current_low = int(arr[0])
    current_segment = [arr[0]]
    segments = []
    
    for value in arr[1:]:
        value_floor = int(np.floor(value))
        if value_floor == current_low or (value_floor == max_value and current_low == max_value - 1):
            current_segment.append(value)
        else:
            segments.append((current_low, np.array(current_segment) - current_low))
            current_segment = [value]
            current_low = value_floor
    
    segments.append((current_low, np.array(current_segment) - current_low))
    return segments


__all__ = [
    'get_segment_lengths',
    'get_segment_indices',
    'merge_endpoint_scaled_u',
    'get_per_channel_coords_merged_endpoints',
    'get_per_channel_coords',
    'get_uniform_per_channel_coords',
    'construct_scaled_u',
    'get_segments_from_scaled_u',
]
