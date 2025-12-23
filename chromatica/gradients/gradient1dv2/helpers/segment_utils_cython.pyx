# cython: boundscheck=False, wraparound=False, nonecheck=False

"""
Cython-optimized segment utilities for gradient1dv2.

This module provides high-performance implementations of segment extraction
from scaled u arrays, which is critical for gradient rendering performance.
"""

import numpy as np
cimport numpy as np

def get_segments_from_scaled_u_cython(
    np.ndarray[np.double_t, ndim=1] arr,
    int max_value
):
    """
    Extract segments from scaled u array using Cython optimization.
    
    This is a direct Cython port of the Python implementation for improved performance.
    
    Args:
        arr: Array of scaled u values (1D double array)
        max_value: Maximum segment index
        
    Returns:
        List of (segment_index, local_u_values) tuples
        
    Example:
        >>> arr = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        >>> segments = get_segments_from_scaled_u_cython(arr, 2)
        >>> # Returns [(0, [0.0, 0.5]), (1, [0.0, 0.5]), (2, [0.0])]
    """
    cdef Py_ssize_t i, n = arr.shape[0]
    if n == 0:
        raise ValueError("Input array must not be empty.")

    cdef int current_low = <int>arr[0]
    cdef double value
    cdef int value_floor

    current_segment = [arr[0]]
    segments = []

    for i in range(1, n):
        value = arr[i]
        value_floor = <int>value  # faster than floor()

        if value_floor == current_low or (
            value_floor == max_value and current_low == max_value - 1
        ):
            current_segment.append(value)
        else:
            segments.append(
                (current_low, np.array(current_segment) - current_low)
            )
            current_segment = [value]
            current_low = value_floor

    segments.append(
        (current_low, np.array(current_segment) - current_low)
    )
    return segments


def get_segments_from_scaled_u_cython_v2(
    np.ndarray[np.double_t, ndim=1] arr,
    int max_value
):
    """
    Extract segments from scaled u array using optimized two-pass algorithm.
    
    This version pre-computes all floor values and uses two passes:
    1. Count segments
    2. Collect segment boundaries
    Then uses fast array slicing to create segments.
    
    Args:
        arr: Array of scaled u values (1D double array)
        max_value: Maximum segment index
        
    Returns:
        List of (segment_index, local_u_values) tuples
        
    Note:
        This version may be faster for large arrays due to reduced allocations.
    """
    cdef Py_ssize_t i, n = arr.shape[0]
    if n == 0:
        raise ValueError("Input array must not be empty.")

    # Pre-compute floors in a typed array
    cdef np.ndarray[np.int32_t, ndim=1] floors = np.empty(n, dtype=np.int32)
    for i in range(n):
        floors[i] = <int>arr[i]

    # First pass: count segments
    cdef Py_ssize_t num_segments = 1
    cdef int curr_floor, next_floor

    for i in range(n - 1):
        curr_floor = floors[i]
        next_floor = floors[i + 1]
        if next_floor != curr_floor:
            if not (next_floor == max_value and curr_floor == max_value - 1):
                num_segments += 1

    # Second pass: collect start indices
    cdef np.ndarray[np.int64_t, ndim=1] starts = np.empty(num_segments, dtype=np.int64)
    cdef Py_ssize_t seg_idx = 1
    starts[0] = 0

    for i in range(n - 1):
        curr_floor = floors[i]
        next_floor = floors[i + 1]
        if next_floor != curr_floor:
            if not (next_floor == max_value and curr_floor == max_value - 1):
                starts[seg_idx] = i + 1
                seg_idx += 1

    # Build segments using array slicing (fast bulk operations)
    cdef Py_ssize_t s, e
    cdef int floor_val
    
    segments = []
    for i in range(num_segments):
        s = starts[i]
        e = starts[i + 1] if i < num_segments - 1 else n
        floor_val = floors[s]
        segments.append((floor_val, arr[s:e] - floor_val))

    return segments
