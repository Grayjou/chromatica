#chromatica\gradients\gradient2dv2\generators\helpers.py
from ....types.transform_types import PerChannelCoords
from typing import Union, Tuple, List
import numpy as np


def _add_padding(
        slice1: PerChannelCoords,
        slice2: PerChannelCoords,
        edge_column: PerChannelCoords,
        padding: int
    ):
    # 0 padding means slice1 gets the edge column on the right, slice2 doesn't get any padding
    # 1 padding means slice1 gets edge column on right, slice2 gets edge column on left
    # 2 padding means slice1 gets edge column on right twice, slice2 gets edge column on left once
    # 3 padding means slice1 gets edge column on right twice, slice2 gets edge column on left twice
    # So padding for slice1 is (padding + 1) // 2 and for slice2 is padding // 2
    if padding <= 0:
        raise ValueError("Padding must be a positive integer.")
    pad1 = (padding + 1) // 2
    pad2 = padding // 2
    # Slice1, Slice2 and edge column share typing becasue they come from the same per_channel_coords
    if isinstance(slice1, list):
        new_slice1 = [
            np.concatenate(
                [s1] + [edge_column[i]] * pad1,
                axis=1
            ) for i, s1 in enumerate(slice1)
        ]
        new_slice2 = [
            np.concatenate(
                [edge_column[i]] * pad2 + [s2],
                axis=1
            ) for i, s2 in enumerate(slice2)
        ]
    elif isinstance(slice1, np.ndarray):
        new_slice1 = np.concatenate(
            [slice1] + [edge_column] * pad1,
            axis=1
        )
        new_slice2 = np.concatenate(
            [edge_column] * pad2 + [slice2],
            axis=1
        )
    else:
        raise TypeError("Unsupported type for per-channel coordinates.")
    return new_slice1, new_slice2

def slice_pcc(
    per_channel_coords: PerChannelCoords,
    start_idx: int,
    end_idx: int,
    padding: int = 0
) -> Tuple[PerChannelCoords, PerChannelCoords]:
    """Slice per-channel coordinates with optional padding.

    Args:
        per_channel_coords (PerChannelCoords): The per-channel coordinates to slice.
        start_idx (int): The starting index for slicing.
        end_idx (int): The ending index for slicing.
        padding (int): Number of times the edge values should be padded.

    Returns:
        Tuple[PerChannelCoords, PerChannelCoords]: (left_slice, right_slice)
    """
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError("Invalid start_idx / end_idx combination.")

    # ---- list of channels ----
    if isinstance(per_channel_coords, list):
        slice1 = [c[:, start_idx:end_idx] for c in per_channel_coords]
        slice2 = [c[:, end_idx:] for c in per_channel_coords]

        if padding > 0:
            edge_column = [c[:, end_idx - 1:end_idx] for c in per_channel_coords]
            slice1, slice2 = _add_padding(slice1, slice2, edge_column, padding)

    # ---- single ndarray ----
    elif isinstance(per_channel_coords, np.ndarray):
        slice1 = per_channel_coords[:, start_idx:end_idx]
        slice2 = per_channel_coords[:, end_idx:]

        if padding > 0:
            edge_column = per_channel_coords[:, end_idx - 1:end_idx]
            slice1, slice2 = _add_padding(slice1, slice2, edge_column, padding)

    else:
        raise TypeError("Unsupported type for per-channel coordinates.")

    return slice1, slice2
