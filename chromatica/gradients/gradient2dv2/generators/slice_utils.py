from __future__ import annotations
from typing import Union, List
import numpy as np

from ....types.transform_types import PerChannelCoords


def slice_pcc_with_padding(
	pcc: Union[List[np.ndarray], np.ndarray],
	start_idx: int,
	end_idx: int,
	pad_left: int,
	pad_right: int,
) -> Union[List[np.ndarray], np.ndarray]:
	"""Slice per_channel_coords with boundary padding.

	Args:
		pcc: Per-channel coordinates (list of arrays or single array)
		start_idx: Start index (inclusive)
		end_idx: End index (exclusive)
		pad_left: Number of columns to pad on the left using boundary values
		pad_right: Number of columns to pad on the right using boundary values

	Returns:
		Sliced coordinates keeping the original structure type.
	"""
	if isinstance(pcc, list):
		return _slice_list_pcc(pcc, start_idx, end_idx, pad_left, pad_right)
	elif isinstance(pcc, np.ndarray):
		return _slice_array_pcc(pcc, start_idx, end_idx, pad_left, pad_right)
	else:
		raise TypeError(f"Unsupported type for per_channel_coords: {type(pcc)}")


def _slice_list_pcc(
	pcc: List[np.ndarray],
	start_idx: int,
	end_idx: int,
	pad_left: int,
	pad_right: int,
) -> List[np.ndarray]:
	"""Slice list-based per_channel_coords with optional boundary padding."""
	base_slices = [c[:, start_idx:end_idx] for c in pcc]

	if pad_left == 0 and pad_right == 0:
		return base_slices

	result: List[np.ndarray] = []
	for ch_idx, base_slice in enumerate(base_slices):
		parts = []

		if pad_left > 0:
			left_boundary = pcc[ch_idx][:, start_idx - 1:start_idx]
			parts.extend([left_boundary] * pad_left)

		parts.append(base_slice)

		if pad_right > 0:
			right_boundary = base_slice[:, -1:]
			parts.extend([right_boundary] * pad_right)

		result.append(np.concatenate(parts, axis=1))

	return result


def _slice_array_pcc(
	pcc: np.ndarray,
	start_idx: int,
	end_idx: int,
	pad_left: int,
	pad_right: int,
) -> np.ndarray:
	"""Slice array-based per_channel_coords with optional boundary padding."""
	base_slice = pcc[:, start_idx:end_idx]

	if pad_left == 0 and pad_right == 0:
		return base_slice

	parts = []

	if pad_left > 0:
		left_boundary = pcc[:, start_idx - 1:start_idx]
		parts.extend([left_boundary] * pad_left)

	parts.append(base_slice)

	if pad_right > 0:
		right_boundary = base_slice[:, -1:]
		parts.extend([right_boundary] * pad_right)

	return np.concatenate(parts, axis=1)


def _slice_line_with_padding(
	line: np.ndarray,
	start_idx: int,
	end_idx: int,
	pad_left: int,
	pad_right: int,
) -> np.ndarray:
	"""Slice a 1D line array with optional boundary padding.

	The padding replicates boundary elements:
	- pad_left: copies element at (start_idx - 1), prepended to left
	- pad_right: copies last element of base slice, appended to right

	Args:
		line: 1D array of colors or values
		start_idx: Start index (inclusive)
		end_idx: End index (exclusive)
		pad_left: Number of times to repeat left boundary element
		pad_right: Number of times to repeat right boundary element

	Returns:
		Sliced and padded 1D array
	"""
	base_slice = line[start_idx:end_idx]

	if pad_left == 0 and pad_right == 0:
		return base_slice.copy()

	parts: List[np.ndarray] = []

	if pad_left > 0:
		left_boundary = line[start_idx - 1:start_idx]
		parts.extend([left_boundary] * pad_left)

	parts.append(base_slice)

	if pad_right > 0:
		right_boundary = base_slice[-1:]
		parts.extend([right_boundary] * pad_right)

	return np.concatenate(parts, axis=0)


def slice_lines_with_padding(
	top_line: np.ndarray,
	bottom_line: np.ndarray,
	start_idx: int,
	end_idx: int,
	pad_left: int,
	pad_right: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Slice top and bottom 1D line arrays with boundary padding.

	Returns:
		Tuple of (sliced_top_line, sliced_bottom_line)
	"""
	return (
		_slice_line_with_padding(top_line, start_idx, end_idx, pad_left, pad_right),
		_slice_line_with_padding(bottom_line, start_idx, end_idx, pad_left, pad_right),
	)

