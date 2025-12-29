from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any
import numpy as np

from ..partitions import (
	PerpendicularPartition,
	IndexRoundingMode,
	index_rounding_mode_functions,
)


@dataclass
class SliceSpec:
	"""Specification for a single partition slice."""
	index: int
	start_frac: float
	end_frac: float
	px_start: int
	px_end: int
	width: int
	pad_left: int
	pad_right: int
	interval: Any  # PartitionInterval or similar


def compute_partition_slices(
	partition: PerpendicularPartition,
	total_width: int,
	padding: int = 1,
	index_rounding_mode: IndexRoundingMode = IndexRoundingMode.ROUND,
) -> List[SliceSpec]:
	"""Compute slice specifications for a partition.

	Centralizes the shared geometry logic used by factories when slicing
	into multiple intervals with optional boundary padding.
	"""
	intervals = list(partition.intervals())
	num_intervals = len(intervals)
	if num_intervals == 0:
		return []

	# Compute slice boundaries (exclusive end indices, Python slicing semantics)
	rounding_fn = index_rounding_mode_functions[index_rounding_mode]
	slice_ends = (
		[0]
		+ [int(rounding_fn(bp * (total_width - 1))) + 1 for bp in partition.breakpoints]
		+ [total_width]
	)

	# Base widths
	widths = [slice_ends[i + 1] - slice_ends[i] for i in range(num_intervals)]

	# Padding split: left/right based on shared-edge duplication policy
	padding = max(0, padding)
	extra_left = padding // 2
	extra_right = padding - extra_left

	# Pad amounts per interval for coord slicing helpers
	pad_lefts = [0] + [extra_right] * (num_intervals - 1)
	pad_rights = [extra_left] * (num_intervals - 1) + [0]

	# Final widths after padding is applied to neighbors
	final_widths = widths.copy()
	for i in range(num_intervals - 1):
		final_widths[i] += extra_left
	for i in range(1, num_intervals):
		final_widths[i] += extra_right

	specs: List[SliceSpec] = []
	for i, (start_frac, end_frac, interval) in enumerate(intervals):
		specs.append(
			SliceSpec(
				index=i,
				start_frac=start_frac,
				end_frac=end_frac,
				px_start=slice_ends[i],
				px_end=slice_ends[i + 1],
				width=final_widths[i],
				pad_left=pad_lefts[i],
				pad_right=pad_rights[i],
				interval=interval,
			)
		)

	return specs

