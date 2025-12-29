from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Union, List, Any
import numpy as np

from ....types.transform_types import PerChannelCoords
from unitfield import upbm_2d


class BaseCellFactoryProperties(ABC):
	"""Base class for shared factory property and cache management.

	This consolidates common logic used by Corners/Lines/Dual factories:
	- Dimension changes invalidate cell and coords
	- Per-channel coords lazy initialization and syncing with cell
	- Per-channel transforms invalidation
	- Cache helpers and base coord generation
	"""

	def __init__(
		self,
		width: int,
		height: int,
		*,
		per_channel_coords: Optional[PerChannelCoords] = None,
	) -> None:
		if width <= 0 or height <= 0:
			raise ValueError("Width and height must be positive integers.")
		self._width = width
		self._height = height
		self._cell: Optional[Any] = None
		self._per_channel_coords: Optional[PerChannelCoords] = per_channel_coords
		self._per_channel_transforms: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]] = None

	# === Dimension Properties ===
	@property
	def width(self) -> int:
		return self._width

	@width.setter
	def width(self, value: int) -> None:
		if self._width != value:
			self._width = value
			self._invalidate_for_dimension_change()

	@property
	def height(self) -> int:
		return self._height

	@height.setter
	def height(self, value: int) -> None:
		if self._height != value:
			self._height = value
			self._invalidate_for_dimension_change()

	def _invalidate_for_dimension_change(self) -> None:
		self._cell = None
		self._per_channel_coords = None

	# === Per-Channel Coordinates ===
	@property
	def per_channel_coords(self) -> PerChannelCoords:
		if self._cell is not None:
			return self._cell.per_channel_coords
		if self._per_channel_coords is None:
			self._per_channel_coords = self.base_coords()
		return self._per_channel_coords

	@per_channel_coords.setter
	def per_channel_coords(self, value: Optional[PerChannelCoords]) -> None:
		if value is not None and not isinstance(value, (list, np.ndarray)):
			raise TypeError("per_channel_coords must be a list, ndarray, or None")
		self._per_channel_coords = value
		if self._cell is not None:
			if value is not None:
				self._cell.per_channel_coords = value
				self._cell.invalidate_cache()
			else:
				# Reset cell so base coords will be used on next render
				self._cell = None

	# === Per-Channel Transforms ===
	@property
	def per_channel_transforms(self) -> Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]]:
		return self._per_channel_transforms

	@per_channel_transforms.setter
	def per_channel_transforms(self, value: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]]) -> None:
		self._per_channel_transforms = value
		self._cell = None
		self._per_channel_coords = None

	# === Abstract Properties ===
	@property
	@abstractmethod
	def num_channels(self) -> int:
		"""Number of channels in the working color space."""
		raise NotImplementedError

	@abstractmethod
	def _get_color_space_for_repr(self) -> str:
		"""Return color space info for __repr__."""
		raise NotImplementedError

	# === Cache Management ===
	def invalidate_cell_cache(self) -> None:
		if self._cell is not None:
			self._cell.invalidate_cache()

	def invalidate_cell(self) -> None:
		self._cell = None

	def reset_per_channel_coords(self) -> None:
		self._per_channel_coords = None
		self._cell = None

	# === Coordinate Creation ===
	def base_coords(self) -> PerChannelCoords:
		return upbm_2d(width=self._width, height=self._height) 

	def get_per_channel_coords(self) -> PerChannelCoords:
		if self._per_channel_coords is None:
			return self.base_coords()
		return self._per_channel_coords

	# === Representation ===
	def __repr__(self) -> str:
		cell_status = "cached" if self._cell is not None else "not created"
		return (
			f"{self.__class__.__name__}("
			f"width={self._width}, height={self._height}, "
			f"{self._get_color_space_for_repr()}, "
			f"cell={cell_status})"
		)

