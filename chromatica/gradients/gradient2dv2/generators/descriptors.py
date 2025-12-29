#chromatica\gradients\gradient2dv2\generators\descriptors.py
from __future__ import annotations
from typing import Optional


class SyncedCellPropertyDescriptor:
	"""Unified descriptor for properties that sync with underlying cells.

	Use this in factory property classes to keep the factory's private state
	and the underlying cell instance in sync, while handling cache invalidation
	and optional per-channel coord resets.
	"""

	def __init__(
		self,
		attr_name: str,
		cell_attr: Optional[str] = None,
		invalidates_cell: bool = False,
		invalidates_cache: bool = True,
		resets_coords: bool = False,
	) -> None:
		self.attr_name = attr_name
		self.private_name = f"_{attr_name}"
		self.cell_attr = cell_attr or attr_name
		self.invalidates_cell = invalidates_cell
		self.invalidates_cache = invalidates_cache
		self.resets_coords = resets_coords

	def __set_name__(self, owner: type, name: str) -> None:
		# Store the public name if needed for debugging/introspection
		self.public_name = name

	def __get__(self, obj, objtype=None):
		if obj is None:
			return self
		return getattr(obj, self.private_name)

	def __set__(self, obj, value) -> None:
		# Update factory private attribute
		setattr(obj, self.private_name, value)

		# If changing this property requires a full cell rebuild
		if self.invalidates_cell:
			obj._cell = None
		elif obj._cell is not None:
			# Sync the property to the underlying cell if it has the same attribute
			if hasattr(obj._cell, self.cell_attr):
				setattr(obj._cell, self.cell_attr, value)
			# Invalidate rendered cache by default
			if self.invalidates_cache:
				obj._cell.invalidate_cache()

		# Optionally reset stored per-channel coords
		if self.resets_coords:
			obj._per_channel_coords = None

