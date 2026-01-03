# chromatica/gradients/gradient2dv2/cell/_descriptors.py
"""Property descriptors for cell classes."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class CellPropertyDescriptor:
    """Descriptor for cell properties with automatic cache invalidation.
    
    Provides a declarative way to define properties that automatically
    invalidate caches when modified.
    
    Args:
        attr_name: Name of the attribute (stored as _{attr_name})
        invalidates_segments: Whether to call _invalidate_segments() on set
        invalidates_cache: Whether to call invalidate_cache() on set
        readonly: If True, raises AttributeError on set attempts
    
    Example:
        class MyCell(CellBase):
            # Writable property that invalidates cache
            color: np.ndarray = CellPropertyDescriptor('color')
            
            # Read-only property
            space: ColorModes = CellPropertyDescriptor('space', readonly=True)
            
            # Property that also invalidates segment caches
            corner: np.ndarray = CellPropertyDescriptor(
                'corner', invalidates_segments=True
            )
    """
    
    def __init__(
        self,
        attr_name: str,
        *,
        invalidates_segments: bool = False,
        invalidates_cache: bool = True,
        readonly: bool = False,
    ):
        self.attr_name = attr_name
        self.private_name = f"_{attr_name}"
        self.invalidates_segments = invalidates_segments
        self.invalidates_cache = invalidates_cache
        self.readonly = readonly
        self.public_name: str = attr_name  # May be overwritten by __set_name__
    
    def __set_name__(self, owner: type, name: str) -> None:
        """Capture the attribute name as defined on the class."""
        self.public_name = name
    
    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return getattr(obj, self.private_name)
    
    def __set__(self, obj: Any, value: Any) -> None:
        if self.readonly:
            raise AttributeError(
                f"'{self.public_name}' is read-only on {obj.__class__.__name__}"
            )
        
        setattr(obj, self.private_name, value)
        
        # Invalidate segment cache if applicable
        if self.invalidates_segments and hasattr(obj, '_invalidate_segments'):
            obj._invalidate_segments()
        
        # Invalidate main cache
        if self.invalidates_cache and hasattr(obj, 'invalidate_cache'):
            obj.invalidate_cache()
    
    def __repr__(self) -> str:
        flags = []
        if self.readonly:
            flags.append("readonly")
        if self.invalidates_segments:
            flags.append("invalidates_segments")
        if self.invalidates_cache:
            flags.append("invalidates_cache")
        flags_str = ", ".join(flags) if flags else "writable"
        return f"CellPropertyDescriptor({self.attr_name!r}, {flags_str})"