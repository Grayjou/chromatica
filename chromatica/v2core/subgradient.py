"""
Base classes for gradient segments and cells.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from ..types.color_types import ColorSpaces


class SubGradient(ABC):
    """
    Abstract base class for gradient subsections (segments and cells).
    
    This class provides common functionality for both 1D gradient segments
    and 2D gradient cells, including lazy value computation and color space conversion.
    """
    
    __slots__ = ('_value',)
    
    def __init__(self):
        """Initialize with no cached value."""
        self._value: Optional[np.ndarray] = None
    
    def get_value(self) -> np.ndarray:
        """
        Get the interpolated color values, computing them if needed.
        
        Returns:
            Array of interpolated color values
        """
        if self._value is None:
            self._value = self._render_value()
        return self._value
    
    @abstractmethod
    def _render_value(self) -> np.ndarray:
        """
        Compute the interpolated color values.
        
        Must be implemented by subclasses.
        
        Returns:
            Array of interpolated color values
        """
        pass
    
    @property
    def format_type(self) -> str:
        """Get the format type of the color values."""
        return "float"
    
    @abstractmethod
    def convert_to_space(self, color_space: ColorSpaces) -> 'SubGradient':
        """
        Convert this gradient subsection to a different color space.
        
        Args:
            color_space: Target color space
            
        Returns:
            New SubGradient instance in the target color space
        """
        pass

    def invalidate_cache(self):
        """Invalidate the cached value, forcing recomputation on next access."""
        self._value = None

__all__ = ['SubGradient']
