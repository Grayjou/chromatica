"""Base class for 2D gradient cells."""

from __future__ import annotations
from abc import abstractmethod
from ....v2core.subgradient import SubGradient
from ..helpers import CellMode


class CellBase(SubGradient):
    """Abstract base class for 2D gradient cells, extending SubGradient."""
    
    mode: CellMode
    
    def __init__(self):
        """Initialize with no cached value."""
        super().__init__()
