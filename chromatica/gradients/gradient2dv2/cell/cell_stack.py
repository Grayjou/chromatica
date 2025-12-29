from .base import CellBase
from typing import List, Optional
import numpy as np


class CellStackBottom:
    def __init__(self, cells: List[CellBase]) -> None:
        self.cells = cells
        self._value: Optional[np.ndarray] = None