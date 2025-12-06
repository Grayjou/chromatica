from typing import Any
from collections.abc import Sized

def get_dimension(element: Any) -> int:
    if element is None:
        return 0
    if isinstance(element, Sized):
        return len(element)
    return 1
