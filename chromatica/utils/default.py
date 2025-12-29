from typing import Optional, TypeVar

T = TypeVar('T')

def value_or_default(value: Optional[T], default: T) -> T:
    """Return the value if it is not None, otherwise return the default."""
    return value if value is not None else default