from typing import Any
from collections.abc import Sized

def get_dimension(element: Any) -> int:
    if element is None:
        return 0
    if isinstance(element, Sized):
        return len(element)
    return 1

# Do I keep this function with the most efficient algo, or simply use collections.Counter because it might be faster due to python loop overhead?
# Yeah I noticed Booyer-Moore is slower in python

from collections import Counter
def most_common_element(elements: list) -> Any:
    """
    Return the most common element in the list using collections.Counter.
    If there's a tie, return one of the most common elements.
    """
    if not elements:
        return None
    counter = Counter(elements)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None