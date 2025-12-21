
from enum import Enum
from typing import List, Sequence, Optional, Callable, Any
import numpy as np
class IncreaseStrategy(Enum):
    EXTEND = "extend"

class DecreaseStrategy(Enum):
    TRUNCATE = "truncate"

def handle_list_size_mismatch(
        input_list: Sequence,
        target_size: int,
        increase_strategy: IncreaseStrategy = IncreaseStrategy.EXTEND,
        decrease_strategy: DecreaseStrategy = DecreaseStrategy.TRUNCATE,
        fill_value: Any = None,
        mutate: bool = False,):
    """Adjusts the size of input_list to match target_size using specified strategies.
    Args:
        input_list (Sequence): The original list to be adjusted.
        target_size (int): The desired size of the list.
        increase_strategy (IncreaseStrategy): Strategy to use when increasing size.
        decrease_strategy (DecreaseStrategy): Strategy to use when decreasing size.
        fill_value (Any): Value to use for extending the list.
        mutate (bool): If True, modifies the original list; otherwise returns a new list.
    Returns:
        Sequence: The adjusted list with size equal to target_size.
    """
    if mutate:
        lst = input_list
    else:
        lst = list(input_list)
    current_size = len(lst)
    if fill_value is None and len(lst) > 0:
        fill_value = lst[-1]
    if current_size < target_size:
        if increase_strategy == IncreaseStrategy.EXTEND:
            lst.extend([fill_value] * (target_size - current_size))
    elif current_size > target_size:
        if decrease_strategy == DecreaseStrategy.TRUNCATE:
            del lst[target_size:]
    return lst

#TODO: 
# INTERPOLATE strategy for increasing and decreasing list size
# Let fill_value be a callable that generates values based on index or other criteria (or perhaps create a separate parameter for this ?)

def split_and_distribute_remainder(total_amount: int, num_intervals: int) -> np.ndarray:
    """
    Calculate how to distribute frames across intervals.
    
    Uses the remainder distribution algorithm:
    - Base frames per interval = floor(total_frames / num_intervals)
    - Remainder = total_frames mod num_intervals
    - Distribute remainder by adding 1 to intervals, biased toward end
    
    Args:
        total_frames: Total number of frames to distribute
        num_intervals: Number of intervals to distribute across
        
    Returns:
        List of frame counts per interval
    """
    base = total_amount // num_intervals
    remainder = total_amount % num_intervals

    distribution = np.full(num_intervals, base)

    if remainder:
        idx = np.linspace(
            num_intervals - remainder,
            num_intervals - 1,
            remainder,
            dtype=int
        )
        distribution[idx] += 1

    return distribution