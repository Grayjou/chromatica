
from typing import Callable, Dict, Optional, Tuple, TypeAlias, Union, List
from numpy.typing import NDArray

from unitfield import Unit2DMappedEndomorphism, UnitNdimField
import numpy as np

PerChannelCoords = Union[np.ndarray, List[np.ndarray]]
UnitTransform = Callable[[NDArray], NDArray]
CoordsArray: TypeAlias = Tuple[NDArray, NDArray]
Remap2D = Callable[[NDArray, NDArray], CoordsArray] | Unit2DMappedEndomorphism
BiVariableSpaceTransform = Dict[int, Remap2D]
BiVariableColorTransform = Dict[int, Callable[[NDArray, NDArray, NDArray], NDArray]]

PerChannelTransform = Dict[int, UnitTransform]

def get_bivar_space_transforms(transform_dict: Optional[BiVariableSpaceTransform] = None) -> Dict[int, Callable[[NDArray, NDArray], Tuple[NDArray, NDArray]]]:
    """Convert BiVariableSpaceTransform to a uniform callable dictionary."""
    if transform_dict is None:
        return {}
    
    result: Dict[int, Callable[[NDArray, NDArray], Tuple[NDArray, NDArray]]] = {}
    for key, transform in transform_dict.items():
        if isinstance(transform, Unit2DMappedEndomorphism):
            result[key] = transform.remap
        else:
            result[key] = transform
    return result