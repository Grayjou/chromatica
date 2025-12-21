# No dependencies
from enum import Enum
import numpy as np
class FormatType(str, Enum):
    INT = "int"
    FLOAT = "float"
    PERCENTAGE = "percentage"

max_non_hue = {
    FormatType.INT: 255,
    FormatType.FLOAT: 1.0,
    FormatType.PERCENTAGE: 100.0
}

format_classes = {
    FormatType.INT: int,
    FormatType.FLOAT: float,
    FormatType.PERCENTAGE: float,
}

default_format_dtypes = {
    FormatType.INT: np.uint64,
    FormatType.FLOAT: np.float32,
    FormatType.PERCENTAGE: np.float32,
}

format_valid_dtypes = {
    FormatType.INT: (int, np.integer),
    FormatType.FLOAT: (float, np.floating),
    FormatType.PERCENTAGE: (float, np.floating),
}

HUE_360 = 360