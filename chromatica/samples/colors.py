import numpy as np
# RED
RED_FLOAT_RGB = np.array([1.0, 0.0, 0.0], dtype=np.float32)
RED_INT_RGB = np.array([255, 0, 0], dtype=np.uint8)
RED_FLOAT_HSV = np.array([0.0, 1.0, 1.0], dtype=np.float32)

# GREEN
GREEN_FLOAT_RGB = np.array([0.0, 1.0, 0.0], dtype=np.float32)
GREEN_INT_RGB = np.array([0, 255, 0], dtype=np.uint8)
GREEN_FLOAT_HSV = np.array([120, 1.0, 1.0], dtype=np.float32)

# BLUE
BLUE_FLOAT_RGB = np.array([0.0, 0.0, 1.0], dtype=np.float32)
BLUE_INT_RGB = np.array([0, 0, 255], dtype=np.uint8)
BLUE_FLOAT_HSV = np.array([240, 1.0, 1.0], dtype=np.float32)

# YELLOW
YELLOW_FLOAT_RGB = np.array([1.0, 1.0, 0.0], dtype=np.float32)
YELLOW_INT_RGB = np.array([255, 255, 0], dtype=np.uint8)
YELLOW_FLOAT_HSV = np.array([60, 1.0, 1.0], dtype=np.float32)

#MAGENTA
MAGENTA_FLOAT_RGB = np.array([1.0, 0.0, 1.0], dtype=np.float32)
MAGENTA_INT_RGB = np.array([255, 0, 255], dtype=np.uint8)
MAGENTA_FLOAT_HSV = np.array([300, 1.0, 1.0], dtype=np.float32)

# CYAN
CYAN_FLOAT_RGB = np.array([0.0, 1.0, 1.0], dtype=np.float32)
CYAN_INT_RGB = np.array([0, 255, 255], dtype=np.uint8)
CYAN_FLOAT_HSV = np.array([180, 1.0, 1.0], dtype=np.float32)

#WHITE
WHITE_FLOAT_RGB = np.array([1.0, 1.0, 1.0], dtype=np.float32)
WHITE_INT_RGB = np.array([255, 255, 255], dtype=np.uint8)
WHITE_FLOAT_HSV = np.array([0.0, 0.0, 1.0], dtype=np.float32)

#BLACK
BLACK_FLOAT_RGB = np.array([0.0, 0.0, 0.0], dtype=np.float32)
BLACK_INT_RGB = np.array([0, 0, 0], dtype=np.uint8)
BLACK_FLOAT_HSV = np.array([0.0, 0.0, 0.0], dtype=np.float32)

def get_white_hsv(hue: float) -> np.ndarray:
    """Get white color in HSV with specified hue."""
    return np.array([hue, 0.0, 1.0], dtype=np.float32)

def get_black_hsv(hue: float) -> np.ndarray:
    """Get black color in HSV with specified hue."""
    return np.array([hue, 0.0, 0.0], dtype=np.float32)

__all__ = [
    "RED_FLOAT_RGB",
    "RED_INT_RGB",
    "RED_FLOAT_HSV",
    "GREEN_FLOAT_RGB",
    "GREEN_INT_RGB",
    "GREEN_FLOAT_HSV",
    "BLUE_FLOAT_RGB",
    "BLUE_INT_RGB",
    "BLUE_FLOAT_HSV",
    "YELLOW_FLOAT_RGB",
    "YELLOW_INT_RGB",
    "YELLOW_FLOAT_HSV",
    "MAGENTA_FLOAT_RGB",
    "MAGENTA_INT_RGB",
    "MAGENTA_FLOAT_HSV",
    "CYAN_FLOAT_RGB",
    "CYAN_INT_RGB",
    "CYAN_FLOAT_HSV",
    "WHITE_FLOAT_RGB",
    "WHITE_INT_RGB",
    "WHITE_FLOAT_HSV",
    "BLACK_FLOAT_RGB",
    "BLACK_INT_RGB",
    "BLACK_FLOAT_HSV",
]