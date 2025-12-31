from setuptools import setup
from Cython.Build import cythonize
import numpy as np

paths = [
    "chromatica/v2core/interp_2d_/border_handling.pyx",
    "chromatica/v2core/interp_2d_/interp_2d_fast_.pyx",
    "chromatica/v2core/interp_2d_/interp_2d_array_border.pyx",
    "chromatica/v2core/interp_2d_/corner_interp_2d_fast_.pyx",
    "chromatica/v2core/interp_hue_/interp_hue_utils.pyx",
    "chromatica/v2core/interp_utils.pyx",
    "chromatica/v2core/interp_2d_/corner_interp_2d_border_.pyx",
    "chromatica/v2core/interp_hue_/interp_hue.pyx",
    "chromatica/v2core/interp_hue_/interp_hue2d.pyx",
    "chromatica/v2core/interp_hue_/interp_hue2d_array_border.pyx",
    "chromatica/v2core/interp_hue_/interp_hue_array_border.pyx",
    "chromatica/v2core/interp_hue_/interp_hue_corners.pyx",
    "chromatica/v2core/interp_hue_/interp_hue_corners_array_border.pyx",
]

setup(
    ext_modules=cythonize(
        paths,
        language_level=3
    ),
    include_dirs=[np.get_include()],
)
