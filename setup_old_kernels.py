from setuptools import setup
from Cython.Build import cythonize
import numpy as np

paths = [
    "chromatica/v2core/interp.pyx",
    "chromatica/v2core/border_handling.pyx",
    "chromatica/v2core/interp_hue/interp_hue_utils.pyx",
    "chromatica/v2core/interp_hue/interp_hue2d.pyx",
    "chromatica/v2core/interp_hue/interp_hue.pyx",
    "chromatica/v2core/interp_2d/helpers.pyx",
    "chromatica/v2core/interp_2d/corner_interp_2d_fast.pyx",
    "chromatica/v2core/interp_2d/interp_2d_fast.pyx",
    "chromatica/v2core/interp_2d/interp_planes.pyx",
]

setup(
    ext_modules=cythonize(
        paths,
        language_level=3
    ),
    include_dirs=[np.get_include()],
)
