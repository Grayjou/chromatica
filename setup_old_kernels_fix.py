from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

extensions = [
    Extension(
        "chromatica.v2core.interp",
        ["chromatica/v2core/interp.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "chromatica.v2core.border_handling",
        ["chromatica/v2core/border_handling.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="chromatica-old-kernels",
    ext_modules=cythonize(
        extensions,
        language_level=3,
    ),
    zip_safe=False,
)
