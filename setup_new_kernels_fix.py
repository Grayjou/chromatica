from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build with proper package structure
extensions = [
    Extension(
        "chromatica.v2core.interp_utils",
        ["chromatica/v2core/interp_utils.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "chromatica.v2core.interp_2d_.border_handling",
        ["chromatica/v2core/interp_2d_/border_handling.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "chromatica.v2core.interp_2d_.interp_2d_fast_",
        ["chromatica/v2core/interp_2d_/interp_2d_fast_.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "chromatica.v2core.interp_2d_.interp_2d_array_border",
        ["chromatica/v2core/interp_2d_/interp_2d_array_border.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "chromatica.v2core.interp_2d_.corner_interp_2d_fast_",
        ["chromatica/v2core/interp_2d_/corner_interp_2d_fast_.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "chromatica.v2core.interp_2d_.corner_interp_2d_border_",
        ["chromatica/v2core/interp_2d_/corner_interp_2d_border_.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "chromatica.v2core.interp_hue_.interp_hue_utils",
        ["chromatica/v2core/interp_hue_/interp_hue_utils.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "chromatica.v2core.interp_hue_.interp_hue",
        ["chromatica/v2core/interp_hue_/interp_hue.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "chromatica.v2core.interp_hue_.interp_hue2d",
        ["chromatica/v2core/interp_hue_/interp_hue2d.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "chromatica.v2core.interp_hue_.interp_hue2d_array_border",
        ["chromatica/v2core/interp_hue_/interp_hue2d_array_border.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "chromatica.v2core.interp_hue_.interp_hue_array_border",
        ["chromatica/v2core/interp_hue_/interp_hue_array_border.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "chromatica.v2core.interp_hue_.interp_hue_corners",
        ["chromatica/v2core/interp_hue_/interp_hue_corners.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "chromatica.v2core.interp_hue_.interp_hue_corners_array_border",
        ["chromatica/v2core/interp_hue_/interp_hue_corners_array_border.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name="chromatica-new-kernels",
    ext_modules=cythonize(
        extensions,
        language_level=3,
    ),
    zip_safe=False,
)
