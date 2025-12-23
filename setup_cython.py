"""
Setup script for building Cython extensions in chromatica.v2core

Usage:
    python setup_cython.py build_ext --inplace

This will compile all .pyx files in the chromatica/v2core directory.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V2CORE_DIR = os.path.join(BASE_DIR, "chromatica", "v2core")

# Define extensions
extensions = [
    Extension(
        "chromatica.v2core.border_handling",
        [os.path.join(V2CORE_DIR, "border_handling.pyx")],
        include_dirs=[],
        extra_compile_args=['-O3'],
    ),
    Extension(
        "chromatica.v2core.interp",
        [os.path.join(V2CORE_DIR, "interp.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
    Extension(
        "chromatica.v2core.interp_2d",
        [os.path.join(V2CORE_DIR, "interp_2d.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
    Extension(
        "chromatica.v2core.interp_hue",
        [os.path.join(V2CORE_DIR, "interp_hue.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
]

# Compiler directives for Cython
compiler_directives = {
    'language_level': '3',
    'boundscheck': False,
    'wraparound': False,
    'nonecheck': False,
    'cdivision': True,
    'initializedcheck': False,
}

setup(
    name="chromatica-cython-extensions",
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=False,  # Set to True to generate HTML annotation files
    ),
    include_dirs=[np.get_include()],
)

print("\n" + "=" * 70)
print("Cython extensions built successfully!")
print("=" * 70)
print("\nBuilt extensions:")
for ext in extensions:
    print(f"  - {ext.name}")
print("\nYou can now import these modules in Python:")
print("  from chromatica.v2core import border_handling")
print("  from chromatica.v2core import interp")
print("  from chromatica.v2core import interp_2d")
print("  from chromatica.v2core import interp_hue")
print("=" * 70)
