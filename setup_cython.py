"""
Setup script for building Cython extensions in chromatica.v2core

Usage:
    python setup_cython.py build_ext --inplace

This will compile all .pyx files in the chromatica/v2core/interp_modules directory.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V2CORE_DIR = os.path.join(BASE_DIR, "chromatica", "v2core")
INTERP_MODULES_DIR = os.path.join(V2CORE_DIR, "interp_modules")

# Define extensions
extensions = [
    # Border handling
    Extension(
        "chromatica.v2core.interp_modules.border_handling",
        [os.path.join(INTERP_MODULES_DIR, "border_handling.pyx")],
        include_dirs=[],
        extra_compile_args=['-O3'],
    ),
    # Core interpolation
    Extension(
        "chromatica.v2core.interp_modules.interp",
        [os.path.join(INTERP_MODULES_DIR, "interp.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
    # 2D interpolation - single channel
    Extension(
        "chromatica.v2core.interp_modules.interp_2d_1ch",
        [os.path.join(INTERP_MODULES_DIR, "interp_2d_1ch.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
    # 2D interpolation - multi-channel
    Extension(
        "chromatica.v2core.interp_modules.interp_2d_multichannel",
        [os.path.join(INTERP_MODULES_DIR, "interp_2d_multichannel.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
    # 2D interpolation - dispatcher
    Extension(
        "chromatica.v2core.interp_modules.interp_2d",
        [os.path.join(INTERP_MODULES_DIR, "interp_2d.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
    # Hue interpolation - simple
    Extension(
        "chromatica.v2core.interp_modules.interp_hue_simple",
        [os.path.join(INTERP_MODULES_DIR, "interp_hue_simple.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
    # Hue interpolation - spatial
    Extension(
        "chromatica.v2core.interp_modules.interp_hue_spatial",
        [os.path.join(INTERP_MODULES_DIR, "interp_hue_spatial.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
    # Hue interpolation - between lines
    Extension(
        "chromatica.v2core.interp_modules.interp_hue_between_lines",
        [os.path.join(INTERP_MODULES_DIR, "interp_hue_between_lines.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
    ),
    # Hue interpolation - dispatcher
    Extension(
        "chromatica.v2core.interp_modules.interp_hue",
        [os.path.join(INTERP_MODULES_DIR, "interp_hue.pyx")],
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
print("  from chromatica.v2core.interp_modules import border_handling")
print("  from chromatica.v2core.interp_modules import interp")
print("  from chromatica.v2core.interp_modules import interp_2d")
print("  from chromatica.v2core.interp_modules import interp_hue")
print("=" * 70)
