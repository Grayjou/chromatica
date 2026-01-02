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

# Relative paths from chromatica/ directory
EXTENSION_PATHS = [
    # interp_2d modules
    "v2core/interp_2d/corner_interp_2d_fast_.pyx",
    "v2core/interp_2d/corner_interp_2d_border_.pyx",
    "v2core/interp_2d/interp_2d_fast_.pyx",
    "v2core/interp_2d/interp_2d_array_border.pyx",
    
    # interp_hue modules
    "v2core/interp_hue/interp_hue.pyx",
    "v2core/interp_hue/interp_hue2d.pyx",
    "v2core/interp_hue/interp_hue2d_array_border.pyx",
    "v2core/interp_hue/interp_hue_array_border.pyx",
    "v2core/interp_hue/interp_hue_corners.pyx",
    "v2core/interp_hue/interp_hue_corners_array_border.pyx",
    "v2core/interp_hue/interp_hue_utils.pyx",
    
    # Core utilities
    "v2core/interp_utils.pyx",
]


def create_extension(rel_path: str) -> Extension:
    """
    Create an Extension object from a relative path.
    
    Args:
        rel_path: Path relative to chromatica/ (e.g., "v2core/interp_2d/file.pyx")
    
    Returns:
        Configured Extension object
    """
    # Convert path to module name
    # e.g., "v2core/interp_2d/file.pyx" -> "chromatica.v2core.interp_2d.file"
    module_name = "chromatica." + rel_path.replace("/", ".").replace(".pyx", "")
    full_path = os.path.join(BASE_DIR, "chromatica", rel_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Cython source not found: {full_path}")
    
    return Extension(
        name=module_name,
        sources=[full_path],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],  # Adjust for your needs
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )


def build_extensions():
    """Build all extension modules."""
    extensions = [create_extension(path) for path in EXTENSION_PATHS]
    
    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
        annotate=True,  # Generates HTML annotation files for optimization review
    )


if __name__ == "__main__":
    setup(
        name="chromatica-v2core",
        ext_modules=build_extensions(),
        zip_safe=False,
    )