"""
Build new kernel Cython extensions only (those ending in _ or in *_ directories)
"""
import os
import sys
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.dist import Distribution

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def should_build(pyx_path, module_name):
    """Check if this module should be built."""
    # Skip __init__ files
    if '__init__' in pyx_path:
        return False
    
    # Build modules in directories that end with _
    parts = pyx_path.split(os.sep)
    for part in parts:
        if part.endswith('_') and not part.startswith('__'):
            return True
    
    # Build base modules
    base_modules = [
        'border_handling.pyx',
        'border_handling_.pyx', 
        'interp_utils.pyx',
        'interp.pyx',
    ]
    for bm in base_modules:
        if pyx_path.endswith(bm):
            return True
    
    return False

def find_pyx_files():
    """Find all .pyx files to build."""
    pyx_files = []
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, 'chromatica')):
        for f in files:
            if f.endswith('.pyx'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, BASE_DIR)
                module_name = rel_path.replace(os.sep, '.').replace('.pyx', '')
                if should_build(full_path, module_name):
                    pyx_files.append((full_path, module_name))
    return pyx_files

def build_extensions():
    """Build all eligible extensions."""
    pyx_files = find_pyx_files()
    print(f"Building {len(pyx_files)} extensions:")
    for path, mod in pyx_files:
        print(f"  - {mod}")
    
    extensions = []
    for pyx_path, module_name in pyx_files:
        ext = Extension(
            module_name,
            [pyx_path],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O3', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        )
        extensions.append(ext)
    
    compiler_directives = {
        'language_level': '3',
        'boundscheck': False,
        'wraparound': False,
        'nonecheck': False,
        'cdivision': True,
        'initializedcheck': False,
    }
    
    ext_modules = cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=False,
        quiet=False,
    )
    
    dist = Distribution({'ext_modules': ext_modules})
    dist.package_dir = {'': BASE_DIR}
    
    build_ext_cmd = dist.get_command_obj('build_ext')
    build_ext_cmd.inplace = True
    build_ext_cmd.ensure_finalized()
    build_ext_cmd.run()
    
    print("\nBuild complete!")

if __name__ == '__main__':
    build_extensions()
