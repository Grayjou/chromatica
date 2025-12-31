"""
Build all Cython extensions for chromatica
"""
import os
import sys
import subprocess
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_pyx_files():
    """Find all .pyx files in the chromatica package."""
    pyx_files = []
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, 'chromatica')):
        for f in files:
            if f.endswith('.pyx') and not f.startswith('_'):
                pyx_files.append(os.path.join(root, f))
    return pyx_files

def build_with_cython():
    """Try to build all .pyx files using cythonize command."""
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension, setup
    from setuptools.dist import Distribution
    
    pyx_files = find_pyx_files()
    print(f"Found {len(pyx_files)} .pyx files")
    
    extensions = []
    for pyx_path in pyx_files:
        # Get module name from path
        rel_path = os.path.relpath(pyx_path, BASE_DIR)
        module_name = rel_path.replace(os.sep, '.').replace('.pyx', '')
        
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
    
    # Build extensions
    ext_modules = cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=False,
        quiet=False,
    )
    
    # Build in place
    dist = Distribution({'ext_modules': ext_modules})
    dist.package_dir = {'': BASE_DIR}
    
    build_ext_cmd = dist.get_command_obj('build_ext')
    build_ext_cmd.inplace = True
    build_ext_cmd.ensure_finalized()
    build_ext_cmd.run()

if __name__ == '__main__':
    build_with_cython()
