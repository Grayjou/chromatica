# Building Cython Extensions

This document explains how to build the Cython extensions for chromatica's v2core module.

## Quick Start

To build all Cython extensions, run:

```bash
python setup_cython.py build_ext --inplace
```

This will compile all `.pyx` files in `chromatica/v2core/interp_modules/` and place the resulting `.so` (or `.pyd` on Windows) files in the appropriate locations.

## Requirements

- Python 3.8+
- Cython 0.29+
- NumPy
- A C compiler (gcc on Linux, MSVC on Windows, clang on macOS)

Install requirements:
```bash
pip install cython numpy
```

## What Gets Built

The setup script compiles the following modules from `chromatica/v2core/interp_modules/`:

### Core Modules
1. **border_handling.pyx** - Border handling for 2D interpolation (no numpy dependency)
2. **interp.pyx** - Core interpolation functions (requires numpy)

### 2D Interpolation (Split for Maintainability)
3. **interp_2d_1ch.pyx** - Single-channel 2D interpolation (~327 lines)
4. **interp_2d_multichannel.pyx** - Multi-channel 2D interpolation (~267 lines)
5. **interp_2d.pyx** - Main 2D dispatcher (~136 lines)

### Hue Interpolation (Split for Maintainability)
6. **interp_hue_simple.pyx** - Simple hue lerp functions (~100 lines)
7. **interp_hue_spatial.pyx** - 1D and 2D spatial hue interpolation (~283 lines)
8. **interp_hue_between_lines.pyx** - Between-lines hue interpolation (~296 lines)
9. **interp_hue.pyx** - Main hue dispatcher (~70 lines)

### Shared Definitions
- **hue_common.pxd** - Common hue interpolation helpers and constants (cimport only)

## Architecture

The large Cython files (`interp_2d.pyx` and `interp_hue.pyx`) have been split into smaller, 
more maintainable modules grouped by functionality. Each module compiles to its own shared library,
and the main dispatcher modules (`interp_2d.pyx`, `interp_hue.pyx`) provide a unified interface.

This architecture provides:
- **Better maintainability**: Smaller files are easier to understand and modify
- **Faster compilation**: Changes to one module don't require recompiling everything
- **Modular design**: Clear separation of concerns

## Troubleshooting

### Missing numpy headers

If you get errors about `numpy/arrayobject.h` not found:
```bash
pip install --upgrade numpy
```

### No C compiler

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install build-essential
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
Install Visual Studio Build Tools from https://visualstudio.microsoft.com/downloads/

## Development Workflow

1. Make changes to `.pyx` files in `chromatica/v2core/interp_modules/`
2. Rebuild: `python setup_cython.py build_ext --inplace`
3. Test your changes
4. Commit both `.pyx` and generated `.c` files (optional, depending on your workflow)

## CI/CD Integration

For continuous integration, you can add this to your workflow:

```yaml
- name: Build Cython extensions
  run: python setup_cython.py build_ext --inplace
```

## Clean Build

To force a complete rebuild:

```bash
rm -rf build/
rm chromatica/v2core/interp_modules/*.so chromatica/v2core/interp_modules/*.c
python setup_cython.py build_ext --inplace
```

## Notes

- The `.so` files are platform-specific and should not be committed to version control
- The generated `.c` files can be committed to allow building without Cython installed
- Build artifacts are placed in the `build/` directory
- All modules are compiled with `-O3` optimization for maximum performance
