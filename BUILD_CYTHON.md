# Building Cython Extensions

This document explains how to build the Cython extensions for chromatica's v2core module.

## Quick Start

To build all Cython extensions, run:

```bash
python setup_cython.py build_ext --inplace
```

This will compile all `.pyx` files in `chromatica/v2core/` and place the resulting `.so` (or `.pyd` on Windows) files in the appropriate locations.

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

The setup script compiles the following modules:

1. **border_handling.pyx** - Border handling for 2D interpolation (no numpy dependency)
2. **interp.pyx** - Core interpolation functions (requires numpy)
3. **interp_2d.pyx** - 2D interpolation between lines (requires numpy)
4. **interp_hue.pyx** - Hue interpolation for cyclical color spaces (requires numpy)

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

1. Make changes to `.pyx` files
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
rm chromatica/v2core/*.so chromatica/v2core/*.c
python setup_cython.py build_ext --inplace
```

## Notes

- The `.so` files are platform-specific and should not be committed to version control
- The generated `.c` files can be committed to allow building without Cython installed
- Build artifacts are placed in the `build/` directory
