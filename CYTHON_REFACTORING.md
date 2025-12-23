# Cython File Refactoring Summary

## Overview

The large Cython files in `chromatica/v2core/` have been refactored into smaller, more maintainable modules organized in the new `interp_modules/` subdirectory.

## Motivation

- **interp_2d.pyx** was 721 lines - too large for easy maintenance
- **interp_hue.pyx** was 731 lines - difficult to navigate and understand
- Large files are harder to review, debug, and modify
- Compilation times are slower when everything is in one file

## Changes Made

### New Directory Structure

```
chromatica/v2core/
├── interp_modules/              # NEW: All Cython interpolation code
│   ├── __init__.py
│   ├── hue_common.pxd          # Shared hue helper functions
│   ├── border_handling.pyx     # Border handling (moved)
│   ├── border_mode.pyx         # Border modes (moved)
│   ├── interp.pyx              # Core interpolation (moved)
│   ├── interp_2d_1ch.pyx       # NEW: Single-channel 2D interp
│   ├── interp_2d_multichannel.pyx  # NEW: Multi-channel 2D interp
│   ├── interp_2d.pyx           # NEW: 2D dispatcher/wrapper
│   ├── interp_hue_simple.pyx   # NEW: Simple hue functions
│   ├── interp_hue_spatial.pyx  # NEW: Spatial hue interpolation
│   ├── interp_hue_between_lines.pyx  # NEW: Hue between-lines
│   └── interp_hue.pyx          # NEW: Hue dispatcher/wrapper
├── core.py                      # Updated imports
├── core2d.py                    # Updated imports
└── __init__.py
```

### File Splitting Details

#### interp_2d.pyx (721 lines) → Split into 3 files:

1. **interp_2d_1ch.pyx** (327 lines)
   - `lerp_between_lines_1ch` - Single channel line interpolation
   - `lerp_between_lines_flat_1ch` - Flat coordinate version
   - `lerp_between_planes_1ch` - 3D plane interpolation
   - `lerp_between_lines_x_discrete_1ch` - Discrete x-sampling

2. **interp_2d_multichannel.pyx** (267 lines)
   - `lerp_between_lines_multichannel` - Multi-channel line interpolation
   - `lerp_between_lines_flat_multichannel` - Flat coordinate version
   - `lerp_between_lines_x_discrete_multichannel` - Discrete x-sampling

3. **interp_2d.pyx** (136 lines)
   - `lerp_between_planes` - Dispatcher for plane interpolation
   - `lerp_between_lines` - Main dispatcher that routes to appropriate function

#### interp_hue.pyx (731 lines) → Split into 4 files:

1. **interp_hue_simple.pyx** (100 lines)
   - `hue_lerp_simple` - Simple 1D hue interpolation
   - `hue_lerp_arrays` - Vectorized hue interpolation

2. **interp_hue_spatial.pyx** (283 lines)
   - `hue_lerp_1d_spatial` - 1D spatial hue interpolation
   - `hue_lerp_2d_spatial` - 2D spatial hue interpolation

3. **interp_hue_between_lines.pyx** (296 lines)
   - `hue_lerp_2d_with_modes` - Per-pixel mode interpolation
   - `hue_lerp_between_lines` - Hue interpolation between two lines
   - `hue_lerp_between_lines_x_discrete` - Discrete x-sampling version

4. **interp_hue.pyx** (70 lines)
   - `hue_multidim_lerp` - Main dispatcher

5. **hue_common.pxd** (80 lines)
   - Shared constants: `HUE_CW`, `HUE_CCW`, `HUE_SHORTEST`, `HUE_LONGEST`
   - Shared inline functions: `wrap_hue`, `adjust_end_for_mode`, `lerp_hue_single`

## Benefits

1. **Better Organization**: Related functions are grouped together
2. **Easier Maintenance**: Smaller files are easier to understand and modify
3. **Faster Incremental Builds**: Changing one module doesn't require recompiling everything
4. **Clearer Dependencies**: Import structure makes dependencies explicit
5. **Code Reuse**: Common helpers (hue_common.pxd) can be cimported by multiple modules

## Backward Compatibility

The public API remains unchanged. All functions are still importable from:
- `chromatica.v2core.core`
- `chromatica.v2core.core2d`

Internal imports have been updated but the external interface is preserved.

## Build System Changes

`setup_cython.py` has been updated to:
1. Compile all modules in `interp_modules/` directory
2. Handle the new modular structure
3. Build 9 separate extensions instead of 4

## Testing

All modules successfully compile with:
```bash
python setup_cython.py build_ext --inplace
```

Compilation produces 9 shared libraries (.so files on Linux):
- border_handling.so
- interp.so
- interp_2d_1ch.so
- interp_2d_multichannel.so
- interp_2d.so
- interp_hue_simple.so
- interp_hue_spatial.so
- interp_hue_between_lines.so
- interp_hue.so

## Migration Notes for Developers

### Before
```python
from chromatica.v2core.interp_2d import lerp_between_lines
```

### After
Still works the same way (imports updated internally):
```python
from chromatica.v2core.interp_2d import lerp_between_lines
```

Or import from the high-level API:
```python
from chromatica.v2core.core2d import sample_between_lines_continuous
```

## Files Removed

The following original files have been removed from `v2core/`:
- `interp_2d.pyx` (replaced by split modules)
- `interp_hue.pyx` (replaced by split modules)
- `interp.pyx` (moved to interp_modules/)
- `border_handling.pyx` (moved to interp_modules/)
- `border_mode.pyx` (moved to interp_modules/)

## Future Work

- Consider further splitting if any module exceeds ~300 lines
- Add module-level docstrings
- Consider splitting interp.pyx if it grows too large
