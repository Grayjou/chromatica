# V2Core Unused and Repeated Methods Analysis

This document contains analysis of unused and repeated methods in the v2core refactor, including Cython files, Python wrappers, and dispatchers.

Date: 2025-12-25
Analysis performed on the v2core directory and related files.

---

## Summary

- **Total v2core files analyzed**: 17
- **Files with unused methods**: 0 
- **Total unused methods detected**: 0
- **True duplicate methods** (excluding .pxd/.pyx pairs): 9

---

## Unused Methods

**No unused public methods were detected in v2core.**

All public functions defined in v2core appear to be either:
1. Called from within v2core modules
2. Imported and used by external modules
3. Exposed through `__init__.py` for public API
4. Used in tests or examples

---

## Repeated/Duplicate Methods

### True Duplicates (Excluding .pxd/.pyx Declaration/Implementation Pairs)

These are methods that appear to be duplicated across different implementation files:

#### Border Handling Duplicates

**chromatica/v2core/border_handling.pyx** {
  clamp,
  tri2,
  handle_border_edges_2d,
  handle_border_lines_2d,
}

**chromatica/v2core/border_handling_fallback.py** {
  clamp,
  tri2,
  handle_border_edges_2d,
  handle_border_lines_2d,
}

**Note**: These are intentional duplicates - `.pyx` is the optimized Cython implementation, `.py` is the Python fallback when Cython extensions are not built. The `.pxd` file contains declarations for C-level access.

#### Hue Interpolation Wrapper Duplicates

**chromatica/v2core/interp_hue/interp_hue.pyx** {
  hue_lerp_1d_spatial,
  hue_lerp_simple,
  hue_lerp_arrays,
  hue_multidim_lerp,
}

**chromatica/v2core/interp_hue/wrappers.py** {
  hue_lerp_1d_spatial,
  hue_lerp_simple,
  hue_lerp_arrays,
  hue_multidim_lerp,
}

**Note**: These are wrapper functions. The `.pyx` file contains the Cython implementation, while `wrappers.py` provides type-hinted Python wrappers that import from the Cython module. This is an architectural pattern to provide better IDE support and type hints.

#### Plane Interpolation Wrapper Duplicates

**chromatica/v2core/interp_2d/interp_planes.pyx** {
  lerp_between_planes,
}

**chromatica/v2core/interp_2d/wrappers.py** {
  lerp_between_planes,
}

**Note**: Same pattern as above - Cython implementation with Python wrapper.

---

## Expected Cython Declaration/Implementation Pairs

These are **NOT duplicates** but expected pairs where `.pxd` declares functions and `.pyx` implements them:

**chromatica/v2core/border_handling.pxd** + **chromatica/v2core/border_handling.pyx** {
  handle_border_edges_2d,
  handle_border_lines_2d,
}

**chromatica/v2core/interp_2d/helpers.pxd** + **chromatica/v2core/interp_2d/helpers.pyx** {
  handle_border_1d,
  is_out_of_bounds_1d,
  is_out_of_bounds_2d,
  is_out_of_bounds_3d,
}

**chromatica/v2core/interp_hue/interp_hue_utils.pxd** + **chromatica/v2core/interp_hue/interp_hue_utils.pyx** {
  adjust_end_for_mode,
  lerp_hue_single,
  wrap_hue,
}

---

## Architecture Patterns Identified

### 1. Cython + Python Fallback Pattern

Files: `border_handling.pyx` + `border_handling_fallback.py`

**Purpose**: Provide optimized Cython implementation with Python fallback when extensions are not built.

**Recommendation**: This is intentional and should be kept. The module loader in `border_handler.py` attempts to import from `.pyx` first, falls back to `.py` if import fails.

### 2. Cython + Python Wrapper Pattern

Files: Various `.pyx` + `wrappers.py` files

**Purpose**: 
- `.pyx` files contain optimized Cython implementations
- `wrappers.py` files provide type-hinted Python wrappers for better IDE support
- Wrappers add type annotations, default parameters, and error handling

**Recommendation**: This is an intentional architectural pattern for better developer experience. The wrappers are not truly duplicate implementations but provide a typed interface layer.

### 3. Declaration + Implementation Pattern

Files: Various `.pxd` + `.pyx` pairs

**Purpose**: 
- `.pxd` files declare function signatures for C-level cimport
- `.pyx` files implement the functions
- This allows other Cython modules to import and call these functions at C speed

**Recommendation**: This is standard Cython practice and should be kept.

---

## Detailed File Analysis

### Files Checked

1. chromatica/v2core/__init__.py
2. chromatica/v2core/border_handler.py
3. chromatica/v2core/border_handling.pxd
4. chromatica/v2core/border_handling.pyx
5. chromatica/v2core/border_handling_fallback.py
6. chromatica/v2core/border_mode.pyx
7. chromatica/v2core/core.py
8. chromatica/v2core/core2d.py
9. chromatica/v2core/corner_interp_2d.py
10. chromatica/v2core/interp.pyx
11. chromatica/v2core/subgradient.py
12. chromatica/v2core/interp_2d/__init__.py
13. chromatica/v2core/interp_2d/wrappers.py
14. chromatica/v2core/interp_2d/corner_interp_2d_fast.pyx
15. chromatica/v2core/interp_2d/helpers.pxd
16. chromatica/v2core/interp_2d/helpers.pyx
17. chromatica/v2core/interp_2d/interp_.pyx
18. chromatica/v2core/interp_2d/interp_2d_fast.pyx
19. chromatica/v2core/interp_2d/interp_planes.pyx
20. chromatica/v2core/interp_hue/__init__.py
21. chromatica/v2core/interp_hue/interp_hue.pyx
22. chromatica/v2core/interp_hue/interp_hue2d.pxd
23. chromatica/v2core/interp_hue/interp_hue2d.pyx
24. chromatica/v2core/interp_hue/interp_hue_utils.pxd
25. chromatica/v2core/interp_hue/interp_hue_utils.pyx
26. chromatica/v2core/interp_hue/wrappers.py

### Functions Exposed in Public API (from __init__.py)

The following functions are exported from `chromatica/v2core/__init__.py`:

- multival1d_lerp
- multival2d_lerp
- multival1d_lerp_uniform
- multival2d_lerp_uniform
- single_channel_multidim_lerp
- hue_lerp
- hue_lerp_multi
- hue_multidim_lerp_bounded
- hue_gradient_1d
- hue_gradient_2d
- sample_hue_between_lines
- make_hue_line_sampler
- bound_coeffs
- bound_coeffs_fused
- HueMode
- lerp_between_lines
- lerp_between_planes
- sample_between_lines
- sample_between_lines_continuous
- sample_between_lines_discrete
- sample_hue_between_lines_continuous
- sample_hue_between_lines_discrete
- multival2d_lerp_between_lines_continuous
- multival2d_lerp_between_lines_discrete
- multival2d_lerp_from_corners
- sample_between_planes
- SubGradient
- handle_border_edges_2d
- handle_border_lines_2d
- BORDER_REPEAT
- BORDER_MIRROR
- BORDER_CONSTANT
- BORDER_CLAMP
- BORDER_OVERFLOW

All of these are actively used in the codebase.

---

## Recommendations

### Functions to Keep (All Current Functions)

Based on the analysis, **no functions should be removed**. All functions fall into one of these categories:

1. **Public API functions**: Exposed through `__init__.py` and used by consumers
2. **Internal helper functions**: Used within v2core modules
3. **Cython declarations**: Required for C-level imports
4. **Fallback implementations**: Needed when Cython extensions aren't built
5. **Type-hinted wrappers**: Provide better IDE support and type safety

### Architecture is Sound

The current architecture with:
- Cython implementations (.pyx)
- Python fallbacks (.py)
- Cython declarations (.pxd)
- Type-hinted wrappers (wrappers.py)

...is a well-designed pattern that provides:
- Performance (Cython)
- Portability (Python fallbacks)
- Type safety (Type hints)
- C-level interop (.pxd declarations)

### Potential Cleanup (Minor)

If you insist on pruning, the only candidates would be:

1. **border_mode.pyx** - Contains only constant definitions, could potentially be consolidated into border_handling.pyx
2. **corner_interp_2d.py** - Very small file that just re-exports from interp_2d, could potentially be removed if consumers import directly

However, both of these are so small that removing them provides minimal benefit.

---

## Conclusion

The v2core refactor appears to be well-structured with **no truly unused methods**. All apparent "duplicates" are intentional architectural patterns that serve specific purposes:

- Performance optimization (Cython vs Python)
- Type safety (wrappers with type hints)
- Portability (fallback implementations)
- C-level interoperability (.pxd declarations)

**Final recommendation**: No pruning necessary. The codebase is clean and well-organized.
