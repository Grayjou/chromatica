# Redundant Functions Documentation

This document identifies functions in the codebase that have overlapping or redundant functionality but are still being called by other parts of the code. These functions are flagged for potential future refactoring.

## Purpose

While these functions may appear redundant, they are still actively used. This documentation helps future developers:
- Understand which functions serve similar purposes
- Plan consolidation efforts without breaking existing code
- Identify candidates for deprecation in future versions

## Redundant Function Patterns

### 1. Per-Channel Transform Functions

**Location:** `chromatica/gradients/gradient2dv2/helpers/cell_utils.py`

**Functions:**
- `apply_per_channel_transforms_2d_single(coords, per_channel_transforms, num_channels)` - Lines 20-65
- `apply_per_channel_transforms_2d(coords, per_channel_transforms, num_channels)` - Lines 68-123

**Redundancy:** `apply_per_channel_transforms_2d_single` is a specialized version that only handles single coordinate grids (numpy arrays), while `apply_per_channel_transforms_2d` is the general version that handles both single grids and lists of grids. The general version calls the specialized version internally.

**Current Usage:** Both functions are used, with `apply_per_channel_transforms_2d` being the primary public interface.

**Recommendation:** Keep both functions. The single-grid version is a performance optimization that avoids unnecessary type checking when the input is known to be a single array.

**Future Refactoring:** Consider making `apply_per_channel_transforms_2d_single` a private function (prefix with `_`) to clarify it's an internal implementation detail.

---

### 2. Hue and Non-Hue Channel Split Functions

**Location:** `chromatica/gradients/gradient2dv2/helpers/interpolation/utils.py`

**Functions:**
- `prepare_hue_and_rest_channels(data, is_hue)` - Lines 8-21
- `combine_hue_and_rest_channels(hue_data, rest_data)` - Lines 24-38

**Redundancy:** These are simple utility functions that split and combine hue channels. Similar operations could be done with basic numpy slicing like `data[..., 0]` and `data[..., 1:]`, and `np.concatenate([hue[..., np.newaxis], rest], axis=-1)`.

**Current Usage:** Used throughout the interpolation code to separate hue interpolation from other channel interpolation.

**Recommendation:** Keep these functions as they provide a clear, semantic interface and make the code more readable. They abstract away the details of hue channel positioning.

**Future Refactoring:** No action needed. These simple wrappers improve code clarity.

---

### 3. Color Space Conversion Wrappers

**Location:** `chromatica/utils/color_utils.py`

**Function:**
- `convert_to_space_float(color, from_space, format_type, to_space)` - Lines 13-35

**Redundancy:** This function wraps the color conversion functionality that's already available through ColorBase methods. It's a convenience function that combines tuple-to-class conversion with color space conversion in one call.

**Current Usage:** Heavily used in cell factory functions (`cell/factory.py`) to convert corner/line colors to the target color space.

**Recommendation:** Keep this function. It provides a clean interface for the common pattern of "convert input color to target space in float format" which is used throughout gradient generation.

**Future Refactoring:** No action needed. This wrapper reduces boilerplate in gradient code.

---

### 4. Grayscale Detection Functions

**Location:** `chromatica/utils/color_utils.py`

**Functions:**
- `is_hue_color_grayscale(color, thresh)` - Lines 37-39
- `is_hue_color_arr_grayscale(color, thresh)` - Lines 41-43

**Redundancy:** These two functions do the same thing (check if saturation is near zero) but one handles scalar colors and one handles arrays of colors.

**Current Usage:** Used when determining whether to perform hue interpolation (grayscale colors have undefined hue).

**Recommendation:** Keep both functions. They provide a semantic interface for grayscale detection and the dual implementation (scalar vs array) is necessary for performance.

**Future Refactoring:** Consider using numpy's `np.atleast_2d` pattern to unify these into a single function that handles both cases, but only if it doesn't impact performance.

---

### 5. Line Interpolation Method Dispatch

**Location:** `chromatica/gradients/gradient2dv2/helpers/interpolation/lines.py`

**Functions:**
- `_interp_transformed_non_hue_space_2d_lines_discrete()` - Lines 41-62
- `_interp_transformed_non_hue_space_2d_lines_continuous()` - Lines 65-85
- `_interp_transformed_hue_space_2d_lines_continuous()` - Lines 88-100+
- Similar pattern for discrete/continuous variants

**Redundancy:** Multiple similar functions that differ only in whether they use discrete or continuous sampling, and whether they handle hue or non-hue channels.

**Current Usage:** Called based on the `LineInterpMethods` enum value and whether the color space has hue.

**Recommendation:** Keep separate functions. This is not redundancy but proper separation of concerns. Each function has a specific, well-defined purpose.

**Future Refactoring:** No action needed. This pattern provides clear dispatch logic and type safety.

---

### 6. Corner vs Line Cell Interpolation

**Location:** `chromatica/gradients/gradient2dv2/helpers/interpolation/corners.py` and `lines.py`

**Functions:**
- Corner interpolation functions in `corners.py`
- Line interpolation functions in `lines.py`

**Redundancy:** Both sets of functions perform 2D interpolation, but corners use bilinear interpolation from 4 corner values while lines interpolate between two 1D lines.

**Current Usage:** Used by different cell types (`CornersCell` vs `LinesCell`).

**Recommendation:** Keep separate. These are fundamentally different interpolation methods even though they both produce 2D gradients.

**Future Refactoring:** No action needed. The separation reflects different mathematical approaches.

---

## Summary

Most apparent "redundancies" in the codebase are actually:
1. **Performance optimizations** (specialized vs general implementations)
2. **Semantic wrappers** (making common operations more readable)
3. **Proper separation of concerns** (different methods for different use cases)

None of these functions should be removed in the current codebase version. If consolidation is desired in the future, it should be done carefully to preserve both functionality and performance.

## Guidelines for Future Development

When considering whether to consolidate similar functions:

1. **Check performance impact:** Specialized functions often exist for performance reasons
2. **Consider readability:** Semantic wrappers can make code more maintainable
3. **Evaluate type safety:** Separate functions can provide better type checking
4. **Look at call sites:** If a function has many callers, consolidation may be risky
5. **Test thoroughly:** Any consolidation must be validated with comprehensive tests

## Maintenance Notes

This document should be updated when:
- New functions are added that duplicate existing functionality
- Functions are deprecated or removed
- Refactoring consolidates multiple functions into one
- Performance characteristics change significantly

**Last Updated:** 2026-01-03

---

## Analysis of Core Module Functions (v2core)

This section documents functions from `chromatica/v2core/core.py`, `core2d.py`, and wrapper modules, identifying which are used only in testing or are candidates for deprecation.

### Core.py Functions Analysis

**Location:** `chromatica/v2core/core.py`

#### Actively Used Functions (Production Code)

1. **`multival1d_lerp()`** - Used in gradient1dv2 modules (9 non-test occurrences)
   - **Status:** Essential, actively used in 1D gradient generation
   - **Action:** Keep

2. **`hue_lerp()`** - Widely used (107 occurrences)
   - **Status:** Core function for hue interpolation
   - **Action:** Keep

3. **`_prepare_bound_types()`** - Internal utility (10 occurrences)
   - **Status:** Essential for bound type normalization
   - **Action:** Keep as internal utility

4. **`_apply_bound()`** - Used internally (9 occurrences)
   - **Status:** Essential for coordinate bounding
   - **Action:** Keep

5. **`apply_bounds()`** - Used in core2d and wrappers (5 occurrences)
   - **Status:** Essential for multi-array bounding
   - **Action:** Keep

6. **`bound_coeffs()`** - Used internally (5 occurrences)
   - **Status:** Essential for coefficient bounding
   - **Action:** Keep

7. **`hue_gradient_1d()` / `hue_gradient_2d()`** - Used in core2d and exports (5 occurrences each)
   - **Status:** Public API functions
   - **Action:** Keep

8. **`sample_hue_between_lines()` / `sample_between_lines()`** - Used in core modules (7-9 occurrences)
   - **Status:** Essential interpolation functions
   - **Action:** Keep

#### Unused or Export-Only Functions

1. **`_prepare_hue_modes()`** - Lines 46-60
   - **Current Usage:** Only defined, never called outside core.py
   - **Status:** Internal utility not used
   - **Recommendation:** Mark as potentially unused or verify if it's needed for future features

2. **`bound_coeffs_fused()`** - Lines 133-159
   - **Current Usage:** Only exported in `__init__.py`, not called anywhere
   - **Status:** Alternative implementation of `bound_coeffs` with in-place operations
   - **Recommendation:** Consider deprecating if not used. Document if kept for API compatibility.

3. **`multival1d_lerp_uniform()`** - Lines 207-217
   - **Current Usage:** Only exported in `__init__.py` (2 occurrences)
   - **Status:** Convenience wrapper around `multival1d_lerp`
   - **Recommendation:** Keep for API completeness, or mark for deprecation if truly unused

4. **`make_hue_line_sampler()`** - Lines 379-402
   - **Current Usage:** Exported in `__init__.py` only (2 occurrences)
   - **Status:** Factory function for creating reusable samplers
   - **Recommendation:** Document or deprecate if not used in practice

5. **`_bound_stacked()`** - Lines 81-86
   - **Current Usage:** Only called by `_apply_bound` (2 occurrences)
   - **Status:** Internal helper
   - **Recommendation:** Keep as internal utility

---

### Core2d.py Functions Analysis

**Location:** `chromatica/v2core/core2d.py`

#### Border Mode Optimization

1. **`_optimize_border_mode()`** - Lines 38-43
   - **Current Usage:** Called 6 times within core2d.py
   - **Purpose:** Converts bound_type to appropriate border_mode
   - **Status:** Essential optimization function
   - **Action:** Keep and ensure bound_types are set to IGNORE where BORDER_MODE is used
   - **Note:** This function implements the pattern requested in the issue - ensuring BORDER_MODE is used directly in Cython instead of legacy numpy-based bound_types

#### Active Functions

2. **`sample_between_lines_continuous()` / `sample_hue_between_lines_continuous()`**
   - **Status:** Core 2D interpolation functions
   - **Action:** Keep

3. **`multival2d_lerp_between_lines_continuous()` / `multival2d_lerp_between_lines_discrete()`**
   - **Status:** Multi-channel 2D interpolation
   - **Action:** Keep

4. **`multival2d_lerp_from_corners()`**
   - **Status:** Corner-based 2D interpolation
   - **Action:** Keep

#### Internal Utilities

5. **`_prepare_border_constant()`** - Lines 46-76
   - **Status:** Essential utility for border constant preparation
   - **Action:** Keep

6. **`_ensure_list_ndarray()`** - Lines 86-92
   - **Status:** Type conversion utility
   - **Action:** Keep as internal helper

7. **`sample_hue_between_lines_discrete()`**
   - **Status:** Discrete sampling variant
   - **Action:** Keep

---

### Wrapper Functions Analysis

**Location:** `chromatica/v2core/interp_2d/wrappers.py` and `interp_hue/wrappers.py`

These wrapper functions provide typed interfaces to Cython kernels and are actively used. Analysis shows:

1. **`lerp_between_lines()` / `lerp_from_corners()`**
   - **Status:** Primary 2D interpolation wrappers
   - **Usage:** Called from core2d.py and gradient modules
   - **Action:** Keep - essential public API

2. **`lerp_between_lines_inplace()` / `lerp_between_lines_onto_array()`**
   - **Status:** Memory-efficient variants for in-place operations
   - **Usage:** Used in specific performance-critical scenarios
   - **Action:** Keep - provides important optimization

3. **Border handling utilities (`_normalize_border_mode`, `_normalize_border_constant`)**
   - **Status:** Input validation and normalization
   - **Action:** Keep - essential for robust API

4. **Coordinate utilities (`_ensure_coords_array`, `_ensure_float64_contiguous`)**
   - **Status:** Type and memory layout utilities
   - **Action:** Keep - ensures Cython performance

---

## Flat Conversion Functions Status

**Issue Note:** The problem statement mentions that "due to backwards compatibility of imports, a lot of unused functions, especially around flat conversions have been preserved."

### Investigation Required

A thorough grep of "flat" related functions shows:
- Flat array variants exist in interp_hue wrappers
- These provide flattened coordinate array interfaces
- Usage appears limited to specific scenarios

### Recommendation

Create a separate analysis document `FLAT_CONVERSION_ANALYSIS.md` to:
1. Enumerate all flat conversion functions
2. Map their dependencies
3. Identify which are used only in tests
4. Propose a deprecation path

---

## Border Mode vs Bound Types Migration

**Issue Context:** The problem statement notes that `bound_types` are numpy-based legacy, and the system should use `BORDER_MODE` directly in Cython.

### Current Status

✅ **Already Implemented:** The `_optimize_border_mode()` function in core2d.py (lines 38-43) already implements this pattern:

```python
def _optimize_border_mode(bound_type: BoundType, border_mode: BorderMode) -> BorderMode:
    if bound_type != BoundType.IGNORE:
        #if we are bounding, no need for complex border handling
        return BorderMode.OVERFLOW
    return border_mode
```

### Analysis

- When `bound_type` is not `IGNORE`, the function returns `OVERFLOW` mode
- This allows Cython kernels to use direct `BORDER_MODE` logic
- The optimization is already applied in 6 places in core2d.py

### Recommendation

✅ **No changes needed** - The requested pattern is already implemented correctly.

### Documentation Note

Consider adding docstring to `_optimize_border_mode()` explaining:
1. Why bound_types should be IGNORE when using BORDER_MODE
2. The performance benefit of direct Cython border handling
3. The legacy numpy-based bound_types compatibility

---

## Removed Files

### core_lines.py - REMOVED ✅

**Location:** `chromatica/v2core/core_lines.py`

**Status:** Deleted as requested in problem statement

**Justification:**
- No imports found in codebase (`grep -r "from.*core_lines import\|import.*core_lines"` returned nothing)
- Created as test harness for new kernel functions
- Superseded by direct kernel usage with intelligent dispatcher
- No production code dependencies

**Action Taken:** File removed in this PR

---

## Summary of Actions

### Completed
1. ✅ Removed `core_lines.py` (zero dependencies)
2. ✅ Updated `.gitignore` to exclude generated `.c` and `.html` files
3. ✅ Added `interp.pyx` to `setup_cython.py` for proper compilation
4. ✅ Documented border mode optimization pattern (already implemented)

### Identified for Future Consideration
1. ⚠️ `_prepare_hue_modes()` - unused internal function
2. ⚠️ `bound_coeffs_fused()` - exported but not called
3. ⚠️ `make_hue_line_sampler()` - potentially unused factory function
4. ⚠️ Flat conversion functions - need separate analysis

### Keep (Active Use)
- All public API functions with production usage
- All internal utilities called by active functions
- All wrapper functions providing Cython interfaces
- Border mode optimization functions

---

**Last Updated:** 2026-01-03
