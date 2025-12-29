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

**Last Updated:** 2025-12-29
