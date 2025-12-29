# Changelog

All notable changes to Chromatica will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Cell Generator System**: Factory classes for creating and managing 2D gradient cells
  - `CornersCellFactory`: Generate cells from four corner colors with bilinear interpolation
  - `LinesCellFactory`: Generate cells from two horizontal lines
  - `CornersCellDualFactory`: Advanced cells with independent top/bottom edge interpolation
  - Cell partitioning: Split cells into regions with different color spaces and interpolation modes
  - Per-channel coordinate transformations for complex gradient effects
  
- **Enhanced Gradient Features**
  - Border handling modes: clamp, wrap, constant value, and overflow
  - Discrete and continuous sampling methods for line interpolation
  - Multi-channel interpolation with independent bound types per channel
  - Support for mixed color space gradients through partitioning

- **Documentation Improvements**
  - Added `REDUNDANT_FUNCTIONS.md` documenting functions with overlapping functionality
  - Added `OPTIMIZATION_HOTSPOTS.md` identifying performance-critical code paths
  - Enhanced README with cell generator examples and usage patterns
  - Improved type hints throughout cell and interpolation modules

### Changed
- **Build System**: Added Cython and numpy to build-system requirements in pyproject.toml
- **Dependencies**: Updated project dependencies to include boundednumbers and unitfield
- **Type Hints**: Improved type annotations for coordinate types in gradient2dv2 modules

### Removed
- Outdated documentation files: PR_SUMMARY.md, CODE_QUALITY_RECOMMENDATIONS.md, REFACTORING_SUMMARY.md, 
  LIBRARY_REVIEW_V2.md, BORDER_HANDLING_IMPLEMENTATION.md, BUILD_CYTHON.md, 
  CELL_REFACTORING_COMPLETE.md, UNUSED_METHODS_ANALYSIS.md

## [1.0.0] - 2025-12-05

### Added
- **Complete 1.X Framework Overhaul**
  - Rebuilt color system from ground up with ColorBase as foundation
  - Full array support throughout the entire library
  - Vectorized operations on color arrays using numpy
  
- **ColorBase Architecture**
  - Immutable color instances with frozen state after initialization
  - Support for both scalar colors and color arrays
  - Automatic dtype validation and enforcement
  - Value clamping to valid ranges for all formats
  - `is_array` and `shape` properties for array detection
  - `has_alpha` and `has_hue` properties for color space introspection

- **Color Space Classes**
  - RGB variants: ColorRGBINT, ColorRGBAINT, ColorUnitRGB, ColorUnitRGBA, ColorPercentageRGB, ColorPercentageRGBA
  - HSV variants: ColorHSVINT, ColorHSVAINT, UnitHSV, UnitHSVA, PercentageHSV, PercentageHSVA
  - HSL variants: ColorHSLINT, ColorHSLAINT, UnitHSL, UnitHSLA, PercentageHSL, PercentageHSLA
  - All classes support INT (0-255), FLOAT (0.0-1.0), and PERCENTAGE (0-100) formats

- **Advanced Arithmetic System**
  - `make_arithmetic()` function for wrapping colors with arithmetic behavior
  - Auto-arithmetic operators injected into ColorBase (`+`, `-`, `*`, `/`, and reverse operations)
  - Custom overflow functions: `clamp` (default), `bounce`, `cyclic_wrap_float`
  - Industry-standard hue wrapping for HSV/HSL (cyclic 0-360 degrees)
  - Automatic conversion to float for multiplication/division operations
  - Automatic tuple-to-array conversion for arithmetic operations
  - Automatic alpha channel matching in arithmetic (RGBA + RGB automatically converts RGB to RGBA)
  - Arithmetic persistence through color space conversions

- **Conversion System**
  - Unified conversion API with `convert()` method on all ColorBase instances
  - CSS Color 4 / Culori algorithm support via `use_css_algo` parameter
  - Automatic detection of scalar vs array values with appropriate conversion routing
  - Support for RGB ↔ HSV, RGB ↔ HSL, HSV ↔ HSL conversions
  - Format conversion between INT, FLOAT, and PERCENTAGE
  - Wrapper API for direct numpy array conversions

- **Alpha Channel Support**
  - `with_alpha()` method for adding/modifying alpha channels
  - Support for scalar alpha (applies to all colors) or array alpha (per-element)
  - Automatic shape validation for array alphas
  - Alpha preservation across color space conversions
  - WithAlpha mixin using array-compatible indexing (`[..., -1]`)

- **Color Array Generation (ColorArr Rework)**
  - Complete redesign as wrapper classes around ColorBase
  - `Color1DArr`: 1D color arrays with gradient generation
    - `repeat()`: Tile gradients horizontally and vertically
    - `wrap_around()`: Create angular gradients around a point
    - `rotate_around()`: Create radial gradients from a center
    - Support for easing functions via `unit_transform` parameter
  - `Color2DArr`: 2D color arrays with image manipulation
    - `repeat()`: Tile images horizontally and vertically
    - `resize()`: Resize with anti-aliasing (requires scikit-image)
  - Inherits all ColorBase features via delegation: conversions, arithmetic, alpha support
  - Numpy array interface support via `__array__()`

- **Gradient Generation (Complete Refactor)**
  - `Gradient1D`: Enhanced 1D gradient generation
    - `from_colors()`: Create gradients with hue direction control
    - **Hue Direction Control**: 'cw' (clockwise), 'ccw' (counter-clockwise), None (shortest path)
    - Circular hue interpolation for HSV/HSL color spaces
    - Linear interpolation for RGB color spaces
    - Custom interpolation transforms via `unit_transform` parameter
    - Accepts ColorBase instances or tuples as input
  - `Gradient2D`: 2D gradient from corner colors
    - `from_colors()`: Bilinear interpolation from four corner colors
    - Independent transforms for X and Y axes (`unit_transform_x`, `unit_transform_y`)
    - Supports all color spaces and format types
  - `radial_gradient()`: Radial gradient function
    - Radiates from center point with customizable radius
    - Optional `outside_fill` for areas beyond gradient
    - Optional `base` array for compositing
    - Custom easing via `unit_transform`
    - Start/end/offset parameters for gradient positioning
  - All gradient functions:
    - Work with ColorBase instances or tuples
    - Support all color spaces (RGB, RGBA, HSV, HSVA, HSL, HSLA)
    - Support all format types (INT, FLOAT, PERCENTAGE)
    - Automatic color space conversion when mixing ColorBase types

- **Comprehensive Testing**
  - Array-based conversion tests with vectorized sample data
  - Round-trip conversion tests for accuracy validation
  - Alpha channel functionality tests (11 comprehensive tests)
  - Arithmetic operation tests with color space conversions
  - Multiplication and division tests with overflow handling
  - Chained arithmetic tests for complex operations
  - Gradient generation tests (hue direction, bilinear interpolation, radial gradients)

- **Documentation**
  - Comprehensive module docstrings with usage examples
  - Detailed API documentation for all color classes
  - Examples for both scalar and array usage patterns
  - Conversion algorithm documentation (CSS vs direct)

### Changed
- **Breaking Changes**
  - Removed old ColorMode/ColorFormat enum system
  - Removed dependency on old ColorInt and color() factory function
  - ColorArr classes now wrapper classes (not subclasses) around ColorBase
  - All color classes now immutable (frozen after initialization)
  - Conversion methods now return new instances instead of modifying in place
  - **Gradient API Changes**:
    - `Gradient1D/2D.from_colors()`: Now use `color_space` and `format_type` instead of `color_mode` and `color_format`
    - Removed old `__init__` methods; use `from_colors()` classmethod instead
    - `radial_gradient()`: Now uses `color_space`/`format_type` parameters
    - All gradient functions return ColorBase-wrapped results or numpy arrays
    - `direction` parameter added for hue gradient control ('cw'/'ccw'/None)

- **API Improvements**
  - Simplified color creation: direct class instantiation instead of factory functions
  - Consistent API across all color spaces and formats
  - Better type hints and static type checking support
  - More intuitive arithmetic operations (no wrapper required for basic operations)

### Removed
- Old 0.1.2 ColorArr implementation with ColorMode/ColorFormat enums
- ColorInt base class (replaced by ColorBase)
- get_color_mode() and color() factory functions
- Dependency on legacy color system
- ColorFormat enum (INT, PIL_INT, UNIT_FLOAT) replaced by FormatType enum

### Fixed
- CSS Color 4 hue calculation consistency between scalar and vectorized versions
- Integer division issues in multiplicative arithmetic operations
- Tuple/array incompatibility in arithmetic operations
- Alpha channel mismatches in arithmetic between RGBA and non-alpha colors
- Overflow function type errors with tuple maxima values
- Array dtype validation and automatic conversion to proper types

### Technical Details
- **Performance**: Vectorized operations significantly faster for batch color processing
- **Memory**: Immutable design allows for safe sharing of color instances
- **Compatibility**: Full numpy integration with array interface protocol
- **Type Safety**: Comprehensive dtype validation prevents silent errors
- **Industry Standards**: Hue wrapping matches standard graphics libraries (0-360° cyclic)

---

## [0.1.2] - Previous Version

Legacy version with basic color array support (deprecated).
