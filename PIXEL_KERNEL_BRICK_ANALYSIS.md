# Pixel-Based Kernel and Brick Stack Analysis

**Created:** 2026-01-03  
**Purpose:** Evaluate existing plans and suggest improvements for pixel_based_kernel and brick_stack modules

---

## Current State Analysis

### 1. Pixel-Based Kernel Module

**Location:** `chromatica/v2core/pixel_based_kernel/`

#### Files Present
1. `__init__.py` / `__init__.pyx` - Module initialization
2. `pixel_interp.pyx` - Pixel interpolation implementations
3. `plans.txt` - 4-phase development plan
4. `unified_cell_plan.txt` - UnifiedCell integration plan

#### Plans.txt Analysis

The existing 4-phase plan is well-structured:

```
Phase 1: Create pixel kernels (new file, no risk)
├── _pixel_kernels.pxd - declarations
├── _pixel_kernels.pyx - implementations for lines/corners
├── _hue_pixel_kernels.pyx - hue-aware versions
└── Test: Verify pixel output matches array kernel output

Phase 2: Brick dispatch (new file, no risk)  
├── _brick_topology.pxd - CSR grid structures
├── _brick_dispatch.pyx - main dispatch loop
└── Test: Verify brick membership and local coordinates

Phase 3: Integration (new code, minimal risk)
├── Create Python BrickGrid class
├── Create factory to build CellPtr arrays from UnifiedCell
└── Test: Full pipeline with various cell types

Phase 4: Optional array kernel refactor (low priority)
├── Gradually replace inline logic with pixel kernel calls
├── Benchmark before/after
└── Only do if maintenance benefit outweighs risk
```

**Evaluation:**

✅ **Strengths:**
- Clear separation of concerns
- Risk-aware phasing
- Comprehensive testing at each phase
- Optional optimization phase

⚠️ **Gaps Identified:**
- No memory management strategy documented
- Thread safety not addressed
- Error handling not specified
- Performance targets not quantified

#### Unified_cell_plan.txt Analysis

**Content:** Complete UnifiedCell structure specification with:
- Python dataclass definition
- Cython struct mapping (CellData)
- Type discriminators (CellType enum)
- Field usage by cell type

**Evaluation:**

✅ **Strengths:**
- Comprehensive type specifications
- Clear C-compatibility path
- All cell types covered

⚠️ **Gaps:**
- No serialization/deserialization plan
- Memory layout not optimized for cache locality
- No discussion of struct alignment

---

### 2. Brick Stack Module

**Location:** `chromatica/v2core/brick_stack/`

#### Files Present
1. `brick_stack.c` - C implementation
2. `brick_stack.h` - Header file

#### Current Status

**No Python Wrapper:** The C implementation exists but lacks:
- Python bindings
- Cython interface
- Documentation
- Test coverage

**Missing Components:**
1. `brick_stack.pyx` - Cython wrapper
2. `__init__.py` - Python module initialization
3. `brick_stack.pxd` - Cython declarations
4. Tests
5. Documentation

---

## Improvement Recommendations

### 1. Enhanced Pixel Kernel Plan

#### Add Phase 0: Foundation (Pre-implementation)

```markdown
Phase 0: Foundation & Design
├── Define performance benchmarks
│   ├── Target: 1M pixels/sec minimum
│   ├── Memory: <1GB for 4K image
│   └── Latency: <16ms for interactive updates
├── Design memory management
│   ├── Arena allocator for cell data
│   ├── Reference counting for shared data
│   └── Cache-friendly memory layout
├── Define error handling
│   ├── Error codes enum
│   ├── Error message buffer
│   └── Graceful degradation strategy
└── Thread safety analysis
    ├── Identify shared state
    ├── Define locking strategy
    └── Consider lock-free alternatives
```

**Rationale:** Setting foundations before implementation prevents costly refactoring.

#### Enhanced Phase 1: Pixel Kernels with Error Handling

```markdown
Phase 1: Create Pixel Kernels (Enhanced)
├── _pixel_kernels.pxd - declarations
│   ├── Error code enum
│   ├── Kernel function signatures
│   └── Result struct (value + error code)
├── _pixel_kernels.pyx - implementations
│   ├── Corner interpolation kernel
│   ├── Line interpolation kernel
│   ├── Error checking at boundaries
│   └── Graceful NaN handling
├── _hue_pixel_kernels.pyx - hue-aware versions
│   ├── Circular interpolation modes
│   ├── Grayscale detection
│   └── Hue wrapping edge cases
└── Tests
    ├── Verify pixel output vs array kernel
    ├── Test boundary conditions
    ├── Test error handling
    └── Benchmark performance
```

#### Enhanced Phase 2: Brick Dispatch with Profiling

```markdown
Phase 2: Brick Dispatch (Enhanced)
├── _brick_topology.pxd
│   ├── CSR grid structures
│   ├── Spatial indexing (quadtree optional)
│   └── Memory pool management
├── _brick_dispatch.pyx
│   ├── Main dispatch loop
│   ├── Parallel dispatch (OpenMP)
│   ├── Cache optimization
│   └── Profiling instrumentation
└── Tests
    ├── Verify brick membership
    ├── Test local coordinates
    ├── Benchmark serial vs parallel
    └── Memory leak detection
```

#### Enhanced Phase 3: Integration with Monitoring

```markdown
Phase 3: Integration (Enhanced)
├── Python BrickGrid class
│   ├── Public API
│   ├── Context manager for resources
│   ├── Progress callbacks
│   └── Performance metrics
├── Factory functions
│   ├── build_from_unified_cell()
│   ├── build_from_gradient_grid()
│   └── Validation on construction
└── Tests
    ├── Full pipeline various cell types
    ├── Memory usage monitoring
    ├── Error propagation
    └── Resource cleanup verification
```

---

### 2. Brick Stack Wrapper Development Plan

#### New File: brick_stack.pyx

```python
# chromatica/v2core/brick_stack/brick_stack.pyx
"""
Cython wrapper for brick_stack C implementation.

Provides Python interface to efficient grid-based cell management.
"""

cimport numpy as np
import numpy as np

# External C functions from brick_stack.c
cdef extern from "brick_stack.h":
    ctypedef struct BrickStackGrid:
        int num_rows
        int num_cols
        int total_cells
        void* cell_data
    
    BrickStackGrid* brick_stack_create(int rows, int cols)
    void brick_stack_destroy(BrickStackGrid* grid)
    int brick_stack_set_cell(BrickStackGrid* grid, int row, int col, void* cell_data)
    void* brick_stack_get_cell(BrickStackGrid* grid, int row, int col)
    int brick_stack_render(BrickStackGrid* grid, double* output, int width, int height)


cdef class BrickGrid:
    """
    Python wrapper for BrickStackGrid C structure.
    
    Manages a grid of cells with efficient memory layout and rendering.
    """
    
    cdef BrickStackGrid* _grid
    cdef int _num_rows
    cdef int _num_cols
    
    def __cinit__(self, int num_rows, int num_cols):
        """Initialize brick grid."""
        self._grid = brick_stack_create(num_rows, num_cols)
        if self._grid is NULL:
            raise MemoryError("Failed to create brick stack grid")
        self._num_rows = num_rows
        self._num_cols = num_cols
    
    def __dealloc__(self):
        """Clean up resources."""
        if self._grid is not NULL:
            brick_stack_destroy(self._grid)
    
    @property
    def num_rows(self):
        return self._num_rows
    
    @property
    def num_cols(self):
        return self._num_cols
    
    def set_cell(self, int row, int col, cell_data):
        """Set cell data at position."""
        # Convert Python cell_data to C structure
        # TODO: Implementation depends on cell data format
        pass
    
    def get_cell(self, int row, int col):
        """Get cell data at position."""
        cdef void* cell_ptr = brick_stack_get_cell(self._grid, row, col)
        if cell_ptr is NULL:
            return None
        # Convert C structure to Python object
        # TODO: Implementation depends on cell data format
        pass
    
    def render(self, int width, int height):
        """
        Render entire grid to numpy array.
        
        Args:
            width: Output width in pixels
            height: Output height in pixels
            
        Returns:
            np.ndarray of shape (height, width, 3)
        """
        cdef np.ndarray[double, ndim=3] output = np.empty(
            (height, width, 3), dtype=np.float64
        )
        
        cdef int result = brick_stack_render(
            self._grid,
            &output[0, 0, 0],
            width,
            height
        )
        
        if result != 0:
            raise RuntimeError(f"Render failed with error code {result}")
        
        return output
```

#### New File: __init__.py

```python
# chromatica/v2core/brick_stack/__init__.py
"""
Brick Stack module for efficient grid-based cell management.

Provides high-performance rendering of cell grids using C backend.
"""

from .brick_stack import BrickGrid

__all__ = ['BrickGrid']
```

#### New File: test_brick_stack.py

```python
# tests/v2core/test_brick_stack.py
"""Tests for brick_stack module."""

import pytest
import numpy as np
from chromatica.v2core.brick_stack import BrickGrid


def test_brick_grid_creation():
    """Test creating brick grid."""
    grid = BrickGrid(num_rows=5, num_cols=5)
    assert grid.num_rows == 5
    assert grid.num_cols == 5


def test_brick_grid_cell_access():
    """Test setting and getting cells."""
    grid = BrickGrid(3, 3)
    # TODO: Add cell data
    # grid.set_cell(1, 1, cell_data)
    # assert grid.get_cell(1, 1) is not None


def test_brick_grid_rendering():
    """Test rendering grid."""
    grid = BrickGrid(2, 2)
    # TODO: Add cells
    result = grid.render(width=200, height=200)
    assert result.shape == (200, 200, 3)
    assert result.dtype == np.float64


def test_brick_grid_cleanup():
    """Test resource cleanup."""
    grid = BrickGrid(5, 5)
    del grid
    # Should not leak memory


def test_brick_grid_error_handling():
    """Test error handling."""
    with pytest.raises(ValueError):
        BrickGrid(num_rows=0, num_cols=5)
```

---

### 3. Integration with Gradient Grid

#### Factory Function: gradient_grid_to_brick

```python
# chromatica/gradients/gradient_grid/brick_integration.py
"""
Integration between GradientGrid and BrickGrid.
"""

from typing import List
import numpy as np
from ..gradient_grid import GradientGrid
from ...v2core.brick_stack import BrickGrid


def gradient_grid_to_brick(grid: GradientGrid) -> BrickGrid:
    """
    Convert GradientGrid to BrickGrid for efficient rendering.
    
    Args:
        grid: GradientGrid to convert
        
    Returns:
        BrickGrid ready for rendering
        
    Example:
        >>> grad_grid = GradientGrid(10, 10, 64, 64)
        >>> # ... populate with cells ...
        >>> brick_grid = gradient_grid_to_brick(grad_grid)
        >>> result = brick_grid.render(640, 640)
    """
    brick_grid = BrickGrid(
        num_rows=grid.num_rows,
        num_cols=grid.num_cols
    )
    
    for row_idx in range(grid.num_rows):
        for col_idx in range(grid.num_cols):
            cell = grid.get_cell(row_idx, col_idx)
            if cell is not None:
                # Convert UnifiedCell to brick format
                brick_cell_data = _unified_cell_to_brick(cell)
                brick_grid.set_cell(row_idx, col_idx, brick_cell_data)
    
    return brick_grid


def _unified_cell_to_brick(cell):
    """Convert UnifiedCell to brick-compatible format."""
    # TODO: Implementation depends on C structure layout
    pass
```

---

### 4. Performance Optimization Strategy

#### Memory Layout Optimization

```c
// Proposed cache-friendly cell structure
typedef struct {
    // Type discriminator (1 byte) + padding (3 bytes)
    uint8_t cell_type;
    uint8_t _pad[3];
    
    // Dimensions (aligned to 8 bytes)
    int32_t width;
    int32_t height;
    
    // Inline small data (64 bytes cache line)
    // For corner cells: 4 corners × 4 channels × 8 bytes = 128 bytes
    // For line cells: pointer to line data
    union {
        struct {
            double corners[16];  // 4 corners × 4 channels
        } corner_data;
        struct {
            double* top_line;
            double* bottom_line;
            int32_t line_length;
        } line_data;
    };
    
    // Pointer to extended data if needed
    void* extended_data;
    
} __attribute__((aligned(64))) CellData;
```

**Benefits:**
- 64-byte alignment for cache efficiency
- Inline small data to avoid pointer chasing
- Discriminated union for type-specific data

#### Parallel Rendering Strategy

```cython
# Using OpenMP for parallel brick dispatch
def render_parallel(grid: BrickGrid, width: int, height: int, 
                   num_threads: int = 4) -> np.ndarray:
    """
    Render grid in parallel using multiple threads.
    
    Args:
        grid: BrickGrid to render
        width: Output width
        height: Output height
        num_threads: Number of threads to use
        
    Returns:
        Rendered array
    """
    cdef np.ndarray[double, ndim=3] output = np.empty(
        (height, width, 3), dtype=np.float64
    )
    
    cdef int r, c
    cdef int rows = grid.num_rows
    cdef int cols = grid.num_cols
    
    # Parallel loop over grid cells
    with nogil, parallel(num_threads=num_threads):
        for r in prange(rows):
            for c in range(cols):
                # Render cell
                render_cell_brick(grid, r, c, output)
    
    return output
```

---

## Implementation Priority Matrix

| Feature | Priority | Effort | Impact | Status |
|---------|----------|--------|--------|--------|
| Phase 0: Foundation | High | Low | High | ⚠️ Add to plan |
| Error handling in kernels | High | Medium | High | ⚠️ Add to plan |
| Brick stack Python wrapper | High | Medium | High | 🚧 Needs implementation |
| Memory layout optimization | Medium | High | High | 💡 Proposed |
| Parallel rendering | Medium | Medium | High | 💡 Proposed |
| Profiling instrumentation | Medium | Low | Medium | ⚠️ Add to plan |
| Quadtree spatial indexing | Low | High | Medium | 🔮 Future |
| Serialization support | Low | Medium | Low | 🔮 Future |

**Legend:**
- ⚠️ Add to existing plans
- 🚧 Needs immediate implementation
- 💡 Design proposal (evaluate first)
- 🔮 Future enhancement

---

## Testing Strategy

### Unit Tests

```python
# tests/v2core/pixel_based_kernel/test_pixel_kernels.py

def test_corner_kernel_basic():
    """Test basic corner interpolation."""
    from chromatica.v2core.pixel_based_kernel import corner_pixel_kernel
    
    # 4 corner colors
    corners = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0]])
    
    # Sample at center
    u_x, u_y = 0.5, 0.5
    result = corner_pixel_kernel(corners, u_x, u_y)
    
    # Should be average of corners
    expected = corners.mean(axis=0)
    np.testing.assert_array_almost_equal(result, expected)


def test_line_kernel_basic():
    """Test basic line interpolation."""
    from chromatica.v2core.pixel_based_kernel import line_pixel_kernel
    
    top_line = np.linspace([1,0,0], [0,1,0], 100)
    bottom_line = np.linspace([0,0,1], [1,1,0], 100)
    
    # Sample at center
    u_x, u_y = 0.5, 0.5
    result = line_pixel_kernel(top_line, bottom_line, u_x, u_y, 
                              discrete_x=True)
    
    # Check result is between top and bottom at x=50
    assert 0 <= result[0] <= 1
    assert 0 <= result[1] <= 1
    assert 0 <= result[2] <= 1


def test_hue_kernel_wraparound():
    """Test hue kernel handles wraparound correctly."""
    from chromatica.v2core.pixel_based_kernel import hue_pixel_kernel
    
    # Hue values near wraparound (350° and 10°)
    h1, h2 = 350.0, 10.0
    
    # Interpolate with shortest path
    result = hue_pixel_kernel(h1, h2, u=0.5, mode=SHORTEST)
    
    # Should be 0° (or 360°), not 180°
    assert result < 5 or result > 355
```

### Integration Tests

```python
# tests/v2core/brick_stack/test_brick_integration.py

def test_gradient_grid_to_brick_conversion():
    """Test converting GradientGrid to BrickGrid."""
    from chromatica.gradients.gradient_grid import GradientGrid
    from chromatica.gradients.gradient_grid.brick_integration import (
        gradient_grid_to_brick
    )
    
    # Create gradient grid
    grad_grid = GradientGrid(5, 5, 64, 64)
    # ... populate with cells ...
    
    # Convert to brick
    brick_grid = gradient_grid_to_brick(grad_grid)
    
    assert brick_grid.num_rows == grad_grid.num_rows
    assert brick_grid.num_cols == grad_grid.num_cols


def test_brick_rendering_matches_direct():
    """Test brick rendering matches direct rendering."""
    grad_grid = create_test_gradient_grid()
    
    # Direct rendering
    direct_result = grad_grid.render()
    
    # Brick rendering
    brick_grid = gradient_grid_to_brick(grad_grid)
    brick_result = brick_grid.render(grad_grid.width, grad_grid.height)
    
    # Should match (within tolerance)
    np.testing.assert_array_almost_equal(direct_result, brick_result, 
                                        decimal=5)
```

### Performance Benchmarks

```python
# tests/v2core/benchmarks/test_brick_performance.py

def test_brick_rendering_performance(benchmark):
    """Benchmark brick rendering vs direct rendering."""
    grad_grid = create_large_gradient_grid(20, 20)  # 400 cells
    brick_grid = gradient_grid_to_brick(grad_grid)
    
    # Benchmark brick rendering
    def render_brick():
        return brick_grid.render(1280, 1280)
    
    result = benchmark(render_brick)
    
    # Should be faster than direct rendering
    # Target: <100ms for 1280×1280 image


def test_parallel_speedup(benchmark):
    """Test parallel rendering speedup."""
    brick_grid = create_test_brick_grid(50, 50)  # Large grid
    
    # Serial rendering
    t_serial = benchmark_serial_render(brick_grid)
    
    # Parallel rendering (4 threads)
    t_parallel = benchmark_parallel_render(brick_grid, num_threads=4)
    
    # Should achieve >2x speedup
    assert t_parallel < t_serial / 2
```

---

## Documentation Needs

### User Documentation

1. **Brick Stack User Guide** (NEW)
   - What is brick-based rendering?
   - When to use BrickGrid vs direct rendering
   - Performance characteristics
   - Best practices

2. **Pixel Kernel Reference** (NEW)
   - Available kernel functions
   - Parameter descriptions
   - Performance considerations
   - Example usage

3. **Migration Guide** (NEW)
   - Converting existing code to use brick stack
   - Performance comparison
   - Feature parity checklist

### Developer Documentation

1. **Architecture Overview** (UPDATE)
   - Add brick stack architecture diagram
   - Memory layout documentation
   - Thread safety guarantees

2. **Extending Kernels** (NEW)
   - How to add new pixel kernels
   - Testing requirements
   - Performance benchmarking

3. **C API Reference** (NEW)
   - Document brick_stack.h functions
   - Usage examples in C
   - Memory management rules

---

## Conclusion

The existing plans for pixel_based_kernel and brick_stack are solid foundations, but would benefit from:

### Immediate Actions
1. ✅ Add Phase 0 (Foundation) to pixel kernel plan
2. ✅ Enhance error handling in all phases
3. 🚧 Implement brick_stack Python wrapper (HIGH PRIORITY)
4. ✅ Add profiling instrumentation to plans
5. ✅ Define performance benchmarks

### Short-Term Improvements
1. Memory layout optimization
2. Parallel rendering implementation
3. Comprehensive test suite
4. Documentation for users and developers

### Long-Term Enhancements
1. Quadtree spatial indexing for very large grids
2. GPU acceleration exploration
3. Serialization/deserialization support
4. Advanced caching strategies

**Next Step:** Begin implementation of brick_stack Python wrapper as it's the critical missing piece for Gradient Grid integration.

---

**Last Updated:** 2026-01-03  
**Status:** Analysis Complete, Ready for Implementation
