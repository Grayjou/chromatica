# Performance Optimization Hotspots

This document identifies performance-critical code paths and potential optimization opportunities in the Chromatica library.

## Purpose

This document helps developers:
- Understand which parts of the code are performance-critical
- Identify candidates for Cython optimization
- Plan performance improvement efforts
- Benchmark and measure optimization impact

## Current Optimization Status

### Already Optimized with Cython ✅

1. **1D Interpolation** (`v2core/interp.pyx`)
   - Status: Fully optimized
   - Performance: Excellent (10-100x faster than pure Python)
   - Coverage: Complete

2. **2D Line Interpolation** (`v2core/interp_2d.pyx`)
   - Status: Well optimized
   - Performance: Very good (20-50x faster than pure Python)
   - Coverage: Discrete and continuous sampling
   - Functions: `lerp_between_lines`, `lerp_between_lines_x_discrete_1ch`, `lerp_between_lines_x_discrete_multichannel`

3. **Hue Interpolation** (`v2core/interp_hue.pyx`)
   - Status: Fully optimized
   - Performance: Excellent (30-80x faster than pure Python)
   - Coverage: 1D and 2D hue interpolation with direction control

4. **Corner Interpolation** (`v2core/corner_interp_2d.pyx`)
   - Status: Optimized
   - Performance: Very good
   - Coverage: Bilinear interpolation from corner values

## High Priority Optimization Targets 🔥🔥🔥

### 1. Coordinate Transformations

**Location:** `chromatica/gradients/gradient2dv2/helpers/cell_utils.py`

**Function:** `apply_per_channel_transforms_2d()` and `apply_per_channel_transforms_2d_single()`

**Current Status:** Pure Python with numpy

**Estimated Speedup:** 10-50x

**Difficulty:** Medium

**Why Optimize:**
- Called once per cell creation
- Involves heavy array manipulation and copying
- Creates per-channel coordinate grids
- Lambda function evaluation per pixel

**Impact:** Would significantly speed up gradient initialization time, especially for large gradients with custom transforms.

**Implementation Approach:**
```cython
# chromatica/v2core/coord_transforms.pyx

cimport numpy as np
import numpy as np

cpdef list apply_transforms_2d_cython(
    coords,  # Either single array or list
    dict transforms,
    int num_channels
):
    """Cython-optimized per-channel coordinate transformation."""
    cdef int h, w, c
    cdef np.ndarray[np.float64_t, ndim=3] result
    cdef np.ndarray[np.float64_t, ndim=3] base
    
    # Handle list vs single array input
    if isinstance(coords, list):
        # List of coordinate grids
        result_list = []
        for c in range(num_channels):
            if c < len(coords):
                base = coords[c].astype(np.float64)
            else:
                base = coords[0].astype(np.float64)
            
            if c in transforms:
                # Apply transform (still calls Python function)
                transformed = transforms[c](base)
                result_list.append(transformed)
            else:
                result_list.append(base.copy())
        return result_list
    else:
        # Single coordinate grid for all channels
        base = coords.astype(np.float64)
        h, w = base.shape[0], base.shape[1]
        result_list = []
        
        for c in range(num_channels):
            if c in transforms:
                transformed = transforms[c](base.copy())
                result_list.append(transformed)
            else:
                result_list.append(base.copy())
        return result_list
```

**Note:** Full optimization requires handling common transform patterns (power, sine, etc.) directly in Cython rather than calling Python lambdas.

---

### 2. Polar Coordinate Conversion

**Location:** Not yet implemented (needed for future angular-radial gradients)

**Current Status:** Would be pure Python/numpy

**Estimated Speedup:** 20-100x

**Difficulty:** Low-Medium

**Why Optimize:**
- Cartesian to polar conversion is CPU-intensive
- Involves trigonometric functions (atan2, sqrt) per pixel
- Used for every pixel in angular-radial gradients
- Tight inner loop suitable for Cython

**Impact:** Critical for performance of angular-radial gradient generation. Without optimization, large angular-radial gradients would be very slow.

**Implementation Approach:**
```cython
# chromatica/v2core/polar_coords.pyx

cimport numpy as np
import numpy as np
from libc.math cimport sqrt, atan2, cos, sin, M_PI

cpdef tuple cartesian_to_polar_2d(
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t, ndim=2] y,
    double cx,
    double cy
):
    """
    Convert Cartesian coordinates to polar (r, θ).
    
    Args:
        x: X-coordinates (H, W)
        y: Y-coordinates (H, W)
        cx: Center x-coordinate
        cy: Center y-coordinate
    
    Returns:
        Tuple of (radius, theta) arrays
    """
    cdef int h = x.shape[0]
    cdef int w = x.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] r = np.empty((h, w), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] theta = np.empty((h, w), dtype=np.float64)
    cdef int i, j
    cdef double dx, dy
    
    # Release GIL for pure C operations
    with nogil:
        for i in range(h):
            for j in range(w):
                dx = x[i, j] - cx
                dy = y[i, j] - cy
                r[i, j] = sqrt(dx * dx + dy * dy)
                theta[i, j] = atan2(dy, dx)
    
    return (r, theta)

cpdef tuple polar_to_cartesian_2d(
    np.ndarray[np.float64_t, ndim=2] r,
    np.ndarray[np.float64_t, ndim=2] theta,
    double cx,
    double cy
):
    """Convert polar coordinates to Cartesian."""
    cdef int h = r.shape[0]
    cdef int w = r.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] x = np.empty((h, w), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((h, w), dtype=np.float64)
    cdef int i, j
    
    with nogil:
        for i in range(h):
            for j in range(w):
                x[i, j] = cx + r[i, j] * cos(theta[i, j])
                y[i, j] = cy + r[i, j] * sin(theta[i, j])
    
    return (x, y)
```

**Benchmarking:**
```python
def benchmark_polar_conversion():
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for h, w in sizes:
        x = np.linspace(0, 1, w).reshape(1, -1).repeat(h, axis=0)
        y = np.linspace(0, 1, h).reshape(-1, 1).repeat(w, axis=1)
        cx, cy = 0.5, 0.5
        
        # Time pure Python/numpy version
        start = time.time()
        dx = x - cx
        dy = y - cy
        r_py = np.sqrt(dx**2 + dy**2)
        theta_py = np.arctan2(dy, dx)
        time_py = time.time() - start
        
        # Time Cython version
        start = time.time()
        r_cy, theta_cy = cartesian_to_polar_2d(x, y, cx, cy)
        time_cy = time.time() - start
        
        print(f"{h}x{w}: Python {time_py:.4f}s, Cython {time_cy:.4f}s, "
              f"Speedup: {time_py/time_cy:.1f}x")
```

---

## Medium Priority Optimization Targets 🔥🔥

### 3. Cell Partitioning and Slicing

**Location:** `chromatica/gradients/gradient2dv2/cell/_cell_coords.py`

**Functions:** `slice_and_renormalize()`, `slice_coords()`, `extract_edge()`, `lerp_point()`

**Current Status:** Pure Python with numpy

**Estimated Speedup:** 3-10x

**Difficulty:** Medium-High

**Why Optimize:**
- Called frequently when using partitions
- Involves array copying and index calculations
- Memory allocation overhead
- Loop over channels

**Impact:** Would speed up gradient generation when using partitions (multiple color space transitions).

**Recommendation:**
- Profile first to confirm this is a bottleneck
- Consider implementing only if partitions are heavily used
- May not be worth the complexity for occasional use

**When to Optimize:**
- If profiling shows significant time in these functions
- When implementing complex multi-partition gradients
- If user feedback indicates slow partition performance

---

### 4. Border Handling

**Location:** `chromatica/v2core/border_handler.py`

**Current Status:** Mix of Python and optimized code

**Estimated Speedup:** 2-5x

**Difficulty:** Medium

**Why Consider:**
- Called for every out-of-bounds coordinate access
- Can be hot path depending on transform functions
- Currently has some Cython optimization

**Recommendation:**
- Already partially optimized
- Further optimization only if profiling shows issues
- Current implementation is likely sufficient

---

## Low Priority / Already Efficient 🔥

### 5. Color Space Conversions

**Location:** `chromatica/conversions/` module

**Current Status:** Already uses numpy efficiently

**Estimated Speedup:** 1.5-3x

**Difficulty:** High (complex logic)

**Why Low Priority:**
- Already well-optimized with numpy
- Called less frequently than interpolation
- Complex branching logic difficult to optimize
- ROI is low

**Recommendation:**
- Only consider if profiling shows significant time here
- Better to optimize calling patterns (caching, batching)
- Current numpy implementation is good enough

---

### 6. Cell Factory Functions

**Location:** `chromatica/gradients/gradient2dv2/cell/factory.py`

**Current Status:** Pure Python orchestration

**Estimated Speedup:** Minimal (already fast)

**Why Low Priority:**
- Mostly orchestration code, not computation
- Called once per cell initialization
- Heavy lifting done by already-optimized interpolation

**Recommendation:**
- No optimization needed
- Focus on the functions it calls (see above)

---

## Profiling Strategy

### Step 1: Baseline Measurements

```python
# benchmark_tests/profile_gradients.py

import cProfile
import pstats
from pstats import SortKey
import time

def profile_cell_creation():
    """Profile cell creation and rendering."""
    from chromatica.gradients.gradient2dv2.cell import get_transformed_corners_cell
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Create various cells
    for _ in range(100):
        cell = get_transformed_corners_cell(
            top_left=np.array([0., 0., 0.]),
            top_right=np.array([1., 0., 0.]),
            bottom_left=np.array([0., 1., 0.]),
            bottom_right=np.array([1., 1., 0.]),
            per_channel_coords=np.random.random((100, 100, 2)),
            color_space=ColorSpace.RGB,
        )
        result = cell.get_value()
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)

def benchmark_gradient_sizes():
    """Benchmark different gradient sizes."""
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for h, w in sizes:
        start = time.time()
        # Create gradient
        cell = get_transformed_corners_cell(...)
        result = cell.get_value()
        elapsed = time.time() - start
        
        print(f"{h}x{w}: {elapsed:.3f}s ({elapsed*1000/(h*w):.2f} µs/pixel)")
```

### Step 2: Identify Hotspots

Look for:
1. Functions taking >5% of total time
2. Functions called thousands of times
3. Functions with high cumulative time
4. Tight loops over arrays

### Step 3: Optimize and Measure

For each optimization:
1. Write benchmark before optimization
2. Implement Cython version
3. Write benchmark after optimization
4. Document speedup
5. Add performance regression test

---

## Memory Optimization

### Current Memory Usage

**Typical gradient memory usage:**
- 100x100 RGB: ~120 KB
- 500x500 RGB: ~3 MB
- 1000x1000 RGB: ~12 MB
- 2000x2000 RGB: ~48 MB
- 4000x4000 RGB: ~192 MB

**With per-channel coordinates:**
- Multiply by ~2-3x for transformed coordinates

### Optimization Strategies

1. **Lazy Evaluation** ✅ (Already Implemented)
   - SubGradient._value caching works well
   - Only compute when needed

2. **Chunked Rendering** (Future Enhancement)
   ```python
   def render_chunked(self, chunk_size: int = 1000):
       """Render gradient in chunks to reduce memory."""
       for chunk in self._generate_chunks(chunk_size):
           yield chunk
   ```

3. **Coordinate Sharing**
   - Reuse coordinate grids when possible
   - Avoid unnecessary copies
   - Use views instead of copies where safe

4. **Memory Limits**
   ```python
   class CellBase:
       _cache_size_limit = 100_000_000  # 100MB
       
       def get_value(self):
           if self._value is not None and self._value.nbytes > self._cache_size_limit:
               warnings.warn("Cell cache exceeds limit")
   ```

---

## Parallel Processing Opportunities

### Cell Rendering

**Opportunity:** Cell rendering is embarrassingly parallel

**Implementation:**
```python
from concurrent.futures import ProcessPoolExecutor

def render_cells_parallel(cells, num_workers=4):
    """Render multiple cells in parallel."""
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(cell.get_value) for cell in cells]
        return [f.result() for f in futures]
```

**Note:** Use `ProcessPoolExecutor` not `ThreadPoolExecutor` due to Python GIL. Cython with `nogil` could enable threading.

### Partitioned Gradients

When using partitions, each partition slice can be rendered independently:

```python
def render_partitioned_parallel(factory, partition):
    """Render partitioned gradient in parallel."""
    slices = factory.partition_slice(partition)
    rendered = render_cells_parallel(slices)
    return np.concatenate(rendered, axis=1)  # Horizontal concatenation
```

---

## Performance Testing

### Regression Tests

```python
# tests/performance/test_performance.py

import pytest
import time

@pytest.mark.performance
def test_cell_rendering_speed():
    """Ensure cell rendering meets performance targets."""
    cell = create_test_cell(500, 500)
    
    start = time.time()
    result = cell.get_value()
    elapsed = time.time() - start
    
    # Should render 500x500 in <100ms
    assert elapsed < 0.1, f"Rendering too slow: {elapsed:.3f}s"

@pytest.mark.performance
def test_large_gradient_memory():
    """Ensure memory usage stays reasonable."""
    import psutil
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    cell = create_test_cell(2000, 2000)
    result = cell.get_value()
    
    mem_after = process.memory_info().rss
    mem_used = (mem_after - mem_before) / 1024 / 1024
    
    # Should use <200MB for 2000x2000 RGB
    assert mem_used < 200, f"Memory usage too high: {mem_used:.1f}MB"
```

---

## Summary and Roadmap

### Immediate Priorities (High Impact, Feasible)

1. ✅ Document optimization hotspots (this file)
2. 🔥🔥🔥 Implement polar coordinate Cython module (needed for angular-radial)
3. 🔥🔥🔥 Profile coordinate transformations
4. 🔥🔥 Implement coordinate transform Cython module if profiling confirms need

### Future Enhancements

1. 🔥🔥 Cell partitioning optimization (if profiling shows bottleneck)
2. 🔥 Chunked rendering for very large gradients
3. 🔥 Parallel rendering support
4. Performance regression test suite

### When to Optimize

- **Before:** Profile to confirm bottleneck
- **During:** Benchmark before and after
- **After:** Add regression test

### What Not to Optimize

- Functions called infrequently (< 0.1% of total time)
- Already-efficient numpy operations
- Complex orchestration code (focus on computational kernels)

---

**Last Updated:** 2025-12-29
