# Brick Stack Membership Algorithm

**Created:** 2026-01-03  
**Purpose:** Document the brick stack spatial membership algorithm for non-uniform grid partitioning

---

## Problem Statement

Given a **brick stack** - a rectangular grid where:
- Horizontal rows can have DIFFERENT heights
- All bricks in the SAME row have the SAME height
- Bricks within a row can have VARYING widths
- The stack is always rectangular overall

**Input:**
1. Normalized y intervals defining row boundaries: `[0.0, 0.2, 0.5, 1.0]`
2. Per-row normalized x intervals: `[[0.0, 0.3, 0.6, 1.0], [0.0, 0.6, 1.0], ...]`
3. Coordinate matrix: WxHx2 with [ux, uy] normalized positions

**Output:**
1. Pertinency matrix: WxHx2 with [brick_idx, row_idx] 
2. Relative position within brick: [rel_x, rel_y] in [0,1]×[0,1]

---

## Algorithm

### Example

**Given:**
- Y intervals: `[0.0, 0.5, 1.0]` (2 rows: [0,0.5) and [0.5,1.0])
- X intervals row 0: `[0.0, 0.3, 0.6, 1.0]` (3 bricks)
- X intervals row 1: `[0.0, 0.6, 1.0]` (2 bricks)

**Query point:** `(ux=0.4, uy=0.4)`

**Step 1: Find row**
- 0.4 is between y_intervals[0]=0.0 and y_intervals[1]=0.5
- Row index: 0
- Relative y: (0.4 - 0.0) / (0.5 - 0.0) = 0.8

**Step 2: Find brick within row**
- Use x_intervals[0] = [0.0, 0.3, 0.6, 1.0]
- 0.4 is between x_intervals[0][1]=0.3 and x_intervals[0][2]=0.6
- Brick index: 1
- Relative x: (0.4 - 0.3) / (0.6 - 0.3) = 0.333...

**Result:**
- Pertinency: [brick=1, row=0]
- Relative position: [rel_x=0.333, rel_y=0.8]

---

## Python Implementation

### Core Algorithm

```python
import numpy as np
from typing import List, Tuple


def find_brick_membership(
    coords: np.ndarray,  # Shape (H, W, 2) with [ux, uy]
    y_intervals: np.ndarray,  # Shape (num_rows+1,) 
    x_intervals: List[np.ndarray],  # Per-row, varying lengths
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find brick membership and relative positions.
    
    Args:
        coords: Normalized coordinates [ux, uy] in [0,1]
        y_intervals: Row boundaries [0.0, ..., 1.0]
        x_intervals: Per-row brick boundaries, list of arrays
        
    Returns:
        pertinency: (H, W, 2) array with [brick_idx, row_idx]
        relative_pos: (H, W, 2) array with [rel_x, rel_y] in [0,1]
    
    Example:
        >>> y_int = np.array([0.0, 0.5, 1.0])
        >>> x_int = [
        ...     np.array([0.0, 0.3, 0.6, 1.0]),  # Row 0: 3 bricks
        ...     np.array([0.0, 0.6, 1.0])         # Row 1: 2 bricks
        ... ]
        >>> coords = np.array([[[0.4, 0.4]]])  # Single point
        >>> pert, rel = find_brick_membership(coords, y_int, x_int)
        >>> print(pert)  # [[[[1, 0]]]]  (brick 1, row 0)
        >>> print(rel)   # [[[[0.333, 0.8]]]]
    """
    H, W = coords.shape[:2]
    
    # Extract coordinates
    ux = coords[..., 0]  # Shape (H, W)
    uy = coords[..., 1]  # Shape (H, W)
    
    # Output arrays
    pertinency = np.zeros((H, W, 2), dtype=np.int32)
    relative_pos = np.zeros((H, W, 2), dtype=np.float64)
    
    # Find row membership using searchsorted
    row_indices = np.searchsorted(y_intervals[1:], uy, side='right')
    row_indices = np.clip(row_indices, 0, len(y_intervals) - 2)
    
    # Compute relative y position
    y_start = y_intervals[row_indices]
    y_end = y_intervals[row_indices + 1]
    rel_y = (uy - y_start) / (y_end - y_start)
    rel_y = np.clip(rel_y, 0.0, 1.0)
    
    # For each row, find brick membership
    for row_idx in range(len(x_intervals)):
        # Mask for points in this row
        row_mask = (row_indices == row_idx)
        
        if not np.any(row_mask):
            continue
        
        # Get x intervals for this row
        x_int = x_intervals[row_idx]
        
        # Find brick indices within row
        ux_in_row = ux[row_mask]
        brick_indices = np.searchsorted(x_int[1:], ux_in_row, side='right')
        brick_indices = np.clip(brick_indices, 0, len(x_int) - 2)
        
        # Compute relative x position
        x_start = x_int[brick_indices]
        x_end = x_int[brick_indices + 1]
        rel_x = (ux_in_row - x_start) / (x_end - x_start)
        rel_x = np.clip(rel_x, 0.0, 1.0)
        
        # Store results
        pertinency[row_mask, 0] = brick_indices
        pertinency[row_mask, 1] = row_idx
        relative_pos[row_mask, 0] = rel_x
        relative_pos[row_mask, 1] = rel_y[row_mask]
    
    return pertinency, relative_pos
```

---

## Cython Optimized Version

### High-Performance Implementation

```cython
# brick_membership.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport floor

ctypedef np.float64_t f64
ctypedef np.int32_t i32


cdef inline int binary_search_interval(
    double value,
    const double* intervals,
    int n_intervals
) nogil:
    """
    Find which interval contains value.
    Returns index i such that intervals[i] <= value < intervals[i+1]
    """
    cdef int left = 0
    cdef int right = n_intervals - 1
    cdef int mid
    
    # Edge cases
    if value <= intervals[0]:
        return 0
    if value >= intervals[n_intervals - 1]:
        return n_intervals - 2
    
    # Binary search
    while left < right - 1:
        mid = (left + right) // 2
        if intervals[mid] <= value:
            left = mid
        else:
            right = mid
    
    return left


def find_brick_membership_fast(
    np.ndarray[f64, ndim=3] coords,  # (H, W, 2)
    np.ndarray[f64, ndim=1] y_intervals,
    list x_intervals_list,  # List of 1D arrays
):
    """
    Fast Cython implementation of brick membership.
    
    Optimized for:
    - Minimal memory allocation
    - Cache-friendly access patterns
    - Parallel processing (OpenMP optional)
    - No Python overhead in inner loops
    """
    cdef int H = coords.shape[0]
    cdef int W = coords.shape[1]
    cdef int num_rows = len(x_intervals_list)
    
    # Output arrays
    cdef np.ndarray[i32, ndim=3] pertinency = np.zeros(
        (H, W, 2), dtype=np.int32
    )
    cdef np.ndarray[f64, ndim=3] relative_pos = np.zeros(
        (H, W, 2), dtype=np.float64
    )
    
    # Memory views for fast access
    cdef f64[:, :, :] coords_view = coords
    cdef f64[:] y_int_view = y_intervals
    cdef i32[:, :, :] pert_view = pertinency
    cdef f64[:, :, :] rel_view = relative_pos
    
    # Convert x_intervals to contiguous arrays
    cdef list x_intervals = []
    cdef list x_intervals_views = []
    for x_int in x_intervals_list:
        arr = np.ascontiguousarray(x_int, dtype=np.float64)
        x_intervals.append(arr)
        x_intervals_views.append(<f64[:len(arr)]>arr)
    
    cdef int h, w, row_idx, brick_idx
    cdef f64 ux, uy
    cdef f64 y_start, y_end, rel_y
    cdef f64 x_start, x_end, rel_x
    cdef int n_y_int = len(y_intervals)
    cdef int n_x_int
    cdef f64[:] x_int_view
    
    # Main loop - can be parallelized with OpenMP
    with nogil:
        for h in range(H):
            for w in range(W):
                # Get coordinates
                ux = coords_view[h, w, 0]
                uy = coords_view[h, w, 1]
                
                # Find row
                row_idx = binary_search_interval(
                    uy, &y_int_view[0], n_y_int
                )
                
                # Compute relative y
                y_start = y_int_view[row_idx]
                y_end = y_int_view[row_idx + 1]
                rel_y = (uy - y_start) / (y_end - y_start)
                if rel_y < 0.0:
                    rel_y = 0.0
                elif rel_y > 1.0:
                    rel_y = 1.0
                
                # Store row info
                pert_view[h, w, 1] = row_idx
                rel_view[h, w, 1] = rel_y
    
    # Find brick indices (requires Python for list access)
    for h in range(H):
        for w in range(W):
            ux = coords[h, w, 0]
            row_idx = pertinency[h, w, 1]
            
            x_int_view = x_intervals_views[row_idx]
            n_x_int = len(x_int_view)
            
            with nogil:
                # Find brick
                brick_idx = binary_search_interval(
                    ux, &x_int_view[0], n_x_int
                )
                
                # Compute relative x
                x_start = x_int_view[brick_idx]
                x_end = x_int_view[brick_idx + 1]
                rel_x = (ux - x_start) / (x_end - x_start)
                if rel_x < 0.0:
                    rel_x = 0.0
                elif rel_x > 1.0:
                    rel_x = 1.0
            
            # Store brick info
            pertinency[h, w, 0] = brick_idx
            relative_pos[h, w, 0] = rel_x
    
    return pertinency, relative_pos
```

---

## Usage Example

```python
import numpy as np
from chromatica.v2core.brick_stack import find_brick_membership

# Define brick stack structure
y_intervals = np.array([0.0, 0.2, 0.5, 1.0])  # 3 rows
x_intervals = [
    np.array([0.0, 0.3, 0.6, 1.0]),     # Row 0: 3 bricks
    np.array([0.0, 0.4, 0.7, 0.9, 1.0]), # Row 1: 4 bricks
    np.array([0.0, 0.5, 1.0])            # Row 2: 2 bricks
]

# Generate coordinate grid
H, W = 100, 100
ux, uy = np.meshgrid(
    np.linspace(0, 1, W),
    np.linspace(0, 1, H)
)
coords = np.stack([ux, uy], axis=-1)

# Find membership
pertinency, relative_pos = find_brick_membership(
    coords, y_intervals, x_intervals
)

# Extract info for a specific point
h, w = 40, 60
brick_idx = pertinency[h, w, 0]
row_idx = pertinency[h, w, 1]
rel_x = relative_pos[h, w, 0]
rel_y = relative_pos[h, w, 1]

print(f"Point ({ux[h,w]:.3f}, {uy[h,w]:.3f}):")
print(f"  - In row {row_idx}, brick {brick_idx}")
print(f"  - Relative position: ({rel_x:.3f}, {rel_y:.3f})")
```

---

## Integration with Gradient Grid

### Brick-Based Rendering

```python
class GradientGrid:
    """Grid with variable-height rows."""
    
    def __init__(self, y_intervals: List[float], 
                 x_intervals: List[List[float]]):
        """
        Initialize grid with brick stack structure.
        
        Args:
            y_intervals: Row boundaries [0.0, ..., 1.0]
            x_intervals: Per-row brick boundaries
        """
        self.y_intervals = np.array(y_intervals)
        self.x_intervals = [np.array(x) for x in x_intervals]
        self.num_rows = len(self.y_intervals) - 1
        
        # Validate
        assert self.y_intervals[0] == 0.0
        assert self.y_intervals[-1] == 1.0
        for x_int in self.x_intervals:
            assert x_int[0] == 0.0
            assert x_int[-1] == 1.0
    
    def render(self, width: int, height: int) -> np.ndarray:
        """
        Render grid using brick membership algorithm.
        
        Args:
            width: Output width in pixels
            height: Output height in pixels
            
        Returns:
            RGB array of shape (height, width, 3)
        """
        # Generate coordinate grid
        ux, uy = np.meshgrid(
            np.linspace(0, 1, width),
            np.linspace(0, 1, height)
        )
        coords = np.stack([ux, uy], axis=-1)
        
        # Find membership
        pertinency, relative_pos = find_brick_membership(
            coords, self.y_intervals, self.x_intervals
        )
        
        # Render each brick
        output = np.zeros((height, width, 3))
        
        for row_idx in range(self.num_rows):
            num_bricks = len(self.x_intervals[row_idx]) - 1
            
            for brick_idx in range(num_bricks):
                # Mask for this brick
                mask = (
                    (pertinency[..., 0] == brick_idx) &
                    (pertinency[..., 1] == row_idx)
                )
                
                if not np.any(mask):
                    continue
                
                # Get brick cell
                cell = self.get_cell(row_idx, brick_idx)
                
                # Render brick
                rel_pos_brick = relative_pos[mask]
                colors = self._render_cell(cell, rel_pos_brick)
                
                # Write to output
                output[mask] = colors
        
        return output
    
    def _render_cell(self, cell, rel_pos):
        """Render cell at relative positions."""
        # Use existing cell rendering logic
        # cell could be CornersCell or LinesCell
        pass
```

---

## Performance Optimization

### CSR (Compressed Sparse Row) Format

For very large grids, use CSR format for efficient storage:

```python
class BrickStackCSR:
    """
    Brick stack in CSR format for memory efficiency.
    
    Similar to sparse matrix CSR format:
    - row_ptr: [0, 3, 7, 9, ...]  (cumulative brick counts)
    - brick_boundaries: concatenated x intervals
    - cell_data: flat array of all cells
    """
    
    def __init__(self, y_intervals, x_intervals_list):
        self.y_intervals = np.array(y_intervals)
        
        # Build CSR structure
        self.row_ptr = [0]
        self.brick_boundaries = []
        
        for x_int in x_intervals_list:
            num_bricks = len(x_int) - 1
            self.row_ptr.append(self.row_ptr[-1] + num_bricks)
            self.brick_boundaries.extend(x_int)
        
        self.row_ptr = np.array(self.row_ptr, dtype=np.int32)
        self.brick_boundaries = np.array(self.brick_boundaries)
    
    def get_brick_bounds(self, row_idx, brick_idx):
        """Get [x_start, x_end] for brick."""
        brick_offset = self.row_ptr[row_idx] + brick_idx
        return (
            self.brick_boundaries[brick_offset],
            self.brick_boundaries[brick_offset + 1]
        )
```

---

## Testing

```python
def test_brick_membership_basic():
    """Test basic membership finding."""
    y_int = np.array([0.0, 0.5, 1.0])
    x_int = [
        np.array([0.0, 0.3, 0.6, 1.0]),
        np.array([0.0, 0.6, 1.0])
    ]
    
    # Single point
    coords = np.array([[[0.4, 0.4]]])
    pert, rel = find_brick_membership(coords, y_int, x_int)
    
    # Should be brick 1, row 0
    assert pert[0, 0, 0] == 1  # brick_idx
    assert pert[0, 0, 1] == 0  # row_idx
    
    # Relative position
    assert abs(rel[0, 0, 0] - 1/3) < 0.01  # rel_x
    assert abs(rel[0, 0, 1] - 0.8) < 0.01  # rel_y


def test_brick_membership_edges():
    """Test edge cases."""
    y_int = np.array([0.0, 0.5, 1.0])
    x_int = [
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0])
    ]
    
    # Corner points
    test_points = [
        ([0.0, 0.0], [0, 0]),  # Top-left
        ([1.0, 0.0], [1, 0]),  # Top-right
        ([0.0, 1.0], [0, 1]),  # Bottom-left
        ([1.0, 1.0], [0, 1]),  # Bottom-right
    ]
    
    for (ux, uy), (expected_brick, expected_row) in test_points:
        coords = np.array([[[ux, uy]]])
        pert, _ = find_brick_membership(coords, y_int, x_int)
        assert pert[0, 0, 0] == expected_brick
        assert pert[0, 0, 1] == expected_row


def test_brick_membership_performance():
    """Benchmark membership finding."""
    import time
    
    # Large grid
    y_int = np.linspace(0, 1, 50)  # 49 rows
    x_int = [np.linspace(0, 1, np.random.randint(5, 20)) 
             for _ in range(49)]
    
    # 1920×1080 coordinates
    H, W = 1080, 1920
    ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    coords = np.stack([ux, uy], axis=-1)
    
    start = time.time()
    pert, rel = find_brick_membership(coords, y_int, x_int)
    elapsed = time.time() - start
    
    print(f"Processed {H*W:,} points in {elapsed:.3f}s")
    print(f"Throughput: {H*W/elapsed:,.0f} points/sec")
    
    # Should be fast (<100ms for 2M points)
    assert elapsed < 0.1
```

---

## Conclusion

The brick stack membership algorithm provides efficient spatial binning for non-uniform grid partitioning. Key features:

1. **Flexible Structure**: Variable row heights and brick widths
2. **Efficient Lookup**: O(log n) per query using binary search
3. **Vectorized**: Processes entire coordinate arrays at once
4. **Cython-Ready**: Inner loops can run without GIL
5. **Memory-Efficient**: CSR format for large grids

This algorithm is the foundation for rendering Gradient Grids with mixed cell types and variable dimensions.

---

**Next Steps:**
1. Implement Cython optimized version
2. Add OpenMP parallelization
3. Integrate with GradientGrid rendering
4. Benchmark against direct rendering

**Last Updated:** 2026-01-03
