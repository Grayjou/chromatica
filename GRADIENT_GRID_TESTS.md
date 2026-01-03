# Gradient Grid Test Specifications

**Created:** 2026-01-03  
**Purpose:** Test specifications for Gradient Grid feature implementation

## Test Categories

### 1. Grid Structure Tests

#### 1.1 Basic Grid Creation

```python
def test_grid_initialization():
    """
    Test basic grid creation with default parameters.
    
    Given: Grid dimensions (rows=5, cols=5, cell_size=100)
    When: Creating GradientGrid
    Then: Grid should have correct dimensions and empty cells
    """
    grid = GradientGrid(num_rows=5, num_cols=5, 
                       cell_width=100, cell_height=100)
    
    assert grid.num_rows == 5
    assert grid.num_cols == 5
    assert grid.width == 500
    assert grid.height == 500
    assert len(grid.rows) == 5
    assert all(len(row.cells) == 5 for row in grid.rows)


def test_grid_with_varying_cell_sizes():
    """
    Test grid with different cell dimensions.
    
    Given: Non-uniform cell dimensions
    When: Creating grid
    Then: Total dimensions should be sum of cell sizes
    """
    grid = GradientGrid(num_rows=3, num_cols=4, 
                       cell_width=50, cell_height=75)
    
    assert grid.width == 200
    assert grid.height == 225


def test_grid_validation():
    """
    Test input validation for grid creation.
    
    Given: Invalid parameters
    When: Creating grid
    Then: Should raise appropriate ValueError
    """
    with pytest.raises(ValueError, match="rows must be positive"):
        GradientGrid(num_rows=0, num_cols=5, 
                    cell_width=100, cell_height=100)
    
    with pytest.raises(ValueError, match="columns must be positive"):
        GradientGrid(num_rows=5, num_cols=0, 
                    cell_width=100, cell_height=100)
    
    with pytest.raises(ValueError, match="cell_width must be positive"):
        GradientGrid(num_rows=5, num_cols=5, 
                    cell_width=0, cell_height=100)
```

#### 1.2 Cell Access and Modification

```python
def test_get_cell():
    """
    Test retrieving cells from grid.
    
    Given: Grid with populated cells
    When: Getting cell at position
    Then: Should return correct cell or None
    """
    grid = create_test_grid_with_cells()
    cell = grid.get_cell(2, 3)
    
    assert cell is not None
    assert isinstance(cell, UnifiedCell)


def test_set_cell():
    """
    Test setting cells in grid.
    
    Given: Empty grid and new cell
    When: Setting cell at position
    Then: Cell should be stored and retrievable
    """
    grid = GradientGrid(3, 3, 100, 100)
    cell = create_corner_cell()
    
    grid.set_cell(1, 1, cell)
    retrieved = grid.get_cell(1, 1)
    
    assert retrieved is cell


def test_cell_bounds_checking():
    """
    Test bounds checking for cell access.
    
    Given: Grid with bounds
    When: Accessing out-of-bounds position
    Then: Should raise IndexError
    """
    grid = GradientGrid(3, 3, 100, 100)
    
    with pytest.raises(IndexError):
        grid.get_cell(3, 0)  # Row out of bounds
    
    with pytest.raises(IndexError):
        grid.get_cell(0, 3)  # Col out of bounds
    
    with pytest.raises(IndexError):
        grid.set_cell(-1, 0, create_corner_cell())
```

---

### 2. Vertex Synchronization Tests

#### 2.1 Horizontal Movement

```python
def test_vertex_sync_top_left_x():
    """
    Test top_left x-movement synchronizes bottom_left.
    
    Given: Cell with top_left at (0, 0)
    When: Moving top_left by delta_x=10
    Then: bottom_left x should also move by 10
    """
    grid = create_test_grid()
    cell = grid.get_cell(0, 0)
    original_bl_x = cell.bottom_left[0]
    
    grid.move_vertex(0, 0, "top_left", delta_x=10, delta_y=0)
    
    updated_cell = grid.get_cell(0, 0)
    assert updated_cell.bottom_left[0] == original_bl_x + 10
    assert updated_cell.bottom_left[1] == cell.bottom_left[1]  # y unchanged


def test_vertex_sync_top_right_x():
    """
    Test top_right x-movement synchronizes bottom_right.
    
    Given: Cell with top_right
    When: Moving top_right by delta_x=-5
    Then: bottom_right x should move by -5
    """
    grid = create_test_grid()
    cell = grid.get_cell(0, 0)
    original_br_x = cell.bottom_right[0]
    
    grid.move_vertex(0, 0, "top_right", delta_x=-5, delta_y=0)
    
    updated_cell = grid.get_cell(0, 0)
    assert updated_cell.bottom_right[0] == original_br_x - 5


def test_vertex_sync_cross_cell():
    """
    Test vertex movements affect adjacent cells.
    
    Given: Adjacent cells sharing edge
    When: Moving shared vertex
    Then: Both cells should update
    """
    grid = create_test_grid_with_adjacent_cells()
    
    # Move vertex at boundary
    grid.move_vertex(0, 0, "top_right", delta_x=10, delta_y=0)
    
    # Check both affected cells
    cell_00 = grid.get_cell(0, 0)
    cell_01 = grid.get_cell(0, 1)
    
    assert cell_00.top_right[0] == cell_01.top_left[0]  # Synchronized
```

#### 2.2 Vertical Movement

```python
def test_vertex_sync_top_left_y():
    """
    Test top_left y-movement synchronizes top_right.
    
    Given: Cell with top_left
    When: Moving top_left by delta_y=10
    Then: top_right y should also move by 10
    """
    grid = create_test_grid()
    cell = grid.get_cell(0, 0)
    original_tr_y = cell.top_right[1]
    
    grid.move_vertex(0, 0, "top_left", delta_x=0, delta_y=10)
    
    updated_cell = grid.get_cell(0, 0)
    assert updated_cell.top_right[1] == original_tr_y + 10
    assert updated_cell.top_right[0] == cell.top_right[0]  # x unchanged


def test_vertex_sync_bottom_left_y():
    """
    Test bottom_left y-movement synchronizes bottom_right.
    """
    grid = create_test_grid()
    cell = grid.get_cell(0, 0)
    original_br_y = cell.bottom_right[1]
    
    grid.move_vertex(0, 0, "bottom_left", delta_x=0, delta_y=-5)
    
    updated_cell = grid.get_cell(0, 0)
    assert updated_cell.bottom_right[1] == original_br_y - 5
```

#### 2.3 Combined Movement

```python
def test_vertex_sync_diagonal():
    """
    Test diagonal vertex movement synchronizes both dimensions.
    
    Given: Cell with vertex
    When: Moving vertex diagonally
    Then: Appropriate vertices should move in both dimensions
    """
    grid = create_test_grid()
    cell = grid.get_cell(0, 0)
    original_tr_y = cell.top_right[1]
    original_bl_x = cell.bottom_left[0]
    
    # Move top_left diagonally
    grid.move_vertex(0, 0, "top_left", delta_x=10, delta_y=10)
    
    updated_cell = grid.get_cell(0, 0)
    assert updated_cell.top_right[1] == original_tr_y + 10
    assert updated_cell.bottom_left[0] == original_bl_x + 10
```

---

### 3. Cell Type Tests

#### 3.1 Corner Cell Tests

```python
def test_corner_cell_color_editable():
    """
    Test that corner cells allow color editing at all corners.
    
    Given: Corner cell
    When: Checking editability
    Then: All corner colors should be editable
    """
    cell = create_corner_cell()
    editor = CellEditor()
    
    assert editor.can_edit_color(cell, "top_left")
    assert editor.can_edit_color(cell, "top_right")
    assert editor.can_edit_color(cell, "bottom_left")
    assert editor.can_edit_color(cell, "bottom_right")


def test_corner_cell_edit_color():
    """
    Test editing corner colors.
    
    Given: Corner cell with initial colors
    When: Editing a corner color
    Then: Color should update and cache should invalidate
    """
    cell = create_corner_cell()
    editor = CellEditor()
    new_color = np.array([1.0, 0.5, 0.0])
    
    editor.edit_color(cell, "top_left", new_color)
    
    np.testing.assert_array_equal(cell.top_left, new_color)
    assert cell._value is None  # Cache invalidated


def test_corner_cell_rendering():
    """
    Test rendering corner cell produces valid gradient.
    
    Given: Corner cell with distinct colors
    When: Rendering
    Then: Output should be valid gradient array
    """
    cell = create_corner_cell(
        top_left=[1, 0, 0],
        top_right=[0, 1, 0],
        bottom_left=[0, 0, 1],
        bottom_right=[1, 1, 0],
        width=100, height=100
    )
    
    result = render_cell(cell, CellEditor())
    
    assert result.shape == (100, 100, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    
    # Check corners match input colors
    np.testing.assert_array_almost_equal(result[0, 0], [1, 0, 0])
    np.testing.assert_array_almost_equal(result[0, -1], [0, 1, 0])
    np.testing.assert_array_almost_equal(result[-1, 0], [0, 0, 1])
    np.testing.assert_array_almost_equal(result[-1, -1], [1, 1, 0])
```

#### 3.2 Line Cell Tests

```python
def test_line_cell_not_color_editable():
    """
    Test that line cells don't allow color editing.
    
    Given: Line cell
    When: Attempting to edit color
    Then: Should be prevented
    """
    cell = create_line_cell()
    editor = CellEditor()
    
    assert not editor.can_edit_color(cell, "top_line")
    assert not editor.can_edit_color(cell, "bottom_line")
    
    with pytest.raises(ValueError, match="Cannot edit color"):
        editor.edit_color(cell, "top_line", np.array([1, 0, 0]))


def test_line_cell_points_movable():
    """
    Test that line cell endpoints can be moved.
    
    Given: Line cell
    When: Checking movability
    Then: Endpoints should be movable
    """
    cell = create_line_cell()
    editor = CellEditor()
    
    assert editor.can_move_point(cell, "top_line_start")
    assert editor.can_move_point(cell, "top_line_end")
    assert editor.can_move_point(cell, "bottom_line_start")
    assert editor.can_move_point(cell, "bottom_line_end")


def test_line_cell_rendering():
    """
    Test rendering line cell produces valid interpolation.
    
    Given: Line cell with defined top and bottom lines
    When: Rendering
    Then: Output should interpolate between lines
    """
    top_line = np.linspace([1, 0, 0], [0, 1, 0], 100)
    bottom_line = np.linspace([0, 0, 1], [1, 1, 0], 100)
    
    cell = create_line_cell(
        top_line=top_line,
        bottom_line=bottom_line,
        width=100, height=100
    )
    
    result = render_cell(cell, CellEditor())
    
    assert result.shape == (100, 100, 3)
    
    # Check top and bottom match input lines
    np.testing.assert_array_almost_equal(result[0], top_line)
    np.testing.assert_array_almost_equal(result[-1], bottom_line)
```

#### 3.3 Mixed Cell Type Tests

```python
def test_grid_with_mixed_cell_types():
    """
    Test grid containing both corner and line cells.
    
    Given: Grid with alternating cell types
    When: Rendering entire grid
    Then: Should produce valid output with all cell types
    """
    grid = create_grid_with_mixed_cells()
    cache = RenderCache()
    
    result = render_grid_lazy(grid, cache)
    
    assert result.shape == (grid.height, grid.width, 3)
    assert not np.isnan(result).any()


def test_editing_mixed_grid():
    """
    Test editing operations on grid with mixed cell types.
    
    Given: Mixed grid
    When: Editing corner cell colors and moving line cell endpoints
    Then: Both operations should work correctly
    """
    grid = create_grid_with_mixed_cells()
    editor = CellEditor()
    
    # Edit corner cell
    corner_cell = grid.get_cell(0, 0)
    editor.edit_color(corner_cell, "top_left", [1, 1, 1])
    
    # Move line cell endpoint
    line_cell = grid.get_cell(0, 1)
    editor.move_point(line_cell, "top_line_start", 10, 5)
    
    # Render should work
    result = grid.render()
    assert result is not None
```

---

### 4. Caching and Invalidation Tests

#### 4.1 Cache Functionality

```python
def test_cache_stores_rendered_cells():
    """
    Test that cache stores rendered cell data.
    
    Given: Empty cache
    When: Rendering and caching cell
    Then: Cache should contain data
    """
    cache = RenderCache()
    cell_data = np.random.rand(100, 100, 3)
    
    cache.set(0, 0, cell_data)
    
    retrieved = cache.get(0, 0)
    np.testing.assert_array_equal(retrieved, cell_data)


def test_cache_invalidation():
    """
    Test that invalidated cells return None.
    
    Given: Cached cell data
    When: Invalidating cell
    Then: Cache should return None for that cell
    """
    cache = RenderCache()
    cache.set(0, 0, np.random.rand(100, 100, 3))
    
    cache.invalidate(0, 0)
    
    assert cache.get(0, 0) is None
    assert (0, 0) in cache.dirty


def test_cache_region_invalidation():
    """
    Test invalidating rectangular region.
    
    Given: Grid with cached cells
    When: Invalidating region
    Then: All cells in region should be dirty
    """
    cache = RenderCache()
    
    # Cache some cells
    for r in range(5):
        for c in range(5):
            cache.set(r, c, np.random.rand(10, 10, 3))
    
    # Invalidate 2x2 region
    cache.invalidate_region(1, 1, 2, 2)
    
    # Check dirty cells
    assert (1, 1) in cache.dirty
    assert (1, 2) in cache.dirty
    assert (2, 1) in cache.dirty
    assert (2, 2) in cache.dirty
    
    # Check clean cells
    assert (0, 0) not in cache.dirty
    assert (3, 3) not in cache.dirty
```

#### 4.2 Lazy Rendering

```python
def test_lazy_rendering_uses_cache():
    """
    Test that lazy rendering uses cached data.
    
    Given: Grid with cached renders
    When: Rendering again without changes
    Then: Should use cached data (not re-compute)
    """
    grid = create_test_grid()
    cache = RenderCache()
    
    # First render
    result1 = render_grid_lazy(grid, cache)
    
    # Track if cells were actually rendered
    # (Could use mock to verify render_cell not called)
    
    # Second render should use cache
    result2 = render_grid_lazy(grid, cache)
    
    np.testing.assert_array_equal(result1, result2)


def test_lazy_rendering_recomputes_dirty():
    """
    Test that dirty cells are re-rendered.
    
    Given: Grid with cached renders
    When: Invalidating some cells and rendering
    Then: Only dirty cells should be re-computed
    """
    grid = create_test_grid()
    cache = RenderCache()
    
    # Initial render
    result1 = render_grid_lazy(grid, cache)
    
    # Modify cell and invalidate
    cell = grid.get_cell(1, 1)
    cell.top_left = np.array([1, 1, 1])
    cache.invalidate(1, 1)
    
    # Re-render
    result2 = render_grid_lazy(grid, cache)
    
    # Results should differ
    assert not np.array_equal(result1, result2)


def test_lazy_rendering_performance():
    """
    Test that caching improves performance.
    
    Given: Large grid
    When: Rendering multiple times
    Then: Cached renders should be significantly faster
    """
    grid = create_large_test_grid(rows=20, cols=20)
    cache = RenderCache()
    
    import time
    
    # First render (no cache)
    start = time.time()
    result1 = render_grid_lazy(grid, cache)
    first_time = time.time() - start
    
    # Second render (with cache)
    start = time.time()
    result2 = render_grid_lazy(grid, cache)
    second_time = time.time() - start
    
    # Cached render should be much faster
    assert second_time < first_time * 0.1  # At least 10x faster
```

---

### 5. Row Partitioning Tests

```python
def test_partition_row():
    """
    Test splitting a row into multiple rows.
    
    Given: Grid with row to partition
    When: Partitioning row at index
    Then: Grid should have additional row with correct heights
    """
    grid = GradientGrid(3, 3, 100, 100)
    
    # Partition row 1 into two rows with 60/40 height ratio
    grid.partition_rows(1, height_ratios=(0.6, 0.4))
    
    assert grid.num_rows == 4
    assert grid.rows[1].height == 60
    assert grid.rows[2].height == 40


def test_partition_preserves_cells():
    """
    Test that partitioning preserves cell data.
    
    Given: Grid with populated cells
    When: Partitioning row
    Then: Cell data should be preserved and duplicated
    """
    grid = create_test_grid_with_cells()
    original_cell = grid.get_cell(1, 0)
    
    grid.partition_rows(1, (0.5, 0.5))
    
    # Both new rows should have cell data
    cell_1 = grid.get_cell(1, 0)
    cell_2 = grid.get_cell(2, 0)
    
    assert cell_1 is not None
    assert cell_2 is not None


def test_partition_invalidates_cache():
    """
    Test that partitioning invalidates affected cells.
    
    Given: Grid with cached renders
    When: Partitioning row
    Then: Affected cells should be invalidated
    """
    grid = create_test_grid()
    cache = RenderCache()
    
    # Render to populate cache
    render_grid_lazy(grid, cache)
    
    # Partition
    grid.partition_rows(1, (0.6, 0.4))
    
    # Cache should be dirty for affected rows
    assert (1, 0) in cache.dirty
    assert (2, 0) in cache.dirty
```

---

### 6. Performance Tests

```python
def test_large_grid_rendering():
    """
    Test rendering large grid completes in reasonable time.
    
    Given: Large grid (50x50)
    When: Rendering
    Then: Should complete within time budget
    """
    grid = create_large_test_grid(50, 50)
    cache = RenderCache()
    
    import time
    start = time.time()
    result = render_grid_lazy(grid, cache)
    elapsed = time.time() - start
    
    assert elapsed < 5.0  # Should render in less than 5 seconds
    assert result.shape == (grid.height, grid.width, 3)


def test_incremental_update_performance():
    """
    Test that incremental updates are efficient.
    
    Given: Large grid with cache
    When: Updating single cell
    Then: Update should be fast (only re-render one cell)
    """
    grid = create_large_test_grid(20, 20)
    cache = RenderCache()
    
    # Initial render
    render_grid_lazy(grid, cache)
    
    # Update single cell
    import time
    start = time.time()
    grid.get_cell(10, 10).top_left = [1, 1, 1]
    cache.invalidate(10, 10)
    render_grid_lazy(grid, cache)
    elapsed = time.time() - start
    
    # Should be very fast (only one cell)
    assert elapsed < 0.1


def test_memory_usage():
    """
    Test that cache doesn't consume excessive memory.
    
    Given: Large grid
    When: Rendering and caching
    Then: Memory usage should be reasonable
    """
    import sys
    
    grid = create_large_test_grid(50, 50)
    cache = RenderCache()
    
    initial_size = sys.getsizeof(cache)
    
    render_grid_lazy(grid, cache)
    
    final_size = sys.getsizeof(cache)
    
    # Cache should not be more than 2x grid size
    grid_size = grid.width * grid.height * 3 * 8  # float64
    cache_overhead = final_size - initial_size
    
    assert cache_overhead < grid_size * 2
```

---

### 7. Integration Tests

```python
def test_full_workflow():
    """
    Test complete user workflow.
    
    Given: Empty grid
    When: Creating, populating, editing, and rendering
    Then: All operations should work correctly
    """
    # Create grid
    grid = GradientGrid(5, 5, 64, 64)
    
    # Populate with cells
    for r in range(5):
        for c in range(5):
            if (r + c) % 2 == 0:
                cell = create_corner_cell()
            else:
                cell = create_line_cell()
            grid.set_cell(r, c, cell)
    
    # Edit some cells
    editor = CellEditor()
    editor.edit_color(grid.get_cell(0, 0), "top_left", [1, 0, 0])
    editor.edit_color(grid.get_cell(2, 2), "bottom_right", [0, 1, 0])
    
    # Move some vertices
    grid.move_vertex(3, 3, "top_left", 5, 5)
    
    # Partition a row
    grid.partition_rows(2, (0.5, 0.5))
    
    # Render
    cache = RenderCache()
    result = render_grid_lazy(grid, cache)
    
    # Validate output
    assert result.shape[0] == grid.height
    assert result.shape[1] == grid.width
    assert result.shape[2] == 3
    assert not np.isnan(result).any()
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_brick_kernel_integration():
    """
    Test integration with brick kernel dispatch.
    
    Given: GradientGrid
    When: Converting to brick grid format
    Then: Should produce valid brick structure
    """
    grid = create_test_grid()
    
    brick_grid = create_brick_grid_from_gradient_grid(grid)
    
    assert brick_grid is not None
    assert brick_grid.num_rows == grid.num_rows
    assert brick_grid.num_cols == grid.num_cols
    
    # Render via brick dispatch
    result = brick_grid.render()
    
    # Should match direct rendering
    direct_result = grid.render()
    np.testing.assert_array_almost_equal(result, direct_result)
```

---

## Test Fixtures

```python
# conftest.py or test_helpers.py

import pytest
import numpy as np
from chromatica.gradients.gradient_grid import (
    GradientGrid, UnifiedCell, CellType, CellEditor, RenderCache
)


@pytest.fixture
def empty_grid():
    """Create empty 3x3 grid."""
    return GradientGrid(3, 3, 100, 100)


@pytest.fixture
def test_grid_with_cells():
    """Create grid populated with corner cells."""
    grid = GradientGrid(3, 3, 100, 100)
    for r in range(3):
        for c in range(3):
            cell = create_corner_cell()
            grid.set_cell(r, c, cell)
    return grid


@pytest.fixture
def mixed_cell_grid():
    """Create grid with mixed cell types."""
    grid = GradientGrid(5, 5, 64, 64)
    for r in range(5):
        for c in range(5):
            if (r + c) % 2 == 0:
                cell = create_corner_cell()
            else:
                cell = create_line_cell()
            grid.set_cell(r, c, cell)
    return grid


def create_corner_cell(**kwargs):
    """Helper to create corner cell with defaults."""
    defaults = {
        'cell_type': CellType.CORNERS,
        'width': 100,
        'height': 100,
        'num_channels': 3,
        'top_left': np.array([1.0, 0.0, 0.0]),
        'top_right': np.array([0.0, 1.0, 0.0]),
        'bottom_left': np.array([0.0, 0.0, 1.0]),
        'bottom_right': np.array([1.0, 1.0, 0.0]),
    }
    defaults.update(kwargs)
    return UnifiedCell(**defaults)


def create_line_cell(**kwargs):
    """Helper to create line cell with defaults."""
    defaults = {
        'cell_type': CellType.LINES,
        'width': 100,
        'height': 100,
        'num_channels': 3,
        'top_line': np.linspace([1,0,0], [0,1,0], 100),
        'bottom_line': np.linspace([0,0,1], [1,1,0], 100),
    }
    defaults.update(kwargs)
    return UnifiedCell(**defaults)
```

---

## Coverage Goals

Target coverage by module:

- **Grid Structure:** 100%
- **Vertex Synchronization:** 95%+
- **Cell Types:** 100%
- **Caching:** 95%+
- **Rendering:** 90%+
- **Integration:** 85%+

---

## Performance Benchmarks

### Target Metrics

| Operation | Grid Size | Target Time | Max Memory |
|-----------|-----------|-------------|------------|
| Grid Creation | 10x10 | < 10ms | < 1MB |
| Cell Set/Get | Any | < 1ms | - |
| Vertex Move | Any | < 5ms | - |
| First Render | 10x10 | < 500ms | < 50MB |
| Cached Render | 10x10 | < 50ms | - |
| Incremental Update | 1 cell | < 10ms | - |
| Row Partition | 10x10 | < 100ms | - |

### Stress Tests

- [ ] 100x100 grid rendering
- [ ] 1000 vertex movements
- [ ] 10,000 cache operations
- [ ] Memory leak detection (render 1000 times)
- [ ] Concurrent access (if multithreading)

---

## Test Execution Plan

### Phase 1: Unit Tests (Week 1)
- Grid structure tests
- Cell type tests
- Cache tests

### Phase 2: Integration Tests (Week 2)
- Vertex synchronization
- Lazy rendering
- Row partitioning

### Phase 3: Performance Tests (Week 3)
- Benchmarks
- Stress tests
- Memory profiling

### Phase 4: Validation (Week 4)
- Visual validation (render sample grids)
- Edge case testing
- Regression testing

---

## Continuous Integration

### CI Pipeline

```yaml
# .github/workflows/gradient_grid_tests.yml
name: Gradient Grid Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        python setup_cython.py build_ext --inplace
    
    - name: Run tests
      run: |
        pytest tests/gradient_grid/ -v --cov=chromatica.gradients.gradient_grid
    
    - name: Run performance benchmarks
      run: |
        pytest tests/gradient_grid/test_performance.py --benchmark-only
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

**Status:** Specification Complete  
**Next Step:** Implement test fixtures and begin unit testing  
**Review Date:** 2026-01-10
