# Gradient Grid Feature - Implementation Plan

**Created:** 2026-01-03  
**Purpose:** Plan integration of Gradient Grid feature with cell generators

## Feature Requirements

### Core Concept: Gradient Grid

A **Gradient Grid** is a stack of rows of gradient cells with the following characteristics:

1. **Structure:**
   - Stack is always rectangular
   - Composed of rows of gradient cells
   - All rows have the same height
   - Can be partitioned dynamically

2. **Interactive Vertices:**
   - Border vertices move their respective counterparts
   - `top_left` moves `top_right` if y changes, `bottom_left` if x changes
   - This maintains rectangular structure

3. **Cell Types:**
   - **Corner Cells:** Have editable color points at corners
   - **Line Cells:** Have movable points that determine start and end (no color editing)
   - Different cell types can coexist in the same grid

4. **Performance Features:**
   - Lazy rendering (compute only when needed)
   - Invalidation after editing (mark dirty regions)
   - Efficient updates when vertices change

---

## Current Architecture Analysis

### Existing Cell System

**Location:** `chromatica/gradients/gradient2dv2/cell/`

The current system has:

1. **UnifiedCell structure** (`unified_cell_plan.txt`)
   - Already supports different cell types via enum discriminator
   - `CellType.CORNERS`, `CellType.CORNERS_DUAL`, `CellType.LINES`
   - Has caching support (`_value`, `_top_segment`, `_bottom_segment`)

2. **Cell Interface:**
   ```python
   class CellType(IntEnum):
       CORNERS = 0
       CORNERS_DUAL = 1
       LINES = 2
   ```

3. **Existing Features:**
   - Per-channel coordinates
   - Multiple color modes (RGB, HSL, HSV, OKLCH)
   - Hue direction control
   - Border handling (bound_types and border_mode)

### Brick Stack Architecture

**Location:** `chromatica/v2core/brick_stack/`

Currently contains:
- `brick_stack.c`, `brick_stack.h` - C implementation (needs review)
- No Python wrapper yet
- Designed for efficient grid management

### Pixel-Based Kernel

**Location:** `chromatica/v2core/pixel_based_kernel/`

**Plans:**
- `plans.txt` - 4-phase development plan
- `unified_cell_plan.txt` - Integration with UnifiedCell

**Current Status:** Implementation staged but not integrated

---

## Integration Architecture

### Phase 1: Grid Structure

#### 1.1 Grid Container Class

```python
class GradientGrid:
    """
    Rectangular grid of gradient cells with interactive vertices.
    
    Attributes:
        rows: List[GridRow]
        width: int  # Total width in pixels
        height: int  # Total height in pixels
        row_heights: List[int]  # Heights of each row (all same initially)
    """
    
    def __init__(self, num_rows: int, num_cols: int, 
                 cell_width: int, cell_height: int):
        """Initialize rectangular grid."""
        pass
    
    def get_cell(self, row: int, col: int) -> UnifiedCell:
        """Get cell at position."""
        pass
    
    def set_cell(self, row: int, col: int, cell: UnifiedCell):
        """Set cell at position."""
        pass
    
    def partition_rows(self, split_row: int, height_ratios: tuple):
        """Split a row into multiple rows."""
        pass
    
    def move_vertex(self, row: int, col: int, vertex: str, 
                   delta_x: float, delta_y: float):
        """Move a vertex and update connected cells."""
        pass
    
    def invalidate_region(self, top_row: int, left_col: int,
                         bottom_row: int, right_col: int):
        """Mark region as dirty for re-rendering."""
        pass
    
    def render(self, force: bool = False) -> np.ndarray:
        """Render grid with lazy evaluation."""
        pass
```

#### 1.2 Grid Row Structure

```python
class GridRow:
    """
    A single row in the gradient grid.
    
    Attributes:
        cells: List[UnifiedCell]
        height: int
        y_offset: int  # Position in grid
        dirty: bool  # Needs re-rendering
    """
    
    def render(self) -> np.ndarray:
        """Render all cells in row."""
        pass
```

### Phase 2: Vertex Synchronization

#### 2.1 Vertex Movement Logic

When a border vertex moves:

```python
def sync_vertices(grid: GradientGrid, row: int, col: int, 
                 vertex: str, delta_x: float, delta_y: float):
    """
    Synchronize vertex movements to maintain rectangular structure.
    
    Rules:
    - top_left movement:
        - delta_y: affects top_right of same cell
        - delta_x: affects bottom_left of same cell
    - top_right movement:
        - delta_y: affects top_left of same cell
        - delta_x: affects bottom_right of same cell
    - etc.
    """
    
    cell = grid.get_cell(row, col)
    
    if vertex == "top_left":
        if delta_y != 0:
            # Move top_right's y coordinate
            cell.top_right = update_vertex_y(cell.top_right, delta_y)
        if delta_x != 0:
            # Move bottom_left's x coordinate
            cell.bottom_left = update_vertex_x(cell.bottom_left, delta_x)
            
    # Similar logic for other vertices...
    
    # Invalidate affected cells
    grid.invalidate_region(row, col, row, col)
```

#### 2.2 Constraint System

Implement constraint solver for maintaining:
- Rectangular structure
- Equal row heights (configurable)
- Cell alignment
- Boundary consistency

### Phase 3: Cell Type Differentiation

#### 3.1 Editable vs Movable Points

```python
class CellEditor:
    """
    Handles editing operations based on cell type.
    """
    
    def can_edit_color(self, cell: UnifiedCell, point: str) -> bool:
        """Check if color at point can be edited."""
        if cell.cell_type == CellType.CORNERS:
            return point in ["top_left", "top_right", 
                           "bottom_left", "bottom_right"]
        elif cell.cell_type == CellType.LINES:
            return False  # Lines only movable, not color-editable
        return False
    
    def can_move_point(self, cell: UnifiedCell, point: str) -> bool:
        """Check if point can be moved."""
        # All cell types allow position movement
        return True
    
    def edit_color(self, cell: UnifiedCell, point: str, 
                  new_color: np.ndarray):
        """Edit color at point (if allowed)."""
        if not self.can_edit_color(cell, point):
            raise ValueError(f"Cannot edit color for {cell.cell_type}")
        
        # Update color
        setattr(cell, point, new_color)
        cell._value = None  # Invalidate cache
```

#### 3.2 Rendering Dispatch

```python
def render_cell(cell: UnifiedCell, 
               editor: CellEditor) -> np.ndarray:
    """
    Render cell based on type with appropriate constraints.
    """
    
    if cell.cell_type == CellType.CORNERS:
        # Full corner interpolation with editable colors
        return render_corners_cell(cell)
        
    elif cell.cell_type == CellType.LINES:
        # Line interpolation with fixed colors, movable endpoints
        return render_lines_cell(cell)
        
    elif cell.cell_type == CellType.CORNERS_DUAL:
        # Dual-mode corner interpolation
        return render_corners_dual_cell(cell)
    
    raise ValueError(f"Unknown cell type: {cell.cell_type}")
```

### Phase 4: Lazy Rendering and Invalidation

#### 4.1 Cache Management

```python
class RenderCache:
    """
    Manages rendered cell data with invalidation.
    """
    
    def __init__(self):
        self.cache: Dict[Tuple[int, int], np.ndarray] = {}
        self.dirty: Set[Tuple[int, int]] = set()
    
    def get(self, row: int, col: int) -> Optional[np.ndarray]:
        """Get cached render if valid."""
        if (row, col) in self.dirty:
            return None
        return self.cache.get((row, col))
    
    def set(self, row: int, col: int, data: np.ndarray):
        """Cache rendered data."""
        self.cache[(row, col)] = data
        self.dirty.discard((row, col))
    
    def invalidate(self, row: int, col: int):
        """Mark cell as dirty."""
        self.dirty.add((row, col))
    
    def invalidate_region(self, top_row: int, left_col: int,
                         bottom_row: int, right_col: int):
        """Invalidate rectangular region."""
        for r in range(top_row, bottom_row + 1):
            for c in range(left_col, right_col + 1):
                self.dirty.add((r, c))
```

#### 4.2 Lazy Evaluation

```python
def render_grid_lazy(grid: GradientGrid, 
                    cache: RenderCache) -> np.ndarray:
    """
    Render grid with lazy evaluation of cells.
    """
    
    output = np.empty((grid.height, grid.width, 
                      grid.rows[0].cells[0].num_channels))
    
    y_offset = 0
    for row_idx, row in enumerate(grid.rows):
        x_offset = 0
        for col_idx, cell in enumerate(row.cells):
            # Check cache
            cached = cache.get(row_idx, col_idx)
            if cached is None:
                # Render and cache
                cached = render_cell(cell)
                cache.set(row_idx, col_idx, cached)
            
            # Copy to output
            h, w = cached.shape[:2]
            output[y_offset:y_offset+h, 
                  x_offset:x_offset+w] = cached
            
            x_offset += w
        y_offset += row.height
    
    return output
```

---

## Integration with Existing Pixel Kernel

### Leveraging Existing Plans

The `pixel_based_kernel/plans.txt` outlines a 4-phase approach:

1. **Phase 1: Create pixel kernels** ✅ (foundation exists)
   - Pixel-level interpolation functions
   - Hue-aware variants

2. **Phase 2: Brick dispatch** 🔄 (needs Python wrapper)
   - CSR grid structures
   - Main dispatch loop

3. **Phase 3: Integration** 🚀 (this plan)
   - Python BrickGrid class (→ GradientGrid)
   - Factory for CellPtr arrays from UnifiedCell
   - Full pipeline with various cell types

4. **Phase 4: Array kernel refactor** (optional)
   - Replace inline logic with pixel kernel calls
   - Benchmark improvements

### Adaptation for Gradient Grid

The brick dispatch system can be adapted:

```python
def create_brick_grid_from_gradient_grid(grid: GradientGrid) -> BrickGrid:
    """
    Convert GradientGrid to brick dispatch structure.
    """
    
    # Create cell pointer array
    cell_ptrs = []
    for row in grid.rows:
        for cell in row.cells:
            cell_ptr = create_cell_ptr_from_unified(cell)
            cell_ptrs.append(cell_ptr)
    
    # Create brick topology (CSR format)
    brick_grid = BrickGrid(
        num_rows=len(grid.rows),
        num_cols=len(grid.rows[0].cells),
        cell_pointers=cell_ptrs,
        row_heights=[row.height for row in grid.rows],
        col_widths=[cell.width for cell in grid.rows[0].cells],
    )
    
    return brick_grid
```

---

## Implementation Phases

### Phase 1: Core Grid Structure (Weeks 1-2)
- [ ] Implement `GradientGrid` class
- [ ] Implement `GridRow` class
- [ ] Basic cell management (get/set)
- [ ] Write unit tests

### Phase 2: Vertex Synchronization (Week 3)
- [ ] Implement vertex movement logic
- [ ] Add constraint system
- [ ] Test rectangular structure maintenance
- [ ] Add integration tests

### Phase 3: Cell Type Handling (Week 4)
- [ ] Implement `CellEditor` class
- [ ] Add per-cell-type rendering dispatch
- [ ] Test corner cells vs line cells
- [ ] Validate color editing constraints

### Phase 4: Lazy Rendering (Week 5)
- [ ] Implement `RenderCache` class
- [ ] Add invalidation system
- [ ] Optimize rendering pipeline
- [ ] Performance benchmarks

### Phase 5: Brick Kernel Integration (Week 6)
- [ ] Create Python wrapper for brick_stack C code
- [ ] Implement `create_brick_grid_from_gradient_grid()`
- [ ] Benchmark brick dispatch vs direct rendering
- [ ] Document performance characteristics

### Phase 6: Polish & Documentation (Week 7)
- [ ] API documentation
- [ ] Usage examples
- [ ] Tutorial notebooks
- [ ] Performance guide

---

## Testing Strategy

### Unit Tests

```python
# tests/gradient_grid/test_grid_structure.py
def test_grid_creation():
    """Test basic grid initialization."""
    grid = GradientGrid(num_rows=3, num_cols=4, 
                       cell_width=100, cell_height=100)
    assert grid.width == 400
    assert grid.height == 300

def test_cell_access():
    """Test getting and setting cells."""
    grid = GradientGrid(3, 3, 100, 100)
    cell = create_corner_cell(...)
    grid.set_cell(1, 1, cell)
    retrieved = grid.get_cell(1, 1)
    assert retrieved.cell_type == CellType.CORNERS

# tests/gradient_grid/test_vertices.py
def test_vertex_sync_horizontal():
    """Test vertex synchronization on horizontal movement."""
    grid = create_test_grid()
    grid.move_vertex(0, 0, "top_left", delta_x=10, delta_y=0)
    # Verify bottom_left moved accordingly
    cell = grid.get_cell(0, 0)
    assert cell.bottom_left[0] == original_x + 10

def test_vertex_sync_vertical():
    """Test vertex synchronization on vertical movement."""
    grid = create_test_grid()
    grid.move_vertex(0, 0, "top_left", delta_x=0, delta_y=10)
    # Verify top_right moved accordingly
    cell = grid.get_cell(0, 0)
    assert cell.top_right[1] == original_y + 10

# tests/gradient_grid/test_cell_types.py
def test_corner_cell_color_editing():
    """Test that corner cells allow color editing."""
    cell = create_corner_cell()
    editor = CellEditor()
    assert editor.can_edit_color(cell, "top_left")
    new_color = np.array([1.0, 0.0, 0.0])
    editor.edit_color(cell, "top_left", new_color)
    np.testing.assert_array_equal(cell.top_left, new_color)

def test_line_cell_no_color_editing():
    """Test that line cells don't allow color editing."""
    cell = create_line_cell()
    editor = CellEditor()
    assert not editor.can_edit_color(cell, "top_line")
    with pytest.raises(ValueError):
        editor.edit_color(cell, "top_line", np.array([1.0, 0.0, 0.0]))

# tests/gradient_grid/test_caching.py
def test_lazy_rendering():
    """Test that cells are rendered lazily."""
    grid = create_test_grid()
    cache = RenderCache()
    
    # First render
    result1 = render_grid_lazy(grid, cache)
    assert (0, 0) not in cache.dirty
    
    # Second render should use cache
    result2 = render_grid_lazy(grid, cache)
    np.testing.assert_array_equal(result1, result2)

def test_invalidation():
    """Test cache invalidation on edit."""
    grid = create_test_grid()
    cache = RenderCache()
    
    # Initial render
    render_grid_lazy(grid, cache)
    assert (0, 0) not in cache.dirty
    
    # Edit cell
    grid.invalidate_region(0, 0, 0, 0)
    assert (0, 0) in cache.dirty
    
    # Re-render should compute new value
    result = render_grid_lazy(grid, cache)
    assert (0, 0) not in cache.dirty
```

### Integration Tests

```python
# tests/gradient_grid/test_integration.py
def test_full_pipeline():
    """Test complete workflow from creation to rendering."""
    # Create grid
    grid = GradientGrid(5, 5, 64, 64)
    
    # Populate with mixed cell types
    for r in range(5):
        for c in range(5):
            if (r + c) % 2 == 0:
                cell = create_corner_cell()
            else:
                cell = create_line_cell()
            grid.set_cell(r, c, cell)
    
    # Move some vertices
    grid.move_vertex(2, 2, "top_left", 5, 5)
    
    # Render
    cache = RenderCache()
    result = render_grid_lazy(grid, cache)
    
    assert result.shape == (320, 320, 3)
    assert not np.isnan(result).any()
```

---

## Performance Considerations

### Optimization Strategies

1. **Spatial Partitioning**
   - Only render visible regions
   - Use quadtree for large grids

2. **Incremental Updates**
   - Track changed cells
   - Only re-render dirty regions

3. **Vectorization**
   - Batch cell renders where possible
   - Use numpy broadcasting

4. **Cython Acceleration**
   - Move hot paths to Cython
   - Leverage existing pixel kernels

### Benchmarking

Track performance metrics:
- Grid creation time
- Vertex movement latency
- Render time (cached vs uncached)
- Memory usage
- Cache hit rate

---

## API Design

### Public Interface

```python
# High-level API
from chromatica.gradients.gradient_grid import (
    GradientGrid,
    CellEditor,
    create_corner_cell,
    create_line_cell,
)

# Create grid
grid = GradientGrid(rows=10, cols=10, 
                   cell_width=64, cell_height=64)

# Add cells
for r in range(10):
    for c in range(10):
        if c % 2 == 0:
            cell = create_corner_cell(
                top_left=(1.0, 0.0, 0.0),
                top_right=(0.0, 1.0, 0.0),
                bottom_left=(0.0, 0.0, 1.0),
                bottom_right=(1.0, 1.0, 0.0),
                width=64, height=64,
            )
        else:
            cell = create_line_cell(
                top_line=np.linspace([1,0,0], [0,1,0], 64),
                bottom_line=np.linspace([0,0,1], [1,1,0], 64),
                width=64, height=64,
            )
        grid.set_cell(r, c, cell)

# Move vertex
grid.move_vertex(row=5, col=5, vertex="top_left", 
                delta_x=10, delta_y=5)

# Render
result = grid.render()

# Save
from PIL import Image
img = Image.fromarray((result * 255).astype(np.uint8))
img.save("gradient_grid.png")
```

---

## Future Enhancements

1. **Interactive UI**
   - Drag vertices with mouse
   - Real-time preview
   - Color picker for corner cells

2. **Animation**
   - Interpolate between grid states
   - Keyframe system
   - Export to video

3. **Export Formats**
   - SVG with embedded gradients
   - Shader code generation
   - CSS gradient export

4. **Advanced Constraints**
   - Custom vertex dependencies
   - Automatic alignment
   - Symmetry modes

---

## References

- `pixel_based_kernel/plans.txt` - Kernel architecture
- `pixel_based_kernel/unified_cell_plan.txt` - UnifiedCell structure
- `chromatica/gradients/gradient2dv2/cell/` - Current cell implementation
- `REDUNDANT_FUNCTIONS.md` - Function usage analysis

---

**Status:** Planning Complete  
**Next Step:** Begin Phase 1 implementation  
**Owner:** Development Team  
**Review Date:** 2026-01-10
