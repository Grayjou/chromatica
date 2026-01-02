# border_handling.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Border handling modes for 2D interpolation.

This module provides border handling functions for gradient interpolation
with different wrapping/clamping behaviors at boundaries.
"""



from libc.math cimport fmod, fabs



# =============================================================================
# Border Mode Constants
# =============================================================================
# Border modes for handling coordinates outside [0, 1] range
# Using cdef for internal use and module-level Python ints for export
cdef int _BORDER_REPEAT = 0    # Modulo repeat
cdef int _BORDER_MIRROR = 1    # Mirror repeat
cdef int _BORDER_CONSTANT = 2  # Constant color fill
cdef int _BORDER_CLAMP = 3     # Clamp to edge
cdef int _BORDER_OVERFLOW = 4  # Allow overflow (no border handling)

# Export constants to Python
BORDER_REPEAT = 0
BORDER_MIRROR = 1
BORDER_CONSTANT = 2
BORDER_CLAMP = 3
BORDER_OVERFLOW = 4

# =============================================================================
# Helper Functions
# =============================================================================
cdef inline f64 tri2(f64 x) nogil:
    cdef f64 m = fmod(x, 2.0)
    if m < 0:
        m += 2.0
    return 1.0 - fabs(m - 1.0)


cdef inline f64 clamp(f64 value, f64 min_val, f64 max_val) nogil:
    """
    Clamp value to range [min_val, max_val].
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value

# =============================================================================
# Border Handling Functions
# =============================================================================
cpdef handle_border_edges_2d(f64 x, f64 y, int border_mode):
    """
    Handle border conditions for 2D edge-based interpolation.
    
    This function applies border handling to coordinates that may fall outside
    the [0, 1] normalized range. It's used for general 2D interpolation where
    both axes are treated equally.
    
    Args:
        x: X coordinate (normalized to [0, 1])
        y: Y coordinate (normalized to [0, 1])
        border_mode: Border mode constant (BORDER_REPEAT, BORDER_MIRROR, etc.)
        
    Returns:
        Tuple of (x, y) with border handling applied, or None if BORDER_CONSTANT
        and coordinates are out of bounds.
        
    Border modes:
        BORDER_REPEAT (0): Modulo repeat - coordinates wrap around
        BORDER_MIRROR (1): Mirror repeat - coordinates reflect at boundaries
        BORDER_CONSTANT (2): Returns None for out-of-bounds (caller handles constant fill)
        BORDER_CLAMP (3): Clamp to [0, 1] range
        BORDER_OVERFLOW (4): No handling, allows overflow
    """
    cdef f64 new_x = x
    cdef f64 new_y = y
    cdef bint out_of_bounds = False
    
    # Check if coordinates are out of bounds
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        out_of_bounds = True
    
    # BORDER_CONSTANT: return None for out of bounds
    if border_mode == BORDER_CONSTANT:
        if out_of_bounds:
            return None
        return (x, y)
    
    # BORDER_OVERFLOW: no handling
    if border_mode == BORDER_OVERFLOW:
        return (x, y)
    
    # Apply border handling to x coordinate
    if x < 0.0 or x > 1.0 or (border_mode == _BORDER_REPEAT and x == 1.0):
        if border_mode == _BORDER_REPEAT:
            new_x = fmod(x, 1.0)
            if new_x < 0.0:
                new_x += 1.0
        elif border_mode == _BORDER_MIRROR:
            new_x = tri2(x)
        elif border_mode == _BORDER_CLAMP:
            new_x = clamp(x, 0.0, 1.0)
    
    # Apply border handling to y coordinate
    if y < 0.0 or y > 1.0 or (border_mode == _BORDER_REPEAT and y == 1.0):
        if border_mode == _BORDER_REPEAT:
            new_y = fmod(y, 1.0)
            if new_y < 0.0:
                new_y += 1.0
        elif border_mode == _BORDER_MIRROR:
            new_y = tri2(y)
        elif border_mode == _BORDER_CLAMP:
            new_y = clamp(y, 0.0, 1.0)
    
    return (new_x, new_y)


cpdef handle_border_lines_2d(f64 x, f64 y, int border_mode):
    """
    Handle border conditions for 2D line-based interpolation.
    
    This function is used when interpolating between lines, where the line axis
    (typically x) uses index-based interpolation. For BORDER_OVERFLOW mode,
    it converts to BORDER_CLAMP for the line axis since the interpolation is
    done based on index calculation.
    
    Args:
        x: X coordinate (normalized to [0, 1], along the line)
        y: Y coordinate (normalized to [0, 1], between lines)
        border_mode: Border mode constant
        
    Returns:
        Tuple of (x, y) with border handling applied, or None if BORDER_CONSTANT
        and coordinates are out of bounds.
        
    Note:
        BORDER_OVERFLOW is treated as BORDER_CLAMP for the line axis (x) because
        line interpolation is index-based.
    """
    cdef f64 new_x = x
    cdef f64 new_y = y
    cdef bint out_of_bounds = False
    cdef int effective_mode_X = border_mode
    cdef int effective_mode_Y = border_mode
    # Check if coordinates are out of bounds
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        out_of_bounds = True
    
    # BORDER_CONSTANT: return None for out of bounds
    if border_mode == BORDER_CONSTANT:
        if out_of_bounds:
            return None
        return (x, y)
    
    # BORDER_OVERFLOW becomes BORDER_CLAMP for line axis
    if border_mode == BORDER_OVERFLOW:
        effective_mode_X = _BORDER_CLAMP
    
    # Apply border handling to x coordinate (line axis)
    if x < 0.0 or x > 1.0 or (effective_mode_X == _BORDER_REPEAT and x == 1.0):
        if effective_mode_X == _BORDER_REPEAT:
            new_x = fmod(x, 1.0)
            if new_x < 0.0:
                new_x += 1.0
        elif effective_mode_X == _BORDER_MIRROR:
            new_x = tri2(x)
        elif effective_mode_X == _BORDER_CLAMP:
            new_x = clamp(x, 0.0, 1.0)
    
    # Apply border handling to y coordinate (perpendicular axis)
    if y < 0.0 or y > 1.0 or (effective_mode_Y == _BORDER_REPEAT and y == 1.0):
        if effective_mode_Y == _BORDER_REPEAT:
            new_y = fmod(y, 1.0)
            if new_y < 0.0:
                new_y += 1.0
        elif effective_mode_Y == _BORDER_MIRROR:
            new_y = tri2(y)
        elif effective_mode_Y == _BORDER_CLAMP:
            new_y = clamp(y, 0.0, 1.0)
    
    return (new_x, new_y)