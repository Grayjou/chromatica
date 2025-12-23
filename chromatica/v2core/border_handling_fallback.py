"""
Border handling modes for 2D interpolation - Python fallback implementation.

This module provides border handling functions for gradient interpolation
with different wrapping/clamping behaviors at boundaries.
"""

import math

# Border Mode Constants
BORDER_REPEAT = 0    # Modulo repeat
BORDER_MIRROR = 1    # Mirror repeat
BORDER_CONSTANT = 2  # Constant color fill
BORDER_CLAMP = 3     # Clamp to edge
BORDER_OVERFLOW = 4  # Allow overflow (no border handling)


def tri2(x):
    """
    Triangle wave function for mirror repeat.
    Returns value in [0, 1] that mirrors back and forth.
    
    Args:
        x: Input value
        
    Returns:
        Mirrored value in [0, 1]
    """
    return 1.0 - abs(math.fmod(x, 2.0) - 1.0)


def clamp(value, min_val, max_val):
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


def handle_border_edges_2d(x, y, border_mode):
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
    new_x = x
    new_y = y
    out_of_bounds = False
    
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
    if x < 0.0 or x > 1.0 or (border_mode == BORDER_REPEAT and x == 1.0):
        if border_mode == BORDER_REPEAT:
            new_x = math.fmod(x, 1.0)
            if new_x < 0.0:
                new_x += 1.0
        elif border_mode == BORDER_MIRROR:
            new_x = tri2(x)
        elif border_mode == BORDER_CLAMP:
            new_x = clamp(x, 0.0, 1.0)
    
    # Apply border handling to y coordinate
    if y < 0.0 or y > 1.0 or (border_mode == BORDER_REPEAT and y == 1.0):
        if border_mode == BORDER_REPEAT:
            new_y = math.fmod(y, 1.0)
            if new_y < 0.0:
                new_y += 1.0
        elif border_mode == BORDER_MIRROR:
            new_y = tri2(y)
        elif border_mode == BORDER_CLAMP:
            new_y = clamp(y, 0.0, 1.0)
    
    return (new_x, new_y)


def handle_border_lines_2d(x, y, border_mode):
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
    new_x = x
    new_y = y
    out_of_bounds = False
    effective_mode = border_mode
    
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
        effective_mode = BORDER_CLAMP
    
    # Apply border handling to x coordinate (line axis)
    if x < 0.0 or x > 1.0 or (effective_mode == BORDER_REPEAT and x == 1.0):
        if effective_mode == BORDER_REPEAT:
            new_x = math.fmod(x, 1.0)
            if new_x < 0.0:
                new_x += 1.0
        elif effective_mode == BORDER_MIRROR:
            new_x = tri2(x)
        elif effective_mode == BORDER_CLAMP:
            new_x = clamp(x, 0.0, 1.0)
    
    # Apply border handling to y coordinate (perpendicular axis)
    if y < 0.0 or y > 1.0 or (effective_mode == BORDER_REPEAT and y == 1.0):
        if effective_mode == BORDER_REPEAT:
            new_y = math.fmod(y, 1.0)
            if new_y < 0.0:
                new_y += 1.0
        elif effective_mode == BORDER_MIRROR:
            new_y = tri2(y)
        elif effective_mode == BORDER_CLAMP:
            new_y = clamp(y, 0.0, 1.0)
    
    return (new_x, new_y)
