# __init__.py or main module file

# Define constants once at module level
class BorderMode:
    """Border mode constants."""
    REPEAT = 0
    MIRROR = 1
    CONSTANT = 2
    CLAMP = 3
    OVERFLOW = 4
BORDER_REPEAT = 0
BORDER_MIRROR = 1
BORDER_CONSTANT = 2
BORDER_CLAMP = 3
BORDER_OVERFLOW = 4

# Try to import Cython implementation first
try:
    from .border_handling import (
        handle_border_edges_2d,
        handle_border_lines_2d,
    )
    # Constants are already defined above, no need to reimport
except ImportError:
    # If Cython extension fails, use Python fallback
    import warnings
    warnings.warn(
        "Cython border handling extension not found, using Python fallback. "
        "Performance may be reduced.",
        ImportWarning
    )
    
    from .border_handling_fallback import (
        handle_border_edges_2d,
        handle_border_lines_2d,
        # No need to import constants since they're defined above
    )