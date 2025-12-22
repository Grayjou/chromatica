from boundednumbers import clamp, bounce, cyclic_wrap_float
import numpy as np
from typing import Callable
from .color_base import ColorBase

def make_arithmetic(color, overflow_function: Callable = clamp):
    """
    Wrap a ColorBase instance with arithmetic behavior.
    Arithmetic persists through operations and conversions.
    
    Industry standard behavior:
    - HSV/HSL hue channel: Uses cyclic wrapping (0-360 degrees)
    - Other channels: Uses specified overflow function (default: clamp)
    - Multiplication/Division: Converts to float format to avoid integer division issues
    - Tuple values: Automatically converted to arrays for arithmetic
    """
    class ArithmeticProxy:
        __slots__ = ("_base", "_overflow_fn")

        def __init__(self, base: ColorBase):
            self._base = base
            self._overflow_fn = overflow_function

        # -----------------------
        # Transparent forwarding
        # -----------------------
        def __getattr__(self, name):
            """Forward attribute access to the wrapped ColorBase instance,
            except convert(), which is wrapped to return another proxy."""
            attr = getattr(self._base, name)

            # Wrap convert() so that it returns arithmetic-enabled instances
            if name == "convert" and callable(attr):
                def wrapped_convert(*args, **kwargs):
                    new_base = attr(*args, **kwargs)
                    return ArithmeticProxy(new_base)
                return wrapped_convert

            return attr

        def unwrap(self):
            """Return the underlying raw ColorBase instance."""
            return self._base

        # -----------------------
        # Core arithmetic engine
        # -----------------------
        def _operate(self, other, op, is_multiplicative=False):
            from .color_base import ColorBase  # local import to avoid cycles

            # Extract base from another proxy
            if isinstance(other, ArithmeticProxy):
                other = other._base

            # Handle alpha channel compatibility
            if isinstance(other, ColorBase):
                # Match alpha channels: if one has alpha and other doesn't, adjust
                if self._base.has_alpha and not other.has_alpha:
                    # Self has alpha, other doesn't -> add alpha to other
                    if self._base.is_array:
                        # Use default alpha matching self's format
                        other = other.with_alpha()
                    else:
                        other = other.with_alpha()
                elif not self._base.has_alpha and other.has_alpha:
                    # Other has alpha, self doesn't -> remove alpha from other
                    # Convert other to non-alpha version by slicing
                    non_alpha_mode = other.mode[:-1] if other.mode.endswith('a') else other.mode
                    other = other.convert(non_alpha_mode, other.format_type)
                
                # Now convert to same space and format
                other = other.convert(
                    self._base.mode, self._base.format_type
                ).value
            
            # Get values and ensure they're arrays
            a = self._base.value
            b = other
            
            # Convert tuples to arrays for arithmetic
            if isinstance(a, tuple):
                a = np.array(a, dtype=float)
            else:
                a = np.asarray(a, dtype=float)
            
            if isinstance(b, tuple):
                b = np.array(b, dtype=float)
            else:
                b = np.asarray(b, dtype=float)
            
            # For multiplicative operations, ensure float dtype to avoid integer division
            if is_multiplicative and a.dtype.kind in ('i', 'u'):
                a = a.astype(float)
            
            # Perform arithmetic
            result = op(a, b)
            
            # Apply overflow handling per channel
            # Convert maxima to array for consistent handling
            maxima_array = np.array(self._base.maxima) if isinstance(self._base.maxima, tuple) else self._base.maxima
            
            # HSV/HSL: hue (channel 0) uses cyclic wrap, others use overflow_fn
            if self._base.has_hue:
                # Separate hue channel from other channels
                if result.ndim == 1:
                    # Single color
                    hue = cyclic_wrap_float(result[0], 0, maxima_array[0])
                    other_channels = self._overflow_fn(result[1:], 0, maxima_array[1:])
                    result = np.concatenate([[hue], other_channels])
                else:
                    # Array of colors
                    hue = cyclic_wrap_float(result[..., 0], 0, maxima_array[0])
                    other_channels = self._overflow_fn(result[..., 1:], 0, maxima_array[1:])
                    result = np.concatenate([hue[..., np.newaxis], other_channels], axis=-1)
            else:
                # RGB and RGBA: apply overflow function to all channels
                result = self._overflow_fn(result, 0, maxima_array)

            # Instantiate a new ColorBase of same class
            new_base = self._base.__class__(result)

            # Return another arithmetic-enabled instance
            return ArithmeticProxy(new_base)

        # -----------------------
        # Operator overloads
        # -----------------------
        def __add__(self, other):
            return self._operate(other, np.add, is_multiplicative=False)

        def __sub__(self, other):
            return self._operate(other, np.subtract, is_multiplicative=False)

        def __mul__(self, other):
            return self._operate(other, np.multiply, is_multiplicative=True)

        def __truediv__(self, other):
            return self._operate(other, np.divide, is_multiplicative=True)
        
        def __radd__(self, other):
            return self.__add__(other)
        
        def __rsub__(self, other):
            # other - self = -(self - other)
            return self._operate(other, lambda a, b: np.subtract(b, a), is_multiplicative=False)
        
        def __rmul__(self, other):
            return self.__mul__(other)
        
        def __rtruediv__(self, other):
            # other / self
            return self._operate(other, lambda a, b: np.divide(b, a), is_multiplicative=True)

        # -----------------------
        # Representation
        # -----------------------
        def __repr__(self):
            return f"ArithmeticProxy({self._base!r})"

    return ArithmeticProxy(color)


# Add arithmetic operators to ColorBase to auto-wrap on first use
def _auto_arithmetic_operation(op_name, is_multiplicative=False):
    """Create an operator that auto-wraps ColorBase with default arithmetic."""
    def operation(self, other):
        # Auto-wrap with default arithmetic (clamp)
        proxy = make_arithmetic(self, overflow_function=clamp)
        # Delegate to the proxy's operator
        return getattr(proxy, op_name)(other)
    return operation


# Inject arithmetic operators into ColorBase
ColorBase.__add__ = _auto_arithmetic_operation('__add__')
ColorBase.__sub__ = _auto_arithmetic_operation('__sub__')
ColorBase.__mul__ = _auto_arithmetic_operation('__mul__', is_multiplicative=True)
ColorBase.__truediv__ = _auto_arithmetic_operation('__truediv__', is_multiplicative=True)
ColorBase.__radd__ = _auto_arithmetic_operation('__radd__')
ColorBase.__rsub__ = _auto_arithmetic_operation('__rsub__')
ColorBase.__rmul__ = _auto_arithmetic_operation('__rmul__', is_multiplicative=True)
ColorBase.__rtruediv__ = _auto_arithmetic_operation('__rtruediv__', is_multiplicative=True)

