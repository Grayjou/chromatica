import numpy as np
from enum import Enum
from typing import Sequence, Callable, Optional, Union, List, Any, Dict
from boundednumbers.functions import clamp, bounce, cyclic_wrap_float
from boundednumbers import UnitFloat
from abc import ABC, abstractmethod
import warnings


def get_value_envelope_from_map(unit_map: np.ndarray, t: Union[UnitFloat, float]) -> Union[UnitFloat, float]:
    """Get the modulated value from a unit map based on input t in [0, 1]."""
    if not 0 <= t <= 1:
        raise ValueError(f"t must be in [0, 1], got {t}")
    index = int(t * (len(unit_map) - 1))
    return unit_map[index]


class Envelope(ABC):
    """Base class for all envelope functions."""
    
    @abstractmethod
    def __call__(self, value: Union[UnitFloat, float, np.ndarray]) -> Union[UnitFloat, float, np.ndarray]:
        """Apply envelope to value(s)."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class UnitMapEnvelope(Envelope):
    """Envelope defined by a unit map (lookup table)."""
    
    def __init__(self, unit_map: np.ndarray):
        if not isinstance(unit_map, np.ndarray):
            unit_map = np.array(unit_map)
        if unit_map.ndim != 1:
            raise ValueError(f"unit_map must be 1D, got shape {unit_map.shape}")
        self.unit_map = unit_map
    
    def __call__(self, value: Union[UnitFloat, float, np.ndarray]) -> Union[UnitFloat, float, np.ndarray]:
        """Apply envelope to value(s)."""
        if isinstance(value, (float, int, UnitFloat)):
            scalar_val = float(value)
            result = get_value_envelope_from_map(self.unit_map, scalar_val)
            if isinstance(value, UnitFloat):
                return UnitFloat(result)
            return result
        else:
            # Vectorized version for arrays
            result = np.zeros_like(value)
            flat_val = value.flatten()
            flat_result = np.zeros_like(flat_val)
            for i, v in enumerate(flat_val):
                flat_result[i] = get_value_envelope_from_map(self.unit_map, v)
            result = flat_result.reshape(value.shape)
            return result
    
    def __repr__(self) -> str:
        return f"UnitMapEnvelope(map_shape={self.unit_map.shape})"


class FunctionEnvelope(Envelope):
    """Envelope defined by a callable function."""
    
    def __init__(self, func: Callable[[Union[UnitFloat, float]], Union[UnitFloat, float]]):
        self.func = func
    
    def __call__(self, value: Union[UnitFloat, float, np.ndarray]) -> Union[UnitFloat, float, np.ndarray]:
        """Apply envelope function to value(s)."""
        if isinstance(value, np.ndarray):
            return np.vectorize(lambda x: self.func(x))(value)
        return self.func(value)
    
    def __repr__(self) -> str:
        return f"FunctionEnvelope(func={self.func.__name__ if hasattr(self.func, '__name__') else 'lambda'})"


class CompositeEnvelope(Envelope):
    """Composite envelope applying multiple envelopes in sequence."""
    
    def __init__(self, envelopes: Sequence[Envelope]):
        self.envelopes = envelopes
    
    def __call__(self, value: Union[UnitFloat, float, np.ndarray]) -> Union[UnitFloat, float, np.ndarray]:
        """Apply all envelopes in sequence."""
        result = value
        for envelope in self.envelopes:
            result = envelope(result)
        return result
    
    def __repr__(self) -> str:
        return f"CompositeEnvelope(envelopes={len(self.envelopes)})"


class OutOfBoundsBehavior(Enum):
    """Behavior when interpolation parameter is outside expected range."""
    CLAMP = "clamp"
    RAISE = "raise"
    BOUNCE = "bounce"
    CYCLIC_WRAP = "cyclic_wrap_float"
    MIRROR = "mirror"  # Add mirror behavior
    IGNORE = "ignore"  # Continue extrapolation


def _apply_out_of_bounds_behavior(
    t: float,
    length: int,
    behavior: OutOfBoundsBehavior,
) -> float:
    """Apply out-of-bounds behavior to normalized coordinate."""
    if behavior == OutOfBoundsBehavior.CLAMP:
        return clamp(t, 0.0, length - 1)
    if behavior == OutOfBoundsBehavior.BOUNCE:
        return bounce(t, 0.0, length - 1)
    if behavior == OutOfBoundsBehavior.CYCLIC_WRAP:
        return cyclic_wrap_float(t, 0.0, length - 1)
    if behavior == OutOfBoundsBehavior.MIRROR:
        # Mirror reflection at boundaries
        period = 2 * (length - 1)
        t_mod = t % period
        if t_mod > length - 1:
            t_mod = period - t_mod
        return t_mod
    if behavior == OutOfBoundsBehavior.RAISE:
        if t < 0 or t >= length:
            raise ValueError(
                f"Interpolation parameter t={t} is out of bounds for length={length}."
            )
        return t
    if behavior == OutOfBoundsBehavior.IGNORE:
        return t  # Allow extrapolation
    
    warnings.warn(f"Unknown out-of-bounds behavior: {behavior}, defaulting to CLAMP")
    return clamp(t, 0.0, length - 1)


def multiple_envelope(
    values: Sequence[np.ndarray],
    t: Union[float, np.ndarray],
    unit_envelopes: Optional[Sequence[Optional[Callable[[UnitFloat], UnitFloat]]]] = None,
    out_of_bounds_behavior: OutOfBoundsBehavior = OutOfBoundsBehavior.CLAMP,
    cyclic: bool = False,
    direction: Optional[str] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Interpolate between multiple values with optional envelope transforms.
    
    Args:
        values: Sequence of value arrays to interpolate between
        t: Interpolation parameter(s). Can be scalar or array.
        unit_envelopes: Optional envelope functions for each segment
        out_of_bounds_behavior: How to handle t outside [0, len(values)-1]
        cyclic: Whether to treat values as cyclic (for hue-like values)
        direction: Direction for cyclic interpolation ('cw', 'ccw', or None for shortest)
    
    Returns:
        Interpolated value(s)
    """
    if len(values) == 0:
        raise ValueError("values must contain at least one element.")
    
    # Handle single value case
    if len(values) == 1:
        if isinstance(t, np.ndarray):
            return np.full_like(t, values[0])
        return values[0].copy()
    
    unit_envelopes = list(unit_envelopes or [])
    if len(unit_envelopes) < len(values):
        unit_envelopes += [None] * (len(values) - len(unit_envelopes))
    
    length = len(values)
    
    # Vectorized version for array t
    if isinstance(t, np.ndarray):
        t = np.asarray(t, dtype=float)
        
        # Apply out-of-bounds behavior
        if out_of_bounds_behavior != OutOfBoundsBehavior.IGNORE:
            t = np.vectorize(lambda x: _apply_out_of_bounds_behavior(x, length, out_of_bounds_behavior))(t)
        
        # Special handling for cyclic hue interpolation
        if cyclic and values[0].size == 1:  # Assuming hue values are scalars
            if direction in ['cw', 'ccw']:
                result = _cyclic_interpolate_directional(values, t, length, direction)
                return result.reshape(t.shape + (1,))
        
        # Regular interpolation
        lower_indices = np.floor(t).astype(int)
        upper_indices = np.minimum(lower_indices + 1, length - 1)
        
        # Create mask for valid interpolation (not at boundaries)
        interpolate_mask = lower_indices != upper_indices
        
        result_shape = t.shape + values[0].shape
        result = np.zeros(result_shape, dtype=float)
        
        # Handle boundary points
        boundary_mask = ~interpolate_mask
        if np.any(boundary_mask):
            result[boundary_mask] = values[lower_indices[boundary_mask][0]]
        
        # Handle interpolation points
        if np.any(interpolate_mask):
            t_interp = t[interpolate_mask]
            lower_idx = lower_indices[interpolate_mask]
            upper_idx = upper_indices[interpolate_mask]
            
            # Get interpolation factors
            interp_factors = t_interp - lower_idx
            
            # Apply segment envelopes if specified
            for i, env in enumerate(unit_envelopes):
                if env is not None and i < length - 1:
                    mask = lower_idx == i
                    if np.any(mask):
                        env_factors = np.vectorize(lambda x: float(env(UnitFloat(x))))(interp_factors[mask])
                        interp_factors[mask] = env_factors
            
            # Perform interpolation
            lower_vals = np.array([values[i] for i in lower_idx])
            upper_vals = np.array([values[i] for i in upper_idx])
            
            interp_result = lower_vals + (upper_vals - lower_vals) * interp_factors[:, None]
            
            # Assign results
            result[interpolate_mask] = interp_result
        
        return result
    
    # Scalar version for single t
    else:
        t_scalar = float(t)
        
        if t_scalar < 0 or t_scalar >= length:
            t_scalar = _apply_out_of_bounds_behavior(t_scalar, length, out_of_bounds_behavior)
        
        lower_index = int(np.floor(t_scalar))
        upper_index = min(lower_index + 1, length - 1)
        
        if lower_index == upper_index:
            return values[lower_index].copy()
        
        interp_factor: UnitFloat = UnitFloat(t_scalar - lower_index)
        
        envelope = unit_envelopes[lower_index]
        if envelope is not None:
            interp_factor = envelope(interp_factor)
        
        lower = values[lower_index]
        upper = values[upper_index]
        
        return lower + (upper - lower) * float(interp_factor)


def _cyclic_interpolate_directional(
    values: Sequence[np.ndarray],
    t: np.ndarray,
    length: int,
    direction: str
) -> np.ndarray:
    """Special cyclic interpolation for directional values (like hue)."""
    # Extract scalar values
    scalars = np.array([v.item() for v in values])
    
    # Adjust values for direction
    adjusted = [scalars[0]]
    for i in range(1, len(scalars)):
        h0 = adjusted[-1]
        h1 = scalars[i]
        
        if direction == 'cw':
            if h1 <= h0:
                h1 += 360.0
        elif direction == 'ccw':
            if h1 >= h0:
                h1 -= 360.0
        else:  # shortest
            delta = h1 - h0
            if delta > 180.0:
                h1 -= 360.0
            elif delta < -180.0:
                h1 += 360.0
        
        adjusted.append(h1)
    
    adjusted = np.array(adjusted)
    
    # Map t to segment space
    t_scaled = (t % 1.0) * length
    
    # Interpolate
    lower_idx = np.floor(t_scaled).astype(int) % length
    upper_idx = (lower_idx + 1) % length
    
    interp_factors = t_scaled - np.floor(t_scaled)
    
    result = adjusted[lower_idx] + (adjusted[upper_idx] - adjusted[lower_idx]) * interp_factors
    
    # Wrap result
    return result % 360.0


def global_envelope_multiple_interp(
    values: Sequence[np.ndarray],
    t: Union[UnitFloat, float, np.ndarray],
    global_envelope: Optional[Callable[[UnitFloat], UnitFloat]] = None,
    unit_envelopes: Optional[Sequence[Optional[Callable[[UnitFloat], UnitFloat]]]] = None,
    out_of_bounds_behaviors: Optional[Sequence[OutOfBoundsBehavior]] = None,
    global_out_of_bounds_behavior: OutOfBoundsBehavior = OutOfBoundsBehavior.CLAMP,
    cyclic: bool = False,
    direction: Optional[str] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Interpolate between multiple values with global envelope and segment-specific behaviors.
    
    Args:
        values: Sequence of value arrays
        t: Interpolation parameter(s) in [0, 1]
        global_envelope: Global envelope function applied to t before interpolation
        unit_envelopes: Segment-specific envelope functions
        out_of_bounds_behaviors: Segment-specific out-of-bounds behaviors
        global_out_of_bounds_behavior: Global out-of-bounds behavior
        cyclic: Whether values are cyclic (for hue)
        direction: Direction for cyclic interpolation
    
    Returns:
        Interpolated value(s)
    """
    if len(values) == 0:
        raise ValueError("values must contain at least one element.")
    
    # Convert t to float/array
    if isinstance(t, UnitFloat):
        t_float = float(t)
    else:
        t_float = t
    
    # Apply global out-of-bounds behavior
    if global_out_of_bounds_behavior != OutOfBoundsBehavior.IGNORE:
        if isinstance(t_float, np.ndarray):
            t_float = np.vectorize(
                lambda x: _apply_out_of_bounds_behavior(x, 1.0, global_out_of_bounds_behavior)
            )(t_float)
        else:
            t_float = _apply_out_of_bounds_behavior(t_float, 1.0, global_out_of_bounds_behavior)
    
    # Apply global envelope
    if global_envelope is not None:
        if isinstance(t_float, np.ndarray):
            t_float = np.vectorize(lambda x: float(global_envelope(UnitFloat(x))))(t_float)
        else:
            t_float = float(global_envelope(UnitFloat(t_float)))
    
    # Scale to value index space
    if isinstance(t_float, np.ndarray):
        t_scaled = t_float * len(values)
    else:
        t_scaled = t_float * len(values)
    
    # Handle out-of-bounds behaviors per segment
    if out_of_bounds_behaviors is None:
        out_of_bounds_behaviors = [OutOfBoundsBehavior.CLAMP] * len(values)
    else:
        out_of_bounds_behaviors = list(out_of_bounds_behaviors)
        if len(out_of_bounds_behaviors) < len(values):
            out_of_bounds_behaviors += [OutOfBoundsBehavior.CLAMP] * (len(values) - len(out_of_bounds_behaviors))
    
    # For cyclic interpolation, we need to handle differently
    if cyclic:
        return multiple_envelope(
            values=values,
            t=t_scaled,
            unit_envelopes=unit_envelopes,
            out_of_bounds_behavior=OutOfBoundsBehavior.IGNORE,  # Handled by cyclic logic
            cyclic=True,
            direction=direction,
        )
    
    # For array t, we need to apply per-segment behaviors
    if isinstance(t_scaled, np.ndarray):
        # This is complex - for simplicity, use default behavior
        # In practice, you might want to implement per-segment handling
        warnings.warn("Per-segment out-of-bounds behaviors not fully implemented for array t")
        return multiple_envelope(
            values=values,
            t=t_scaled,
            unit_envelopes=unit_envelopes,
            out_of_bounds_behavior=global_out_of_bounds_behavior,
            cyclic=False,
            direction=None,
        )
    
    # Scalar version with per-segment behavior
    index = min(int(np.floor(t_scaled)), len(values) - 1)
    segment_behavior = out_of_bounds_behaviors[index]
    
    return multiple_envelope(
        values=values,
        t=t_scaled,
        unit_envelopes=unit_envelopes,
        out_of_bounds_behavior=segment_behavior,
        cyclic=False,
        direction=None,
    )