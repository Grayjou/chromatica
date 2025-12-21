# ===================== v2_core.py =====================
"""
Core interpolation functions with Cython-accelerated backends.
"""

import numpy as np
from typing import List, Tuple, Union

from boundednumbers import BoundType, bound_type_to_np_function
from ..types.array_types import ndarray_1d
from .v2core_interp import ( # type: ignore
    lerp_bounded_1d_spatial_fast,
    lerp_bounded_2d_spatial_fast,
    single_channel_multidim_lerp_bounded_cython_fast,
)

# =============================================================================
# Type Aliases
# =============================================================================
BoundTypeSequence = Union[List[BoundType], Tuple[BoundType, ...]]


# =============================================================================
# Bounding Utilities
# =============================================================================
def _prepare_bound_types(
    bound_types: Union[BoundType, BoundTypeSequence],
) -> BoundTypeSequence:
    """Normalize bound_types to a sequence."""
    if isinstance(bound_types, BoundType):
        return [bound_types]
    if isinstance(bound_types, (list, tuple)):
        return bound_types
    raise ValueError("Invalid bound_types argument")


def _bound_stacked(U: np.ndarray, bound_type: BoundType) -> np.ndarray:
    """Apply a single bound type to an array."""
    fn = bound_type_to_np_function[bound_type]
    return fn(U, 0.0, 1.0)


def bound_coeffs(
    coeffs: List[ndarray_1d],
    bound_types: BoundTypeSequence,
) -> List[ndarray_1d]:
    """
    Apply per-dimension bounding to coefficient arrays.
    
    Args:
        coeffs: List of D coefficient arrays, each with shape (L1, ..., Ln)
        bound_types: Sequence of BoundType for each dimension
        
    Returns:
        List of bounded coefficient arrays
    """
    coeffs = [np.asarray(c) for c in coeffs]
    D = len(coeffs)

    if len(bound_types) < D:
        bound_types = list(bound_types) + [BoundType.CLAMP] * (D - len(bound_types))

    # Fast path: all IGNORE
    if all(bt is BoundType.IGNORE for bt in bound_types):
        return coeffs

    U = np.stack(coeffs, axis=-1)  # (L1, ..., Ln, D)
    U_out = np.empty_like(U)

    for bt in set(bound_types):
        idx = [i for i, t in enumerate(bound_types) if t is bt]

        if bt is BoundType.IGNORE:
            U_out[..., idx] = U[..., idx]
        else:
            U_out[..., idx] = _bound_stacked(U[..., idx], bt)

    return [U_out[..., i] for i in range(D)]


def bound_coeffs_fused(
    coeffs: List[np.ndarray],
    bound_types: BoundTypeSequence,
) -> List[np.ndarray]:
    """
    Apply per-dimension bounding with in-place operations where possible.
    """
    coeffs = [np.asarray(c) for c in coeffs]
    D = len(coeffs)

    if len(bound_types) < D:
        bound_types = list(bound_types) + [BoundType.CLAMP] * (D - len(bound_types))

    # Fast path: all IGNORE
    if all(bt is BoundType.IGNORE for bt in bound_types):
        return coeffs

    U = np.stack(coeffs, axis=-1)  # (L, D)

    for i, bt in enumerate(bound_types):
        if bt is BoundType.IGNORE:
            continue

        fn = bound_type_to_np_function[bt]
        try:
            fn(U[..., i], 0.0, 1.0, out=U[..., i])
        except TypeError:
            U[..., i] = fn(U[..., i], 0.0, 1.0)

    return [U[..., i] for i in range(D)]


# =============================================================================
# Multi-dimensional Interpolation (Generic)
# =============================================================================
def single_channel_multidim_lerp(
    starts: np.ndarray,
    ends: np.ndarray,
    coeffs: np.ndarray,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    Vectorized multi-dimensional linear interpolation.
    
    Args:
        starts: Array of start values with shape (2^{N-1},)
        ends: Array of end values with shape (2^{N-1},)
        coeffs: (D_1, D_2, ..., D_k, N) Tensor of coefficient arrays
        bound_type: Bound type to apply to all coefficients
        
    Returns:
        Array of interpolated values with shape (D_1, D_2, ..., D_k)
        
    Note:
        Uses a single bound type for all dimensions. For per-dimension bounds,
        pre-bound with bound_coeffs() and pass BoundType.IGNORE here.
    """
    bounded = _bound_stacked(coeffs, bound_type)
    return single_channel_multidim_lerp_bounded_cython_fast(starts, ends, bounded)


# =============================================================================
# 1D Multi-value Interpolation
# =============================================================================
def multival1d_lerp(
    starts: np.ndarray,
    ends: np.ndarray,
    coeffs: List[np.ndarray],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
    prefer_float64: bool = True,
) -> np.ndarray:
    """
    Multi-channel 1D linear interpolation.
    
    Args:
        starts: Start values, shape (num_channels,)
        ends: End values, shape (num_channels,)
        coeffs: List of num_channels arrays, each shape (num_steps,)
        bound_types: Single BoundType or per-channel sequence
        prefer_float64: If True, use float64 for computation
        
    Returns:
        Interpolated values, shape (num_steps, num_channels)
    """
    # Normalize inputs
    starts = np.asarray(starts, dtype=np.float64).ravel()
    ends = np.asarray(ends, dtype=np.float64).ravel()
    coeffs = [np.asarray(c, dtype=np.float64) for c in coeffs]

    num_channels = len(starts)
    num_steps = coeffs[0].shape[0]

    # Validate
    if len(ends) != num_channels:
        raise ValueError("starts and ends must have same length")
    if len(coeffs) != num_channels:
        raise ValueError("coeffs must have one array per channel")

    # Apply bounds
    if bound_types is BoundType.IGNORE:
        bounded = coeffs
    else:
        bounded = bound_coeffs(coeffs, _prepare_bound_types(bound_types))

    # Output: (num_steps, num_channels)
    out = np.empty((num_steps, num_channels), dtype=np.float64)

    # Process each channel with fast Cython kernel
    for ch in range(num_channels):
        U = bounded[ch][:, np.newaxis]  # shape (num_steps, 1)
        out[:, ch] = lerp_bounded_1d_spatial_fast(
            np.array([starts[ch]], dtype=np.float64),
            np.array([ends[ch]], dtype=np.float64),
            U,
        )

    return out


# =============================================================================
# 2D Multi-value Interpolation
# =============================================================================
def multival2d_lerp(
    starts: np.ndarray,
    ends: np.ndarray,
    coeffs: List[np.ndarray],
    bound_types: Union[BoundType, BoundTypeSequence] = BoundType.CLAMP,
) -> np.ndarray:
    """
    Multi-channel 2D linear interpolation over a spatial grid.
    
    Args:
        starts: Start values, shape (num_channels,)
        ends: End values, shape (num_channels,)
        coeffs: List of num_channels arrays, each shape (H, W)
        bound_types: Single BoundType or per-channel sequence
        
    Returns:
        Interpolated values, shape (H, W, num_channels)
        
    Example:
        >>> starts = np.array([0.0, 100.0, 50.0])  # RGB start
        >>> ends = np.array([255.0, 0.0, 200.0])   # RGB end
        >>> u = np.random.rand(1080, 1920)         # Same coeff for all
        >>> coeffs = [u, u, u]
        >>> result = multival2d_lerp(starts, ends, coeffs)
        >>> result.shape
        (1080, 1920, 3)
    """
    # Normalize inputs
    starts = np.asarray(starts, dtype=np.float64).ravel()
    ends = np.asarray(ends, dtype=np.float64).ravel()
    coeffs = [np.asarray(c, dtype=np.float64) for c in coeffs]

    num_channels = len(starts)
    H, W = coeffs[0].shape[:2]

    # Validate
    if len(ends) != num_channels:
        raise ValueError("starts and ends must have same length")
    if len(coeffs) != num_channels:
        raise ValueError("coeffs must have one array per channel")
    if any(c.shape[:2] != (H, W) for c in coeffs):
        raise ValueError("All coeff arrays must have same spatial shape")

    # Apply bounds
    if bound_types is BoundType.IGNORE:
        bounded = coeffs
    else:
        bounded = bound_coeffs(coeffs, _prepare_bound_types(bound_types))

    # Output: (H, W, num_channels)
    out = np.empty((H, W, num_channels), dtype=np.float64)

    # Process each channel with fast Cython kernel
    for ch in range(num_channels):
        U = bounded[ch][:, :, np.newaxis]  # shape (H, W, 1)
        out[:, :, ch] = lerp_bounded_2d_spatial_fast(
            np.array([starts[ch]], dtype=np.float64),
            np.array([ends[ch]], dtype=np.float64),
            U,
        )

    return out


# =============================================================================
# Convenience: Uniform coefficient interpolation
# =============================================================================
def multival2d_lerp_uniform(
    starts: np.ndarray,
    ends: np.ndarray,
    coeff: np.ndarray,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    2D interpolation with same coefficient for all channels.
    
    Args:
        starts: Start values, shape (num_channels,)
        ends: End values, shape (num_channels,)
        coeff: Single coefficient array, shape (H, W)
        bound_type: Bound type (same for all channels)
        
    Returns:
        Interpolated values, shape (H, W, num_channels)
        
    Example:
        >>> starts = np.array([0.0, 0.0, 0.0])
        >>> ends = np.array([255.0, 128.0, 64.0])
        >>> u = np.random.rand(100, 100)
        >>> result = multival2d_lerp_uniform(starts, ends, u)
    """
    num_channels = len(np.atleast_1d(starts))
    coeffs = [coeff] * num_channels
    return multival2d_lerp(starts, ends, coeffs, bound_types=bound_type)


def multival1d_lerp_uniform(
    starts: np.ndarray,
    ends: np.ndarray,
    coeff: np.ndarray,
    bound_type: BoundType = BoundType.CLAMP,
) -> np.ndarray:
    """
    1D interpolation with same coefficient for all channels.
    
    Args:
        starts: Start values, shape (num_channels,)
        ends: End values, shape (num_channels,)
        coeff: Single coefficient array, shape (num_steps,)
        bound_type: Bound type (same for all channels)
        
    Returns:
        Interpolated values, shape (num_steps, num_channels)
    """
    num_channels = len(np.atleast_1d(starts))
    coeffs = [coeff] * num_channels
    return multival1d_lerp(starts, ends, coeffs, bound_types=bound_type)