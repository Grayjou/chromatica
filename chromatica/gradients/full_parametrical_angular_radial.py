from typing import Optional, Tuple, Callable, List, Dict
import warnings
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from .angular_radial_helpers import (
    CoordinateGridCache,
    compute_center,
    interpolate_hue,
    normalize_angle,
    process_outside_fill,
)
from .gradient import Gradient2D
from ..colors.color import unified_tuple_to_class
from ..normalizers.color_normalizer import normalize_color_input, ColorInput
from ..format_type import FormatType


GradientEnds = Tuple[ColorInput, ...]


@lru_cache(maxsize=256)
def cached_normalize_color(color_tuple: Tuple) -> Tuple:
    """Cache normalized color conversions for performance."""
    return tuple(normalize_color_input(color_tuple))


GRID_CACHE = CoordinateGridCache()


def build_angular_mask(theta: NDArray, deg_start: float, deg_end: float, normalize_theta: bool) -> NDArray:
    """Create an angular mask honoring the configured angular range."""
    angular_mask = np.ones_like(theta, dtype=bool)
    if not normalize_theta:
        return angular_mask

    if deg_end >= deg_start:
        angular_range = deg_end - deg_start
        if angular_range >= 360.0:
            return angular_mask
        theta_start = deg_start % 360.0
        theta_end = deg_end % 360.0
        if theta_end >= theta_start:
            return (theta >= theta_start) & (theta <= theta_end)
        return (theta >= theta_start) | (theta <= theta_end)

    theta_start = deg_start % 360.0
    theta_end = deg_end % 360.0
    return (theta >= theta_start) | (theta <= theta_end)


def normalize_theta_range(
    theta: NDArray, deg_start: float, deg_end: float, angular_mask: NDArray, normalize_theta: bool
) -> Tuple[NDArray, float]:
    """Normalize theta values into [0, 1] respecting the configured range."""
    if not normalize_theta:
        return theta / 360.0, 360.0

    if deg_end >= deg_start:
        theta_range = deg_end - deg_start
        if theta_range <= 0:
            theta_range = 360.0
    else:
        theta_range = (360.0 - normalize_angle(deg_start)) + normalize_angle(deg_end)

    theta_adjusted = (theta - deg_start + 360.0) % 360.0

    if deg_end < deg_start:
        return (theta_adjusted / theta_range) % 1.0, theta_range

    return np.where(angular_mask, np.clip(theta_adjusted / theta_range, 0.0, 1.0), 0), theta_range


def normalize_radial_distances(
    distances: NDArray, inner_radius: NDArray, outer_radius: NDArray, normalize_radius: bool
) -> Tuple[NDArray, NDArray]:
    """Normalize radial distances and provide the in-bounds mask."""
    denominator = outer_radius - inner_radius
    denominator = np.where(np.abs(denominator) < 1e-3, 1e-3, denominator)
    u_r = (distances - inner_radius) / denominator
    if not normalize_radius:
        return np.clip(u_r, 0.0, 1.0), (distances >= inner_radius) & (distances <= outer_radius)

    return np.clip(u_r, 0.0, 1.0), (distances >= inner_radius) & (distances <= outer_radius)


class FullParametricalAngularRadialGradient(Gradient2D):
    """
    Full parametric angular-radial gradient generator with maximum mathematical control.
    
    This is the most advanced gradient class, providing:
    - Arbitrary number of color rings with arbitrary stops per ring
    - Parametric radius functions r(θ) for both inner and outer boundaries
    - Per-ring hue direction control in both angular and radial dimensions
    - Bivariable space transforms: (r, θ) → (r', θ') before color mapping
    - Bivariable color transforms: (r, θ, color) → color' after color mapping
    - Full control over angular ranges and masking
    
    Example Use Cases:
    - Complex multi-ring gradients with different angular patterns per ring
    - Non-circular gradients (ellipses, spirals, flowers, stars)
    - Channel-dependent spatial transformations
    - Post-processing color effects based on position
    """
    
    @staticmethod
    def _validate_inputs(
        width: int,
        height: int,
        inner_r_theta: Callable,
        outer_r_theta: Callable,
        color_rings: List[GradientEnds]
    ) -> None:
        """Validate input parameters."""
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        if not callable(inner_r_theta) or not callable(outer_r_theta):
            raise TypeError("inner_r_theta and outer_r_theta must be callable")

        if not color_rings or len(color_rings) == 0:
            raise ValueError("color_rings must contain at least one ring")

        for i, ring in enumerate(color_rings):
            if not ring or len(ring) == 0:
                raise ValueError(f"color_rings[{i}] must contain at least one color")

    @staticmethod
    def _validate_radius_functions(
        inner_r_theta: Callable,
        outer_r_theta: Callable
    ) -> None:
        """Validate radius functions work correctly."""
        try:
            test_theta = np.array([0.0, 90.0, 180.0, 270.0])
            inner_test = inner_r_theta(test_theta)
            outer_test = outer_r_theta(test_theta)

            if not isinstance(inner_test, (np.ndarray, int, float)):
                raise ValueError("inner_r_theta must return a number or array")
            if not isinstance(outer_test, (np.ndarray, int, float)):
                raise ValueError("outer_r_theta must return a number or array")

            inner_test = np.atleast_1d(inner_test)
            outer_test = np.atleast_1d(outer_test)

            if np.any(inner_test < 0) or np.any(outer_test < 0):
                raise ValueError("Radius functions must return non-negative values")

            if np.any(outer_test < inner_test):
                warnings.warn(
                    "outer_r_theta returns values less than inner_r_theta at some angles. "
                    "This may cause unexpected behavior."
                )
        except Exception as e:
            if isinstance(e, (ValueError, TypeError)):
                raise
            warnings.warn(f"Could not validate radius functions: {e}")

    @staticmethod
    def _validate_hue_directions(
        hue_directions_theta: Optional[List[List[Optional[str]]]],
        hue_directions_r: Optional[List[Optional[str]]],
        num_rings: int
    ) -> None:
        """Validate hue direction specifications."""
        if hue_directions_theta is not None:
            if len(hue_directions_theta) != num_rings:
                raise ValueError(f"hue_directions_theta must have {num_rings} entries (one per ring)")
        
        if hue_directions_r is not None:
            if len(hue_directions_r) != num_rings - 1:
                raise ValueError(f"hue_directions_r must have {num_rings - 1} entries (between rings)")

    @staticmethod
    def _normalize_color_rings(color_rings: List[GradientEnds]) -> List[List]:
        """Normalize all colors in rings with caching for tuples."""
        normalized_rings = []
        for ring in color_rings:
            normalized_ring = []
            for color in ring:
                if isinstance(color, tuple):
                    try:
                        normalized_ring.append(cached_normalize_color(color))
                    except TypeError:
                        # Unhashable type, use regular normalization
                        normalized_ring.append(normalize_color_input(color))
                else:
                    normalized_ring.append(normalize_color_input(color))
            normalized_rings.append(normalized_ring)
        return normalized_rings

    @staticmethod
    def _compute_transformed_coordinates(
        distances: NDArray,
        theta: NDArray,
        num_channels: int,
        easing_theta: Optional[Dict[int, Callable]],
        bivariable_space_transforms: Optional[Dict[int, Callable]]
    ) -> Dict[int, Tuple[NDArray, NDArray]]:
        """Compute per-channel coordinate transformations."""
        has_transforms = bool(easing_theta or bivariable_space_transforms)
        
        if has_transforms:
            transformed_coords = {}
            for ch in range(num_channels):
                theta_ch = theta
                if easing_theta and ch in easing_theta:
                    theta_ch = easing_theta[ch](theta)
                
                r_ch = distances
                if bivariable_space_transforms and ch in bivariable_space_transforms:
                    r_ch, theta_ch = bivariable_space_transforms[ch](distances, theta_ch)
                
                transformed_coords[ch] = (r_ch, theta_ch)
        else:
            # Reuse the same coordinates for all channels (memory efficient)
            transformed_coords = {ch: (distances, theta) for ch in range(num_channels)}
        
        return transformed_coords

    @staticmethod
    def _compute_radius_arrays(
        theta: NDArray,
        inner_r_theta: Callable,
        outer_r_theta: Callable,
        distances: NDArray,
        deg_start: float,
        rotate_r_theta_with_theta_normalization: bool
    ) -> Tuple[NDArray, NDArray]:
        """Compute inner and outer radius arrays."""
        # Handle rotation of r_theta functions if requested
        if rotate_r_theta_with_theta_normalization:
            theta_for_r_theta = (theta - deg_start + 360.0) % 360.0
        else:
            theta_for_r_theta = theta
        
        # Calculate inner and outer radius for each pixel
        inner_radius = np.atleast_1d(inner_r_theta(theta_for_r_theta))
        outer_radius = np.atleast_1d(outer_r_theta(theta_for_r_theta))
        
        # Ensure arrays are properly shaped
        if inner_radius.size == 1:
            inner_radius = np.full_like(distances, inner_radius.item())
        if outer_radius.size == 1:
            outer_radius = np.full_like(distances, outer_radius.item())
        
        # Edge case handling: ensure inner <= outer
        if np.any(inner_radius > outer_radius):
            warnings.warn(
                "inner_radius > outer_radius at some angles. Clipping inner_radius to match outer_radius."
            )
            inner_radius = np.minimum(inner_radius, outer_radius)
        
        return inner_radius, outer_radius

    @staticmethod
    def _apply_radial_easing(
        u_r: NDArray,
        num_channels: int,
        easing_r: Optional[Dict[int, Callable]]
    ) -> Dict[int, NDArray]:
        """Apply radial easing per channel."""
        u_r_eased = {}
        for ch in range(num_channels):
            if easing_r and ch in easing_r:
                u_r_eased[ch] = easing_r[ch](u_r)
            else:
                u_r_eased[ch] = u_r
        return u_r_eased

    @staticmethod
    def _initialize_result_array(
        width: int,
        height: int,
        num_channels: int,
        outside_fill_processed: np.ndarray
    ) -> np.ndarray:
        """Initialize result array with outside fill."""
        if isinstance(outside_fill_processed, np.ndarray):
            if outside_fill_processed.ndim == 1:
                base = np.tile(outside_fill_processed, (height, width, 1))
            else:
                base = outside_fill_processed.copy()
        else:
            base = np.tile(np.array(outside_fill_processed), (height, width, 1))
        
        return base.astype(np.float32, copy=False)

    @staticmethod
    def _compute_normalized_theta_for_channel(
        theta_ch: NDArray,
        deg_start: float,
        deg_end: float,
        theta_range: float,
        angular_mask: NDArray,
        normalize_theta: bool
    ) -> NDArray:
        """Compute normalized theta for a specific channel."""
        if normalize_theta:
            # Adjust transformed theta relative to deg_start
            theta_ch_adjusted = (theta_ch - deg_start + 360.0) % 360.0
            
            # Handle wrap-around case
            if deg_end < deg_start:
                # Wrap-around case
                theta_ch_normalized = theta_ch_adjusted / theta_range
                # For wrap-around, values > 1 should wrap back to [0, 1]
                theta_ch_normalized = theta_ch_normalized % 1.0
            else:
                # Normal case - clip to [0, 1] within the angular range
                theta_ch_normalized = np.where(
                    angular_mask, 
                    np.clip(theta_ch_adjusted / theta_range, 0.0, 1.0), 
                    0
                )
        else:
            theta_ch_normalized = theta_ch / 360.0
        
        return theta_ch_normalized

    @staticmethod
    def _interpolate_colors_angularly(
        ring: List,
        theta_ch_normalized: NDArray,
        ch: int,
        is_hue_space: bool,
        ring_idx: int,
        hue_directions_theta: Optional[List[List[Optional[str]]]]
    ) -> NDArray:
        """Interpolate colors within a ring angularly."""
        num_colors = len(ring)
        
        if num_colors == 1:
            # Single color ring
            return np.full_like(theta_ch_normalized, ring[0][ch])
        
        # Interpolate colors angularly
        segment_u = np.clip(theta_ch_normalized * (num_colors - 1), 0, num_colors - 1 - 1e-7)
        segment_idx = np.floor(segment_u).astype(int)
        segment_idx = np.clip(segment_idx, 0, num_colors - 2)
        local_u = segment_u - segment_idx
        
        # Get colors for each segment
        color_start = np.array([ring[i][ch] for i in range(num_colors)])
        color_end = np.array([ring[i][ch] for i in range(1, num_colors)] + [ring[0][ch]])
        
        # Use flat indexing to avoid creating extra dimensions
        flat_idx = segment_idx.ravel()
        c0 = color_start[flat_idx].reshape(segment_idx.shape)
        c1 = color_end[flat_idx].reshape(segment_idx.shape)
        
        # Apply hue interpolation if needed
        if ch == 0 and is_hue_space and hue_directions_theta:
            ring_hue_dirs = hue_directions_theta[ring_idx]
            
            # Apply hue interpolation per segment
            ring_color = np.zeros_like(theta_ch_normalized)
            for seg in range(num_colors - 1):
                mask_seg = segment_idx == seg
                if np.any(mask_seg):
                    direction = ring_hue_dirs[seg] if seg < len(ring_hue_dirs) else None
                    ring_color[mask_seg] = interpolate_hue(
                        np.full(np.sum(mask_seg), color_start[seg]),
                        np.full(np.sum(mask_seg), color_start[seg + 1]),
                        local_u[mask_seg],
                        direction
                    )
        else:
            # Linear interpolation
            ring_color = c0 * (1 - local_u) + c1 * local_u
        
        return ring_color

    @staticmethod
    def _build_angular_colors_for_rings(
        normalized_rings: List[List],
        theta_ch_normalized: NDArray,
        ch: int,
        is_hue_space: bool,
        hue_directions_theta: Optional[List[List[Optional[str]]]]
    ) -> List[NDArray]:
        """Build angular colors for each ring."""
        ring_colors = []
        for ring_idx, ring in enumerate(normalized_rings):
            ring_color = FullParametricalAngularRadialGradient._interpolate_colors_angularly(
                ring, theta_ch_normalized, ch, is_hue_space, ring_idx, hue_directions_theta
            )
            ring_colors.append(ring_color)
        return ring_colors

    @staticmethod
    def _interpolate_rings_radially(
        ring_colors: List[NDArray],
        u_r_ch_eased: NDArray,
        ch: int,
        is_hue_space: bool,
        hue_directions_r: Optional[List[Optional[str]]]
    ) -> NDArray:
        """Interpolate between rings radially."""
        num_rings = len(ring_colors)
        
        if num_rings == 1:
            return ring_colors[0]
        
        # Determine which ring pair to interpolate between
        ring_u = u_r_ch_eased * (num_rings - 1)
        ring_idx = np.floor(ring_u).astype(int)
        ring_idx = np.clip(ring_idx, 0, num_rings - 2)
        local_u_r = ring_u - ring_idx
        
        # Interpolate between ring pairs
        if ch == 0 and is_hue_space and hue_directions_r:
            # Apply hue interpolation for radial dimension
            channel_result = np.zeros_like(u_r_ch_eased)
            for ring_i in range(num_rings - 1):
                mask_ring = ring_idx == ring_i
                if np.any(mask_ring):
                    direction = hue_directions_r[ring_i]
                    channel_result[mask_ring] = interpolate_hue(
                        ring_colors[ring_i][mask_ring],
                        ring_colors[ring_i + 1][mask_ring],
                        local_u_r[mask_ring],
                        direction
                    )
        else:
            # Linear interpolation between rings
            channel_result = np.zeros_like(u_r_ch_eased)
            for ring_i in range(num_rings - 1):
                mask_ring = ring_idx == ring_i
                if np.any(mask_ring):
                    channel_result[mask_ring] = (
                        ring_colors[ring_i][mask_ring] * (1 - local_u_r[mask_ring]) +
                        ring_colors[ring_i + 1][mask_ring] * local_u_r[mask_ring]
                    )
        
        return channel_result

    @staticmethod
    def _process_single_channel(
        ch: int,
        transformed_coords: Dict[int, Tuple[NDArray, NDArray]],
        inner_r_theta: Callable,
        outer_r_theta: Callable,
        easing_r: Optional[Dict[int, Callable]],
        deg_start: float,
        deg_end: float,
        theta_range: float,
        angular_mask: NDArray,
        normalize_theta: bool,
        normalized_rings: List[List],
        is_hue_space: bool,
        hue_directions_theta: Optional[List[List[Optional[str]]]],
        hue_directions_r: Optional[List[Optional[str]]],
        bivariable_color_transforms: Optional[Dict[int, Callable]]
    ) -> NDArray:
        """Process a single color channel."""
        # Get transformed coordinates for this channel
        r_ch, theta_ch = transformed_coords[ch]
        
        # Recalculate inner/outer radius for transformed theta
        inner_radius_ch = np.atleast_1d(inner_r_theta(theta_ch))
        outer_radius_ch = np.atleast_1d(outer_r_theta(theta_ch))
        
        if inner_radius_ch.size == 1:
            inner_radius_ch = np.full_like(r_ch, inner_radius_ch.item())
        if outer_radius_ch.size == 1:
            outer_radius_ch = np.full_like(r_ch, outer_radius_ch.item())
        
        inner_radius_ch = np.minimum(inner_radius_ch, outer_radius_ch)
        denominator_ch = outer_radius_ch - inner_radius_ch
        denominator_ch = np.where(np.abs(denominator_ch) < 1e-3, 1e-3, denominator_ch)
        u_r_ch = (r_ch - inner_radius_ch) / denominator_ch
        
        # Apply radial easing for this channel
        if easing_r and ch in easing_r:
            u_r_ch_eased = easing_r[ch](np.clip(u_r_ch, 0.0, 1.0))
        else:
            u_r_ch_eased = np.clip(u_r_ch, 0.0, 1.0)
        
        # Compute normalized theta for this channel
        theta_ch_normalized = FullParametricalAngularRadialGradient._compute_normalized_theta_for_channel(
            theta_ch, deg_start, deg_end, theta_range, angular_mask, normalize_theta
        )
        
        # Build angular colors for each ring
        ring_colors = FullParametricalAngularRadialGradient._build_angular_colors_for_rings(
            normalized_rings, theta_ch_normalized, ch, is_hue_space, hue_directions_theta
        )
        
        # Interpolate between rings radially
        channel_result = FullParametricalAngularRadialGradient._interpolate_rings_radially(
            ring_colors, u_r_ch_eased, ch, is_hue_space, hue_directions_r
        )
        
        # Apply bivariable color transform if specified
        if bivariable_color_transforms and ch in bivariable_color_transforms:
            channel_result = bivariable_color_transforms[ch](r_ch, theta_ch, channel_result)
        
        return channel_result

    @staticmethod
    def _apply_outside_fill(
        result: np.ndarray,
        combined_mask: NDArray,
        outside_fill_processed: np.ndarray,
        num_channels: int
    ) -> None:
        """Apply outside fill to masked areas."""
        outside_areas = ~combined_mask
        if isinstance(outside_fill_processed, np.ndarray):
            if outside_fill_processed.ndim == 1:
                # Single color tuple - apply to all outside areas
                for ch in range(num_channels):
                    result[outside_areas, ch] = outside_fill_processed[ch]
            else:
                # Full image array - apply to masked areas
                result[outside_areas] = outside_fill_processed[outside_areas]

    @classmethod
    def generate(
        cls,
        width: int,
        height: int,
        inner_r_theta: Callable[[NDArray], NDArray],
        outer_r_theta: Callable[[NDArray], NDArray],
        color_rings: List[GradientEnds],
        color_space: str = 'rgb',
        format_type: FormatType = FormatType.FLOAT,
        center: Optional[Tuple[int, int]] = None,
        relative_center: Optional[Tuple[float, float]] = None,
        deg_start: float = 0.0,
        deg_end: float = 360.0,
        normalize_radius: bool = True,
        normalize_theta: bool = True,
        hue_directions_theta: Optional[List[List[Optional[str]]]] = None,
        hue_directions_r: Optional[List[Optional[str]]] = None,
        outside_fill: Optional[ColorInput] = None,
        bivariable_space_transforms: Optional[Dict[int, Callable[[NDArray, NDArray], Tuple[NDArray, NDArray]]]] = None,
        bivariable_color_transforms: Optional[Dict[int, Callable[[NDArray, NDArray, NDArray], NDArray]]] = None,
        easing_theta: Optional[Dict[int, Callable[[NDArray], NDArray]]] = None,
        easing_r: Optional[Dict[int, Callable[[NDArray], NDArray]]] = None,
        rotate_r_theta_with_theta_normalization: bool = False
    ) -> 'Gradient2D':
        """
        Generate a fully parametric angular-radial gradient.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            inner_r_theta: Function mapping angle (degrees) to inner radius
            outer_r_theta: Function mapping angle (degrees) to outer radius
            color_rings: List of color tuples, one per radial ring
            color_space: Color space ('rgb', 'hsv', 'hsl', 'rgba', etc.)
            format_type: Format type (INT or FLOAT)
            center: Absolute (x, y) center position
            relative_center: Relative (x, y) center as fractions of width/height
            deg_start: Starting angle in degrees
            deg_end: Ending angle in degrees
            normalize_radius: If True, normalize radial distances to [0, 1]
            normalize_theta: If True, constrain gradient to angular range
            hue_directions_theta: Hue interpolation direction per ring (angular)
            hue_directions_r: Hue interpolation direction between rings (radial)
            outside_fill: Color to fill areas outside the gradient
            bivariable_space_transforms: Channel-specific (r, θ) → (r', θ') transforms
            bivariable_color_transforms: Channel-specific (r, θ, color) → color' transforms
            easing_theta: Channel-specific angular easing functions
            easing_r: Channel-specific radial easing functions
            rotate_r_theta_with_theta_normalization: Apply deg_start offset to pattern
        
        Returns:
            Gradient2D instance containing the generated gradient
        """
        # Validate inputs
        cls._validate_inputs(width, height, inner_r_theta, outer_r_theta, color_rings)
        cls._validate_radius_functions(inner_r_theta, outer_r_theta)
        
        # Setup
        center = compute_center(width, height, center, relative_center)
        respective_class = unified_tuple_to_class[(color_space, format_type)]
        num_channels = respective_class.num_channels
        is_hue_space = color_space.lower() in ('hsv', 'hsl', 'hsva', 'hsla')
        
        # Process colors
        num_rings = len(color_rings)
        cls._validate_hue_directions(hue_directions_theta, hue_directions_r, num_rings)
        normalized_rings = cls._normalize_color_rings(color_rings)
        outside_fill_processed = process_outside_fill(outside_fill, width, height, format_type, color_space)
        
        # Compute coordinates
        distances, theta = GRID_CACHE.get_grid(width, height, center)
        transformed_coords = cls._compute_transformed_coordinates(
            distances, theta, num_channels, easing_theta, bivariable_space_transforms
        )
        
        # Compute angular masks and normalization
        angular_mask = build_angular_mask(theta, deg_start, deg_end, normalize_theta)
        theta_normalized, theta_range = normalize_theta_range(
            theta, deg_start, deg_end, angular_mask, normalize_theta
        )
        
        # Compute radial boundaries
        inner_radius, outer_radius = cls._compute_radius_arrays(
            theta, inner_r_theta, outer_r_theta, distances, deg_start,
            rotate_r_theta_with_theta_normalization
        )
        u_r, radial_mask = normalize_radial_distances(distances, inner_radius, outer_radius, normalize_radius)
        u_r_eased = cls._apply_radial_easing(u_r, num_channels, easing_r)
        
        combined_mask = angular_mask & radial_mask if normalize_theta else radial_mask
        
        # Initialize result array
        result = cls._initialize_result_array(width, height, num_channels, outside_fill_processed)
        
        # Process each channel
        for ch in range(num_channels):
            channel_result = cls._process_single_channel(
                ch, transformed_coords, inner_r_theta, outer_r_theta, easing_r,
                deg_start, deg_end, theta_range, angular_mask, normalize_theta,
                normalized_rings, is_hue_space, hue_directions_theta, hue_directions_r,
                bivariable_color_transforms
            )
            result[..., ch] = np.where(combined_mask, channel_result, 0)
        
        # Apply outside fill
        cls._apply_outside_fill(result, combined_mask, outside_fill_processed, num_channels)
        
        # Convert to appropriate format
        if format_type == FormatType.INT:
            result = np.round(result).astype(np.uint16)
        
        # Create ColorBase instance and wrap in Gradient2D
        assigned_class = unified_tuple_to_class[(color_space, format_type)]
        result_color = assigned_class(result)
        gradient_obj = cls(result_color)
        
        return gradient_obj
    
    @classmethod
    def create_elliptical(
        cls,
        width: int,
        height: int,
        colors: GradientEnds,
        eccentricity: float = 0.5,
        **kwargs
    ) -> 'Gradient2D':
        """
        Create an elliptical gradient (simplified interface).
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            colors: Tuple of colors to interpolate
            eccentricity: Ellipse eccentricity (0 = circle, 1 = line)
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            Gradient2D instance with elliptical gradient
        """
        a = min(width, height) / 2
        b = a * (1 - np.clip(eccentricity, 0, 0.99))
        
        def ellipse_r(theta):
            theta_rad = np.radians(theta)
            return a * b / np.sqrt(
                (b * np.cos(theta_rad))**2 + 
                (a * np.sin(theta_rad))**2
            )
        
        return cls.generate(
            width=width,
            height=height,
            inner_r_theta=lambda theta: np.zeros_like(theta),
            outer_r_theta=ellipse_r,
            color_rings=[colors],
            **kwargs
        )
    
    @classmethod
    def create_star(
        cls,
        width: int,
        height: int,
        colors: GradientEnds,
        points: int = 5,
        inner_ratio: float = 0.4,
        **kwargs
    ) -> 'Gradient2D':
        """
        Create a star-shaped gradient (simplified interface).
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            colors: Tuple of colors to interpolate
            points: Number of star points
            inner_ratio: Ratio of inner to outer radius
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            Gradient2D instance with star-shaped gradient
        """
        outer_radius = min(width, height) / 2
        inner_radius = outer_radius * inner_ratio
        
        def star_r(theta):
            # Create star shape by alternating between inner and outer radius
            angle_per_point = 360.0 / (points * 2)
            normalized_angle = (theta % (2 * angle_per_point)) / (2 * angle_per_point)
            # Smooth transition using sine
            t = (np.sin((normalized_angle - 0.5) * np.pi) + 1) / 2
            return inner_radius + (outer_radius - inner_radius) * t
        
        return cls.generate(
            width=width,
            height=height,
            inner_r_theta=lambda theta: np.zeros_like(theta),
            outer_r_theta=star_r,
            color_rings=[colors],
            **kwargs
        )
