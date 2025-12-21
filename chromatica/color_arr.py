"""
Color Array Generation Module
==============================

This module provides color array classes that extend ColorBase with generation
methods for creating 1D and 2D color gradients and patterns.

Features
--------
- Inherits all ColorBase functionality (conversions, arithmetic, alpha support)
- 1D color arrays with repeat and projection methods
- 2D color arrays with repeat and resize methods
- Angular gradients (wrap_around)
- Radial gradients (rotate_around)
- Easing/transform support for custom gradient curves

Classes
-------
Color1DArr: 1D color array with gradient generation methods
Color2DArr: 2D color array with tiling and resizing methods
"""

from numpy import ndarray as NDArray
import numpy as np
from typing import List, Union, Optional, Callable, Tuple
from .colors.color_base import ColorBase
from .types.color_types import ColorValue


class Color1DArr(ColorBase):
    """
    1D Color Array with gradient generation capabilities.
    
    Wraps a ColorBase instance, so it has full support for:
    - Color space conversion
    - Arithmetic operations
    - Alpha channel manipulation
    - Array operations
    
    Additional methods:
    - repeat: Tile the gradient horizontally and vertically
    - wrap_around: Create angular gradients around a point
    - rotate_around: Create radial gradients from a center
    """
    
    def __init__(self, color_base: ColorBase) -> None:
        """
        Initialize a 1D color array.
        
        Args:
            color_base: ColorBase instance with 2D array value (N, channels)
                       where N is the number of color samples.
        """

        if not isinstance(color_base, ColorBase):
            raise TypeError("Color1DArr requires a ColorBase instance")
        
        # Validate that we have a proper 1D array (N, channels)
        if not color_base.is_array:
            raise ValueError("Color1DArr requires an array value, not a scalar")
        
        if color_base.value.ndim != 2:
            raise ValueError(
                f"Color1DArr requires 2D array (N, channels), got shape {color_base.value.shape}"
            )

        self._color = color_base

    
    def __getattr__(self, name):
        """Forward attribute access to wrapped ColorBase."""
        return getattr(self._color, name)
    
    @property
    def channels(self) -> List[NDArray]:
        """Get individual color channels as separate arrays."""
        return [self._color.value[..., i] for i in range(self._color.num_channels)]
    
    def __array__(self) -> NDArray:
        """Enable numpy array interface."""
        return self._color.value
    
    def repeat(self, horizontally: float = 1.0, vertically: int = 1) -> 'Color2DArr':
        """
        Repeat the 1D color array horizontally and vertically to create a 2D array.

        Horizontally can be a float — repeats the array fully `int(horizontally)` times
        and appends a proportional fraction of the array.

        Args:
            horizontally (float): How many times to repeat horizontally.
            vertically (int): How many times to stack vertically.

        Returns:
            Color2DArr: Repeated 2D color array as a new Color2DArr instance.
        """
        if horizontally <= 0:
            raise ValueError("horizontally must be > 0")
        if vertically <= 0:
            raise ValueError("vertically must be > 0")

        n = self._color.value.shape[0]
        full_repeats = int(horizontally)
        partial_fraction = horizontally - full_repeats
        partial_count = int(round(partial_fraction * n))

        # Build one row
        row = np.concatenate(
            [self._color.value] * full_repeats + ([self._color.value[:partial_count]] if partial_count > 0 else [])
        )

        # Stack vertically
        result = np.tile(row, (vertically, 1, 1))
        
        # Return as Color2DArr with same color class
        return Color2DArr(self._color.__class__(result))
    
    def wrap_around(
        self,
        width: int,
        height: int,
        center: Union[Tuple[int, int], List[int]] = (0, 0),
        *,
        angle_start: float = 0.0,
        angle_end: float = 2 * np.pi,
        unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
        outside_fill: Optional[ColorValue] = None,
        radius_offset: float = 0.0,
        base: Optional[NDArray] = None
    ) -> 'Color2DArr':
        """
        Wrap the 1-D gradient around a center point, producing a 2-D
        (height, width, channels) array whose color varies with *angle*.

        Args:
            width, height: Size of the output image.
            center: (cx, cy) – origin in pixel coordinates.
            angle_start: Angle (rad) mapped to the first entry of the
                        1-D gradient.
            angle_end: Angle (rad) mapped to the last entry of the
                      1-D gradient. If angle_end-angle_start == 2π
                      the gradient completes a full circle.
            unit_transform: Optional easing function applied to the
                           normalized angle before color lookup.
            outside_fill: Color value used for pixels whose angle lies
                         outside [angle_start, angle_end). Can be ColorBase,
                         tuple, or scalar.
            radius_offset: Minimum distance from *center* that must be
                          exceeded for a pixel to be considered
                          "inside" the gradient.
            base: Optional image to paint *onto*; must have the
                 same shape and dtype as the output.

        Returns:
            Color2DArr: Angular gradient as 2D color array.
        """
        n_steps, n_channels = self._color.value.shape

        # Output container – start from *base* or zero-array
        if base is not None:
            if base.shape != (height, width, n_channels):
                raise ValueError(
                    f"`base` shape {base.shape} does not match "
                    f"expected {(height, width, n_channels)}")
            out = base.copy()
        else:
            out = np.zeros((height, width, n_channels), dtype=self._color.value.dtype)

        # Fallback color for "outside" pixels
        if outside_fill is not None:
            if isinstance(outside_fill, ColorBase):
                # Convert to same color space and format
                outside_color_obj = outside_fill.convert(self._color.mode, self._color.format_type)
                outside_colour = np.asarray(outside_color_obj.value, dtype=self._color.value.dtype)
            else:
                outside_colour = np.asarray(outside_fill, dtype=self._color.value.dtype)
        else:
            outside_colour = None

        # Compute angle of each pixel relative to center
        yy, xx = np.indices((height, width), dtype=float)
        cx, cy = center
        dx, dy = xx - cx, yy - cy
        distance = np.hypot(dx, dy)

        angle = np.arctan2(dy, dx)  # [-π, π]
        angle = (angle + 2 * np.pi) % (2 * np.pi)  # → [0, 2π)

        # Map angle into the gradient's [0, 1] domain
        span = (angle_end - angle_start) % (2 * np.pi)
        span = 2 * np.pi if span == 0 else span
        unit = ((angle - angle_start) % (2 * np.pi)) / span

        # Mask: pixels whose angle lies inside the active arc and
        # whose radius is beyond `radius_offset`
        inside_mask = (unit <= 1.0) & (distance >= radius_offset)

        # Optional easing / warping
        if unit_transform is not None:
            unit = np.where(inside_mask, unit_transform(unit), unit)

        # Look-up / interpolate color for every pixel inside_mask
        idx_f = unit * (n_steps - 1)
        idx0 = np.floor(idx_f).astype(int).clip(0, n_steps - 1)
        idx1 = np.clip(idx0 + 1, 0, n_steps - 1)
        t = (idx_f - idx0)[..., None]  # shape (..., 1)

        # Linear interpolation between neighboring color steps
        col = (1 - t) * self._color.value[idx0] + t * self._color.value[idx1]

        # Write colors into the output
        out[inside_mask] = col[inside_mask]

        # Pixels not inside_mask
        outside_mask = ~inside_mask
        if outside_colour is not None:
            out[outside_mask] = outside_colour

        return Color2DArr(self._color.__class__(out))
    
    def rotate_around(
        self,
        width: int,
        height: int,
        center: Union[Tuple[int, int], List[int]] = (0, 0),
        *,
        angle_start: float = 0.0,
        angle_end: float = 2 * np.pi,
        unit_transform: Optional[Callable[[NDArray], NDArray]] = None,
        outside_fill: Optional[ColorValue] = None,
        radius_offset: float = 0.0,
        base: Optional[NDArray] = None
    ) -> 'Color2DArr':
        """
        Cast the 1-D gradient radially: color now varies with *radius* from
        `center`. Angles may still be limited via `angle_start/angle_end`.

        Parameters are intentionally identical to `wrap_around` so you can
        swap the two calls with no code changes.

        Returns:
            Color2DArr: Radial gradient as 2D color array.
        """
        n_steps, n_channels = self._color.value.shape

        # Prepare output
        if base is not None:
            if base.shape != (height, width, n_channels):
                raise ValueError(
                    f"`base` shape {base.shape} != required {(height, width, n_channels)}")
            out = base.copy()
        else:
            out = np.zeros((height, width, n_channels), dtype=self._color.value.dtype)

        if outside_fill is not None:
            if isinstance(outside_fill, ColorBase):
                outside_color_obj = outside_fill.convert(self._color.mode, self._color.format_type)
                outside_colour = np.asarray(outside_color_obj.value, dtype=self._color.value.dtype)
            else:
                outside_colour = np.asarray(outside_fill, dtype=self._color.value.dtype)
        else:
            outside_colour = None

        # Geometry
        yy, xx = np.indices((height, width), dtype=float)
        cx, cy = center
        dx, dy = xx - cx, yy - cy
        distance = np.hypot(dx, dy)  # radius of each pixel

        angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)  # 0‥2π
        span = (angle_end - angle_start) % (2 * np.pi)
        span = 2 * np.pi if span == 0 else span
        angle_mask = ((angle - angle_start) % (2 * np.pi)) <= span

        # Normalize radius to [0,1]
        max_radius = np.max(distance) if base is None else np.hypot(
            max(cx, width - cx), max(cy, height - cy))

        unit = (distance - radius_offset) / max(max_radius - radius_offset, 1e-9)
        unit = np.clip(unit, 0.0, 1.0)

        if unit_transform is not None:
            unit = unit_transform(unit)

        # Gradient lookup (linear interp)
        idx_f = unit * (n_steps - 1)
        idx0 = np.floor(idx_f).astype(int)
        idx1 = np.clip(idx0 + 1, 0, n_steps - 1)
        t = (idx_f - idx0)[..., None]  # (...,1)

        col = (1 - t) * self._color.value[idx0] + t * self._color.value[idx1]

        # Compose result
        inside_mask = (distance >= radius_offset) & angle_mask
        out[inside_mask] = col[inside_mask]

        outside_mask = ~inside_mask
        if outside_colour is not None:
            out[outside_mask] = outside_colour

        return Color2DArr(self._color.__class__(out))
    def __add__(self, other):

        if isinstance(other, Color1DArr):

            color_class = self._color.__class__
            other_color = other._color.convert(self._color.mode, self._color.format_type)
            return Color1DArr(
                color_class(np.concatenate([self._color.value, other_color.value], axis=0))
            )
        
    def _radd__(self, other):
        return self.__add__(other)


class Color2DArr:
    """
    2D Color Array with image manipulation capabilities.
    
    Wraps a ColorBase instance, so it has full support for:
    - Color space conversion
    - Arithmetic operations
    - Alpha channel manipulation
    - Array operations
    
    Additional methods:
    - repeat: Tile the image horizontally and vertically
    - resize: Resize the image to new dimensions (requires scikit-image)
    """
    
    def __init__(self, color_base: ColorBase) -> None:
        """
        Initialize a 2D color array.
        
        Args:
            color_base: ColorBase instance with 3D array value (H, W, channels)
                       where H is height and W is width.
        """
        if not isinstance(color_base, ColorBase):
            raise TypeError("Color2DArr requires a ColorBase instance")
        
        # Validate that we have a proper 2D array (H, W, channels)
        if not color_base.is_array:
            raise ValueError("Color2DArr requires an array value, not a scalar")
        
        if color_base.value.ndim != 3:
            raise ValueError(
                f"Color2DArr requires 3D array (H, W, channels), got shape {color_base.value.shape}"
            )
        
        self._color = color_base
    
    def __getattr__(self, name):
        """Forward attribute access to wrapped ColorBase."""
        return getattr(self._color, name)
    
    @property
    def channels(self) -> List[NDArray]:
        """Get individual color channels as separate arrays."""
        return [self._color.value[..., i] for i in range(self._color.num_channels)]
    
    def __array__(self) -> NDArray:
        """Enable numpy array interface."""
        return self._color.value
    
    def repeat(self, horizontally: int = 1, vertically: int = 1) -> 'Color2DArr':
        """
        Repeat the 2D color array horizontally and vertically.

        Both factors must be positive integers.

        Args:
            horizontally (int): How many times to repeat horizontally.
            vertically (int): How many times to repeat vertically.

        Returns:
            Color2DArr: Repeated 2D color array.
        """
        if horizontally <= 0:
            raise ValueError("horizontally must be > 0")
        if vertically <= 0:
            raise ValueError("vertically must be > 0")

        # Repeat each axis
        reps = (vertically, horizontally) + (1,) * (self._color.value.ndim - 2)
        result = np.tile(self._color.value, reps)
        return Color2DArr(self._color.__class__(result))

    def resize(self, new_shape: Tuple[int, int]) -> 'Color2DArr':
        """
        Resize the 2D color array to a new shape.

        Args:
            new_shape (tuple[int, int]): New shape (height, width).

        Returns:
            Color2DArr: Resized color array.
        
        Raises:
            ImportError: If scikit-image is not installed.
        """
        if len(new_shape) != 2:
            raise ValueError("new_shape must be a tuple of (height, width)")
        
        try:
            from skimage.transform import resize
        except ImportError:
            raise ImportError(
                "resize() requires scikit-image. Install with: pip install scikit-image"
            )
        
        resized_colors = resize(
            self._color.value, 
            new_shape + (self._color.num_channels,), 
            anti_aliasing=True, 
            mode='reflect'
        )
        
        # Convert back to original dtype
        if self._color.value.dtype.kind in ('u', 'i'):
            resized_colors = (resized_colors * np.iinfo(self._color.value.dtype).max).astype(self._color.value.dtype)
        
        return Color2DArr(self._color.__class__(resized_colors.astype(self._color.value.dtype)))
