from __future__ import annotations
from typing import Any, ClassVar, Tuple, cast, Self, Callable, Union
from ..conversions import convert, FormatType, np_convert
from ..types.format_type import format_classes, format_valid_dtypes, default_format_dtypes
from ..types.color_types import ColorElement, ColorValue, Scalar, ScalarVector, HUE_SPACES, ColorMode
from ..utils import get_dimension
from abc import ABC
from numpy import ndarray
import numpy as np
class ColorBase:
    __slots__ = ('_value',)  # prevents adding new attributes → immutability

    num_channels: ClassVar[int] = 1
    mode:       ClassVar[ColorMode]
    _type:      ClassVar[type]
    maxima:     ClassVar[ColorElement]
    null_value: ClassVar[ColorElement]
    format_type: ClassVar[FormatType] 
    _is_frozen: bool = False   # class-level default (instance gets its own slot)
    # def color_convert(self: ColorBase, to_space: ColorMode, to_format: FormatType | None = None) -> ColorBase:
    convert: Callable[[ColorBase, ColorMode, FormatType | None], ColorBase]
    with_alpha: Callable[[ColorBase, Union[Scalar, ndarray]], ColorBase]

    def __setattr__(self, name, value):
        """Block attribute changes after __init__ finishes."""
        if getattr(self, '_is_frozen', False):
            raise AttributeError(f"{self.__class__.__name__} is immutable; cannot assign to {name}")
        super().__setattr__(name, value)

    def __init__(self, value: ColorValue) -> None:
        maxima_dim = get_dimension(self.maxima)

        if self.num_channels != maxima_dim:
            raise ValueError(f"{self.mode} expects {self.maxima!r}-shaped maxima")
        
        is_array = isinstance(value, ndarray)
        
        # ---- Handle ColorBase input ----
        if isinstance(value, ColorBase):
            input_is_array = isinstance(value.value, ndarray)
            
            if value.mode == self.mode and value.format_type == self.format_type:
                # Same mode and format, just copy value
                value = value.value
                is_array = input_is_array
            else:
                # Need conversion
                if input_is_array:
                    # Use np_convert for arrays
                    converted = np_convert(
                        color=value.value,
                        from_space=value.mode,
                        to_space=self.mode,
                        input_type=value.format_type.value,
                        output_type=self.format_type.value,
                    )
                    value = converted
                    is_array = True
                else:
                    # Use scalar convert
                    converted = convert(
                        color=value.value,
                        from_space=value.mode,
                        to_space=self.mode,
                        input_type=value.format_type,
                        output_type=self.format_type,
                    )
                    value = converted
                    is_array = False
        
        # ---- Handle array input ----
        if is_array:
            arr = cast(ndarray, value)
            
            # Validate dtype
            valid_types = format_valid_dtypes[self.format_type]
            if not isinstance(arr.dtype.type(0), valid_types):
                raise TypeError(
                    f"{self.mode} with format {self.format_type} expects dtype compatible with {valid_types}, "
                    f"got {arr.dtype}"
                )
            
            # Validate shape: last dimension should match num_channels
            if arr.shape[-1] != self.num_channels:
                raise ValueError(
                    f"{self.mode} expects last dimension to be {self.num_channels}, "
                    f"got shape {arr.shape}"
                )
            
            # Clamp values to maxima
            if isinstance(self.maxima, tuple):
                # Per-channel maxima
                maxima_array = np.array(self.maxima)
                arr = np.clip(arr, 0, maxima_array)
            else:
                # Single maximum for all channels
                arr = np.clip(arr, 0, self.maxima)
            
            # Ensure proper dtype
            target_dtype = default_format_dtypes[self.format_type]
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)
            
            value = arr
        
        # ---- Handle scalar/tuple input ----
        else:
            value_dim = get_dimension(value)
            if maxima_dim != value_dim:
                raise ValueError(f"{self.mode} expects {self.maxima!r}-shaped value")
            
            # type enforcement
            if value_dim == 1:
                value = format_classes[self.format_type](value) 
            else:
                value = tuple(
                    format_classes[self.format_type](v) for v in cast(Tuple[Any, ...], value)
                )
            
            # clamp value
            if isinstance(self.maxima, tuple):
                value = cast(Tuple[Scalar, ...], value)
                value = tuple(
                    max(0, min(v, m)) for v, m in zip(value, cast(Tuple[Scalar, ...], self.maxima))
                )
            elif isinstance(self.maxima, (int, float)):
                value = max(0, min(value, self.maxima))

        # safe assignment; __setattr__ still allows it during init
        self._value = value

        # freeze instance — no more writes allowed
        super().__setattr__('_is_frozen', True)

    # ------------------ READ-ONLY PROPERTIES ------------------
    @property
    def value(self) -> ColorValue:
        return self._value
    
    @property
    def is_array(self) -> bool:
        """Check if this color contains an array of colors."""
        return isinstance(self._value, ndarray)
    
    @property
    def shape(self) -> Tuple[int, ...] | None:
        """Return shape of the array, or None if scalar."""
        if isinstance(self._value, ndarray):
            return self._value.shape
        return None
    @property
    def has_alpha(self) -> bool:
        """Check if this color space includes an alpha channel."""
        return self.mode.endswith('a')

    @property
    def has_hue(self) -> bool:
        """Check if this color space includes a hue channel."""
        return self.mode in HUE_SPACES

class WithAlpha(ABC):
    """
    Mixin for a ColorInt subclass that includes an alpha channel.
    Assumes alpha is the *last* channel.
    
    Note: For array values, alpha operations work on the entire array.
    Use array indexing arr[..., -1] to access alpha channel.
    """

    # Tell static checkers these come from the real subclass (ColorInt)
    num_channels: ClassVar[int]
    maxima: ClassVar[ColorElement]
    mode: ClassVar[ColorMode]
    value: ColorValue  # Can be scalar tuple or ndarray
    is_array: bool

    alpha_index: ClassVar[int] = -1
    alpha_max:   ClassVar[Scalar] 

    @property
    def alpha(self) -> Union[Scalar, ndarray]:
        """
        Get alpha channel value.
        
        Returns:
            Scalar if value is tuple/scalar, ndarray if value is array.
        """
        if isinstance(self.value, ndarray):
            return self.value[..., self.alpha_index]
        else:
            # Scalar/tuple case
            return cast(Tuple[Scalar, ...], self.value)[self.alpha_index]

    @classmethod
    def _validate_alpha_shape(cls, vals: Union[ScalarVector, ndarray]) -> None:
        if isinstance(vals, ndarray):
            if vals.shape[-1] != cls.num_channels:
                raise ValueError(f"{cls.mode} expects last dimension to be {cls.num_channels}")
        else:
            if len(vals) != cls.num_channels:
                raise ValueError(f"{cls.mode} expects {cls.num_channels}-channel tuple")
            if isinstance(cls.maxima, tuple) and len(vals) != len(cls.maxima):
                raise ValueError(f"{cls.mode} expects {cls.maxima!r}-shaped tuple")

    def with_alpha(self, alpha: Union[Scalar, ndarray]) -> Self:
        """
        Return a new instance with modified alpha channel.
        
        Args:
            alpha: New alpha value(s). Can be scalar or array matching shape.
        
        Returns:
            New color instance with updated alpha.
        """
        if isinstance(self.value, ndarray):
            # Array case
            if isinstance(alpha, ndarray):
                # Validate shape matches (except last dimension)
                if alpha.shape != self.value.shape[:-1]:
                    raise ValueError(
                        f"Alpha shape {alpha.shape} doesn't match color shape {self.value.shape[:-1]}"
                    )
                a = np.clip(alpha, 0, self.alpha_max)
            else:
                # Broadcast scalar alpha to all elements
                a = np.clip(alpha, 0, self.alpha_max)
            
            # Replace last channel with new alpha
            new_vals = np.concatenate([
                self.value[..., :-1],
                np.expand_dims(a, axis=-1)
            ], axis=-1)
            
        else:
            # Scalar/tuple case
            if isinstance(alpha, ndarray):
                raise TypeError("Cannot use array alpha with scalar color value")
            
            a = max(0, min(alpha, self.alpha_max))
            values = cast(Tuple[Scalar, ...], self.value)
            new_vals = values[:-1] + (a,)
        
        self._validate_alpha_shape(new_vals)
        return self.__class__(new_vals)  # type: ignore
    

def build_registry(*classes: type[ColorBase]):
    return {
        (cls.mode, cls.format_type): cls
        for cls in classes
    }

