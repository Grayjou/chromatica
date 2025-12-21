
from __future__ import annotations
from typing import List, Optional
import numpy as np
from ...types.color_types import ColorSpace, is_hue_space
from abc import ABC, abstractmethod
from ...utils.interpolate_hue import interpolate_hue
from ..v2core import multival1d_lerp
from boundednumbers import BoundType
from ...conversions import np_convert