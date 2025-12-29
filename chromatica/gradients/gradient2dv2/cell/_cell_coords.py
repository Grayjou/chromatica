# chromatica/gradients/gradient2dv2/cell/_cell_coords.py
"""Canonical handling of per-channel coordinate shapes."""

from __future__ import annotations
from typing import List, Union
import numpy as np

PerChannelCoords = Union[np.ndarray, List[np.ndarray]]


def get_shape(coords: PerChannelCoords) -> tuple[int, int]:
    """Return (height, width). Works with list or ndarray, 3D or 4D."""
    if isinstance(coords, list):
        return coords[0].shape[:2]
    return coords.shape[:2]


def extract_edge(coords: PerChannelCoords, row: int) -> List[np.ndarray]:
    """
    Extract a single row, always returning list of arrays [1, W, 2].
    Handles all input shapes:
    - List of [H,W,2] -> list of [1,W,2]
    - Array [H,W,2] -> [ [1,W,2] ]
    - Array [H,W,2,C] -> C entries of [1,W,2]
    """
    if isinstance(coords, list):
        return [pc[row:row+1, :, :] for pc in coords]
    
    if coords.ndim == 3:  # [H,W,2]
        return [coords[row:row+1, :, :]]
    
    if coords.ndim == 4:  # [H,W,2,C]
        return [coords[row:row+1, :, :, c] for c in range(coords.shape[3])]
    
    raise ValueError(f"Unexpected shape: {coords.shape}")


def extract_point(edge_list: List[np.ndarray], x_idx: int) -> List[np.ndarray]:
    """Extract coords at specific x position: [1,1,2] per channel."""
    return [pc[:, x_idx:x_idx+1, :] for pc in edge_list]


def lerp_point(edge_list: List[np.ndarray], exact_x: float) -> List[np.ndarray]:
    """Interpolate coords between pixels: [1,1,2] per channel."""
    w = edge_list[0].shape[1]
    exact_x = float(np.clip(exact_x, 0.0, w - 1.0))
    lo = int(np.floor(exact_x))
    hi = int(np.ceil(exact_x))
    
    if hi == lo:
        return extract_point(edge_list, lo)
    
    t = (exact_x - lo) / (hi - lo)
    return [edge_list[i][:, lo:lo+1, :] + t * (edge_list[i][:, hi:hi+1, :] - edge_list[i][:, lo:lo+1, :]) 
            for i in range(len(edge_list))]


def slice_and_renormalize(
    coords: PerChannelCoords,
    start_idx: int,
    end_idx: int,
    start_u: float,
    end_u: float
) -> PerChannelCoords:
    """
    Slice coords in x-direction and renormalize to [0,1].
    Preserves input type (list vs array).
    """
    interval = end_u - start_u
    if interval <= 0:
        raise ValueError(f"Invalid interval: [{start_u}, {end_u}]")
    
    if isinstance(coords, list):
        sliced = [pc[:, start_idx:end_idx, :].copy() for pc in coords]
        for pc in sliced:
            pc[..., 0] = (pc[..., 0] - start_u) / interval
        return sliced
    
    sliced = coords[:, start_idx:end_idx, ...].copy()
    sliced[..., 0] = (sliced[..., 0] - start_u) / interval
    return sliced



def slice_coords(
    coords: PerChannelCoords,
    start_idx: int,
    end_idx: int,
) -> PerChannelCoords:
    """
    Slice coords in x-direction and renormalize to [0,1].
    Preserves input type (list vs array).
    """
    interval = end_idx - start_idx
    if interval <= 0:
        raise ValueError(f"Invalid interval: [{start_idx}, {end_idx}]")
    
    if isinstance(coords, list):
        sliced = [pc[:, start_idx:end_idx, :].copy() for pc in coords]
        return sliced
    
    sliced = coords[:, start_idx:end_idx, ...].copy()
    return sliced