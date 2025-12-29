# chromatica/gradients/gradient2dv2/cell/_cell_coords.py
"""Canonical handling of per-channel coordinate shapes."""

from __future__ import annotations
from typing import List, Union
import numpy as np

PerChannelCoords = Union[np.ndarray, List[np.ndarray]]


def get_shape(coords: PerChannelCoords) -> tuple[int, int]:
    """
    Return (height, width) from per-channel coordinates.
    
    Works with list or ndarray, 3D or 4D coordinate arrays.
    
    Args:
        coords: Per-channel coordinate arrays. Can be:
            - List[np.ndarray]: List of coordinate arrays, one per channel, each shape (H, W, 2)
            - np.ndarray: Single array shape (H, W, 2) or (H, W, 2, C)
    
    Returns:
        Tuple of (height, width)
    """
    if isinstance(coords, list):
        return coords[0].shape[:2]
    return coords.shape[:2]


def extract_edge(coords: PerChannelCoords, row: int) -> List[np.ndarray]:
    """
    Extract a single row from per-channel coordinates.
    
    Always returns a list of arrays with shape [1, W, 2], one per channel.
    
    Args:
        coords: Per-channel coordinate arrays. Can be:
            - List of [H, W, 2]: Returns list of [1, W, 2], one per input array
            - Array [H, W, 2]: Returns single-element list with [1, W, 2]
            - Array [H, W, 2, C]: Returns C arrays of [1, W, 2]
        row: Row index to extract (0 to H-1)
    
    Returns:
        List of coordinate arrays, one per channel, each shape [1, W, 2]
        
    Raises:
        ValueError: If coords has unexpected shape
    """
    if isinstance(coords, list):
        return [pc[row:row+1, :, :] for pc in coords]
    
    if coords.ndim == 3:  # [H,W,2]
        return [coords[row:row+1, :, :]]
    
    if coords.ndim == 4:  # [H,W,2,C]
        return [coords[row:row+1, :, :, c] for c in range(coords.shape[3])]
    
    raise ValueError(f"Unexpected shape: {coords.shape}")


def extract_point(edge_list: List[np.ndarray], x_idx: int) -> List[np.ndarray]:
    """
    Extract coordinates at a specific x position from edge list.
    
    Args:
        edge_list: List of edge coordinate arrays, each shape [1, W, 2]
        x_idx: X-coordinate index (0 to W-1)
    
    Returns:
        List of coordinate arrays, one per channel, each shape [1, 1, 2]
    """
    return [pc[:, x_idx:x_idx+1, :] for pc in edge_list]


def lerp_point(edge_list: List[np.ndarray], exact_x: float) -> List[np.ndarray]:
    """
    Interpolate coordinates between pixels at a fractional x position.
    
    Performs linear interpolation between adjacent pixel coordinates when
    exact_x falls between integer indices.
    
    Args:
        edge_list: List of edge coordinate arrays, each shape [1, W, 2]
        exact_x: Fractional x-coordinate position (0.0 to W-1.0)
    
    Returns:
        List of interpolated coordinate arrays, one per channel, each shape [1, 1, 2]
    """
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
    Slice coordinates in x-direction and renormalize to [0,1].
    
    Extracts a horizontal slice of the coordinate arrays and renormalizes
    the x-coordinates so they span [0, 1] in the new coordinate system.
    
    Args:
        coords: Per-channel coordinate arrays (list or array)
        start_idx: Starting x-index (inclusive)
        end_idx: Ending x-index (exclusive)
        start_u: Original u-coordinate at start_idx
        end_u: Original u-coordinate at end_idx
    
    Returns:
        Sliced and renormalized coordinates, preserving input type (list or array)
        
    Raises:
        ValueError: If interval is invalid (end_u <= start_u)
    
    Note:
        Only the x-coordinate (coords[..., 0]) is renormalized. The y-coordinate
        remains unchanged.
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
    Slice coordinates in x-direction without renormalization.
    
    Extracts a horizontal slice of the coordinate arrays without modifying
    the coordinate values.
    
    Args:
        coords: Per-channel coordinate arrays (list or array)
        start_idx: Starting x-index (inclusive)
        end_idx: Ending x-index (exclusive)
    
    Returns:
        Sliced coordinates, preserving input type (list or array)
        
    Raises:
        ValueError: If interval is invalid (end_idx <= start_idx)
    
    Note:
        This is similar to slice_and_renormalize but without the renormalization
        step. Coordinate values remain in their original range.
    """
    interval = end_idx - start_idx
    if interval <= 0:
        raise ValueError(f"Invalid interval: [{start_idx}, {end_idx}]")
    
    if isinstance(coords, list):
        sliced = [pc[:, start_idx:end_idx, :].copy() for pc in coords]
        return sliced
    
    sliced = coords[:, start_idx:end_idx, ...].copy()
    return sliced