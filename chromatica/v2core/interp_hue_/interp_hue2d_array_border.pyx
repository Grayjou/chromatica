# interp_hue2d_array_border.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

"""
Hue interpolation with array-based border blending.

Provides feathered hue interpolation that blends against a per-pixel border
array using the shortest hue path for the blend step. Supports optional
nearest-neighbor sampling along x (x_discrete) and flat coordinate layouts.
"""

import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport floor

from ..interp_utils cimport (
    BorderResult,
    MAX_NORM,
    MANHATTAN,
    SCALED_MANHATTAN,
    ALPHA_MAX,
    ALPHA_MAX_SIMPLE,
    TAYLOR,
    EUCLIDEAN,
    WEIGHTED_MINMAX,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_MIRROR,
    BORDER_REPEAT,
)
from .interp_hue_utils cimport (
    f64,
    i32,
    HUE_SHORTEST,
    wrap_hue,
    lerp_hue_single,
    _interp_line_1ch_hue,
    _interp_line_discrete_hue,
    process_hue_border_2d,
)


cdef i32 _distance_mode_to_int(object distance_mode):
    """Convert distance mode identifier (str/int) to integer constant."""
    if isinstance(distance_mode, str):
        if distance_mode == 'max_norm':
            return MAX_NORM
        elif distance_mode == 'manhattan':
            return MANHATTAN
        elif distance_mode == 'scaled_manhattan':
            return SCALED_MANHATTAN
        elif distance_mode == 'alpha_max':
            return ALPHA_MAX
        elif distance_mode == 'alpha_max_simple':
            return ALPHA_MAX_SIMPLE
        elif distance_mode == 'taylor':
            return TAYLOR
        elif distance_mode == 'euclidean':
            return EUCLIDEAN
        elif distance_mode == 'weighted_minmax':
            return WEIGHTED_MINMAX
        else:
            return ALPHA_MAX
    try:
        return <i32>distance_mode
    except Exception:
        return ALPHA_MAX


def _border_mode_to_int(object border_mode):
    """Convert border mode identifier (str/int) to integer constant."""
    if isinstance(border_mode, str):
        if border_mode == 'constant':
            return BORDER_CONSTANT
        elif border_mode == 'clamp':
            return BORDER_CLAMP
        elif border_mode == 'reflect':
            return BORDER_MIRROR
        elif border_mode == 'periodic':
            return BORDER_REPEAT
        else:
            return BORDER_CONSTANT
    try:
        return <int>border_mode
    except Exception:
        return BORDER_CONSTANT


# =============================================================================
# Kernels
# =============================================================================
cdef inline void _lerp_1ch_hue_array_border_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, :, ::1] c, f64[:, ::1] out_mv,
    const f64[:, ::1] border_array_mv, f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t L,
    int border_mode, int num_threads,
    int mode_x, int mode_y,
    i32 distance_mode,
) noexcept nogil:
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val, border_val_wrapped
    cdef BorderResult border_res

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]

            border_res = process_hue_border_2d(u_x, u_y, border_mode,
                                               border_feathering, distance_mode)

            if border_res.use_border_directly:
                out_mv[h, w] = wrap_hue(border_array_mv[h, w])
            else:
                edge_val = _interp_line_1ch_hue(l0, l1, border_res.u_x_final,
                                                border_res.u_y_final, L,
                                                mode_x, mode_y)
                if border_res.blend_factor > 0.0:
                    border_val_wrapped = wrap_hue(border_array_mv[h, w])
                    out_mv[h, w] = lerp_hue_single(edge_val, border_val_wrapped,
                                                   border_res.blend_factor,
                                                   HUE_SHORTEST)
                else:
                    out_mv[h, w] = edge_val


cdef inline void _lerp_flat_1ch_hue_array_border_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, ::1] c, f64[::1] out_mv,
    const f64[::1] border_array_mv, f64 border_feathering,
    Py_ssize_t N, Py_ssize_t L,
    int border_mode, int num_threads,
    int mode_x, int mode_y,
    i32 distance_mode,
) noexcept nogil:
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, edge_val, border_val_wrapped
    cdef BorderResult border_res

    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]

        border_res = process_hue_border_2d(u_x, u_y, border_mode,
                                           border_feathering, distance_mode)

        if border_res.use_border_directly:
            out_mv[n] = wrap_hue(border_array_mv[n])
        else:
            edge_val = _interp_line_1ch_hue(l0, l1, border_res.u_x_final,
                                            border_res.u_y_final, L,
                                            mode_x, mode_y)
            if border_res.blend_factor > 0.0:
                border_val_wrapped = wrap_hue(border_array_mv[n])
                out_mv[n] = lerp_hue_single(edge_val, border_val_wrapped,
                                            border_res.blend_factor,
                                            HUE_SHORTEST)
            else:
                out_mv[n] = edge_val


cdef inline void _lerp_x_discrete_hue_array_border_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, :, ::1] c, f64[:, ::1] out_mv,
    const f64[:, ::1] border_array_mv, f64 border_feathering,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t L,
    int border_mode, int num_threads,
    int mode_y,
    i32 distance_mode,
) noexcept nogil:
    cdef Py_ssize_t h, w
    cdef f64 u_x, u_y, edge_val, border_val_wrapped
    cdef BorderResult border_res

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]

            border_res = process_hue_border_2d(u_x, u_y, border_mode,
                                               border_feathering, distance_mode)

            if border_res.use_border_directly:
                out_mv[h, w] = wrap_hue(border_array_mv[h, w])
            else:
                edge_val = _interp_line_discrete_hue(
                    l0, l1, border_res.u_x_final, border_res.u_y_final, L, mode_y
                )
                if border_res.blend_factor > 0.0:
                    border_val_wrapped = wrap_hue(border_array_mv[h, w])
                    out_mv[h, w] = lerp_hue_single(edge_val, border_val_wrapped,
                                                   border_res.blend_factor,
                                                   HUE_SHORTEST)
                else:
                    out_mv[h, w] = edge_val


cdef inline void _lerp_x_discrete_flat_hue_array_border_kernel(
    const f64[::1] l0, const f64[::1] l1,
    const f64[:, ::1] c, f64[::1] out_mv,
    const f64[::1] border_array_mv, f64 border_feathering,
    Py_ssize_t N, Py_ssize_t L,
    int border_mode, int num_threads,
    int mode_y,
    i32 distance_mode,
) noexcept nogil:
    cdef Py_ssize_t n
    cdef f64 u_x, u_y, edge_val, border_val_wrapped
    cdef BorderResult border_res

    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]

        border_res = process_hue_border_2d(u_x, u_y, border_mode,
                                           border_feathering, distance_mode)

        if border_res.use_border_directly:
            out_mv[n] = wrap_hue(border_array_mv[n])
        else:
            edge_val = _interp_line_discrete_hue(
                l0, l1, border_res.u_x_final, border_res.u_y_final, L, mode_y
            )
            if border_res.blend_factor > 0.0:
                border_val_wrapped = wrap_hue(border_array_mv[n])
                out_mv[n] = lerp_hue_single(edge_val, border_val_wrapped,
                                            border_res.blend_factor,
                                            HUE_SHORTEST)
            else:
                out_mv[n] = edge_val


# =============================================================================
# Public APIs
# =============================================================================
cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_array_border(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    f64 border_feathering=0.0,
    object border_mode='constant',
    int mode_x=HUE_SHORTEST,
    int mode_y=HUE_SHORTEST,
    object distance_mode='euclidean',
    int num_threads=1,
):
    """Hue interpolation with array-based border values on a grid."""
    if border_array.shape[0] != coords.shape[0] or border_array.shape[1] != coords.shape[1]:
        raise ValueError("border_array shape must match coords[..., 0]")

    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    cdef Py_ssize_t L = l0.shape[0]

    if l1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    if not l0.flags['C_CONTIGUOUS']:
        l0 = np.ascontiguousarray(l0)
    if not l1.flags['C_CONTIGUOUS']:
        l1 = np.ascontiguousarray(l1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    cdef int border_mode_int = _border_mode_to_int(border_mode)
    cdef i32 dist_mode_int = _distance_mode_to_int(distance_mode)

    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    _lerp_1ch_hue_array_border_kernel(
        l0, l1, coords, out_mv, border_array, border_feathering,
        H, W, L, border_mode_int, n_threads,
        mode_x, mode_y, dist_mode_int,
    )
    return out


cpdef np.ndarray[f64, ndim=1] hue_lerp_between_lines_array_border_flat(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    f64 border_feathering=0.0,
    object border_mode='constant',
    int mode_x=HUE_SHORTEST,
    int mode_y=HUE_SHORTEST,
    object distance_mode='euclidean',
    int num_threads=1,
):
    """Hue interpolation with array-based borders for flat coords (N, 2)."""
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if border_array.shape[0] != coords.shape[0]:
        raise ValueError("border_array length must match coords")

    cdef Py_ssize_t N = coords.shape[0]
    cdef Py_ssize_t L = l0.shape[0]

    if l1.shape[0] != L:
        raise ValueError("Lines must have same length")

    if not l0.flags['C_CONTIGUOUS']:
        l0 = np.ascontiguousarray(l0)
    if not l1.flags['C_CONTIGUOUS']:
        l1 = np.ascontiguousarray(l1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    cdef int border_mode_int = _border_mode_to_int(border_mode)
    cdef i32 dist_mode_int = _distance_mode_to_int(distance_mode)

    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out

    _lerp_flat_1ch_hue_array_border_kernel(
        l0, l1, coords, out_mv, border_array, border_feathering,
        N, L, border_mode_int, n_threads,
        mode_x, mode_y, dist_mode_int,
    )
    return out


cpdef np.ndarray[f64, ndim=2] hue_lerp_between_lines_array_border_x_discrete(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=3] coords,
    np.ndarray[f64, ndim=2] border_array,
    f64 border_feathering=0.0,
    object border_mode='constant',
    int mode_y=HUE_SHORTEST,
    object distance_mode='euclidean',
    int num_threads=1,
):
    """Hue interpolation with array borders using nearest-neighbor x."""
    if border_array.shape[0] != coords.shape[0] or border_array.shape[1] != coords.shape[1]:
        raise ValueError("border_array shape must match coords[..., 0]")

    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]
    cdef Py_ssize_t L = l0.shape[0]

    if l1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    if not l0.flags['C_CONTIGUOUS']:
        l0 = np.ascontiguousarray(l0)
    if not l1.flags['C_CONTIGUOUS']:
        l1 = np.ascontiguousarray(l1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    cdef int border_mode_int = _border_mode_to_int(border_mode)
    cdef i32 dist_mode_int = _distance_mode_to_int(distance_mode)

    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    _lerp_x_discrete_hue_array_border_kernel(
        l0, l1, coords, out_mv, border_array, border_feathering,
        H, W, L, border_mode_int, n_threads,
        mode_y, dist_mode_int,
    )
    return out


cpdef np.ndarray[f64, ndim=1] hue_lerp_between_lines_array_border_flat_x_discrete(
    np.ndarray[f64, ndim=1] l0,
    np.ndarray[f64, ndim=1] l1,
    np.ndarray[f64, ndim=2] coords,
    np.ndarray[f64, ndim=1] border_array,
    f64 border_feathering=0.0,
    object border_mode='constant',
    int mode_y=HUE_SHORTEST,
    object distance_mode='euclidean',
    int num_threads=1,
):
    """Hue interpolation with array borders, flat coords, nearest x."""
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if border_array.shape[0] != coords.shape[0]:
        raise ValueError("border_array length must match coords")

    cdef Py_ssize_t N = coords.shape[0]
    cdef Py_ssize_t L = l0.shape[0]

    if l1.shape[0] != L:
        raise ValueError("Lines must have same length")

    if not l0.flags['C_CONTIGUOUS']:
        l0 = np.ascontiguousarray(l0)
    if not l1.flags['C_CONTIGUOUS']:
        l1 = np.ascontiguousarray(l1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_array.flags['C_CONTIGUOUS']:
        border_array = np.ascontiguousarray(border_array)

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    cdef int border_mode_int = _border_mode_to_int(border_mode)
    cdef i32 dist_mode_int = _distance_mode_to_int(distance_mode)

    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out

    _lerp_x_discrete_flat_hue_array_border_kernel(
        l0, l1, coords, out_mv, border_array, border_feathering,
        N, L, border_mode_int, n_threads,
        mode_y, dist_mode_int,
    )
    return out
