# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from cython.parallel cimport prange, parallel
from libc.math cimport floor, fmod, fabs
from libc.stdlib cimport malloc, free

ctypedef np.float64_t f64

cdef int BORDER_REPEAT = 0    # Modulo repeat
cdef int BORDER_MIRROR = 1    # Mirror repeat
cdef int BORDER_CONSTANT = 2  # Constant color fill
cdef int BORDER_CLAMP = 3     # Clamp to edge
cdef int BORDER_OVERFLOW = 4  # Allow overflow (no border handling)


from .helpers cimport (
    handle_border_1d,
    is_out_of_bounds_1d,
    is_out_of_bounds_2d,
)

# =============================================================================
# Single-Channel Kernels
# =============================================================================
cdef inline void _lerp_1ch_kernel_parallel(
    const f64[::1] l0,
    const f64[::1] l1,
    const f64[:, :, ::1] c,
    f64[:, ::1] out_mv,
    f64 border_const,
    Py_ssize_t H,
    Py_ssize_t W,
    Py_ssize_t L,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """
    Parallel single-channel linear interpolation kernel.
    
    Parallelizes over rows (H dimension) for optimal cache utilization.
    """
    cdef Py_ssize_t h, w, idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac, v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]

            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    out_mv[h, w] = border_const
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)

            idx_f = new_u_x * L_minus_1
            idx_lo = <Py_ssize_t>floor(idx_f)

            if idx_lo < 0:
                idx_lo = 0
            if idx_lo >= L - 1:
                idx_lo = L - 2

            idx_hi = idx_lo + 1
            frac = idx_f - <f64>idx_lo

            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0

            v0 = l0[idx_lo] + frac * (l0[idx_hi] - l0[idx_lo])
            v1 = l1[idx_lo] + frac * (l1[idx_hi] - l1[idx_lo])
            out_mv[h, w] = v0 + new_u_y * (v1 - v0)


cdef inline void _lerp_x_discrete_1ch_kernel_parallel(
    const f64[::1] l0,
    const f64[::1] l1,
    const f64[:, :, ::1] c,
    f64[:, ::1] out_mv,
    f64 border_const,
    Py_ssize_t H,
    Py_ssize_t W,
    Py_ssize_t L,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """Parallel discrete x-sampling single-channel kernel."""
    cdef Py_ssize_t h, w, idx
    cdef f64 u_x, u_y, idx_f
    cdef f64 new_u_x, new_u_y
    cdef f64 L_minus_1 = <f64>(L - 1)

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]

            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    out_mv[h, w] = border_const
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)

            idx_f = new_u_x * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)

            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1

            out_mv[h, w] = l0[idx] + new_u_y * (l1[idx] - l0[idx])


# =============================================================================
# Multi-Channel Kernels
# =============================================================================
cdef inline void _lerp_multichannel_kernel_parallel(
    const f64[:, ::1] l0,
    const f64[:, ::1] l1,
    const f64[:, :, ::1] c,
    f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv,
    Py_ssize_t H,
    Py_ssize_t W,
    Py_ssize_t L,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """
    Parallel multi-channel linear interpolation kernel.
    
    Uses row-based parallelization for cache-friendly access patterns.
    """
    cdef Py_ssize_t h, w, ch, idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac, v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]

            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    for ch in range(C):
                        out_mv[h, w, ch] = border_const_mv[ch]
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)

            idx_f = new_u_x * L_minus_1
            idx_lo = <Py_ssize_t>floor(idx_f)

            if idx_lo < 0:
                idx_lo = 0
            if idx_lo >= L - 1:
                idx_lo = L - 2

            idx_hi = idx_lo + 1
            frac = idx_f - <f64>idx_lo

            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0

            for ch in range(C):
                v0 = l0[idx_lo, ch] + frac * (l0[idx_hi, ch] - l0[idx_lo, ch])
                v1 = l1[idx_lo, ch] + frac * (l1[idx_hi, ch] - l1[idx_lo, ch])
                out_mv[h, w, ch] = v0 + new_u_y * (v1 - v0)


cdef inline void _lerp_x_discrete_multichannel_kernel_parallel(
    const f64[:, ::1] l0,
    const f64[:, ::1] l1,
    const f64[:, :, ::1] c,
    f64[:, :, ::1] out_mv,
    const f64[::1] border_const_mv,
    Py_ssize_t H,
    Py_ssize_t W,
    Py_ssize_t L,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """Parallel discrete x-sampling multi-channel kernel."""
    cdef Py_ssize_t h, w, ch, idx
    cdef f64 u_x, u_y, idx_f
    cdef f64 new_u_x, new_u_y
    cdef f64 L_minus_1 = <f64>(L - 1)

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]

            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    for ch in range(C):
                        out_mv[h, w, ch] = border_const_mv[ch]
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)

            idx_f = new_u_x * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)

            if idx < 0:
                idx = 0
            elif idx >= L:
                idx = L - 1

            for ch in range(C):
                out_mv[h, w, ch] = l0[idx, ch] + new_u_y * (l1[idx, ch] - l0[idx, ch])

# =============================================================================
# Flat Coordinate Kernels (N, 2) -> (N,) or (N, C)
# =============================================================================
cdef inline void _lerp_flat_1ch_kernel_parallel(
    const f64[::1] l0,
    const f64[::1] l1,
    const f64[:, ::1] c,
    f64[::1] out_mv,
    f64 border_const,
    Py_ssize_t N,
    Py_ssize_t L,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """Parallel flat single-channel kernel."""
    cdef Py_ssize_t n, idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac, v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y

    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]

        if border_mode == BORDER_CONSTANT:
            if is_out_of_bounds_2d(u_x, u_y):
                out_mv[n] = border_const
                continue
            new_u_x = u_x
            new_u_y = u_y
        elif border_mode == BORDER_OVERFLOW:
            new_u_x = u_x
            new_u_y = u_y
        else:
            new_u_x = handle_border_1d(u_x, border_mode)
            new_u_y = handle_border_1d(u_y, border_mode)

        idx_f = new_u_x * L_minus_1
        idx_lo = <Py_ssize_t>floor(idx_f)

        if idx_lo < 0:
            idx_lo = 0
        if idx_lo >= L - 1:
            idx_lo = L - 2

        idx_hi = idx_lo + 1
        frac = idx_f - <f64>idx_lo

        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0

        v0 = l0[idx_lo] + frac * (l0[idx_hi] - l0[idx_lo])
        v1 = l1[idx_lo] + frac * (l1[idx_hi] - l1[idx_lo])
        out_mv[n] = v0 + new_u_y * (v1 - v0)


cdef inline void _lerp_flat_multichannel_kernel_parallel(
    const f64[:, ::1] l0,
    const f64[:, ::1] l1,
    const f64[:, ::1] c,
    f64[:, ::1] out_mv,
    const f64[::1] border_const_mv,
    Py_ssize_t N,
    Py_ssize_t L,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """Parallel flat multi-channel kernel."""
    cdef Py_ssize_t n, ch, idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac, v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y

    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        u_x = c[n, 0]
        u_y = c[n, 1]

        if border_mode == BORDER_CONSTANT:
            if is_out_of_bounds_2d(u_x, u_y):
                for ch in range(C):
                    out_mv[n, ch] = border_const_mv[ch]
                continue
            new_u_x = u_x
            new_u_y = u_y
        elif border_mode == BORDER_OVERFLOW:
            new_u_x = u_x
            new_u_y = u_y
        else:
            new_u_x = handle_border_1d(u_x, border_mode)
            new_u_y = handle_border_1d(u_y, border_mode)

        idx_f = new_u_x * L_minus_1
        idx_lo = <Py_ssize_t>floor(idx_f)

        if idx_lo < 0:
            idx_lo = 0
        if idx_lo >= L - 1:
            idx_lo = L - 2

        idx_hi = idx_lo + 1
        frac = idx_f - <f64>idx_lo

        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0

        for ch in range(C):
            v0 = l0[idx_lo, ch] + frac * (l0[idx_hi, ch] - l0[idx_lo, ch])
            v1 = l1[idx_lo, ch] + frac * (l1[idx_hi, ch] - l1[idx_lo, ch])
            out_mv[n, ch] = v0 + new_u_y * (v1 - v0)




# =============================================================================
# Public API - Single Channel
# =============================================================================
def lerp_between_lines_1ch_fast(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    int num_threads=-1,
):
    """
    Fast parallel single-channel interpolation between two lines.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
        border_mode: Border handling mode
        border_constant: Value for BORDER_CONSTANT mode
        num_threads: Number of threads (-1 = auto, 0 = serial, >0 = specific count)
    
    Returns:
        Interpolated values, shape (H, W)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    # Handle thread count
    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    # Ensure contiguity
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, :, ::1] c = coords

    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_1ch_kernel_parallel(l0, l1, c, out_mv, border_constant, H, W, L, border_mode, n_threads)

    return out


def lerp_between_lines_x_discrete_1ch_fast(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    int num_threads=-1,
):
    """Fast parallel discrete x-sampling single-channel interpolation."""
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, :, ::1] c = coords

    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_x_discrete_1ch_kernel_parallel(l0, l1, c, out_mv, border_constant, H, W, L, border_mode, n_threads)

    return out


# =============================================================================
# Public API - Multi-Channel
# =============================================================================
def lerp_between_lines_multichannel_fast(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """
    Fast parallel multi-channel interpolation between two lines.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Coordinate grid, shape (H, W, 2)
        border_mode: Border handling mode
        border_constant: Pre-resolved border values, shape (C,)
        num_threads: Number of threads (-1 = auto)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    # Handle default border constant
    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] bc = border_constant

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_multichannel_kernel_parallel(l0, l1, c, out_mv, bc, H, W, L, C, border_mode, n_threads)

    return out


def lerp_between_lines_x_discrete_multichannel_fast(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """Fast parallel discrete x-sampling multi-channel interpolation."""
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] bc = border_constant

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_x_discrete_multichannel_kernel_parallel(l0, l1, c, out_mv, bc, H, W, L, C, border_mode, n_threads)

    return out


# =============================================================================
# Public API - Flat Coordinates
# =============================================================================
def lerp_between_lines_flat_1ch_fast(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    f64 border_constant=0.0,
    int num_threads=-1,
):
    """Fast parallel flat single-channel interpolation."""
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)

    cdef f64[::1] l0 = line0
    cdef f64[::1] l1 = line1
    cdef f64[:, ::1] c = coords

    cdef np.ndarray[f64, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef f64[::1] out_mv = out

    with nogil:
        _lerp_flat_1ch_kernel_parallel(l0, l1, c, out_mv, border_constant, N, L, border_mode, n_threads)

    return out


def lerp_between_lines_flat_multichannel_fast(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=2] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """Fast parallel flat multi-channel interpolation."""
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t N = coords.shape[0]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    if n_threads < 0:
        import os
        n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, ::1] c = coords
    cdef f64[::1] bc = border_constant

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_flat_multichannel_kernel_parallel(l0, l1, c, out_mv, bc, N, L, C, border_mode, n_threads)

    return out

# =============================================================================
# Smart Dispatchers
# =============================================================================
# Threshold for enabling parallelization (avoid overhead for small arrays)
DEF MIN_PARALLEL_SIZE = 10000
# =============================================================================
# Per-Channel Coords Kernels - FIXED
# =============================================================================


cdef inline void _lerp_lines_multichannel_per_channel_kernel_parallel(
    const f64[:, ::1] lines0,  # (L, C) - FIXED: was [:, :, ::1]
    const f64[:, ::1] lines1,  # (L, C) - FIXED: was [:, :, ::1]
    const f64[:, :, :, ::1] coords_mv,  # (C, H, W, 2)
    f64[:, :, ::1] out_mv,  # (H, W, C)
    const f64[::1] border_const_mv,
    Py_ssize_t H,
    Py_ssize_t W,
    Py_ssize_t L,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """
    Parallel multi-channel line interpolation with per-channel coordinates.
    
    lines0, lines1: shape (L, C)
    coords: shape (C, H, W, 2)
    """
    cdef Py_ssize_t h, w, ch, idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac, v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y
    cdef f64 val0_lo, val0_hi, val1_lo, val1_hi  # Temp variables for subtraction

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            for ch in range(C):
                u_x = coords_mv[ch, h, w, 0]
                u_y = coords_mv[ch, h, w, 1]

                if border_mode == BORDER_CONSTANT:
                    if is_out_of_bounds_2d(u_x, u_y):
                        out_mv[h, w, ch] = border_const_mv[ch]
                        continue
                    new_u_x = u_x
                    new_u_y = u_y
                elif border_mode == BORDER_OVERFLOW:
                    new_u_x = u_x
                    new_u_y = u_y
                else:
                    new_u_x = handle_border_1d(u_x, border_mode)
                    new_u_y = handle_border_1d(u_y, border_mode)

                idx_f = new_u_x * L_minus_1
                idx_lo = <Py_ssize_t>floor(idx_f)

                if idx_lo < 0:
                    idx_lo = 0
                if idx_lo >= L - 1:
                    idx_lo = L - 2

                idx_hi = idx_lo + 1
                frac = idx_f - <f64>idx_lo

                if frac < 0.0:
                    frac = 0.0
                elif frac > 1.0:
                    frac = 1.0

                # Extract scalars to avoid memoryview slice subtraction
                val0_lo = lines0[idx_lo, ch]
                val0_hi = lines0[idx_hi, ch]
                val1_lo = lines1[idx_lo, ch]
                val1_hi = lines1[idx_hi, ch]
                
                v0 = val0_lo + frac * (val0_hi - val0_lo)
                v1 = val1_lo + frac * (val1_hi - val1_lo)
                out_mv[h, w, ch] = v0 + new_u_y * (v1 - v0)

cdef inline void _lerp_lines_flat_multichannel_per_channel_kernel_parallel(
    const f64[:, ::1] lines0,  # (L, C)
    const f64[:, ::1] lines1,  # (L, C)
    const f64[:, :, ::1] coords_mv,  # (C, N, 2)
    f64[:, ::1] out_mv,  # (N, C)
    const f64[::1] border_const_mv,
    Py_ssize_t N,
    Py_ssize_t L,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """Flat coords with per-channel coordinates."""
    cdef Py_ssize_t n, ch, idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac, v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 new_u_x, new_u_y
    cdef f64 val0_lo, val0_hi, val1_lo, val1_hi

    for n in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for ch in range(C):
            u_x = coords_mv[ch, n, 0]
            u_y = coords_mv[ch, n, 1]

            if border_mode == BORDER_CONSTANT:
                if is_out_of_bounds_2d(u_x, u_y):
                    out_mv[n, ch] = border_const_mv[ch]
                    continue
                new_u_x = u_x
                new_u_y = u_y
            elif border_mode == BORDER_OVERFLOW:
                new_u_x = u_x
                new_u_y = u_y
            else:
                new_u_x = handle_border_1d(u_x, border_mode)
                new_u_y = handle_border_1d(u_y, border_mode)

            idx_f = new_u_x * L_minus_1
            idx_lo = <Py_ssize_t>floor(idx_f)

            if idx_lo < 0:
                idx_lo = 0
            if idx_lo >= L - 1:
                idx_lo = L - 2

            idx_hi = idx_lo + 1
            frac = idx_f - <f64>idx_lo

            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0

            # Extract scalars
            val0_lo = lines0[idx_lo, ch]
            val0_hi = lines0[idx_hi, ch]
            val1_lo = lines1[idx_lo, ch]
            val1_hi = lines1[idx_hi, ch]
            
            v0 = val0_lo + frac * (val0_hi - val0_lo)
            v1 = val1_lo + frac * (val1_hi - val1_lo)
            out_mv[n, ch] = v0 + new_u_y * (v1 - v0)


# =============================================================================
# Per-Channel Discrete X-Sampling Kernel
# =============================================================================
cdef inline void _lerp_lines_multichannel_per_channel_x_discrete_kernel_parallel(
    const f64[:, ::1] lines0,  # (L, C)
    const f64[:, ::1] lines1,  # (L, C)
    const f64[:, :, :, ::1] coords_mv,  # (C, H, W, 2)
    f64[:, :, ::1] out_mv,  # (H, W, C)
    const f64[::1] border_const_mv,
    Py_ssize_t H,
    Py_ssize_t W,
    Py_ssize_t L,
    Py_ssize_t C,
    int border_mode,
    int num_threads,
) noexcept nogil:
    """Discrete x-sampling with per-channel coordinates."""
    cdef Py_ssize_t h, w, ch, idx
    cdef f64 u_x, u_y, idx_f
    cdef f64 new_u_x, new_u_y
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 val0, val1

    for h in prange(H, nogil=True, schedule='static', num_threads=num_threads):
        for w in range(W):
            for ch in range(C):
                u_x = coords_mv[ch, h, w, 0]
                u_y = coords_mv[ch, h, w, 1]

                if border_mode == BORDER_CONSTANT:
                    if is_out_of_bounds_2d(u_x, u_y):
                        out_mv[h, w, ch] = border_const_mv[ch]
                        continue
                    new_u_x = u_x
                    new_u_y = u_y
                elif border_mode == BORDER_OVERFLOW:
                    new_u_x = u_x
                    new_u_y = u_y
                else:
                    new_u_x = handle_border_1d(u_x, border_mode)
                    new_u_y = handle_border_1d(u_y, border_mode)

                idx_f = new_u_x * L_minus_1
                idx = <Py_ssize_t>floor(idx_f + 0.5)

                if idx < 0:
                    idx = 0
                elif idx >= L:
                    idx = L - 1

                val0 = lines0[idx, ch]
                val1 = lines1[idx, ch]
                out_mv[h, w, ch] = val0 + new_u_y * (val1 - val0)


# =============================================================================
# Public API - Per-Channel Coords
# =============================================================================
def lerp_between_lines_multichannel_per_channel_fast(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=4] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """
    Fast parallel multi-channel line interpolation with per-channel coordinates.
    
    Args:
        line0: First line, shape (L, C)
        line1: Second line, shape (L, C)
        coords: Per-channel coordinate grids, shape (C, H, W, 2)
        border_mode: Border handling mode
        border_constant: Pre-resolved border values, shape (C,)
        num_threads: Thread count (-1=auto)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t H = coords.shape[1]
    cdef Py_ssize_t W = coords.shape[2]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[3] != 2:
        raise ValueError("coords must have shape (C, H, W, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match lines ({C})")

    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    cdef Py_ssize_t total_size = H * W
    if n_threads < 0:
        if total_size < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, :, ::1] c = coords
    cdef f64[::1] bc = border_constant

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_lines_multichannel_per_channel_kernel_parallel(
            l0, l1, c, out_mv, bc, H, W, L, C, border_mode, n_threads
        )

    return out


def lerp_between_lines_x_discrete_multichannel_per_channel_fast(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=4] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """Fast parallel discrete x-sampling with per-channel coordinates."""
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t H = coords.shape[1]
    cdef Py_ssize_t W = coords.shape[2]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[3] != 2:
        raise ValueError("coords must have shape (C, H, W, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match lines ({C})")

    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    cdef Py_ssize_t total_size = H * W
    if n_threads < 0:
        if total_size < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, :, ::1] c = coords
    cdef f64[::1] bc = border_constant

    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    with nogil:
        _lerp_lines_multichannel_per_channel_x_discrete_kernel_parallel(
            l0, l1, c, out_mv, bc, H, W, L, C, border_mode, n_threads
        )

    return out


def lerp_between_lines_flat_multichannel_per_channel_fast(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """Fast parallel flat coords with per-channel coordinates."""
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t C_coords = coords.shape[0]
    cdef Py_ssize_t N = coords.shape[1]

    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (C, N, 2)")
    if C_coords != C:
        raise ValueError(f"coords channels ({C_coords}) must match lines ({C})")

    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    cdef int n_threads = num_threads
    if n_threads < 0:
        if N < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] bc = border_constant

    cdef np.ndarray[f64, ndim=2] out = np.empty((N, C), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    with nogil:
        _lerp_lines_flat_multichannel_per_channel_kernel_parallel(
            l0, l1, c, out_mv, bc, N, L, C, border_mode, n_threads
        )

    return out


# =============================================================================
# Updated Smart Dispatchers (with per-channel support)
# =============================================================================
def lerp_between_lines_full_fast(
    np.ndarray line0,
    np.ndarray line1,
    np.ndarray coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
    int num_threads=-1,
):
    """
    Smart dispatcher for fast line interpolation with full per-channel support.
    
    Handles all coordinate configurations:
    - Single channel: (H, W, 2) or (N, 2)
    - Multi-channel same coords: (H, W, 2) or (N, 2)  
    - Multi-channel per-channel coords: (C, H, W, 2) or (C, N, 2)
    """
    # ALL cdef declarations at top
    cdef Py_ssize_t total_size
    cdef int use_threads = num_threads
    cdef Py_ssize_t C
    cdef np.ndarray[f64, ndim=1] bc_arr
    cdef f64 bc_scalar
    cdef int coords_ndim = coords.ndim
    
    # Convert to proper types
    if line0.dtype != np.float64:
        line0 = np.ascontiguousarray(line0, dtype=np.float64)
    if line1.dtype != np.float64:
        line1 = np.ascontiguousarray(line1, dtype=np.float64)
    if coords.dtype != np.float64:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    # Calculate size for parallelization
    if coords_ndim == 3:
        total_size = coords.shape[0] * coords.shape[1]
    elif coords_ndim == 4:
        total_size = coords.shape[1] * coords.shape[2]
    else:
        total_size = coords.shape[0]
    
    if total_size < MIN_PARALLEL_SIZE and num_threads < 0:
        use_threads = 1
    
    # Single channel
    if line0.ndim == 1:
        bc_scalar = 0.0 if border_constant is None else float(border_constant)
        
        if coords_ndim == 3 and coords.shape[2] == 2:
            return lerp_between_lines_1ch_fast(
                line0, line1, coords, border_mode, bc_scalar, use_threads
            )
        elif coords_ndim == 2 and coords.shape[1] == 2:
            return lerp_between_lines_flat_1ch_fast(
                line0, line1, coords, border_mode, bc_scalar, use_threads
            )
        else:
            raise ValueError(f"Invalid coords shape for single channel: ndim={coords_ndim}")
    
    # Multi-channel
    elif line0.ndim == 2:
        C = line0.shape[1]
        
        if border_constant is None:
            bc_arr = np.zeros(C, dtype=np.float64)
        elif isinstance(border_constant, (int, float)):
            bc_arr = np.full(C, float(border_constant), dtype=np.float64)
        else:
            bc_arr = np.ascontiguousarray(border_constant, dtype=np.float64)
            if bc_arr.shape[0] != C:
                raise ValueError(f"border_constant must have length {C}")
        
        # 3D coords: same for all channels (H, W, 2)
        if coords_ndim == 3 and coords.shape[2] == 2:
            return lerp_between_lines_multichannel_fast(
                line0, line1, coords, border_mode, bc_arr, use_threads
            )
        
        # 2D coords: flat, same for all channels (N, 2)
        elif coords_ndim == 2 and coords.shape[1] == 2:
            return lerp_between_lines_flat_multichannel_fast(
                line0, line1, coords, border_mode, bc_arr, use_threads
            )
        
        # 4D coords: per-channel grid (C, H, W, 2)
        elif coords_ndim == 4 and coords.shape[3] == 2:
            if coords.shape[0] != C:
                raise ValueError(f"coords channels ({coords.shape[0]}) must match lines ({C})")
            return lerp_between_lines_multichannel_per_channel_fast(
                line0, line1, coords, border_mode, bc_arr, use_threads
            )
        
        # 3D coords: per-channel flat (C, N, 2)
        elif coords_ndim == 3 and coords.shape[2] == 2 and coords.shape[0] == C:
            return lerp_between_lines_flat_multichannel_per_channel_fast(
                line0, line1, coords, border_mode, bc_arr, use_threads
            )
        
        else:
            raise ValueError(f"Invalid coords shape for multi-channel: ndim={coords_ndim}")
    
    raise ValueError(f"Unsupported line dimensions: {line0.ndim}")


def lerp_between_lines_x_discrete_full_fast(
    np.ndarray line0,
    np.ndarray line1,
    np.ndarray coords,
    int border_mode=BORDER_CLAMP,
    object border_constant=None,
    int num_threads=-1,
):
    """Smart dispatcher for discrete x-sampling with full per-channel support."""
    cdef Py_ssize_t total_size
    cdef int use_threads = num_threads
    cdef Py_ssize_t C
    cdef np.ndarray[f64, ndim=1] bc_arr
    cdef f64 bc_scalar
    cdef int coords_ndim = coords.ndim
    
    if line0.dtype != np.float64:
        line0 = np.ascontiguousarray(line0, dtype=np.float64)
    if line1.dtype != np.float64:
        line1 = np.ascontiguousarray(line1, dtype=np.float64)
    if coords.dtype != np.float64:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
    
    # Calculate size
    if coords_ndim == 3:
        total_size = coords.shape[0] * coords.shape[1]
    elif coords_ndim == 4:
        total_size = coords.shape[1] * coords.shape[2]
    else:
        total_size = coords.shape[0]
    
    if total_size < MIN_PARALLEL_SIZE and num_threads < 0:
        use_threads = 1
    
    # Single channel
    if line0.ndim == 1:
        bc_scalar = 0.0 if border_constant is None else float(border_constant)
        
        if coords_ndim == 3 and coords.shape[2] == 2:
            return lerp_between_lines_x_discrete_1ch_fast(
                line0, line1, coords, border_mode, bc_scalar, use_threads
            )
        else:
            raise ValueError(f"Invalid coords shape: ndim={coords_ndim}")
    
    # Multi-channel
    elif line0.ndim == 2:
        C = line0.shape[1]
        
        if border_constant is None:
            bc_arr = np.zeros(C, dtype=np.float64)
        elif isinstance(border_constant, (int, float)):
            bc_arr = np.full(C, float(border_constant), dtype=np.float64)
        else:
            bc_arr = np.ascontiguousarray(border_constant, dtype=np.float64)
        
        # Same coords (H, W, 2)
        if coords_ndim == 3 and coords.shape[2] == 2:
            return lerp_between_lines_x_discrete_multichannel_fast(
                line0, line1, coords, border_mode, bc_arr, use_threads
            )
        
        # Per-channel (C, H, W, 2)
        elif coords_ndim == 4 and coords.shape[3] == 2:
            if coords.shape[0] != C:
                raise ValueError(f"coords channels ({coords.shape[0]}) must match lines ({C})")
            return lerp_between_lines_x_discrete_multichannel_per_channel_fast(
                line0, line1, coords, border_mode, bc_arr, use_threads
            )
        
        else:
            raise ValueError(f"Invalid coords shape: ndim={coords_ndim}")
    
    raise ValueError(f"Unsupported line dimensions: {line0.ndim}")

# =============================================================================
# Public API - Multichannel 2D Interpolation (Same Coords)
# =============================================================================
def lerp_between_lines_2d_multichannel_same_coords(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """
    Fast parallel multichannel line interpolation using 2D grid coordinates.
    
    Interpolates between two lines (L, C) using a 2D coordinate grid (H, W, 2)
    where the same coordinates are used for all channels.
    
    Args:
        line0: First line values, shape (L, C)
        line1: Second line values, shape (L, C)
        coords: 2D coordinate grid, shape (H, W, 2) with (u_x, u_y) values
        border_mode: Border handling mode (BORDER_CLAMP, BORDER_REPEAT, etc.)
        border_constant: Constant value for BORDER_CONSTANT mode, shape (C,)
        num_threads: Number of threads (-1=auto, 0=serial, >0=specific)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    # Validation
    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape (L, C)")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    # Handle border constant
    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    # Thread count logic
    cdef int n_threads = num_threads
    cdef Py_ssize_t total_size = H * W
    if n_threads < 0:
        if total_size < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    # Ensure contiguity
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    # Memory views
    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] bc = border_constant

    # Output array
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    # Call kernel
    with nogil:
        _lerp_multichannel_kernel_parallel(
            l0, l1, c, out_mv, bc, H, W, L, C, border_mode, n_threads
        )

    return out

# =============================================================================
# Public API - Multichannel 2D Interpolation (Same Coords) - Discrete X
# =============================================================================
def lerp_between_lines_2d_x_discrete_multichannel_same_coords(
    np.ndarray[f64, ndim=2] line0,
    np.ndarray[f64, ndim=2] line1,
    np.ndarray[f64, ndim=3] coords,
    int border_mode=BORDER_CLAMP,
    np.ndarray[f64, ndim=1] border_constant=None,
    int num_threads=-1,
):
    """
    Fast parallel multichannel discrete x-sampling interpolation using 2D grid coordinates.
    
    Snaps u_x to nearest line index (no interpolation along x), interpolates continuously along y.
    Uses the same coordinates for all channels.
    
    Args:
        line0: First line values, shape (L, C)
        line1: Second line values, shape (L, C)
        coords: 2D coordinate grid, shape (H, W, 2) with (u_x, u_y) values
        border_mode: Border handling mode (BORDER_CLAMP, BORDER_REPEAT, etc.)
        border_constant: Constant value for BORDER_CONSTANT mode, shape (C,)
        num_threads: Number of threads (-1=auto, 0=serial, >0=specific)
    
    Returns:
        Interpolated values, shape (H, W, C)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t C = line0.shape[1]
    cdef Py_ssize_t H = coords.shape[0]
    cdef Py_ssize_t W = coords.shape[1]

    # Validation
    if line1.shape[0] != L or line1.shape[1] != C:
        raise ValueError("Lines must have same shape (L, C)")
    if coords.shape[2] != 2:
        raise ValueError("coords must have shape (H, W, 2)")

    # Handle border constant
    if border_constant is None:
        border_constant = np.zeros(C, dtype=np.float64)
    elif border_constant.shape[0] != C:
        raise ValueError(f"border_constant must have length {C}")

    # Thread count logic
    cdef int n_threads = num_threads
    cdef Py_ssize_t total_size = H * W
    if n_threads < 0:
        if total_size < MIN_PARALLEL_SIZE:
            n_threads = 1
        else:
            import os
            n_threads = os.cpu_count() or 4
    elif n_threads == 0:
        n_threads = 1

    # Ensure contiguity
    if not line0.flags['C_CONTIGUOUS']:
        line0 = np.ascontiguousarray(line0)
    if not line1.flags['C_CONTIGUOUS']:
        line1 = np.ascontiguousarray(line1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    if not border_constant.flags['C_CONTIGUOUS']:
        border_constant = np.ascontiguousarray(border_constant)

    # Memory views
    cdef f64[:, ::1] l0 = line0
    cdef f64[:, ::1] l1 = line1
    cdef f64[:, :, ::1] c = coords
    cdef f64[::1] bc = border_constant

    # Output array
    cdef np.ndarray[f64, ndim=3] out = np.empty((H, W, C), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out

    # Call kernel (already exists!)
    with nogil:
        _lerp_x_discrete_multichannel_kernel_parallel(
            l0, l1, c, out_mv, bc, H, W, L, C, border_mode, n_threads
        )

    return out