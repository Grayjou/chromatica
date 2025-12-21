# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free

ctypedef np.float64_t f64

# =============================================================================
# 1D spatial: coeffs shape (L, N) -> output shape (L,)
# =============================================================================
def lerp_bounded_1d_spatial_fast(
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=2] coeffs,
):
    cdef Py_ssize_t num_points = coeffs.shape[0]
    cdef Py_ssize_t num_interp_dims = coeffs.shape[1]
    cdef Py_ssize_t num_corners = starts.shape[0]

    if ends.shape[0] != num_corners:
        raise ValueError("starts and ends must have same length")
    if num_corners <= 0:
        raise ValueError("num_corners must be > 0")

    # Ensure contiguous
    if not starts.flags['C_CONTIGUOUS']:
        starts = np.ascontiguousarray(starts, dtype=np.float64)
    if not ends.flags['C_CONTIGUOUS']:
        ends = np.ascontiguousarray(ends, dtype=np.float64)
    if not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)

    cdef f64[::1] starts_mv = starts
    cdef f64[::1] ends_mv = ends
    cdef f64[:, ::1] coeffs_mv = coeffs

    cdef np.ndarray[f64, ndim=1] result = np.empty(num_points, dtype=np.float64)
    cdef f64[::1] result_mv = result

    cdef Py_ssize_t p, i, j, half, curr_size
    cdef f64 u

    # ------------------------------------------------------------
    # ALL cdef declarations MUST be here, not inside if blocks
    # ------------------------------------------------------------
    cdef Py_ssize_t MAX_CORNERS = 256
    cdef f64 a_stack[256]  # Moved outside if block
    cdef f64 b_stack[256]  # Moved outside if block
    cdef f64* a = NULL
    cdef f64* b = NULL

    if num_corners <= MAX_CORNERS:
        # Use stack buffers
        for p in range(num_points):
            memcpy(&a_stack[0], &starts_mv[0], num_corners * sizeof(f64))
            memcpy(&b_stack[0], &ends_mv[0],   num_corners * sizeof(f64))

            curr_size = num_corners
            for i in range(num_interp_dims):
                u = coeffs_mv[p, i]

                for j in range(curr_size):
                    a_stack[j] = a_stack[j] + u * (b_stack[j] - a_stack[j])

                if curr_size > 1:
                    half = curr_size >> 1
                    memcpy(&b_stack[0], &a_stack[half], half * sizeof(f64))
                    curr_size = half

            result_mv[p] = a_stack[0]

        return result

    # ------------------------------------------------------------
    # Heap buffer fallback (for large corner counts)
    # ------------------------------------------------------------
    a = <f64*>malloc(num_corners * sizeof(f64))
    b = <f64*>malloc(num_corners * sizeof(f64))
    if a == NULL or b == NULL:
        if a != NULL: free(a)
        if b != NULL: free(b)
        raise MemoryError("Failed to allocate working buffers")

    try:
        for p in range(num_points):
            memcpy(a, &starts_mv[0], num_corners * sizeof(f64))
            memcpy(b, &ends_mv[0],   num_corners * sizeof(f64))

            curr_size = num_corners
            for i in range(num_interp_dims):
                u = coeffs_mv[p, i]

                for j in range(curr_size):
                    a[j] = a[j] + u * (b[j] - a[j])

                if curr_size > 1:
                    half = curr_size >> 1
                    memcpy(b, a + half, half * sizeof(f64))
                    curr_size = half

            result_mv[p] = a[0]
    finally:
        free(a)
        free(b)

    return result


# =============================================================================
# 2D spatial: coeffs shape (H, W, N) -> output shape (H, W)
# =============================================================================
def lerp_bounded_2d_spatial_fast(
    np.ndarray[f64, ndim=1] starts,
    np.ndarray[f64, ndim=1] ends,
    np.ndarray[f64, ndim=3] coeffs,
):
    cdef Py_ssize_t H = coeffs.shape[0]
    cdef Py_ssize_t W = coeffs.shape[1]
    cdef Py_ssize_t N = coeffs.shape[2]
    cdef Py_ssize_t num_corners = starts.shape[0]

    if ends.shape[0] != num_corners:
        raise ValueError("starts and ends must have same length")
    if num_corners <= 0:
        raise ValueError("num_corners must be > 0")

    # Ensure contiguous
    if not starts.flags['C_CONTIGUOUS']:
        starts = np.ascontiguousarray(starts, dtype=np.float64)
    if not ends.flags['C_CONTIGUOUS']:
        ends = np.ascontiguousarray(ends, dtype=np.float64)
    if not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)

    cdef f64[::1] starts_mv = starts
    cdef f64[::1] ends_mv = ends
    cdef f64[:, :, ::1] coeffs_mv = coeffs

    cdef np.ndarray[f64, ndim=2] out = np.empty((H, W), dtype=np.float64)
    cdef f64[:, ::1] out_mv = out

    cdef Py_ssize_t h, w, i, j, half, curr_size
    cdef f64 u

    # ALL cdef declarations at function scope
    cdef Py_ssize_t MAX_CORNERS = 256
    cdef f64 a_stack[256]
    cdef f64 b_stack[256]
    cdef f64* a = NULL
    cdef f64* b = NULL

    if num_corners <= MAX_CORNERS:
        for h in range(H):
            for w in range(W):
                memcpy(&a_stack[0], &starts_mv[0], num_corners * sizeof(f64))
                memcpy(&b_stack[0], &ends_mv[0],   num_corners * sizeof(f64))

                curr_size = num_corners
                for i in range(N):
                    u = coeffs_mv[h, w, i]

                    for j in range(curr_size):
                        a_stack[j] = a_stack[j] + u * (b_stack[j] - a_stack[j])

                    if curr_size > 1:
                        half = curr_size >> 1
                        memcpy(&b_stack[0], &a_stack[half], half * sizeof(f64))
                        curr_size = half

                out_mv[h, w] = a_stack[0]

        return out

    # Heap fallback
    a = <f64*>malloc(num_corners * sizeof(f64))
    b = <f64*>malloc(num_corners * sizeof(f64))
    if a == NULL or b == NULL:
        if a != NULL: free(a)
        if b != NULL: free(b)
        raise MemoryError("Failed to allocate working buffers")

    try:
        for h in range(H):
            for w in range(W):
                memcpy(a, &starts_mv[0], num_corners * sizeof(f64))
                memcpy(b, &ends_mv[0],   num_corners * sizeof(f64))

                curr_size = num_corners
                for i in range(N):
                    u = coeffs_mv[h, w, i]

                    for j in range(curr_size):
                        a[j] = a[j] + u * (b[j] - a[j])

                    if curr_size > 1:
                        half = curr_size >> 1
                        memcpy(b, a + half, half * sizeof(f64))
                        curr_size = half

                out_mv[h, w] = a[0]
    finally:
        free(a)
        free(b)

    return out



def _lerp_bounded_vectorized(
    np.ndarray starts,
    np.ndarray ends,
    np.ndarray coeffs,
):
    """
    Fallback for spatial dims > 2. Uses NumPy broadcasting.
    """
    cdef Py_ssize_t n_interp_dims = coeffs.shape[coeffs.ndim - 1]
    cdef Py_ssize_t spatial_ndims = coeffs.ndim - 1
    cdef Py_ssize_t i, half

    # Reshape for broadcasting: (num_corners,) -> (num_corners, 1, 1, ..., 1)
    broadcast_shape = (starts.shape[0],) + (1,) * spatial_ndims

    cdef np.ndarray current_starts = starts.reshape(broadcast_shape).copy()
    cdef np.ndarray current_ends = ends.reshape(broadcast_shape).copy()
    cdef np.ndarray u
    cdef np.ndarray diff

    for i in range(n_interp_dims):
        u = coeffs[..., i]

        # Fused lerp: out = start + u * (end - start)
        diff = np.subtract(current_ends, current_starts)
        np.multiply(diff, u, out=diff)
        np.add(current_starts, diff, out=current_starts)

        if current_starts.shape[0] > 1:
            half = current_starts.shape[0] // 2
            current_ends = current_starts[half:].copy()
            current_starts = current_starts[:half].copy()

    return current_starts.squeeze(axis=0)


# =============================================================================
# Dispatcher
# =============================================================================
def single_channel_multidim_lerp_bounded_cython_fast(
    np.ndarray starts,
    np.ndarray ends,
    np.ndarray coeffs,
):
    cdef int spatial_ndims = coeffs.ndim - 1

    # Force float64 contiguous
    if coeffs.dtype != np.float64 or not coeffs.flags['C_CONTIGUOUS']:
        coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
    if starts.dtype != np.float64 or not starts.flags['C_CONTIGUOUS']:
        starts = np.ascontiguousarray(starts, dtype=np.float64)
    if ends.dtype != np.float64 or not ends.flags['C_CONTIGUOUS']:
        ends = np.ascontiguousarray(ends, dtype=np.float64)

    if spatial_ndims == 1:
        return lerp_bounded_1d_spatial_fast(starts, ends, coeffs)
    elif spatial_ndims == 2:
        return lerp_bounded_2d_spatial_fast(starts, ends, coeffs)
    else:
        return _lerp_bounded_vectorized(starts, ends, coeffs)