# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport floor

ctypedef np.float64_t f64

# =============================================================================
# Single channel interpolation between two 1D lines
# =============================================================================
def lerp_between_lines_1ch(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
):
    """
    Interpolate between two 1D lines using a 2D grid of (u_x, u_y) coordinates.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x (position along lines, 0-1)
                coords[h, w, 1] = u_y (blend factor between lines, 0-1)
    
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
    
    # Ensure contiguous
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
    
    cdef Py_ssize_t h, w
    cdef Py_ssize_t idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac
    cdef f64 v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            # Map u_x to line index (continuous)
            idx_f = u_x * L_minus_1
            idx_lo = <Py_ssize_t>floor(idx_f)
            
            # Clamp indices to valid range
            if idx_lo < 0:
                idx_lo = 0
            if idx_lo >= L - 1:
                idx_lo = L - 2
            
            idx_hi = idx_lo + 1
            frac = idx_f - <f64>idx_lo
            
            # Clamp frac for boundary cases
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            
            # Sample line0 at u_x (linear interpolation within line)
            v0 = l0[idx_lo] + frac * (l0[idx_hi] - l0[idx_lo])
            
            # Sample line1 at u_x
            v1 = l1[idx_lo] + frac * (l1[idx_hi] - l1[idx_lo])
            
            # Blend between lines using u_y
            out_mv[h, w] = v0 + u_y * (v1 - v0)
    
    return out


# =============================================================================
# Single channel interpolation with flat coordinates
# =============================================================================
def lerp_between_lines_flat_1ch(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=2] coords,
):
    """
    Interpolate between two 1D lines using flat list of (u_x, u_y) coordinates.
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate list, shape (N, 2)
                coords[n, 0] = u_x (position along lines, 0-1)
                coords[n, 1] = u_y (blend factor between lines, 0-1)
    
    Returns:
        Interpolated values, shape (N,)
    """
    cdef Py_ssize_t L = line0.shape[0]
    cdef Py_ssize_t N = coords.shape[0]
    
    if line1.shape[0] != L:
        raise ValueError("Lines must have same length")
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    
    # Ensure contiguous
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
    
    cdef Py_ssize_t n
    cdef Py_ssize_t idx_lo, idx_hi
    cdef f64 u_x, u_y, idx_f, frac
    cdef f64 v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    for n in range(N):
        u_x = c[n, 0]
        u_y = c[n, 1]
        
        # Map u_x to line index (continuous)
        idx_f = u_x * L_minus_1
        idx_lo = <Py_ssize_t>floor(idx_f)
        
        # Clamp indices to valid range
        if idx_lo < 0:
            idx_lo = 0
        if idx_lo >= L - 1:
            idx_lo = L - 2
        
        idx_hi = idx_lo + 1
        frac = idx_f - <f64>idx_lo
        
        # Clamp frac for boundary cases
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0
        
        # Sample line0 at u_x (linear interpolation within line)
        v0 = l0[idx_lo] + frac * (l0[idx_hi] - l0[idx_lo])
        
        # Sample line1 at u_x
        v1 = l1[idx_lo] + frac * (l1[idx_hi] - l1[idx_lo])
        
        # Blend between lines using u_y
        out_mv[n] = v0 + u_y * (v1 - v0)
    
    return out


# =============================================================================
# Single channel 3D plane interpolation
# =============================================================================
def lerp_between_planes_1ch(
    np.ndarray[f64, ndim=2] plane0,
    np.ndarray[f64, ndim=2] plane1,
    np.ndarray[f64, ndim=4] coords,
):
    """
    Interpolate between two 2D planes using 3D coords with (u_x, u_y, u_z).
    
    Args:
        plane0: First plane, shape (H0, W0)
        plane1: Second plane, shape (H0, W0)
        coords: Coordinate grid, shape (D, H, W, 3)
                coords[d, h, w, 0] = u_x (0-1)
                coords[d, h, w, 1] = u_y (0-1)
                coords[d, h, w, 2] = u_z (blend factor, 0-1)
    
    Returns:
        Interpolated values, shape (D, H, W)
    """
    cdef Py_ssize_t H0 = plane0.shape[0]
    cdef Py_ssize_t W0 = plane0.shape[1]
    cdef Py_ssize_t D = coords.shape[0]
    cdef Py_ssize_t H = coords.shape[1]
    cdef Py_ssize_t W = coords.shape[2]
    
    if plane1.shape[0] != H0 or plane1.shape[1] != W0:
        raise ValueError("Planes must have same shape")
    if coords.shape[3] != 3:
        raise ValueError("coords must have shape (D, H, W, 3)")
    
    # Ensure contiguous
    if not plane0.flags['C_CONTIGUOUS']:
        plane0 = np.ascontiguousarray(plane0)
    if not plane1.flags['C_CONTIGUOUS']:
        plane1 = np.ascontiguousarray(plane1)
    if not coords.flags['C_CONTIGUOUS']:
        coords = np.ascontiguousarray(coords)
    
    cdef f64[:, ::1] p0 = plane0
    cdef f64[:, ::1] p1 = plane1
    cdef f64[:, :, :, ::1] c = coords
    
    cdef np.ndarray[f64, ndim=3] out = np.empty((D, H, W), dtype=np.float64)
    cdef f64[:, :, ::1] out_mv = out
    
    cdef Py_ssize_t d, h, w
    cdef Py_ssize_t x_lo, x_hi, y_lo, y_hi
    cdef f64 u_x, u_y, u_z
    cdef f64 x_idx_f, y_idx_f, frac_x, frac_y
    cdef f64 v00, v01, v10, v11, v0, v1
    cdef f64 H0_minus_1 = <f64>(H0 - 1)
    cdef f64 W0_minus_1 = <f64>(W0 - 1)
    
    for d in range(D):
        for h in range(H):
            for w in range(W):
                u_x = c[d, h, w, 0]
                u_y = c[d, h, w, 1]
                u_z = c[d, h, w, 2]
                
                # Map u_x, u_y to plane indices
                x_idx_f = u_x * W0_minus_1
                y_idx_f = u_y * H0_minus_1
                
                x_lo = <Py_ssize_t>floor(x_idx_f)
                y_lo = <Py_ssize_t>floor(y_idx_f)
                
                # Clamp indices to valid range
                if x_lo < 0:
                    x_lo = 0
                if x_lo >= W0 - 1:
                    x_lo = W0 - 2
                if y_lo < 0:
                    y_lo = 0
                if y_lo >= H0 - 1:
                    y_lo = H0 - 2
                
                x_hi = x_lo + 1
                y_hi = y_lo + 1
                
                frac_x = x_idx_f - <f64>x_lo
                frac_y = y_idx_f - <f64>y_lo
                
                # Clamp fractions for boundary cases
                if frac_x < 0.0:
                    frac_x = 0.0
                elif frac_x > 1.0:
                    frac_x = 1.0
                if frac_y < 0.0:
                    frac_y = 0.0
                elif frac_y > 1.0:
                    frac_y = 1.0
                
                # Bilinear interpolation in plane0
                v00 = p0[y_lo, x_lo]
                v01 = p0[y_lo, x_hi]
                v10 = p0[y_hi, x_lo]
                v11 = p0[y_hi, x_hi]
                
                v0 = v00 + frac_x * (v01 - v00)  # Top edge
                v1 = v10 + frac_x * (v11 - v10)  # Bottom edge
                v0 = v0 + frac_y * (v1 - v0)  # Vertical interpolation
                
                # Bilinear interpolation in plane1
                v00 = p1[y_lo, x_lo]
                v01 = p1[y_lo, x_hi]
                v10 = p1[y_hi, x_lo]
                v11 = p1[y_hi, x_hi]
                
                v1 = v00 + frac_x * (v01 - v00)  # Top edge
                v1 = v1 + frac_y * ((v10 + frac_x * (v11 - v10)) - v1)  # Vertical
                
                # Blend between planes using u_z
                out_mv[d, h, w] = v0 + u_z * (v1 - v0)
    
    return out


# =============================================================================
# Single channel discrete x interpolation
# =============================================================================
def lerp_between_lines_x_discrete_1ch(
    np.ndarray[f64, ndim=1] line0,
    np.ndarray[f64, ndim=1] line1,
    np.ndarray[f64, ndim=3] coords,
):
    """
    Interpolate between two lines with discrete x-sampling.
    
    Instead of linearly interpolating along x within the lines,
    this function directly samples line[floor(u_x * L)].
    
    Args:
        line0: First line, shape (L,)
        line1: Second line, shape (L,)
        coords: Coordinate grid, shape (H, W, 2)
                coords[h, w, 0] = u_x (0-1, sampled discretely)
                coords[h, w, 1] = u_y (blend factor, 0-1)
    
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
    
    # Ensure contiguous
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
    
    cdef Py_ssize_t h, w, idx
    cdef f64 u_x, u_y, idx_f
    cdef f64 v0, v1
    cdef f64 L_minus_1 = <f64>(L - 1)
    
    for h in range(H):
        for w in range(W):
            u_x = c[h, w, 0]
            u_y = c[h, w, 1]
            
            # Map u_x to discrete line index
            idx_f = u_x * L_minus_1
            idx = <Py_ssize_t>floor(idx_f + 0.5)  # Round to nearest
            
            # Clamp index to valid range
            if idx < 0:
                idx = 0
            if idx >= L:
                idx = L - 1
            
            # Sample both lines at the discrete index
            v0 = l0[idx]
            v1 = l1[idx]
            
            # Blend between lines using u_y
            out_mv[h, w] = v0 + u_y * (v1 - v0)
    
    return out
