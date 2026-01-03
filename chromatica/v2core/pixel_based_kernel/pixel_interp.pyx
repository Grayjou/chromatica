# chromatica/v2core/interp_pixel.pxd
"""Pointer-based pixel kernels for brick dispatch."""

from ..interp_utils cimport (
    f64, i32,
    BorderResult,
    process_border_2d,
    compute_interp_idx,
)
from ..interp_hue.interp_hue_utils cimport (
    lerp_hue_single,
    HueDirection,
    HUE_SHORTEST,
)

# =============================================================================
# Lines Interpolation - Pointer Based
# =============================================================================

cdef inline f64 interp_lines_pixel_ch(
    const f64* l0,          # top_line data pointer
    const f64* l1,          # bottom_line data pointer  
    f64 u_x,
    f64 u_y,
    Py_ssize_t L,           # line length
    Py_ssize_t C,           # num channels (stride)
    Py_ssize_t ch,          # which channel
) noexcept nogil:
    """Single channel bilinear interpolation between two lines."""
    cdef f64 frac
    cdef Py_ssize_t idx_lo = compute_interp_idx(u_x, L, &frac)
    cdef Py_ssize_t idx_hi = idx_lo + 1
    
    # Pointer arithmetic: line[idx, ch] = line[idx * C + ch]
    cdef f64 v0 = l0[idx_lo * C + ch] + frac * (l0[idx_hi * C + ch] - l0[idx_lo * C + ch])
    cdef f64 v1 = l1[idx_lo * C + ch] + frac * (l1[idx_hi * C + ch] - l1[idx_lo * C + ch])
    
    return v0 + u_y * (v1 - v0)


cdef inline f64 interp_lines_pixel_ch_discrete(
    const f64* l0,
    const f64* l1,
    f64 u_x,
    f64 u_y,
    Py_ssize_t L,
    Py_ssize_t C,
    Py_ssize_t ch,
) noexcept nogil:
    """Discrete x-sampling (nearest neighbor in X)."""
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f = u_x * L_minus_1
    cdef Py_ssize_t idx = <Py_ssize_t>(idx_f + 0.5)
    
    if idx < 0:
        idx = 0
    elif idx >= L:
        idx = L - 1
    
    return l0[idx * C + ch] + u_y * (l1[idx * C + ch] - l0[idx * C + ch])


cdef inline f64 interp_lines_pixel_hue(
    const f64* l0,
    const f64* l1,
    f64 u_x,
    f64 u_y,
    Py_ssize_t L,
    Py_ssize_t C,
    Py_ssize_t ch,
    int mode_x,
    int mode_y,
) noexcept nogil:
    """Hue-aware bilinear interpolation."""
    cdef f64 frac
    cdef Py_ssize_t idx_lo = compute_interp_idx(u_x, L, &frac)
    cdef Py_ssize_t idx_hi = idx_lo + 1
    
    # Hue lerp along X for both lines
    cdef f64 v0 = lerp_hue_single(l0[idx_lo * C + ch], l0[idx_hi * C + ch], frac, mode_x)
    cdef f64 v1 = lerp_hue_single(l1[idx_lo * C + ch], l1[idx_hi * C + ch], frac, mode_x)
    
    # Hue lerp along Y
    return lerp_hue_single(v0, v1, u_y, mode_y)


cdef inline f64 interp_lines_pixel_hue_discrete(
    const f64* l0,
    const f64* l1,
    f64 u_x,
    f64 u_y,
    Py_ssize_t L,
    Py_ssize_t C,
    Py_ssize_t ch,
    int mode_y,
) noexcept nogil:
    """Discrete X, hue-aware Y."""
    cdef f64 L_minus_1 = <f64>(L - 1)
    cdef f64 idx_f = u_x * L_minus_1
    cdef Py_ssize_t idx = <Py_ssize_t>(idx_f + 0.5)
    
    if idx < 0:
        idx = 0
    elif idx >= L:
        idx = L - 1
    
    return lerp_hue_single(l0[idx * C + ch], l1[idx * C + ch], u_y, mode_y)


# =============================================================================
# Corners Interpolation - Pointer Based
# =============================================================================

cdef inline f64 interp_corners_pixel_ch(
    const f64* corners,     # [TL, TR, BL, BR] packed, each has C channels
    f64 u_x,
    f64 u_y,
    Py_ssize_t C,
    Py_ssize_t ch,
) noexcept nogil:
    """Bilinear from 4 corners."""
    # corners layout: [c0_tl, c1_tl, ..., c0_tr, c1_tr, ..., c0_bl, ...]
    cdef f64 tl = corners[0 * C + ch]
    cdef f64 tr = corners[1 * C + ch]
    cdef f64 bl = corners[2 * C + ch]
    cdef f64 br = corners[3 * C + ch]
    
    cdef f64 top = tl + (tr - tl) * u_x
    cdef f64 bot = bl + (br - bl) * u_x
    return top + (bot - top) * u_y


cdef inline f64 interp_corners_pixel_hue(
    const f64* corners,
    f64 u_x,
    f64 u_y,
    Py_ssize_t C,
    Py_ssize_t ch,
    int mode_x,
    int mode_y,
) noexcept nogil:
    """Hue-aware bilinear from 4 corners."""
    cdef f64 tl = corners[0 * C + ch]
    cdef f64 tr = corners[1 * C + ch]
    cdef f64 bl = corners[2 * C + ch]
    cdef f64 br = corners[3 * C + ch]
    
    cdef f64 top = lerp_hue_single(tl, tr, u_x, mode_x)
    cdef f64 bot = lerp_hue_single(bl, br, u_x, mode_x)
    return lerp_hue_single(top, bot, u_y, mode_y)


# =============================================================================
# Full Pixel with Border Handling
# =============================================================================

cdef inline f64 interp_lines_pixel_full(
    const f64* l0,
    const f64* l1,
    f64 u_x,
    f64 u_y,
    Py_ssize_t L,
    Py_ssize_t C,
    Py_ssize_t ch,
    bint x_discrete,
    bint is_hue,
    int hue_mode_x,
    int hue_mode_y,
    int border_mode,
    f64 border_constant,
    f64 border_feathering,
    i32 distance_mode,
) noexcept nogil:
    """Complete single-pixel interpolation with all features."""
    
    # Process border
    cdef BorderResult border = process_border_2d(
        u_x, u_y, border_mode, border_feathering, distance_mode
    )
    
    if border.use_border_directly:
        return border_constant
    
    # Interpolate
    cdef f64 edge_val
    
    if x_discrete:
        if is_hue:
            edge_val = interp_lines_pixel_hue_discrete(
                l0, l1, border.u_x_final, border.u_y_final, L, C, ch, hue_mode_y
            )
        else:
            edge_val = interp_lines_pixel_ch_discrete(
                l0, l1, border.u_x_final, border.u_y_final, L, C, ch
            )
    else:
        if is_hue:
            edge_val = interp_lines_pixel_hue(
                l0, l1, border.u_x_final, border.u_y_final, L, C, ch,
                hue_mode_x, hue_mode_y
            )
        else:
            edge_val = interp_lines_pixel_ch(
                l0, l1, border.u_x_final, border.u_y_final, L, C, ch
            )
    
    # Blend if feathering
    if border.blend_factor > 0.0:
        if is_hue:
            return lerp_hue_single(edge_val, border_constant, border.blend_factor, hue_mode_y)
        else:
            return edge_val + border.blend_factor * (border_constant - edge_val)
    
    return edge_val


cdef inline f64 interp_corners_pixel_full(
    const f64* corners,
    f64 u_x,
    f64 u_y,
    Py_ssize_t C,
    Py_ssize_t ch,
    bint is_hue,
    int hue_mode_x,
    int hue_mode_y,
    int border_mode,
    f64 border_constant,
    f64 border_feathering,
    i32 distance_mode,
) noexcept nogil:
    """Complete single-pixel corner interpolation."""
    
    cdef BorderResult border = process_border_2d(
        u_x, u_y, border_mode, border_feathering, distance_mode
    )
    
    if border.use_border_directly:
        return border_constant
    
    cdef f64 edge_val
    if is_hue:
        edge_val = interp_corners_pixel_hue(
            corners, border.u_x_final, border.u_y_final, C, ch,
            hue_mode_x, hue_mode_y
        )
    else:
        edge_val = interp_corners_pixel_ch(
            corners, border.u_x_final, border.u_y_final, C, ch
        )
    
    if border.blend_factor > 0.0:
        if is_hue:
            return lerp_hue_single(edge_val, border_constant, border.blend_factor, hue_mode_y)
        else:
            return edge_val + border.blend_factor * (border_constant - edge_val)
    
    return edge_val