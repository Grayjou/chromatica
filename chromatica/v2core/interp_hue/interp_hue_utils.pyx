# interp_hue_utils.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

"""
Hue interpolation utilities with cyclical color space support.

Modernized to support feathering, distance modes, and advanced border handling.
"""

