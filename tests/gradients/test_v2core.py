from ...chromatica.gradients.v2core import multival1d_lerp
import numpy as np
from boundednumbers import BoundType

def test_multival1d_lerp_no_bounds():
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 20.0])
    coeffs = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]

    result = multival1d_lerp(start, end, coeffs, bound_types=BoundType.IGNORE)

    expected = np.array([
        [0.0, 0.0],
        [5.0, 10.0],
        [10.0, 20.0]
    ])

    assert np.allclose(result, expected)

def test_lerp_1d():
    start = np.array([0.0])
    end = np.array([10.0])
    coeffs = [np.array([-1.0, 0.0, 0.5, 1.0, 2.0])]

    result = multival1d_lerp(start, end, coeffs, bound_types=BoundType.CLAMP)

    expected = np.array([
        [0.0],
        [0.0],
        [5.0],
        [10.0],
        [10.0]
    ])

    assert np.allclose(result, expected)

def test_lerp_mixed_bounds():
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 20.0])
    coeffs = [np.array([-0.5, 0.0, 0.5, 1.0, 1.5]), np.array([-1.0, 0.0, 0.5, 1.0, 2.0])]
    bound_types = [BoundType.CLAMP, BoundType.BOUNCE]

    result = multival1d_lerp(start, end, coeffs, bound_types=bound_types)

    expected = np.array([
        [0.0, 20.0],   # coeffs[0] clamped to 0.0, coeffs[1] bounces to 20.0
        [0.0, 0.0],    # coeffs[0] clamped to 0.0, coeffs[1] bounces to 0.0
        [5.0, 10.0],   # coeffs[0] at 0.5, coeffs[1] at 10.0
        [10.0, 20.0],  # coeffs[0] at 1.0, coeffs[1] at 20.0
        [10.0, 0.0]   # coeffs[0] clamped to 1.0, coeffs[1] bounces to 0.0
    ])

    assert np.allclose(result, expected)

def test_red_to_blue_lerp():
    start = np.array([1.0, 0.0, 0.0])  # Red
    end = np.array([0.0, 0.0, 1.0])    # Blue
    coeffs = [np.array([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in range(3)]

    result = multival1d_lerp(start, end, coeffs, bound_types=BoundType.CLAMP)

    expected = np.array([
        [1.0, 0.0, 0.0],      # Red
        [0.75, 0.0, 0.25],
        [0.5, 0.0, 0.5],
        [0.25, 0.0, 0.75],
        [0.0, 0.0, 1.0]       # Blue
    ])

    assert np.allclose(result, expected)

def test_4len_lerp():
    start = np.array([[0.0, 0.0, 1.0, 1.0]])
    end = np.array([[1.0, 1.0, 2.0, 2.0]])
    coeffs = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]),np.array([0.0, 0.5, 1.0]),np.array([0.0, 0.5, 1.0]),]
    print(np.array(coeffs).shape, start.shape, end.shape)
    result = multival1d_lerp(start, end, coeffs, bound_types=BoundType.CLAMP)
    expected = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.5, 0.5, 1.5, 1.5],
        [1.0, 1.0, 2.0, 2.0]
    ])

    assert np.allclose(result, expected)

from ...chromatica.gradients.v2core import single_channel_multidim_lerp

def test_1d_multidim_lerp_equivalent():
    start = np.array([0.0])
    end = np.array([10.0])
    coeffs = [np.array([-1.0, 0.0, 0.5, 1.0, 2.0])]

    result_both = single_channel_multidim_lerp(start, end, np.stack(coeffs, axis=-1), bound_type=BoundType.CLAMP)
    result_multi = multival1d_lerp(start, end, coeffs)
    print("result_both:", result_both)
    print("result_multi:", result_multi)
    assert np.allclose(result_both, result_multi.flatten())