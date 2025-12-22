"""
Benchmark tests comparing different interpolation methods and transforms.

Uses np.sin(2*pi*t) as a transform to cause oscillation in the gradient space.
"""
import numpy as np
import time
from typing import Callable, Tuple
from chromatica.gradients.gradient1dv2.gradient_1dv2 import Gradient1D
from chromatica.types.color_types import ColorSpace
from chromatica.types.format_type import FormatType


def benchmark_simple_gradient(
    start_color: np.ndarray,
    end_color: np.ndarray,
    color_space: ColorSpace,
    total_steps: int,
    transform: Callable[[np.ndarray], np.ndarray],
    num_runs: int = 5
) -> Tuple[float, Gradient1D]:
    """
    Benchmark simple gradient generation with given parameters.
    
    Args:
        start_color: Starting color
        end_color: Ending color
        color_space: Color space to use
        total_steps: Number of steps in the gradient
        transform: Transform function to apply
        num_runs: Number of runs to average
        
    Returns:
        Tuple of (average_time_seconds, result_gradient)
    """
    times = []
    result = None
    
    for _ in range(num_runs):
        start_time = time.time()
        
        result = Gradient1D.from_colors(
            left_color=start_color,
            right_color=end_color,
            steps=total_steps,
            color_space=color_space,
            format_type=FormatType.FLOAT,
            unit_transform=transform,
        )
        
        # Force evaluation by accessing the value
        _ = result.value
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    return avg_time, result


def sin_2pi_transform(t: np.ndarray) -> np.ndarray:
    """
    Transform using sin(2*pi*t).
    
    This causes the gradient to oscillate, going back and forth twice.
    Maps [0, 1] -> oscillates through [0, 1] twice.
    """
    # Normalize to [0, 1]
    return (np.sin(2 * np.pi * t) + 1) / 2


def linear_transform(t: np.ndarray) -> np.ndarray:
    """Linear transform (identity)."""
    return t


def test_benchmark_with_transform_vs_without():
    """
    Benchmark gradient generation with and without transform.
    """
    print("\n" + "="*70)
    print("Benchmark: With Transform vs Without Transform - RGB Gradient")
    print("="*70)
    
    start_color = np.array([1.0, 0.0, 0.0])  # Red
    end_color = np.array([0.0, 0.0, 1.0])    # Blue
    total_steps = 1000
    
    print(f"Start: Red, End: Blue")
    print(f"Total steps: {total_steps}")
    print("-" * 70)
    
    # Benchmark with sin transform
    transform_time, transform_result = benchmark_simple_gradient(
        start_color, end_color, "rgb", total_steps, sin_2pi_transform, num_runs=10
    )
    
    # Benchmark without transform
    no_transform_time, no_transform_result = benchmark_simple_gradient(
        start_color, end_color, "rgb", total_steps, linear_transform, num_runs=10
    )
    
    print(f"With sin(2*pi*t) transform: {transform_time*1000:.2f} ms")
    print(f"Without transform:          {no_transform_time*1000:.2f} ms")
    print(f"Overhead:                   {(transform_time - no_transform_time)*1000:.2f} ms ({((transform_time/no_transform_time - 1)*100):.1f}%)")
    print("="*70)
    
    # Verify results have correct shape
    assert transform_result.value.shape == (total_steps, 3)
    assert no_transform_result.value.shape == (total_steps, 3)


def test_benchmark_rgb_vs_hsv():
    """
    Benchmark RGB vs HSV gradient generation.
    """
    print("\n" + "="*70)
    print("Benchmark: RGB vs HSV Gradient")
    print("="*70)
    
    start_color_rgb = np.array([1.0, 0.0, 0.0])  # Red
    end_color_rgb = np.array([0.0, 1.0, 0.0])    # Green
    
    start_color_hsv = np.array([0.0, 1.0, 1.0])  # Red in HSV
    end_color_hsv = np.array([120.0, 1.0, 1.0])  # Green in HSV
    
    total_steps = 1000
    
    print(f"Start: Red, End: Green")
    print(f"Total steps: {total_steps}")
    print(f"Transform: sin(2*pi*t)")
    print("-" * 70)
    
    # Benchmark RGB
    rgb_time, rgb_result = benchmark_simple_gradient(
        start_color_rgb, end_color_rgb, "rgb", total_steps, sin_2pi_transform, num_runs=10
    )
    
    # Benchmark HSV
    hsv_time, hsv_result = benchmark_simple_gradient(
        start_color_hsv, end_color_hsv, "hsv", total_steps, sin_2pi_transform, num_runs=10
    )
    
    print(f"RGB gradient: {rgb_time*1000:.2f} ms")
    print(f"HSV gradient: {hsv_time*1000:.2f} ms")
    print(f"HSV overhead: {(hsv_time - rgb_time)*1000:.2f} ms ({((hsv_time/rgb_time - 1)*100):.1f}%)")
    print("="*70)
    
    # Verify results have correct shape
    assert rgb_result.value.shape == (total_steps, 3)
    assert hsv_result.value.shape == (total_steps, 3)


def test_benchmark_large_gradient():
    """
    Benchmark large gradient generation.
    """
    print("\n" + "="*70)
    print("Benchmark: Large Gradient Generation")
    print("="*70)
    
    start_color = np.array([1.0, 0.0, 0.0])  # Red
    end_color = np.array([0.0, 0.0, 1.0])    # Blue
    total_steps = 10000
    
    print(f"Start: Red, End: Blue")
    print(f"Total steps: {total_steps}")
    print(f"Transform: sin(2*pi*t)")
    print("-" * 70)
    
    # Benchmark
    gradient_time, gradient_result = benchmark_simple_gradient(
        start_color, end_color, "rgb", total_steps, sin_2pi_transform, num_runs=5
    )
    
    print(f"Generation time: {gradient_time*1000:.2f} ms")
    print(f"Per-step time:   {(gradient_time/total_steps)*1000000:.2f} µs")
    print("="*70)
    
    # Verify result
    assert gradient_result.value.shape == (total_steps, 3)


if __name__ == "__main__":
    print("\nRunning gradient performance benchmarks...")
    print("Note: These tests measure relative performance, not correctness.")
    
    test_benchmark_with_transform_vs_without()
    test_benchmark_rgb_vs_hsv()
    test_benchmark_large_gradient()
    
    print("\nAll benchmarks completed!")
