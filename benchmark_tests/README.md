# Benchmark Tests

This directory contains performance benchmark tests for the Chromatica library.

## Purpose

These tests are separate from the main test suite because:
- They focus on performance, not correctness
- They can take longer to run
- Results may vary based on hardware and system load
- They are not part of CI/CD pipelines

## Running Benchmarks

### Run all benchmarks:
```bash
python -m pytest benchmark_tests/ -v -s
```

### Run specific benchmark:
```bash
python -m pytest benchmark_tests/test_segment_vs_mask.py -v -s
```

### Run benchmark as a standalone script:
```bash
python benchmark_tests/test_segment_vs_mask.py
```

## Available Benchmarks

### test_segment_vs_mask.py
Compares performance between segment mode and mask mode in gradient sequences.

- **Transform used**: `np.sin(2*pi*t)` - causes segments to oscillate, going back and forth twice
- **Scenarios tested**:
  - RGB gradients
  - HSV gradients with hue interpolation
  - Large gradients with many color stops

The `-s` flag is recommended to see detailed output including timing information.

## Adding New Benchmarks

When adding new benchmarks:
1. Place them in this directory
2. Name test files with `test_` prefix
3. Include print statements to show timing results
4. Document the benchmark purpose in docstrings
5. Focus on relative performance comparisons, not absolute times
