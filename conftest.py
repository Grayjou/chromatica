"""
Root conftest.py - Sets up Python path for tests.

This conftest is loaded by pytest before any test collection begins.
"""
import sys
import os

# Get the project root (where this conftest.py lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ensure the project root is the first path, filtering out any incorrect chromatica paths
# This prevents import conflicts between /home/runner/work/chromatica and /home/runner/work/chromatica/chromatica
_filtered_paths = []
for p in sys.path:
    # Keep if it's not a chromatica path OR if it's exactly the project root
    if 'chromatica' not in p or p == PROJECT_ROOT:
        _filtered_paths.append(p)
sys.path = _filtered_paths

# Add project root at the start if not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Debug: print the path setup
print(f"[conftest] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[conftest] sys.path[:3]: {sys.path[:3]}")
