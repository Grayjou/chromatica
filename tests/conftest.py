import sys
import os

# Print to debug
print(f"conftest.py loaded, sys.path: {sys.path[:5]}")

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path")
