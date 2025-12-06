import subprocess
import sys
import os
import re

def get_current_version():
    """Extract current version from pyproject.toml"""
    try:
        with open('pyproject.toml', 'r') as f:
            content = f.read()
            match = re.search(r'version\s*=\s*"(.*?)"', content)
            if match:
                return match.group(1)
            print("❌ Could not find version in pyproject.toml")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ pyproject.toml not found in current directory")
        sys.exit(1)

if __name__ == "__main__":
    version = get_current_version()
    print(f"\n🎨 Chromatica Publishing Utility (v{version})")
    print("🚀 Ready to build and publish Chromatica package")
    print("⚠ WARNING: This will publish the current version to PyPI")
    print(f"   Version: {version}\n")
    
    confirm = input("Are you sure you want to publish to PyPI? (yes/no): ").strip().lower()
    
    if confirm not in ["yes", "y"]:
        print("❌ Publishing aborted.")
        sys.exit(0)
    
    script_path = os.path.join(os.path.dirname(__file__), 'publish.ps1')
    command = ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path]
    
    try:
        subprocess.run(command, check=True)
        print("\n✅ Publish process completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Publish failed with error code {e.returncode}")
        sys.exit(e.returncode)