#!/usr/bin/env python3
"""
Fix Jupyter environment mismatch by installing packages in the correct Python environment.
This script detects which Python Jupyter is using and installs packages there.
"""

import sys
import subprocess
import importlib.util

print("=" * 80)
print("JUPYTER ENVIRONMENT FIX")
print("=" * 80)

# Step 1: Identify the Python interpreter
print(f"\n✓ Python executable: {sys.executable}")
print(f"✓ Python version: {sys.version}")

# Step 2: Check if scipy is installed and working
print("\n" + "=" * 80)
print("CHECKING SCIPY INSTALLATION")
print("=" * 80)

try:
    import scipy
    print(f"✓ Scipy is importable: {scipy.__version__}")
    print(f"✓ Scipy location: {scipy.__file__}")
    
    # Try to import scipy._lib
    try:
        from scipy import _lib
        print("✓ scipy._lib is accessible")
    except ImportError as e:
        print(f"✗ scipy._lib is NOT accessible: {e}")
        print("  → Scipy installation is broken!")
        
except ImportError as e:
    print(f"✗ Scipy is NOT installed: {e}")

# Step 3: Reinstall scipy and dependencies in THIS Python environment
print("\n" + "=" * 80)
print("REINSTALLING PACKAGES IN JUPYTER'S PYTHON ENVIRONMENT")
print("=" * 80)

packages_to_remove = ["scipy", "albumentations", "opencv-python", "opencv-python-headless"]

print("\n[1/4] Uninstalling potentially broken packages...")
for pkg in packages_to_remove:
    print(f"  → Uninstalling {pkg}...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

print("\n[2/4] Installing scipy (this may take 2-3 minutes)...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "scipy>=1.7.0,<1.15.0"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("  ✓ Scipy installed successfully")
else:
    print(f"  ✗ Scipy installation failed: {result.stderr}")

print("\n[3/4] Installing opencv-python-headless...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless>=4.5.0"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("  ✓ opencv-python-headless installed successfully")
else:
    print(f"  ✗ opencv-python-headless installation failed: {result.stderr}")

print("\n[4/4] Installing albumentations...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "albumentations>=1.3.0"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("  ✓ albumentations installed successfully")
else:
    print(f"  ✗ albumentations installation failed: {result.stderr}")

# Step 4: Verify installation
print("\n" + "=" * 80)
print("VERIFYING INSTALLATION")
print("=" * 80)

try:
    import scipy
    from scipy import _lib
    import albumentations
    print(f"✓ scipy {scipy.__version__} - WORKING")
    print(f"✓ scipy._lib - ACCESSIBLE")
    print(f"✓ albumentations {albumentations.__version__} - WORKING")
    print("\n" + "=" * 80)
    print("✅ ALL PACKAGES INSTALLED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Restart your Jupyter kernel: Kernel → Restart Kernel")
    print("2. Re-run all cells from the beginning")
    print("3. The import errors should be gone!")
except Exception as e:
    print(f"\n✗ Verification failed: {e}")
    print("\nPlease run this script FROM JUPYTER:")
    print("  1. Create a new cell in your notebook")
    print("  2. Run: !python3 fix_jupyter_environment.py")
    print("  3. Then restart the kernel")

