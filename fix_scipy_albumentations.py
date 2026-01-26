#!/usr/bin/env python3
"""
Fix scipy and albumentations installation issues.
This fixes: ModuleNotFoundError: No module named 'scipy._lib'
"""

import subprocess
import sys

print("="*80)
print("🔧 FIXING SCIPY AND ALBUMENTATIONS")
print("="*80)
print()

# Step 1: Uninstall broken packages
print("Step 1: Uninstalling potentially broken packages...")
packages_to_remove = ["scipy", "albumentations", "opencv-python", "opencv-python-headless"]
for pkg in packages_to_remove:
    print(f"  Uninstalling {pkg}...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
        capture_output=True,
        check=False
    )
print("  ✅ Uninstalled")
print()

# Step 2: Install scipy first (albumentations depends on it)
print("Step 2: Installing scipy...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "scipy>=1.7.0,<1.15.0"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("  ✅ scipy installed")
else:
    print("  ⚠️  scipy installation had issues:")
    print(result.stderr)
print()

# Step 3: Install opencv-python (albumentations depends on it)
print("Step 3: Installing opencv-python...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless>=4.5.0"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("  ✅ opencv-python-headless installed")
else:
    print("  ⚠️  opencv installation had issues:")
    print(result.stderr)
print()

# Step 4: Install albumentations
print("Step 4: Installing albumentations...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "albumentations>=1.3.0"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("  ✅ albumentations installed")
else:
    print("  ⚠️  albumentations installation had issues:")
    print(result.stderr)
print()

# Step 5: Verify installation
print("Step 5: Verifying installation...")
print()

try:
    import scipy
    print(f"  ✅ scipy {scipy.__version__}")
except Exception as e:
    print(f"  ❌ scipy import failed: {e}")

try:
    import cv2
    print(f"  ✅ opencv {cv2.__version__}")
except Exception as e:
    print(f"  ❌ opencv import failed: {e}")

try:
    import albumentations
    print(f"  ✅ albumentations {albumentations.__version__}")
except Exception as e:
    print(f"  ❌ albumentations import failed: {e}")

print()
print("="*80)
print("✅ FIX COMPLETE!")
print("="*80)
print()
print("Next steps:")
print("1. Restart your Jupyter kernel")
print("2. Run all cells in the notebook")
print()

