#!/usr/bin/env python3
"""
Test script to verify the environment fix works correctly.
This simulates the exact import sequence that was failing.
"""

import sys
import subprocess

print("="*80)
print("TESTING ENVIRONMENT FIX FOR NUMPY/SCIPY/ALBUMENTATIONS")
print("="*80)
print()

# Step 1: Check current NumPy version
print("Step 1: Checking NumPy version...")
try:
    import numpy as np
    numpy_version = np.__version__
    print(f"  ✓ NumPy version: {numpy_version}")
    
    if numpy_version.startswith('2.'):
        print(f"  ⚠️  WARNING: NumPy 2.x detected! This will cause issues.")
        print(f"  ⚠️  Installing NumPy <2.0...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "numpy<2.0"])
        print(f"  ✓ NumPy downgraded successfully")
        # Reimport to get new version
        import importlib
        importlib.reload(np)
        print(f"  ✓ New NumPy version: {np.__version__}")
    else:
        print(f"  ✓ NumPy version is compatible (<2.0)")
except ImportError as e:
    print(f"  ✗ NumPy not installed: {e}")
    sys.exit(1)

print()

# Step 2: Test scipy import (this was failing)
print("Step 2: Testing scipy import...")
try:
    import scipy
    from scipy import special
    from scipy.ndimage import gaussian_filter
    print(f"  ✓ scipy version: {scipy.__version__}")
    print(f"  ✓ scipy.special imported successfully")
    print(f"  ✓ scipy.ndimage imported successfully")
except ImportError as e:
    print(f"  ✗ scipy import failed: {e}")
    print(f"  ⚠️  Installing compatible scipy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy<1.15.0"])
    print(f"  ✓ scipy installed")
    import scipy
    print(f"  ✓ scipy version: {scipy.__version__}")

print()

# Step 3: Test albumentations import (this depends on scipy)
print("Step 3: Testing albumentations import...")
try:
    import albumentations as A
    print(f"  ✓ albumentations version: {A.__version__}")
    
    # Test creating a simple transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"  ✓ albumentations transforms work correctly")
except ImportError as e:
    print(f"  ✗ albumentations import failed: {e}")
    print(f"  ⚠️  Installing albumentations...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "albumentations==1.4.22"])
    print(f"  ✓ albumentations installed")
    import albumentations as A
    print(f"  ✓ albumentations version: {A.__version__}")

print()

# Step 4: Test other critical imports
print("Step 4: Testing other critical imports...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  ✗ PyTorch import failed: {e}")

try:
    import timm
    print(f"  ✓ timm version: {timm.__version__}")
except ImportError as e:
    print(f"  ⚠️  timm not installed (optional)")

try:
    import einops
    print(f"  ✓ einops version: {einops.__version__}")
except ImportError as e:
    print(f"  ⚠️  einops not installed (optional)")

print()
print("="*80)
print("✅ ENVIRONMENT FIX VERIFICATION COMPLETE")
print("="*80)
print()
print("Summary:")
print(f"  ✓ NumPy: {np.__version__} (must be <2.0)")
print(f"  ✓ SciPy: {scipy.__version__}")
print(f"  ✓ Albumentations: {A.__version__}")
print()

if numpy_version.startswith('2.'):
    print("⚠️  WARNING: NumPy 2.x was detected and should be downgraded")
    print("   Run: pip install --force-reinstall 'numpy<2.0'")
else:
    print("✅ All critical packages are compatible!")
    print("✅ The ImportError should be fixed!")
    print()
    print("You can now run the training notebook without errors.")

print("="*80)

