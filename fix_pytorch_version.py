#!/usr/bin/env python3
"""
Quick fix for PyTorch/TorchVision version incompatibility.
This fixes the AttributeError: type object 'torch._C.Tag' has no attribute 'needs_exact_strides'
"""

import subprocess
import sys

print("="*80)
print("🔧 FIXING PYTORCH/TORCHVISION VERSION INCOMPATIBILITY")
print("="*80)
print()

# Check current versions
print("Checking current versions...")
try:
    import torch
    import torchvision
    print(f"  Current PyTorch: {torch.__version__}")
    print(f"  Current torchvision: {torchvision.__version__}")
    print()
except Exception as e:
    print(f"  Error checking versions: {e}")
    print()

# Uninstall existing versions
print("Uninstalling existing PyTorch and torchvision...")
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"],
    check=False
)
print("  ✅ Uninstalled")
print()

# Install compatible versions
print("Installing compatible PyTorch 2.0.1 and torchvision 0.15.2...")
print("This may take a few minutes...")
print()

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "torch==2.0.1", "torchvision==0.15.2"],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("  ✅ Installation successful!")
else:
    print("  ⚠️  Installation had issues:")
    print(result.stderr)

print()

# Verify installation
print("Verifying installation...")
try:
    import torch
    import torchvision
    print(f"  ✅ PyTorch: {torch.__version__}")
    print(f"  ✅ torchvision: {torchvision.__version__}")
    print()
    
    # Test the import that was failing
    print("Testing torchvision.transforms import...")
    import torchvision.transforms as transforms
    print("  ✅ Import successful!")
    print()
    
    print("="*80)
    print("✅ FIX COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Restart your Jupyter kernel")
    print("2. Run all cells in the notebook")
    print()
    
except Exception as e:
    print(f"  ❌ Verification failed: {e}")
    print()
    print("Please try manually:")
    print("  pip uninstall torch torchvision")
    print("  pip install torch==2.0.1 torchvision==0.15.2")

