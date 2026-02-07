#!/usr/bin/env python3
"""
Fix NumPy in Jupyter's Python Environment
This script MUST be run from within Jupyter to fix the correct Python environment
"""

import sys
import subprocess

print("="*80)
print("NUMPY FIX FOR JUPYTER")
print("="*80)
print()

# Check current NumPy version
try:
    import numpy as np
    current_version = np.__version__
    print(f"Current NumPy version: {current_version}")
except ImportError:
    print("NumPy not installed!")
    current_version = None

print(f"Python executable: {sys.executable}")
print()

# Check if NumPy needs fixing
if current_version and current_version.startswith("2."):
    print("❌ NumPy 2.x detected - INCOMPATIBLE with PyTorch 2.x")
    print("Installing NumPy 1.26.4...")
    print()
    
    # Uninstall current NumPy first
    print("Step 1: Uninstalling NumPy 2.x...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "numpy"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    # Install NumPy 1.26.4
    print("Step 2: Installing NumPy 1.26.4...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy==1.26.4"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    if result.returncode == 0:
        print()
        print("="*80)
        print("✅ NumPy 1.26.4 INSTALLED SUCCESSFULLY!")
        print("="*80)
        print()
        print("⚠️  CRITICAL: You MUST restart the Jupyter kernel NOW!")
        print()
        print("Steps:")
        print("  1. Kernel → Restart Kernel")
        print("  2. Run Cell 4 (imports)")
        print("  3. Run Cell 15 (training)")
        print()
        print("After restart, you should see NO NumPy warnings!")
        print("="*80)
    else:
        print()
        print("❌ Installation failed!")
        print(result.stderr)
elif current_version:
    print(f"✅ NumPy {current_version} is compatible!")
    print("No action needed.")
else:
    print("❌ NumPy not found!")
    print("Installing NumPy 1.26.4...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy==1.26.4"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

