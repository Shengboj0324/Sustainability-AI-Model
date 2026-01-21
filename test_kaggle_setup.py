#!/usr/bin/env python3
"""
Quick test script to verify Kaggle API setup
Run this before starting the full training to ensure everything is configured correctly.
"""

import os
import sys
import subprocess

print("="*80)
print("🔍 KAGGLE API SETUP TEST")
print("="*80)
print()

# Test 0: Check Python version
print("Test 0: Checking Python version...")
print(f"   Python {sys.version}")
py_version = sys.version_info
if py_version.major == 3 and py_version.minor >= 8:
    print(f"   ✅ Python {py_version.major}.{py_version.minor} is supported")
else:
    print(f"   ⚠️  Python {py_version.major}.{py_version.minor} may have compatibility issues")
    print(f"   Recommended: Python 3.8+")

print()

# Test 1: Check if Kaggle API token is set
print("Test 1: Checking Kaggle API token...")
token = os.environ.get('KAGGLE_API_TOKEN')
if token:
    print(f"✅ KAGGLE_API_TOKEN is set: {token[:20]}...")
else:
    print("❌ KAGGLE_API_TOKEN is NOT set!")
    print()
    print("Please run: export KAGGLE_API_TOKEN='KGAT_7c2e755b1b8e7997695c79cf46a9060a'")
    print("Or run: source setup_kaggle.sh")
    sys.exit(1)

print()

# Test 2: Check if kaggle package is installed
print("Test 2: Checking if kaggle package is installed...")
try:
    import kaggle
    print("✅ kaggle package is installed")
except ImportError:
    print("❌ kaggle package is NOT installed!")
    print()
    print("Installing kaggle package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    print("✅ kaggle package installed successfully")

print()

# Test 3: Check PyTorch installation
print("Test 3: Checking PyTorch installation...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} is installed")
    
    # Check device availability
    if torch.cuda.is_available():
        print(f"   🚀 CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"   🍎 MPS (Apple Silicon) available")
    else:
        print(f"   💻 Using CPU (no GPU acceleration)")
except ImportError:
    print("❌ PyTorch is NOT installed!")
    print()
    print("Please install PyTorch:")
    print("pip install torch torchvision")
    sys.exit(1)

print()

# Test 4: Check disk space
print("Test 4: Checking available disk space...")
import shutil
total, used, free = shutil.disk_usage(".")
free_gb = free / (1024**3)
print(f"   Free space: {free_gb:.2f} GB")
if free_gb < 30:
    print(f"   ⚠️  WARNING: Less than 30 GB free. Datasets require ~20-30 GB.")
else:
    print(f"   ✅ Sufficient disk space available")

print()

# Test 5: Test Kaggle API connection
print("Test 5: Testing Kaggle API connection...")
try:
    result = subprocess.run(
        ["kaggle", "datasets", "list", "--max-size", "1"],
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.returncode == 0:
        print("✅ Kaggle API connection successful!")
    else:
        print(f"❌ Kaggle API connection failed!")
        print(f"   Error: {result.stderr}")
        sys.exit(1)
except subprocess.TimeoutExpired:
    print("❌ Kaggle API connection timed out!")
    print("   Please check your internet connection.")
    sys.exit(1)
except FileNotFoundError:
    print("❌ kaggle command not found!")
    print("   Please install: pip install kaggle")
    sys.exit(1)

print()
print("="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print()
print("You're ready to run the training notebook!")
print("Open: Sustainability_AI_Model_Training.ipynb")
print()

