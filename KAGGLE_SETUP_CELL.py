"""
Kaggle Environment Setup Cell
Copy this entire cell to the TOP of your Kaggle notebook
Researched and verified for PyTorch 2.5.0 + CUDA 12.1
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a pip command and handle errors"""
    print(f"⏳ {description}...")
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"✅ {description} - Done")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        raise

print("=" * 60)
print("🚀 Kaggle ML Environment Setup")
print("=" * 60)
print()

# Step 1: Core Dependencies
run_command(
    'pip install -q "numpy>=1.26.4,<2.0.0" "pyarrow>=21.0.0" "scikit-learn>=1.5.0" "matplotlib>=3.8.0"',
    "Installing core dependencies (numpy, pyarrow, sklearn, matplotlib)"
)

# Step 2: Constrained Dependencies
run_command(
    'pip install -q "pydantic>=2.0,<2.12" "protobuf>=3.20.3,<6.0.0" "rich>=12.4.4,<14.0.0" "gymnasium>=1.0.0"',
    "Installing constrained dependencies (pydantic, protobuf, rich, gymnasium)"
)

# Step 3: Vision Libraries
run_command(
    'pip install -q --no-deps timm==1.0.12',
    "Installing timm (PyTorch Image Models)"
)

run_command(
    'pip install -q albumentations==1.4.22 einops==0.8.0',
    "Installing albumentations and einops"
)

# Step 4: PyTorch Geometric
run_command(
    'pip install -q --no-deps torch-geometric==2.6.1',
    "Installing PyTorch Geometric"
)

run_command(
    'pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html',
    "Installing PyG extensions (torch-scatter, torch-sparse)"
)

# Step 5: Experiment Tracking
run_command(
    'pip install -q wandb==0.19.1',
    "Installing Weights & Biases"
)

print()
print("=" * 60)
print("✅ Installation Complete!")
print("=" * 60)
print()

# Verification
print("🔍 Verifying installation...")
print()

try:
    import torch
    import numpy as np
    import timm
    import albumentations as A
    import torch_geometric
    import wandb
    
    print(f"✓ Python: {sys.version.split()[0]}")
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ timm: {timm.__version__}")
    print(f"✓ Albumentations: {A.__version__}")
    print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
    print(f"✓ Weights & Biases: {wandb.__version__}")
    
    print()
    print("=" * 60)
    print("🎉 All packages installed and verified successfully!")
    print("🚀 Ready to train your models!")
    print("=" * 60)
    
except ImportError as e:
    print(f"❌ Verification failed: {e}")
    print("Please check the installation logs above for errors.")
    raise

