#!/bin/bash
# Kaggle Environment Setup - Quick Install Script
# Researched and verified for PyTorch 2.5.0 + CUDA 12.1
# Run this in a Kaggle notebook cell with: !bash QUICK_INSTALL.sh

set -e  # Exit on error

echo "=========================================="
echo "Kaggle ML Environment Setup"
echo "=========================================="

echo ""
echo "[1/5] Installing core dependencies..."
pip install -q "numpy>=1.26.4,<2.0.0" "pyarrow>=21.0.0" "scikit-learn>=1.5.0" "matplotlib>=3.8.0"

echo "[2/5] Installing constrained dependencies..."
pip install -q "pydantic>=2.0,<2.12" "protobuf>=3.20.3,<6.0.0" "rich>=12.4.4,<14.0.0" "gymnasium>=1.0.0"

echo "[3/5] Installing vision libraries..."
pip install -q --no-deps timm==1.0.12
pip install -q albumentations==1.4.22 einops==0.8.0

echo "[4/5] Installing PyTorch Geometric..."
pip install -q --no-deps torch-geometric==2.6.1
pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

echo "[5/5] Installing experiment tracking..."
pip install -q wandb==0.19.1

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."

python3 << 'EOF'
import sys
import torch
import numpy as np
import timm
import albumentations as A
import torch_geometric
import wandb

print(f"✓ Python: {sys.version.split()[0]}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
print(f"✓ NumPy: {np.__version__}")
print(f"✓ timm: {timm.__version__}")
print(f"✓ Albumentations: {A.__version__}")
print(f"✓ PyG: {torch_geometric.__version__}")
print(f"✓ W&B: {wandb.__version__}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF

echo ""
echo "=========================================="
echo "🚀 Ready to train!"
echo "=========================================="

