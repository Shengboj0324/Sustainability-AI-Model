#!/bin/bash

# Installation script for Python 3.9 on MacBook
# This script installs all dependencies with proper version constraints

echo "=========================================="
echo "Installing Dependencies for Python 3.9"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip
echo ""

# Install dependencies from requirements file
if [ -f "requirements_py39.txt" ]; then
    echo "Installing from requirements_py39.txt..."
    python3 -m pip install -r requirements_py39.txt
    echo ""
else
    echo "Installing packages individually..."
    
    # Core packages
    python3 -m pip install "numpy>=1.19.0,<2.0"
    python3 -m pip install "scipy>=1.7.0,<1.15.0"
    python3 -m pip install "pandas>=1.3.0"
    python3 -m pip install "scikit-learn>=1.0.0"
    
    # Image processing
    python3 -m pip install "Pillow>=8.0.0"
    
    # Deep learning
    python3 -m pip install torch torchvision
    python3 -m pip install "timm>=0.9.0"
    
    # Augmentation
    python3 -m pip install "albumentations>=1.3.0"
    python3 -m pip install "einops>=0.6.0"
    
    # Visualization
    python3 -m pip install "matplotlib>=3.4.0"
    python3 -m pip install "seaborn>=0.11.0"
    python3 -m pip install "tqdm>=4.62.0"
    
    # Experiment tracking
    python3 -m pip install "wandb>=0.15.0"
    
    # Kaggle API
    python3 -m pip install kaggle
    
    # PyTorch Geometric (may fail, that's OK)
    echo "Installing PyTorch Geometric (optional)..."
    python3 -m pip install torch-geometric || echo "PyTorch Geometric installation failed (optional)"
    
    echo ""
fi

echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: source setup_kaggle.sh"
echo "2. Run: python3 test_kaggle_setup.py"
echo "3. Open the training notebook"
echo ""

