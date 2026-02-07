#!/bin/bash
# Environment Fix Script for Sustainability AI Model Training
# Fixes NumPy version incompatibility and validates setup

set -e  # Exit on error

echo "================================================================================"
echo "SUSTAINABILITY AI MODEL - ENVIRONMENT FIX"
echo "================================================================================"
echo ""

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ ERROR: Python not found!"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Fix NumPy version
echo "================================================================================"
echo "STEP 1: Fixing NumPy Version"
echo "================================================================================"
echo ""
echo "Current NumPy version:"
$PYTHON_CMD -c "import numpy; print(f'  NumPy {numpy.__version__}')" 2>/dev/null || echo "  NumPy not installed"
echo ""

echo "Installing NumPy < 2.0 (compatible with PyTorch 2.x)..."
$PYTHON_CMD -m pip install --upgrade "numpy<2.0" --quiet

echo "✅ NumPy fixed!"
echo "New NumPy version:"
$PYTHON_CMD -c "import numpy; print(f'  NumPy {numpy.__version__}')"
echo ""

# Verify PyTorch
echo "================================================================================"
echo "STEP 2: Verifying PyTorch Installation"
echo "================================================================================"
echo ""

$PYTHON_CMD -c "
import torch
print(f'✅ PyTorch {torch.__version__}')

if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon) available')
else:
    print('⚠️  No GPU acceleration (CPU only)')
"
echo ""

# Verify timm
echo "================================================================================"
echo "STEP 3: Verifying timm Installation"
echo "================================================================================"
echo ""

$PYTHON_CMD -c "
import timm
print(f'✅ timm {timm.__version__}')

# Check if model is available
try:
    model = timm.create_model('eva02_base_patch14_224', pretrained=False, num_classes=30)
    print('✅ EVA02 Base model available')
    del model
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
echo ""

# Run diagnostics
echo "================================================================================"
echo "STEP 4: Running Training Diagnostics"
echo "================================================================================"
echo ""

if [ -f "training_diagnostics.py" ]; then
    $PYTHON_CMD training_diagnostics.py
    DIAG_RESULT=$?
    
    if [ $DIAG_RESULT -eq 0 ]; then
        echo ""
        echo "================================================================================"
        echo "✅ ALL CHECKS PASSED - READY TO TRAIN!"
        echo "================================================================================"
        echo ""
        echo "Next steps:"
        echo "  1. Open Jupyter notebook: Sustainability_AI_Model_Training.ipynb"
        echo "  2. Restart kernel: Kernel → Restart Kernel"
        echo "  3. Run Cell 4: Import statements and function definitions"
        echo "  4. Run Cell 15: Start training"
        echo ""
        echo "Expected behavior:"
        echo "  ✅ Transform validation passed"
        echo "  ✅ Pre-training sanity check passed"
        echo "  ✅ Training starts without errors"
        echo "  ✅ Loss decreases over epochs"
        echo ""
    else
        echo ""
        echo "================================================================================"
        echo "⚠️  SOME CHECKS FAILED"
        echo "================================================================================"
        echo ""
        echo "Please review the diagnostic output above and fix any issues."
        echo ""
    fi
else
    echo "⚠️  training_diagnostics.py not found, skipping diagnostics"
    echo ""
    echo "================================================================================"
    echo "✅ ENVIRONMENT FIXED"
    echo "================================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Open Jupyter notebook: Sustainability_AI_Model_Training.ipynb"
    echo "  2. Restart kernel: Kernel → Restart Kernel"
    echo "  3. Run Cell 4: Import statements and function definitions"
    echo "  4. Run Cell 15: Start training"
    echo ""
fi

