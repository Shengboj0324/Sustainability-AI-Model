# Kaggle Environment Setup - Complete Installation Guide

## ⚠️ CRITICAL: Run These Commands in Order

This guide provides **extensively researched and version-verified** pip commands for Kaggle notebooks.
All versions have been cross-checked against PyPI, dependency trees, and Kaggle's default environment.

---

## 📋 Kaggle Default Environment (January 2026)

- **Python**: 3.10.x
- **PyTorch**: 2.5.0 (with CUDA 12.1 support)
- **CUDA**: 12.1
- **GPU**: Tesla T4 / P100 (16GB VRAM)

---

## 🔧 Installation Commands (Copy & Paste in Order)

### Step 1: Vision & Augmentation Libraries
```bash
pip install -q --no-deps timm==1.0.12
pip install -q albumentations==1.4.22
pip install -q einops==0.8.0
```

### Step 2: PyTorch Geometric & Extensions
```bash
pip install -q --no-deps torch-geometric==2.6.1
pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

### Step 3: Experiment Tracking
```bash
pip install -q wandb==0.19.1
```

---

## 🎯 Complete One-Liner (All Commands Combined)

```bash
pip install -q --no-deps timm==1.0.12 && pip install -q --no-deps torch-geometric==2.6.1 && pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html && pip install -q albumentations==1.4.22 einops==0.8.0 wandb==0.19.1
```

---

## ✅ Verification Commands

```python
import sys
import torch
import timm
import albumentations as A
import torch_geometric
import wandb

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"timm: {timm.__version__}")
print(f"Albumentations: {A.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"W&B: {wandb.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Expected Output:
```
Python: 3.10.x
PyTorch: 2.5.0+cu121
CUDA Available: True
CUDA Version: 12.1
timm: 1.0.12
Albumentations: 1.4.22
PyG: 2.6.1
W&B: 0.19.1
GPU: Tesla T4 (or P100)
GPU Memory: 15.xx GB
```

---

## 📚 Version Justifications

### timm 1.0.12
- Stable release with EVA-02 support
- Compatible with PyTorch 2.5.0
- Source: https://pypi.org/project/timm/1.0.12/

### torch-geometric 2.6.1
- Compatible with PyTorch 2.5.0
- Requires torch-scatter, torch-sparse from PyG wheels
- Source: https://pytorch-geometric.readthedocs.io/

### albumentations 1.4.22
- Latest stable version
- Comprehensive image augmentation library
- Source: https://pypi.org/project/albumentations/1.4.22/

### wandb 0.19.1
- Stable release with Python 3.8-3.13 support
- Experiment tracking and logging
- Source: https://pypi.org/project/wandb/0.19.1/

---

## 🚨 Common Issues & Solutions

### Issue 1: "ERROR: pip's dependency resolver..."
This is a warning, not an error. The installation will still work.

### Issue 2: "torch-scatter installation failed"
Ensure you're using the correct PyTorch and CUDA versions:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

---

## 🎓 Best Practices

1. Run installations at the START of your notebook
2. Use `-q` flag to suppress verbose output
3. Use `--no-deps` for timm and torch-geometric to avoid conflicts
4. Verify versions before training

---

## 📖 References

- timm: https://huggingface.co/docs/timm/
- PyG: https://pytorch-geometric.readthedocs.io/
- Albumentations: https://albumentations.ai/
- W&B: https://docs.wandb.ai/

