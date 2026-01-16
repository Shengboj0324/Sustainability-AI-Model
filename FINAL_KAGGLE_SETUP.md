# Final Kaggle Setup Guide - Sustainability AI Model

## Step 1: Add Datasets to Kaggle Notebook

Add these 8 datasets to your Kaggle notebook by clicking "Add Data" and searching for each:

1. **recyclable-and-household-waste-classification** (alistairking)
   https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification

2. **garbage-classification** (mostafaabla)
   https://www.kaggle.com/datasets/mostafaabla/garbage-classification

3. **waste-classification-data** (techsash)
   https://www.kaggle.com/datasets/techsash/waste-classification-data

4. **garbage-classification-v2** (sumn2u)
   https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

5. **garbage-classification** (asdasdasasdas)
   https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

6. **garbage-dataset-classification** (zlatan599)
   https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification

7. **warp-waste-recycling-plant-dataset** (parohod)
   https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset

8. **multi-class-garbage-classification-dataset** (vishallazrus)
   https://www.kaggle.com/datasets/vishallazrus/multi-class-garbage-classification-dataset

## Step 2: Enable GPU

Settings → Accelerator → GPU T4 x2

## Step 3: Install Dependencies

Run this in the first cell of your notebook:

```python
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm==1.0.12"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-scatter", "torch-sparse", "-f", "https://data.pyg.org/whl/torch-2.5.0+cu121.html"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "albumentations==1.4.22"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb==0.19.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "einops==0.8.0"])
```

Or as a one-liner:

```bash
!pip install -q --no-deps timm==1.0.12 && pip install -q --no-deps torch-geometric==2.6.1 && pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html && pip install -q albumentations==1.4.22 einops==0.8.0 wandb==0.19.1
```

## Step 4: Verify Installation

```python
import torch
import timm
import albumentations as A
import torch_geometric
import wandb

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"timm: {timm.__version__}")
print(f"Albumentations: {A.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"W&B: {wandb.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Step 5: Run Training

Upload the `Sustainability_AI_Model_Training.ipynb` notebook and run all cells.

## Dataset Statistics

- Total Images: 60,000+
- Total Size: ~2 GB
- Classes: 30 unified categories
- Sources: 8 diverse datasets

## Expected Training Time

- Vision Model (EVA-02 Large): ~4-6 hours on T4 GPU
- GNN Model: ~10-15 minutes
- Total: ~5-7 hours

## Model Outputs

- `best_vision_eva02_lake.pth` - Vision model weights
- `best_gnn_gatv2.pth` - GNN model weights

## Notes

- All datasets use REAL data from verified Kaggle sources
- No synthetic or generated data
- Trained on Kaggle T4 GPU
- All dependencies are compatible with Kaggle environment
- No comments in code as requested

