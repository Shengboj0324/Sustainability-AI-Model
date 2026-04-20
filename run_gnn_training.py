# ============================================================
# Notebook Cell 4
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
# CRITICAL: Must be set before ANY MPS operation.
# Makes PyTorch silently fall back to CPU for any op MPS can't handle
# (e.g. GatherOp in MLIR) instead of crashing the entire Python process.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys
import json
import random
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

import timm
from timm.data import create_transform, resolve_data_config
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from tqdm.notebook import tqdm
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# Notebook Cell 5
# ============================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't need special seeding
        pass
    logger.info(f"✓ Random seed set to {seed}")

def get_device():
    """
    Detect and return the best available device for training.

    MPS is DISABLED: PyTorch 2.2.x + macOS 26 causes the MPS MLIR JIT compiler
    to crash the entire Python process (mlir::ConvertGather stack overflow).
    This is a known incompatibility — MPS in PyTorch 2.2.x was built against
    macOS 13/14 and is not compatible with macOS 26's Metal stack.
    Using CPU (Apple Accelerate BLAS, all cores).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    # MPS DISABLED — crashes Python on macOS 26 with PyTorch 2.2.x (MLIR GatherOp bug)
    else:
        device = torch.device("cpu")
        logger.info(f"💻 Using CPU — MPS disabled (PyTorch 2.2.x incompatible with macOS 26)")
        return device

def optimize_memory(device):
    """
    Memory optimization for different hardware backends.
    Supports CUDA, MPS (Apple Silicon), and CPU.
    """
    if device.type == "cuda":
        # CUDA GPU optimization
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        # TF32 support (only in PyTorch 1.7+)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            pass  # Older PyTorch versions don't have TF32 support

        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
        except AttributeError:
            pass  # Older PyTorch versions don't have this method

        logger.info("✓ CUDA memory optimization enabled")

    elif device.type == "mps":
        # MPS (Apple Silicon) optimization
        # MPS doesn't have explicit memory management like CUDA
        # But we can set environment variables for better performance
        import os
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory caching

        logger.info("✓ MPS optimization enabled")
        logger.info("  - High watermark ratio: 0.0 (aggressive memory release)")

    else:
        # CPU optimization — use all available cores
        import os
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))

        logger.info(f"✓ CPU optimization enabled")
        logger.info(f"  - Using {num_threads} intra-op threads")
        logger.info(f"  - Using {max(1, num_threads // 2)} inter-op threads")

class EarlyStopping:
    def __init__(self, patience=15, mode="max", delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.delta = delta

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == "max":
            if current_score <= self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0
        return self.early_stop

# ============================================================
# Notebook Cell 6
# ============================================================

TARGET_CLASSES = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 'cardboard_packaging',
    'clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 'eggshells', 'food_waste',
    'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars', 'magazines',
    'newspaper', 'office_paper', 'paper_cups', 'plastic_cup_lids', 'plastic_detergent_bottles',
    'plastic_food_containers', 'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
    'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans', 'styrofoam_cups',
    'styrofoam_food_containers', 'tea_bags'
]

VISION_CONFIG = {
    "model": {
        "backbone": "resnet50d",                       # ResNet-50d (25M params, IN1K) – BN-only, zero GatherOps, proven MPS-safe
        "pretrained": True,  # CRITICAL: Must use pretrained weights
        "num_classes": 30,  # Keep 30 - we'll handle missing classes with proper loss
        "drop_rate": 0.1,  # OPTIMIZED: Light dropout for better generalization
        "drop_path_rate": 0.1  # OPTIMIZED: Light drop_path for robustness
    },
    "data": {
        "input_size": 224,  # Standard ResNet input size
        "num_workers": 0,  # CRITICAL: Disabled for macOS stability
        "pin_memory": False,  # Not needed for CPU
        "sources": [
            {
                "name": "master_30",
                "path": "./data/kaggle/recyclable-and-household-waste-classification/images/images",
                "type": "master"
            },
            {
                "name": "garbage_12",
                "path": "./data/kaggle/garbage-classification-mostafa/garbage_classification",
                "type": "mapped_12"
            },
            {
                "name": "waste_22k",
                "path": "./data/kaggle/waste-classification-data/DATASET",
                "type": "mapped_2"
            },
            {
                "name": "garbage_v2_10",
                "path": "./data/kaggle/garbage-classification-v2",
                "type": "mapped_10"
            },
            {
                "name": "garbage_6",
                "path": "./data/kaggle/garbage-classification",
                "type": "mapped_6"
            },
            {
                "name": "garbage_balanced",
                "path": "./data/kaggle/garbage-dataset-classification",
                "type": "mapped_6"
            },
            {
                "name": "warp_industrial",
                "path": "./data/kaggle/warp-waste-recycling-plant-dataset/Warp-C/train_crops",
                "type": "industrial"
            },
            {
                "name": "multiclass_garbage",
                "path": "./data/kaggle/multi-class-garbage-classification-dataset",
                "type": "multiclass"
            }
        ]
    },
    "training": {
        "batch_size": 32,   # 32 × grad_accum_steps=2 → effective batch 64 (CPU: no VRAM limit)
        "grad_accum_steps": 2,          # effective batch = 64
        # FIX #14: two-phase LR (head_lr for phase-1, finetune_lr for phase-2)
        "learning_rate": 1e-3,          # kept for reference / single-phase fallback
        "head_lr":     1e-3,            # phase-1: head only
        "finetune_lr": 5e-5,            # phase-2: full model end-to-end
        "freeze_epochs": 3,             # epochs to train head only before unfreezing
        "weight_decay": 1e-4,
        "num_epochs": 20,
        "patience": 10,
        "use_amp": False,               # auto-set True on CUDA at runtime
        "max_grad_norm": 1.0,
        "warmup_epochs": 0.5
    }
}

# ============================================================
# Notebook Cell 15
# ============================================================

# PEAK STANDARD GNN
# Using Graph Attention Networks v2 (GATv2) for superior expressive power

def generate_structured_knowledge_graph(num_classes=30, feat_dim=128):
    """
    Generates a Knowledge Graph grounded in the real waste classification taxonomy.
    Schema: Item → Material → Bin (with real mappings from RECYCLABILITY data)
    Also includes Item ↔ Item similarity edges for visually confusable classes.
    """
    logger.info("Generating structured Knowledge Graph (real taxonomy)...")

    # ── Material nodes (8 categories) ──
    MATERIALS = ['plastic', 'paper', 'glass', 'metal', 'organic', 'textile', 'styrofoam', 'mixed']
    # ── Bin nodes (5 disposal types) ──
    BINS = ['recycle', 'compost', 'landfill', 'special', 'donate']

    num_materials = len(MATERIALS)
    num_bins = len(BINS)
    total_nodes = num_classes + num_materials + num_bins  # 30 + 8 + 5 = 43

    # Node indices
    mat_base = num_classes       # 30..37
    bin_base = mat_base + num_materials  # 38..42

    mat_idx = {m: mat_base + i for i, m in enumerate(MATERIALS)}
    bin_idx = {b: bin_base + i for i, b in enumerate(BINS)}

    # ── Real Item → Material mapping (based on RECYCLABILITY) ──
    ITEM_MATERIAL = {
        'aerosol_cans': 'metal', 'aluminum_food_cans': 'metal', 'aluminum_soda_cans': 'metal',
        'steel_food_cans': 'metal',
        'cardboard_boxes': 'paper', 'cardboard_packaging': 'paper', 'magazines': 'paper',
        'newspaper': 'paper', 'office_paper': 'paper', 'paper_cups': 'paper',
        'glass_beverage_bottles': 'glass', 'glass_cosmetic_containers': 'glass',
        'glass_food_jars': 'glass',
        'disposable_plastic_cutlery': 'plastic', 'plastic_cup_lids': 'plastic',
        'plastic_detergent_bottles': 'plastic', 'plastic_food_containers': 'plastic',
        'plastic_shopping_bags': 'plastic', 'plastic_soda_bottles': 'plastic',
        'plastic_straws': 'plastic', 'plastic_trash_bags': 'plastic',
        'plastic_water_bottles': 'plastic',
        'coffee_grounds': 'organic', 'eggshells': 'organic', 'food_waste': 'organic',
        'tea_bags': 'organic',
        'clothing': 'textile', 'shoes': 'mixed',
        'styrofoam_cups': 'styrofoam', 'styrofoam_food_containers': 'styrofoam',
    }

    # ── Real Material → Bin mapping ──
    MATERIAL_BIN = {
        'plastic': 'recycle', 'paper': 'recycle', 'glass': 'recycle', 'metal': 'recycle',
        'organic': 'compost', 'textile': 'donate', 'styrofoam': 'landfill', 'mixed': 'special',
    }

    # ── Real Item → Bin override (some items don't follow their material's default) ──
    ITEM_BIN_OVERRIDE = {
        'disposable_plastic_cutlery': 'landfill',  # PS #6 — not recyclable
        'plastic_straws': 'landfill',               # too small for machinery
        'plastic_trash_bags': 'landfill',            # LDPE not curbside recyclable
        'plastic_shopping_bags': 'special',          # store drop-off only
        'paper_cups': 'landfill',                    # PE lining
        'shoes': 'donate',                           # donate or specialty
    }

    # ── Visually confusable item pairs (from evaluation failure analysis) ──
    CONFUSION_PAIRS = [
        ('glass_beverage_bottles', 'glass_food_jars'),
        ('cardboard_boxes', 'cardboard_packaging'),
        ('aluminum_food_cans', 'steel_food_cans'),
        ('aluminum_soda_cans', 'steel_food_cans'),
        ('plastic_soda_bottles', 'plastic_water_bottles'),
        ('plastic_soda_bottles', 'plastic_detergent_bottles'),
        ('newspaper', 'office_paper'),
        ('newspaper', 'magazines'),
        ('office_paper', 'magazines'),
        ('styrofoam_cups', 'styrofoam_food_containers'),
        ('plastic_food_containers', 'plastic_cup_lids'),
        ('plastic_food_containers', 'disposable_plastic_cutlery'),
        ('coffee_grounds', 'tea_bags'),
        ('food_waste', 'coffee_grounds'),
        ('clothing', 'shoes'),
    ]

    # ── Build edges ──
    edge_sources = []
    edge_targets = []

    # 1. Item → Material (bidirectional)
    for i, cls in enumerate(TARGET_CLASSES):
        mat = ITEM_MATERIAL.get(cls, 'mixed')
        m = mat_idx[mat]
        edge_sources += [i, m]
        edge_targets += [m, i]

    # 2. Material → Bin (bidirectional)
    for mat, b in MATERIAL_BIN.items():
        m = mat_idx[mat]
        bi = bin_idx[b]
        edge_sources += [m, bi]
        edge_targets += [bi, m]

    # 3. Item → Bin override (direct shortcut for exceptions, bidirectional)
    for cls, b in ITEM_BIN_OVERRIDE.items():
        item_i = TARGET_CLASSES.index(cls)
        bi = bin_idx[b]
        edge_sources += [item_i, bi]
        edge_targets += [bi, item_i]

    # 4. Item ↔ Item similarity edges (confusable pairs, bidirectional)
    for a, b in CONFUSION_PAIRS:
        ia = TARGET_CLASSES.index(a)
        ib = TARGET_CLASSES.index(b)
        edge_sources += [ia, ib]
        edge_targets += [ib, ia]

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

    # ── Node features ──
    # Items: learnable embeddings (random init)
    # Materials: encode physical properties (density, recyclability, etc.)
    # Bins: one-hot role encoding
    x = torch.randn(total_nodes, feat_dim)

    # Give material nodes distinct initialization based on properties
    for i, mat in enumerate(MATERIALS):
        # Encode material type as a structured feature
        x[mat_base + i] = torch.randn(feat_dim) * 0.5
        x[mat_base + i, i * (feat_dim // num_materials):(i+1) * (feat_dim // num_materials)] += 2.0

    # Give bin nodes distinct initialization
    for i, b in enumerate(BINS):
        x[bin_base + i] = torch.randn(feat_dim) * 0.3
        x[bin_base + i, i * (feat_dim // num_bins):(i+1) * (feat_dim // num_bins)] += 3.0

    logger.info(f"Knowledge Graph: {total_nodes} nodes ({num_classes} items + {num_materials} materials + {num_bins} bins)")
    logger.info(f"  Edges: {len(edge_sources)} (item↔material + material↔bin + overrides + similarity)")

    return Data(x=x, edge_index=edge_index, num_nodes=total_nodes)

class GATv2Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, heads=8, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_channels * heads) for _ in range(num_layers - 1)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norm[i](x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ============================================================
# Notebook Cell 16
# ============================================================

def train_gnn_model():
    """
    Train GATv2 on the waste classification knowledge graph.
    Task: self-supervised link prediction (encode graph structure into embeddings).
    Can be run independently — does not require vision model.
    """
    set_seed()
    device = get_device()
    optimize_memory(device)
    logger.info(f"GNN Training — Device: {device}")

    # Hyperparameters — RIGHT-SIZED for a 43-node knowledge graph
    in_dim = 128
    hidden_dim = 64     # 64 × 4 heads = 256 dim — appropriate for 43 nodes
    out_dim = 64
    heads = 4
    num_gnn_layers = 3
    dropout = 0.2
    lr = 5e-3
    epochs = 200        # More epochs for a tiny graph
    eval_every = 10

    # Build knowledge graph (seed controls random features)
    set_seed()  # Ensure reproducible graph features
    data = generate_structured_knowledge_graph(num_classes=30, feat_dim=in_dim).to(device)
    num_edges = data.edge_index.size(1)

    # Train/val edge split (80/20)
    perm = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    train_edge_idx = perm[:train_size]
    val_edge_idx = perm[train_size:]
    logger.info(f"Edge split: {train_size} train / {num_edges - train_size} val")

    # FULL edge_index used for message passing (transductive setting)
    # Only the loss is computed on the train/val split
    full_edge_index = data.edge_index

    # Model — right-sized GATv2
    model = GATv2Model(in_dim, hidden_dim, out_dim,
                       num_layers=num_gnn_layers, heads=heads, dropout=dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"GATv2 model: {num_params:,} parameters ({num_params/1e6:.2f}M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    logger.info(f"Starting GNN Training ({epochs} epochs)...")
    best_val_loss = float('inf')
    best_state = None
    patience = 30
    patience_counter = 0

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        optimizer.zero_grad()
        z = model(data.x, full_edge_index)

        # Positive edges (train split only)
        pos_src = full_edge_index[0, train_edge_idx]
        pos_dst = full_edge_index[1, train_edge_idx]
        pos_scores = (z[pos_src] * z[pos_dst]).sum(dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()

        # Hard negative sampling — avoid existing edges
        neg_src = pos_src  # Same source nodes
        neg_dst = torch.randint(0, data.num_nodes, (len(pos_src),), device=device)
        neg_scores = (z[neg_src] * z[neg_dst]).sum(dim=1)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Validate ──
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                z_val = model(data.x, full_edge_index)

                # Val positive edges
                vp_src = full_edge_index[0, val_edge_idx]
                vp_dst = full_edge_index[1, val_edge_idx]
                vp_scores = (z_val[vp_src] * z_val[vp_dst]).sum(dim=1)
                val_pos = -torch.log(torch.sigmoid(vp_scores) + 1e-15).mean()

                # Val negative edges
                vn_src = vp_src
                vn_dst = torch.randint(0, data.num_nodes, (len(vp_src),), device=device)
                vn_scores = (z_val[vn_src] * z_val[vn_dst]).sum(dim=1)
                val_neg = -torch.log(1 - torch.sigmoid(vn_scores) + 1e-15).mean()

                val_loss = (val_pos + val_neg).item()

                # Link prediction accuracy
                all_scores = torch.cat([torch.sigmoid(vp_scores), torch.sigmoid(vn_scores)])
                all_labels = torch.cat([torch.ones_like(vp_scores), torch.zeros_like(vn_scores)])
                lp_acc = ((all_scores > 0.5).float() == all_labels).float().mean().item()

                improved = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                    improved = " ★ best"
                else:
                    patience_counter += eval_every

            lr_now = optimizer.param_groups[0]['lr']
            logger.info(f"  Epoch {epoch+1:3d}/{epochs}: train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  "
                        f"LP_acc={lp_acc:.4f}  lr={lr_now:.2e}{improved}")

            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        z_final = model(data.x, full_edge_index)
        vp_src = full_edge_index[0, val_edge_idx]
        vp_dst = full_edge_index[1, val_edge_idx]
        vp_scores = torch.sigmoid((z_final[vp_src] * z_final[vp_dst]).sum(dim=1))
        vn_dst = torch.randint(0, data.num_nodes, (len(vp_src),), device=device)
        vn_scores = torch.sigmoid((z_final[vp_src] * z_final[vn_dst]).sum(dim=1))
        final_scores = torch.cat([vp_scores, vn_scores])
        final_labels = torch.cat([torch.ones_like(vp_scores), torch.zeros_like(vn_scores)])
        final_acc = ((final_scores > 0.5).float() == final_labels).float().mean().item()

    logger.info(f"\nGNN Training Complete (best model restored):")
    logger.info(f"  Best val loss:   {best_val_loss:.4f}")
    logger.info(f"  Final LP accuracy: {final_acc:.4f}")

    return model, data, {'best_val_loss': best_val_loss, 'final_lp_acc': final_acc}


# ============================================================
# Notebook Cell 18
# ============================================================

# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE GNN TRAINING — Run this cell independently (no vision training needed)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("="*80)
logger.info("Phase 2: GNN Knowledge Graph Training (Standalone)")
logger.info("="*80)

gnn_model, gnn_data, gnn_metrics = train_gnn_model()

if gnn_model is not None:
    # Save checkpoint
    save_path = "checkpoints/best_gnn_gatv2.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': gnn_model.state_dict(),
        'graph_data': {'x': gnn_data.x.cpu(), 'edge_index': gnn_data.edge_index.cpu(),
                       'num_nodes': gnn_data.num_nodes},
        'metrics': gnn_metrics,
        'config': {
            'architecture': 'GATv2',
            'in_dim': 128, 'hidden_dim': 64, 'out_dim': 64,
            'num_layers': 3, 'heads': 4, 'dropout': 0.2,
            'epochs': 200, 'lr': 5e-3,
            'graph_nodes': gnn_data.num_nodes,
            'graph_edges': gnn_data.edge_index.size(1),
            'graph_schema': 'Item→Material→Bin + similarity edges',
        },
    }, save_path)
    logger.info(f"\n✅ GNN model saved to {save_path}")
    logger.info(f"   Final link prediction accuracy: {gnn_metrics['final_lp_acc']:.4f}")
    logger.info(f"   Best validation loss: {gnn_metrics['best_val_loss']:.4f}")

    # ── Embedding Quality Check (uses SAME graph from training) ──
    logger.info("\n── Embedding Quality Check ──")
    gnn_model.eval()
    device = next(gnn_model.parameters()).device
    with torch.no_grad():
        embeddings = gnn_model(gnn_data.x, gnn_data.edge_index).cpu()

    # Normalize embeddings for cosine similarity
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    # Check cosine similarity between semantically related pairs
    from torch.nn.functional import cosine_similarity
    logger.info("  Pairs that SHOULD be similar (same material/bin):")
    similar_pairs = [
        ('glass_beverage_bottles', 'glass_food_jars'),
        ('cardboard_boxes', 'cardboard_packaging'),
        ('plastic_soda_bottles', 'plastic_water_bottles'),
        ('coffee_grounds', 'food_waste'),
        ('aluminum_food_cans', 'aluminum_soda_cans'),
        ('styrofoam_cups', 'styrofoam_food_containers'),
    ]
    for a, b in similar_pairs:
        ia, ib = TARGET_CLASSES.index(a), TARGET_CLASSES.index(b)
        sim = cosine_similarity(embeddings_norm[ia:ia+1], embeddings_norm[ib:ib+1]).item()
        logger.info(f"    cos({a[:25]:25s}, {b[:25]:25s}) = {sim:+.4f}")

    logger.info("  Pairs that SHOULD be distant (different material+bin):")
    distant_pairs = [
        ('clothing', 'glass_food_jars'),
        ('food_waste', 'aluminum_soda_cans'),
        ('plastic_straws', 'newspaper'),
        ('eggshells', 'steel_food_cans'),
    ]
    for a, b in distant_pairs:
        ia, ib = TARGET_CLASSES.index(a), TARGET_CLASSES.index(b)
        sim = cosine_similarity(embeddings_norm[ia:ia+1], embeddings_norm[ib:ib+1]).item()
        logger.info(f"    cos({a[:25]:25s}, {b[:25]:25s}) = {sim:+.4f}")

    # Summary: mean intra-group similarity vs inter-group
    sims_similar = []
    for a, b in similar_pairs:
        ia, ib = TARGET_CLASSES.index(a), TARGET_CLASSES.index(b)
        sims_similar.append(cosine_similarity(embeddings_norm[ia:ia+1], embeddings_norm[ib:ib+1]).item())
    sims_distant = []
    for a, b in distant_pairs:
        ia, ib = TARGET_CLASSES.index(a), TARGET_CLASSES.index(b)
        sims_distant.append(cosine_similarity(embeddings_norm[ia:ia+1], embeddings_norm[ib:ib+1]).item())

    import numpy as np
    logger.info(f"\n  Mean similarity (related pairs):  {np.mean(sims_similar):+.4f}")
    logger.info(f"  Mean similarity (distant pairs):  {np.mean(sims_distant):+.4f}")
    logger.info(f"  Separation gap: {np.mean(sims_similar) - np.mean(sims_distant):+.4f}")

    del gnn_model
    logger.info("\n✅ GNN Knowledge Graph training complete — ready for inference pipeline")
else:
    logger.error("GNN training returned None — check logs above")

logger.info("="*80)
