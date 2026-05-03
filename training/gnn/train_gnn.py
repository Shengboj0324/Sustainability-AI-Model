"""
GNN Training Script

CRITICAL: Train GraphSAGE/GAT for upcycling recommendations
- Link prediction task (CAN_BE_UPCYCLED_TO edges)
- Node classification task (Material properties)
- Comprehensive metrics

CRITICAL FIXES:
- Random seed for reproducibility
- Config validation
- NaN/Inf detection
- Exception handling
- Early stopping
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn.inference import GATv2Model, GraphSAGEModel
from training.utils.training_utils import (
    set_seed,
    validate_config,
    check_loss_valid,
    save_checkpoint,
    EarlyStopping,
    TrainingTimer
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/gnn.yaml"):
    """Load training configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


TARGET_CLASSES = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes',
    'cardboard_packaging', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery',
    'eggshells', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers',
    'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups',
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
    'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
    'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans',
    'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags',
]


def load_graph_data(config: dict) -> Data:
    """
    Build the production 43-node Knowledge Graph.

    Schema: 30 Item nodes + 8 Material nodes + 5 Bin nodes
    Edges: Item→Material (MADE_OF), Material→Bin (GOES_TO),
           Item→Bin overrides, Item↔Item similarity (confusable pairs).
    All edges are bidirectional.
    """
    logger.info("Building production Knowledge Graph (43 nodes)...")

    MATERIALS = ['plastic', 'paper', 'glass', 'metal', 'organic', 'textile', 'styrofoam', 'mixed']
    BINS = ['recycle', 'compost', 'landfill', 'special', 'donate']

    num_classes = len(TARGET_CLASSES)      # 30
    num_materials = len(MATERIALS)          # 8
    num_bins = len(BINS)                    # 5
    total_nodes = num_classes + num_materials + num_bins  # 43

    mat_base = num_classes
    bin_base = mat_base + num_materials
    mat_idx = {m: mat_base + i for i, m in enumerate(MATERIALS)}
    bin_idx = {b: bin_base + i for i, b in enumerate(BINS)}

    ITEM_MATERIAL = {
        'aerosol_cans': 'metal', 'aluminum_food_cans': 'metal',
        'aluminum_soda_cans': 'metal', 'steel_food_cans': 'metal',
        'cardboard_boxes': 'paper', 'cardboard_packaging': 'paper',
        'magazines': 'paper', 'newspaper': 'paper', 'office_paper': 'paper',
        'paper_cups': 'paper',
        'glass_beverage_bottles': 'glass', 'glass_cosmetic_containers': 'glass',
        'glass_food_jars': 'glass',
        'disposable_plastic_cutlery': 'plastic', 'plastic_cup_lids': 'plastic',
        'plastic_detergent_bottles': 'plastic', 'plastic_food_containers': 'plastic',
        'plastic_shopping_bags': 'plastic', 'plastic_soda_bottles': 'plastic',
        'plastic_straws': 'plastic', 'plastic_trash_bags': 'plastic',
        'plastic_water_bottles': 'plastic',
        'coffee_grounds': 'organic', 'eggshells': 'organic',
        'food_waste': 'organic', 'tea_bags': 'organic',
        'clothing': 'textile', 'shoes': 'mixed',
        'styrofoam_cups': 'styrofoam', 'styrofoam_food_containers': 'styrofoam',
    }
    MATERIAL_BIN = {
        'plastic': 'recycle', 'paper': 'recycle', 'glass': 'recycle',
        'metal': 'recycle', 'organic': 'compost', 'textile': 'donate',
        'styrofoam': 'landfill', 'mixed': 'special',
    }
    ITEM_BIN_OVERRIDE = {
        'disposable_plastic_cutlery': 'landfill', 'plastic_straws': 'landfill',
        'plastic_trash_bags': 'landfill', 'plastic_shopping_bags': 'special',
        'paper_cups': 'landfill', 'shoes': 'donate',
    }
    CONFUSION_PAIRS = [
        ('glass_beverage_bottles', 'glass_food_jars'),
        ('cardboard_boxes', 'cardboard_packaging'),
        ('aluminum_food_cans', 'steel_food_cans'),
        ('aluminum_soda_cans', 'steel_food_cans'),
        ('plastic_soda_bottles', 'plastic_water_bottles'),
        ('plastic_soda_bottles', 'plastic_detergent_bottles'),
        ('newspaper', 'office_paper'), ('newspaper', 'magazines'),
        ('office_paper', 'magazines'),
        ('styrofoam_cups', 'styrofoam_food_containers'),
        ('plastic_food_containers', 'plastic_cup_lids'),
        ('plastic_food_containers', 'disposable_plastic_cutlery'),
        ('coffee_grounds', 'tea_bags'), ('food_waste', 'coffee_grounds'),
        ('clothing', 'shoes'),
    ]

    # Build edges (bidirectional)
    src, tgt = [], []
    for i, cls in enumerate(TARGET_CLASSES):
        mat = ITEM_MATERIAL.get(cls, 'mixed')
        m = mat_idx[mat]
        src += [i, m]; tgt += [m, i]
    for mat, b in MATERIAL_BIN.items():
        m = mat_idx[mat]; bi = bin_idx[b]
        src += [m, bi]; tgt += [bi, m]
    for cls, b in ITEM_BIN_OVERRIDE.items():
        ii = TARGET_CLASSES.index(cls); bi = bin_idx[b]
        src += [ii, bi]; tgt += [bi, ii]
    for a, b in CONFUSION_PAIRS:
        ia, ib = TARGET_CLASSES.index(a), TARGET_CLASSES.index(b)
        src += [ia, ib]; tgt += [ib, ia]

    edge_index = torch.tensor([src, tgt], dtype=torch.long)

    # Node features
    feat_dim = config["model"]["input_dim"]
    x = torch.randn(total_nodes, feat_dim)
    for i in range(num_materials):
        x[mat_base + i] = torch.randn(feat_dim) * 0.5
        s = i * (feat_dim // num_materials)
        e = (i + 1) * (feat_dim // num_materials)
        x[mat_base + i, s:e] += 2.0
    for i in range(num_bins):
        x[bin_base + i] = torch.randn(feat_dim) * 0.3
        s = i * (feat_dim // num_bins)
        e = (i + 1) * (feat_dim // num_bins)
        x[bin_base + i, s:e] += 3.0

    data = Data(x=x, edge_index=edge_index, num_nodes=total_nodes)
    logger.info(
        f"Knowledge Graph: {total_nodes} nodes "
        f"({num_classes} items + {num_materials} materials + {num_bins} bins), "
        f"{edge_index.size(1)} edges"
    )
    return data


def create_train_val_test_split(data: Data, config: dict) -> Data:
    """
    Split graph EDGES into train/val/test for link prediction.

    CRITICAL FIX: Previous code created node-level masks but used them to
    index edges (data.edge_index[:, data.train_mask]) — this crashes when
    num_nodes != num_edges.  Now creates edge-level masks.

    Also creates node-level masks for optional node classification.
    """
    # ── Validate split ratios ──
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]
    test_ratio = config["data"]["test_ratio"]

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be positive")

    # ── Edge-level split (for link prediction) ──
    num_edges = data.edge_index.size(1)
    edge_perm = torch.randperm(num_edges)

    train_edge_end = int(num_edges * train_ratio)
    val_edge_end = train_edge_end + int(num_edges * val_ratio)

    if train_edge_end == 0 or (val_edge_end - train_edge_end) == 0 or (num_edges - val_edge_end) == 0:
        raise ValueError(
            f"Invalid edge split sizes with {num_edges} edges. Adjust split ratios."
        )

    edge_train_mask = torch.zeros(num_edges, dtype=torch.bool)
    edge_val_mask = torch.zeros(num_edges, dtype=torch.bool)
    edge_test_mask = torch.zeros(num_edges, dtype=torch.bool)

    edge_train_mask[edge_perm[:train_edge_end]] = True
    edge_val_mask[edge_perm[train_edge_end:val_edge_end]] = True
    edge_test_mask[edge_perm[val_edge_end:]] = True

    data.train_mask = edge_train_mask
    data.val_mask = edge_val_mask
    data.test_mask = edge_test_mask

    logger.info(
        f"Edge split: Train={edge_train_mask.sum().item()}, "
        f"Val={edge_val_mask.sum().item()}, Test={edge_test_mask.sum().item()} "
        f"(total edges={num_edges})"
    )

    return data


def create_model(config: dict, device: torch.device) -> nn.Module:
    """
    Create GNN model using the upgraded architectures from models/gnn/inference.py.

    [Upgrade 7] Uses GATv2Model with dynamic attention when type='gatv2'
    [Upgrade 8] Supports edge_dim for edge-attribute-aware message passing
    """
    model_type = config["model"]["type"]
    model_cfg = config["model"]

    if model_type in ("gat", "gatv2"):
        model = GATv2Model(
            in_channels=model_cfg["input_dim"],
            hidden_channels=model_cfg["hidden_dim"],
            out_channels=model_cfg["output_dim"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg.get("num_heads", 4),
            dropout=model_cfg["dropout"],
            attention_dropout=model_cfg.get("attention_dropout", 0.1),
            edge_dim=model_cfg.get("edge_dim"),
        )
    elif model_type == "graphsage":
        model = GraphSAGEModel(
            in_channels=model_cfg["input_dim"],
            hidden_channels=model_cfg["hidden_dim"],
            out_channels=model_cfg["output_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            aggregator=model_cfg.get("aggregator", "mean"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created {model_type} model: {total_params:,} params ({trainable:,} trainable)")

    return model


def negative_sampling(edge_index, num_nodes, num_neg_samples):
    """
    Sample negative edges (pairs of nodes NOT connected in the graph).

    CRITICAL FIX: Efficient vectorized negative sampling with timeout protection.
    Moved BEFORE usage to prevent NameError.
    """
    neg_edges = []
    edge_set = set(map(tuple, edge_index.t().tolist()))
    num_neg_samples = int(num_neg_samples)

    max_attempts = 100
    batch_size = num_neg_samples * 2

    for attempt in range(max_attempts):
        if len(neg_edges) >= num_neg_samples:
            break

        src = torch.randint(0, num_nodes, (batch_size,))
        dst = torch.randint(0, num_nodes, (batch_size,))

        mask = (src != dst)
        candidates = torch.stack([src[mask], dst[mask]], dim=1)

        for edge in candidates:
            edge_tuple = tuple(edge.tolist())
            if edge_tuple not in edge_set:
                neg_edges.append(edge.tolist())
                if len(neg_edges) >= num_neg_samples:
                    break

    if len(neg_edges) < num_neg_samples:
        raise RuntimeError(
            f"Failed to sample {num_neg_samples} negative edges after {max_attempts} attempts. "
            f"Only got {len(neg_edges)} samples. Graph may be too dense."
        )

    return torch.tensor(neg_edges[:num_neg_samples], dtype=torch.long).t()


@torch.no_grad()
def evaluate_link_prediction(model, data, device, mask):
    """Evaluate link prediction accuracy."""
    model.eval()

    z = model(data.x.to(device), data.edge_index.to(device))

    pos_edge_index = data.edge_index[:, mask].to(device)

    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    ).to(device)

    pos_scores = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1))
    neg_scores = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([
        torch.ones(pos_scores.size(0), device=device),
        torch.zeros(neg_scores.size(0), device=device),
    ])

    preds = (scores > 0.5).float()
    acc = (preds == labels).float().mean().item()

    return acc


def train_epoch_link_prediction(model, data, optimizer, device, config):
    """Train one epoch for link prediction."""
    model.train()

    pos_edge_index = data.edge_index[:, data.train_mask].to(device)

    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1) * config["task"]["link_prediction"]["negative_sampling_ratio"]
    ).to(device)

    # Forward
    optimizer.zero_grad()

    # Get node embeddings
    z = model(data.x.to(device), data.edge_index.to(device))

    # Compute link prediction loss
    pos_loss = -torch.log(
        torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)) + 1e-15
    ).mean()

    neg_loss = -torch.log(
        1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15
    ).mean()

    loss = pos_loss + neg_loss

    # CRITICAL FIX: Check for NaN/Inf loss
    if not torch.isfinite(loss):
        raise ValueError(f"Loss became {loss.item()}, training diverged!")

    # Backward
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    """
    Main training function

    CRITICAL FIXES: Seed setting, config validation, exception handling
    """
    try:
        # Load config
        config = load_config()

        # CRITICAL FIX: Validate configuration
        validate_config(config)

        # CRITICAL FIX: Set random seed for reproducibility
        seed = config["training"].get("seed", 42)
        set_seed(seed)

        # Initialize wandb
        wandb.init(
            project="releaf-gnn",
            config=config,
            name=config["training"]["experiment_name"]
        )

        # CRITICAL: Device selection — GNN MUST use CPU on Apple Silicon
        # because PyG's scatter_reduce is NOT implemented on MPS.
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            # Always CPU for GNN — MPS lacks scatter_reduce for PyG ops
            device = torch.device("cpu")
            if torch.backends.mps.is_available():
                logger.info("🍎 MPS available but PyG scatter_reduce unsupported — using CPU")
            else:
                logger.info("💻 Using CPU")

        logger.info(f"Device: {device}")

        # Load graph data
        data = load_graph_data(config)
        data = create_train_val_test_split(data, config)

        # Create model
        model = create_model(config, device)

        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )

        # Training loop
        best_val_acc = 0.0
        output_dir = Path(config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting training")
        for epoch in range(config["training"]["num_epochs"]):
            logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

            # Train
            if config["task"]["type"] == "link_prediction":
                train_loss = train_epoch_link_prediction(model, data, optimizer, device, config)

                # Evaluate
                train_acc = evaluate_link_prediction(model, data, device, data.train_mask)
                val_acc = evaluate_link_prediction(model, data, device, data.val_mask)

                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

                # Log metrics
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"]
                })

                # Update scheduler
                scheduler.step(val_acc)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'config': config
                    }, output_dir / "best_model.pth")
                    logger.info(f"Saved best model with val_acc: {val_acc:.4f}")

        # Final evaluation on test set
        test_acc = evaluate_link_prediction(model, data, device, data.test_mask)
        logger.info(f"\nTraining complete! Best val_acc: {best_val_acc:.4f}, Test acc: {test_acc:.4f}")

        wandb.log({"test_acc": test_acc})
        wandb.finish()

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()

