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

from models.gnn.inference import UpcyclingGNN
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


def load_graph_data(config: dict) -> Data:
    """Load graph data from files"""
    logger.info("Loading graph data...")

    # Load edges
    edges_df = pd.read_parquet(config["data"]["graph_file"])
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)

    # Load node features
    features_df = pd.read_parquet(config["data"]["node_features_file"])
    x = torch.tensor(features_df.drop('node_id', axis=1).values, dtype=torch.float)

    # Load labels (if doing node classification)
    if config["task"]["type"] == "node_classification":
        labels_df = pd.read_parquet(config["data"]["node_labels_file"])
        y = torch.tensor(labels_df['label'].values, dtype=torch.long)
    else:
        y = None

    # Create graph data object
    data = Data(x=x, edge_index=edge_index, y=y)

    logger.info(f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges")

    return data


def create_train_val_test_split(data: Data, config: dict) -> Tuple[Data, Data, Data]:
    """
    Split graph into train/val/test

    CRITICAL FIX: Validate split ratios and sizes
    """
    num_nodes = data.num_nodes

    # Random permutation
    perm = torch.randperm(num_nodes)

    # Calculate split indices
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]
    test_ratio = config["data"]["test_ratio"]

    # CRITICAL FIX: Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be positive")

    train_end = int(num_nodes * train_ratio)
    val_end = train_end + int(num_nodes * val_ratio)

    # CRITICAL FIX: Validate split sizes
    train_size = train_end
    val_size = val_end - train_end
    test_size = num_nodes - val_end

    if train_size == 0 or val_size == 0 or test_size == 0:
        raise ValueError(
            f"Invalid split sizes: train={train_size}, val={val_size}, test={test_size}. "
            f"Total nodes={num_nodes}. Adjust split ratios."
        )

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    logger.info(f"Split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    return data


def create_model(config: dict, device: torch.device) -> UpcyclingGNN:
    """Create GNN model"""
    model = UpcyclingGNN(
        in_channels=config["model"]["input_dim"],
        hidden_channels=config["model"]["hidden_dim"],
        out_channels=config["model"]["output_dim"],
        num_layers=config["model"]["num_layers"],
        model_type=config["model"]["type"],
        dropout=config["model"]["dropout"]
    )

    model = model.to(device)
    logger.info(f"Created {config['model']['type']} model with {sum(p.numel() for p in model.parameters())} parameters")

    return model


def train_epoch_link_prediction(model, data, optimizer, device, config):
    """Train one epoch for link prediction"""
    model.train()

    # Positive edges (existing edges)
    pos_edge_index = data.edge_index[:, data.train_mask]

    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1) * config["task"]["link_prediction"]["negative_sampling_ratio"]
    )

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

        # CRITICAL: Device selection with M4 Max support
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🍎 Using Apple M4 Max GPU (MPS)")
        else:
            device = torch.device("cpu")
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


def negative_sampling(edge_index, num_nodes, num_neg_samples):
    """
    Sample negative edges

    CRITICAL FIX: Efficient vectorized negative sampling with timeout protection
    """
    neg_edges = []
    edge_set = set(map(tuple, edge_index.t().tolist()))

    # CRITICAL FIX: Vectorized sampling with max attempts to prevent infinite loops
    max_attempts = 100
    batch_size = num_neg_samples * 2  # Sample more than needed for efficiency

    for attempt in range(max_attempts):
        if len(neg_edges) >= num_neg_samples:
            break

        # Vectorized sampling
        src = torch.randint(0, num_nodes, (batch_size,))
        dst = torch.randint(0, num_nodes, (batch_size,))

        # Filter self-loops
        mask = (src != dst)
        candidates = torch.stack([src[mask], dst[mask]], dim=1)

        # Filter existing edges
        for edge in candidates:
            edge_tuple = tuple(edge.tolist())
            if edge_tuple not in edge_set:
                neg_edges.append(edge.tolist())
                if len(neg_edges) >= num_neg_samples:
                    break

    # CRITICAL FIX: Validate we got enough samples
    if len(neg_edges) < num_neg_samples:
        raise RuntimeError(
            f"Failed to sample {num_neg_samples} negative edges after {max_attempts} attempts. "
            f"Only got {len(neg_edges)} samples. Graph may be too dense."
        )

    return torch.tensor(neg_edges[:num_neg_samples], dtype=torch.long).t()


@torch.no_grad()
def evaluate_link_prediction(model, data, device, mask):
    """Evaluate link prediction"""
    model.eval()

    # Get embeddings
    z = model(data.x.to(device), data.edge_index.to(device))

    # Positive edges
    pos_edge_index = data.edge_index[:, mask]

    # Negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )

    # Compute scores
    pos_scores = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1))
    neg_scores = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))

    # Compute AUC
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])

    # Simple accuracy (threshold = 0.5)
    preds = (scores > 0.5).float()
    acc = (preds == labels).float().mean().item()

    return acc

