"""
Multi-Head Vision Classifier Training

CRITICAL: Train 3-head classifier (item_type, material_type, bin_type)
- Uses actual WasteClassifier from models/vision/classifier.py
- Multi-task learning with weighted losses
- Class balancing
- Comprehensive metrics
"""

import os
import sys
import yaml
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vision.classifier import WasteClassifier
from training.vision.dataset import WasteClassificationDataset, get_balanced_sampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/vision_cls.yaml"):
    """Load training configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(config: dict, device: torch.device) -> WasteClassifier:
    """Create multi-head classifier"""
    model = WasteClassifier(
        model_name=config["model"]["backbone"],
        num_classes_item=config["model"]["num_classes_item"],
        num_classes_material=config["model"]["num_classes_material"],
        num_classes_bin=config["model"]["num_classes_bin"],
        pretrained=config["model"]["pretrained"],
        drop_rate=config["model"]["drop_rate"]
    )

    model = model.to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    return model


def get_dataloaders(config: dict):
    """Create data loaders"""
    # Datasets
    train_dataset = WasteClassificationDataset(
        data_dir=config["data"]["data_dir"],
        split="train",
        img_size=config["data"]["input_size"]
    )

    val_dataset = WasteClassificationDataset(
        data_dir=config["data"]["data_dir"],
        split="val",
        img_size=config["data"]["input_size"]
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create balanced sampler for training
    sampler = get_balanced_sampler(train_dataset) if config["training"]["use_balanced_sampler"] else None

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["val_batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"]
    )

    return train_loader, val_loader


def train_epoch(model, loader, criterions, optimizer, device, config):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    item_correct = 0
    material_correct = 0
    bin_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        item_labels = labels['item_type'].to(device)
        material_labels = labels['material_type'].to(device)
        bin_labels = labels['bin_type'].to(device)

        # Forward
        optimizer.zero_grad()
        item_logits, material_logits, bin_logits = model(images)

        # Calculate losses
        item_loss = criterions['item'](item_logits, item_labels)
        material_loss = criterions['material'](material_logits, material_labels)
        bin_loss = criterions['bin'](bin_logits, bin_labels)

        # Weighted combination
        loss = (
            config["training"]["loss_weights"]["item"] * item_loss +
            config["training"]["loss_weights"]["material"] * material_loss +
            config["training"]["loss_weights"]["bin"] * bin_loss
        )

        # Backward
        loss.backward()

        # Gradient clipping
        if config["training"]["clip_grad_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad_norm"])

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        batch_size = images.size(0)
        total_samples += batch_size

        item_correct += (item_logits.argmax(1) == item_labels).sum().item()
        material_correct += (material_logits.argmax(1) == material_labels).sum().item()
        bin_correct += (bin_logits.argmax(1) == bin_labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'item_acc': 100. * item_correct / total_samples,
            'mat_acc': 100. * material_correct / total_samples,
            'bin_acc': 100. * bin_correct / total_samples
        })

    metrics = {
        'loss': total_loss / len(loader),
        'item_acc': 100. * item_correct / total_samples,
        'material_acc': 100. * material_correct / total_samples,
        'bin_acc': 100. * bin_correct / total_samples
    }

    return metrics


@torch.no_grad()
def validate(model, loader, criterions, device, config):
    """Validate model"""
    model.eval()

    total_loss = 0.0
    item_correct = 0
    material_correct = 0
    bin_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Validation")
    for images, labels in pbar:
        images = images.to(device)
        item_labels = labels['item_type'].to(device)
        material_labels = labels['material_type'].to(device)
        bin_labels = labels['bin_type'].to(device)

        # Forward
        item_logits, material_logits, bin_logits = model(images)

        # Calculate losses
        item_loss = criterions['item'](item_logits, item_labels)
        material_loss = criterions['material'](material_logits, material_labels)
        bin_loss = criterions['bin'](bin_logits, bin_labels)

        loss = (
            config["training"]["loss_weights"]["item"] * item_loss +
            config["training"]["loss_weights"]["material"] * material_loss +
            config["training"]["loss_weights"]["bin"] * bin_loss
        )

        # Metrics
        total_loss += loss.item()
        batch_size = images.size(0)
        total_samples += batch_size

        item_correct += (item_logits.argmax(1) == item_labels).sum().item()
        material_correct += (material_logits.argmax(1) == material_labels).sum().item()
        bin_correct += (bin_logits.argmax(1) == bin_labels).sum().item()

    metrics = {
        'loss': total_loss / len(loader),
        'item_acc': 100. * item_correct / total_samples,
        'material_acc': 100. * material_correct / total_samples,
        'bin_acc': 100. * bin_correct / total_samples
    }

    return metrics



def main():
    """Main training function"""
    # Load config
    config = load_config()

    # Initialize wandb
    wandb.init(
        project="releaf-vision-multihead",
        config=config,
        name=config["training"]["experiment_name"]
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = create_model(config, device)

    # Data loaders
    train_loader, val_loader = get_dataloaders(config)

    # Loss functions
    criterions = {
        'item': nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"]),
        'material': nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"]),
        'bin': nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"])
    }

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        eps=config["training"]["eps"]
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],
        eta_min=config["training"]["min_lr"]
    )

    # Training loop
    best_val_acc = 0.0
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training")
    for epoch in range(config["training"]["num_epochs"]):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterions, optimizer, device, config)

        # Validate
        val_metrics = validate(model, val_loader, criterions, device, config)

        # Update scheduler
        scheduler.step()

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_metrics['loss'],
            "train_item_acc": train_metrics['item_acc'],
            "train_material_acc": train_metrics['material_acc'],
            "train_bin_acc": train_metrics['bin_acc'],
            "val_loss": val_metrics['loss'],
            "val_item_acc": val_metrics['item_acc'],
            "val_material_acc": val_metrics['material_acc'],
            "val_bin_acc": val_metrics['bin_acc'],
            "lr": optimizer.param_groups[0]["lr"]
        })

        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Item: {train_metrics['item_acc']:.2f}%, Material: {train_metrics['material_acc']:.2f}%, Bin: {train_metrics['bin_acc']:.2f}%")
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Item: {val_metrics['item_acc']:.2f}%, Material: {val_metrics['material_acc']:.2f}%, Bin: {val_metrics['bin_acc']:.2f}%")

        # Average accuracy across all heads
        avg_val_acc = (val_metrics['item_acc'] + val_metrics['material_acc'] + val_metrics['bin_acc']) / 3

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / "best_model.pth")
            logger.info(f"Saved best model with avg_val_acc: {avg_val_acc:.2f}%")

        # Save checkpoint every N epochs
        if (epoch + 1) % config["training"]["save_every"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")

    logger.info(f"\nTraining complete! Best avg_val_acc: {best_val_acc:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()

