"""
Vision Classifier Training Script
Train waste and material classification model

CRITICAL FIXES:
- Random seed for reproducibility
- Config validation
- NaN/Inf detection
- Exception handling
- Checkpoint resume
- Early stopping
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from tqdm import tqdm
import logging
from pathlib import Path
import wandb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.utils.training_utils import (
    set_seed,
    validate_config,
    check_loss_valid,
    check_gradients_valid,
    clip_gradients,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    TrainingTimer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/vision_cls.yaml"):
    """Load training configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(config, is_train=True):
    """Get image transforms"""
    if is_train:
        # Training transforms with augmentation
        transform_list = [
            transforms.RandomResizedCrop(
                config["data"]["input_size"],
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["data"]["mean"],
                std=config["data"]["std"]
            )
        ]
    else:
        # Validation transforms (no augmentation)
        transform_list = [
            transforms.Resize((config["data"]["input_size"], config["data"]["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["data"]["mean"],
                std=config["data"]["std"]
            )
        ]
    
    return transforms.Compose(transform_list)


def create_model(config):
    """Create classification model"""
    model_name = config["model"]["backbone"]
    num_classes = config["model"]["num_classes_item"]
    
    logger.info(f"Creating model: {model_name}")
    
    # Create model using timm
    model = timm.create_model(
        model_name,
        pretrained=config["model"]["pretrained"],
        num_classes=num_classes,
        drop_rate=config["model"]["drop_rate"],
        drop_path_rate=config["model"]["drop_path_rate"]
    )
    
    return model


def get_dataloaders(config):
    """Create data loaders"""
    # Transforms
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    # Datasets
    train_dataset = ImageFolder(
        root=config["data"]["train_dir"],
        transform=train_transform
    )
    
    val_dataset = ImageFolder(
        root=config["data"]["val_dir"],
        transform=val_transform
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
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
    
    return train_loader, val_loader, train_dataset.classes


def train_epoch(model, loader, criterion, optimizer, device, config, epoch):
    """
    Train for one epoch

    CRITICAL FIXES: NaN/Inf detection, gradient validation
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # CRITICAL FIX: Check for NaN/Inf loss
        check_loss_valid(loss, epoch, batch_idx)

        # Backward
        loss.backward()

        # CRITICAL FIX: Check for NaN/Inf gradients
        check_gradients_valid(model, epoch, batch_idx)

        # Gradient clipping
        if config["training"]["clip_grad_norm"]:
            total_norm = clip_gradients(model, config["training"]["clip_grad_norm"])
            if total_norm > config["training"]["clip_grad_norm"] * 2:
                logger.warning(f"Large gradient norm: {total_norm:.4f}")

        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Validation")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main():
    """
    Main training function

    CRITICAL FIXES: Seed setting, config validation, exception handling, early stopping
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
            project="releaf-vision-classifier",
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
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Data loaders
    train_loader, val_loader, class_names = get_dataloaders(config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["training"]["label_smoothing"]
    )
    
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
    
        # CRITICAL FIX: Early stopping and training timer
        early_stopping = EarlyStopping(
            patience=config["training"].get("early_stopping_patience", 10),
            mode="max"
        )
        timer = TrainingTimer()

        # Training loop
        best_val_acc = 0.0
        output_dir = Path(config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting training")
        for epoch in range(config["training"]["num_epochs"]):
            try:
                logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
                timer.start_epoch()

                # Train
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, config, epoch
                )

                # Validate
                val_loss, val_acc = validate(model, val_loader, criterion, device)

                # Update scheduler
                scheduler.step()

                # Calculate timing
                timing = timer.end_epoch(epoch + 1, config["training"]["num_epochs"])

                # Log metrics
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_time_minutes": timing["epoch_time_minutes"],
                    "eta_hours": timing["eta_hours"]
                })

                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                logger.info(f"Epoch time: {timing['epoch_time_minutes']:.2f}min, ETA: {timing['eta_hours']:.2f}h")

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(
                        model, optimizer, epoch, output_dir,
                        filename="best_model.pth",
                        scheduler=scheduler,
                        metrics={'val_acc': val_acc, 'val_loss': val_loss},
                        config={'class_names': class_names}
                    )
                    logger.info(f"Saved best model with val_acc: {val_acc:.2f}%")

                # CRITICAL FIX: Early stopping check
                if early_stopping(val_acc):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                save_checkpoint(
                    model, optimizer, epoch, output_dir,
                    filename="interrupted_checkpoint.pth",
                    scheduler=scheduler
                )
                raise
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}", exc_info=True)
                save_checkpoint(
                    model, optimizer, epoch, output_dir,
                    filename="error_checkpoint.pth",
                    scheduler=scheduler
                )
                raise

        logger.info(f"\nTraining complete! Best val_acc: {best_val_acc:.2f}%")
        wandb.finish()

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()

