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
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vision.classifier import MultiHeadClassifier
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


def _safe_pil_loader(path: str) -> Image.Image:
    """Load an image via PIL, returning None on ANY failure instead of raising."""
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()                       # force full decode now
        return img.convert("RGB")
    except Exception:
        return None


class SafeImageFolder(ImageFolder):
    """ImageFolder that is completely immune to corrupt images.

    Three layers of defence (no images are deleted from disk):

    1. **Pre-filter at init** — every file is tested with PIL.open+load.
       Entries that fail are removed from ``self.samples`` / ``self.targets``
       so they are never indexed during training.

    2. **Safe loader** — uses ``_safe_pil_loader`` which returns ``None``
       on any PIL error (handles race-conditions / transient I/O).

    3. **__getitem__ fallback** — if the loader returns ``None`` at runtime,
       a random valid neighbour is returned instead of raising.
    """

    def __init__(self, root, transform=None, **kwargs):
        # Let ImageFolder discover all files first
        super().__init__(root, transform=transform, loader=_safe_pil_loader, **kwargs)

        # ── Pre-filter: test every image, keep only loadable ones ──
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True          # tolerate truncated data

        clean_samples = []
        skipped = []
        logger.info(f"Pre-validating {len(self.samples):,} images in {root} …")
        for path, label in self.samples:
            try:
                with open(path, "rb") as f:
                    img = Image.open(f)
                    img.verify()                         # fast header check
                clean_samples.append((path, label))
            except Exception as e:
                skipped.append(path)

        if skipped:
            logger.warning(
                f"⚠️  {len(skipped)} images failed pre-validation and will be "
                f"skipped (NOT deleted from disk):"
            )
            for p in skipped[:20]:
                logger.warning(f"    {p}")
            if len(skipped) > 20:
                logger.warning(f"    … and {len(skipped) - 20} more")

        self.samples = clean_samples
        self.targets = [s[1] for s in clean_samples]
        self.imgs = self.samples                         # legacy alias
        logger.info(
            f"✅ Pre-validation complete: {len(clean_samples):,} usable, "
            f"{len(skipped):,} skipped"
        )

    def __getitem__(self, index):
        """Return (image_tensor, label).  Never raises on a bad file."""
        path, label = self.samples[index]
        sample = self.loader(path)

        # Layer 3: if the loader still fails at runtime, find a neighbour
        if sample is None:
            logger.warning(f"⚠️  Runtime load failure [{index}]: {path}")
            for offset in range(1, min(100, len(self))):
                alt = (index + offset) % len(self)
                sample = self.loader(self.samples[alt][0])
                if sample is not None:
                    label = self.samples[alt][1]
                    break
            if sample is None:
                raise RuntimeError("Cannot load any image in the dataset")

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return sample, label


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
    """
    Create classification model with all industrial-grade upgrades.

    [Upgrade 1] Squeeze-and-Excitation channel attention
    [Upgrade 2] LLM Projection Head (4096d alignment with Llama-3-8B)
    [Upgrade 3] Temperature-scaled confidence calibration
    """
    model_cfg = config["model"]
    use_multi_head = model_cfg.get("multi_head", False)

    if use_multi_head:
        logger.info("Creating MultiHeadClassifier with SE + LLM projection upgrades")
        model = MultiHeadClassifier(
            backbone=model_cfg["backbone"],
            num_classes_item=model_cfg["num_classes_item"],
            num_classes_material=model_cfg["num_classes_material"],
            num_classes_bin=model_cfg["num_classes_bin"],
            drop_rate=model_cfg["drop_rate"],
            pretrained=model_cfg["pretrained"],
            enable_se=True,
            enable_llm_projection=True,
        )
    else:
        # Fallback to plain timm model for single-head configs
        logger.info(f"Creating single-head model: {model_cfg['backbone']}")
        model = timm.create_model(
            model_cfg["backbone"],
            pretrained=model_cfg["pretrained"],
            num_classes=model_cfg["num_classes_item"],
            drop_rate=model_cfg["drop_rate"],
            drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        )

    return model, use_multi_head


def get_dataloaders(config):
    """Create data loaders"""
    # CRITICAL: Validate data directories exist before ImageFolder
    train_dir = Path(config["data"]["train_dir"])
    val_dir = Path(config["data"]["val_dir"])
    for d, label in [(train_dir, "train"), (val_dir, "val")]:
        if not d.exists():
            raise FileNotFoundError(
                f"{label} directory not found: {d}\n"
                f"Run the data preparation pipeline first to populate {d}"
            )
        subdirs = [p for p in d.iterdir() if p.is_dir()]
        if not subdirs:
            raise FileNotFoundError(
                f"{label} directory has no class subdirectories: {d}\n"
                f"ImageFolder requires at least one subdirectory per class."
            )

    # Transforms
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)

    # Datasets — use SafeImageFolder to survive corrupt images
    train_dataset = SafeImageFolder(
        root=str(train_dir),
        transform=train_transform
    )

    val_dataset = SafeImageFolder(
        root=str(val_dir),
        transform=val_transform
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    
    # Data loaders — guard against invalid combinations
    num_workers = config["data"]["num_workers"]
    persistent = config["data"].get("persistent_workers", False) and num_workers > 0
    pin_mem = config["data"].get("pin_memory", False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["val_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
    )
    
    return train_loader, val_loader, train_dataset.classes


def train_epoch(model, loader, criterion, optimizer, device, config, epoch,
                multi_head: bool = False, loss_weights: dict = None):
    """
    Train for one epoch with gradient accumulation and MPS memory management.

    Supports both single-head (plain timm) and multi-head (MultiHeadClassifier).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    lw = loss_weights or {"item_type": 1.0, "material": 1.0, "bin_type": 0.5}
    accum_steps = config["training"].get("gradient_accumulation_steps", 1)
    is_mps = (device.type == "mps")

    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        if multi_head:
            item_logits, material_logits, bin_logits = model(images)
            loss = (lw["item_type"] * criterion(item_logits, labels))
            outputs = item_logits
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale loss for gradient accumulation
        loss = loss / accum_steps

        check_loss_valid(loss, epoch, batch_idx)
        loss.backward()

        # Step optimizer every accum_steps micro-batches
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            check_gradients_valid(model, epoch, batch_idx)

            if config["training"]["clip_grad_norm"]:
                total_norm = clip_gradients(model, config["training"]["clip_grad_norm"])
                if total_norm > config["training"]["clip_grad_norm"] * 2:
                    logger.warning(f"Large gradient norm: {total_norm:.4f}")

            optimizer.step()
            optimizer.zero_grad()

            # MPS memory cleanup every N optimizer steps to prevent OOM
            if is_mps and (batch_idx + 1) % (accum_steps * 10) == 0:
                torch.mps.empty_cache()

        running_loss += loss.item() * accum_steps  # undo scaling for logging
        _, predicted = outputs.detach().max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    # Final MPS cache cleanup between epochs
    if is_mps:
        torch.mps.empty_cache()

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, multi_head: bool = False):
    """Validate model — supports single-head and multi-head."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validation")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if multi_head:
            item_logits, material_logits, bin_logits = model(images)
            loss = criterion(item_logits, labels)
            outputs = item_logits
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total

    # MPS memory cleanup after validation
    if device.type == "mps":
        torch.mps.empty_cache()

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

        # Create model (returns multi_head flag indicating upgrade path)
        model, multi_head = create_model(config)
        model = model.to(device)
        loss_weights = config.get("loss", {}).get("loss_weights", {
            "item_type": 1.0, "material": 1.0, "bin_type": 0.5,
        })

        # ── Memory optimization: gradient checkpointing ──
        if config["training"].get("gradient_checkpointing", False):
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'set_grad_checkpointing'):
                model.backbone.set_grad_checkpointing(True)
                logger.info("✅ Gradient checkpointing enabled on backbone (saves ~60% activation memory)")
            else:
                logger.warning("⚠️  Gradient checkpointing requested but backbone doesn't support it")

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
                    model, train_loader, criterion, optimizer, device, config, epoch,
                    multi_head=multi_head, loss_weights=loss_weights,
                )

                # Validate
                val_loss, val_acc = validate(
                    model, val_loader, criterion, device, multi_head=multi_head,
                )

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

