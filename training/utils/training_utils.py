"""
Training Utilities

CRITICAL: Production-ready training utilities with comprehensive error handling
- Random seed setting for reproducibility
- NaN/Inf detection
- Gradient validation
- Exception handling
- Checkpoint management
- Config validation
"""

import logging
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility
    
    CRITICAL: Must be called at start of all training scripts
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"✅ Random seed set to {seed} for reproducibility")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate training configuration
    
    CRITICAL: Prevents invalid configs from causing crashes mid-training
    """
    # Check required top-level keys
    required_keys = ["model", "data", "training"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")
    
    # Validate training parameters
    training = config["training"]
    
    # Batch size
    if "per_device_train_batch_size" in training:
        batch_size = training["per_device_train_batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"Invalid per_device_train_batch_size: {batch_size}")
    elif "batch_size" in training:
        batch_size = training["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"Invalid batch_size: {batch_size}")
    
    # Learning rate
    lr = training.get("learning_rate", training.get("lr0"))
    if lr is not None:
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"Invalid learning_rate: {lr}")
    
    # Num epochs
    num_epochs = training.get("num_epochs", training.get("num_train_epochs", training.get("epochs")))
    if num_epochs is not None:
        if not isinstance(num_epochs, int) or num_epochs < 1:
            raise ValueError(f"Invalid num_epochs: {num_epochs}")
    
    # Gradient accumulation
    if "gradient_accumulation_steps" in training:
        grad_accum = training["gradient_accumulation_steps"]
        if not isinstance(grad_accum, int) or grad_accum < 1:
            raise ValueError(f"Invalid gradient_accumulation_steps: {grad_accum}")
        if grad_accum > 1000:
            logger.warning(f"Very large gradient_accumulation_steps: {grad_accum}")
    
    # Warmup ratio
    if "warmup_ratio" in training:
        warmup = training["warmup_ratio"]
        if not isinstance(warmup, (int, float)) or not (0.0 <= warmup <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup}")
        if warmup > 0.5:
            logger.warning(f"Large warmup_ratio: {warmup}, may waste training time")
    
    # Gradient clipping
    if "clip_grad_norm" in training or "max_grad_norm" in training:
        clip_value = training.get("clip_grad_norm", training.get("max_grad_norm"))
        if clip_value is not None and clip_value <= 0:
            raise ValueError(f"clip_grad_norm must be positive, got {clip_value}")
    
    logger.info("✅ Configuration validated successfully")


def check_loss_valid(loss: torch.Tensor, epoch: int, step: int) -> None:
    """
    Check if loss is valid (not NaN/Inf)
    
    CRITICAL: Prevents silent training divergence
    """
    if not torch.isfinite(loss):
        raise ValueError(
            f"Loss became {loss.item()} at epoch {epoch}, step {step}. "
            f"Training diverged!"
        )


def check_gradients_valid(model: nn.Module, epoch: int, step: int) -> None:
    """
    Check if gradients are valid (not NaN/Inf)
    
    CRITICAL: Detects gradient explosion/vanishing early
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                raise ValueError(
                    f"NaN/Inf gradient in '{name}' at epoch {epoch}, step {step}"
                )


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Clip gradients and return total norm

    Returns:
        Total gradient norm before clipping
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm
    )

    return total_norm.item()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: Path,
    filename: str = "checkpoint.pth",
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save training checkpoint

    CRITICAL: Comprehensive checkpoint saving for crash recovery
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    if config is not None:
        checkpoint['config'] = config

    checkpoint_path = output_dir / filename
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"💾 Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> int:
    """
    Load training checkpoint

    CRITICAL: Resume training from checkpoint

    Returns:
        Epoch to resume from (checkpoint epoch + 1)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        logger.info("No checkpoint found, starting from scratch")
        return 0

    logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    logger.info(f"✅ Resuming from epoch {start_epoch}")

    return start_epoch


class EarlyStopping:
    """
    Early stopping to prevent overfitting

    CRITICAL: Saves compute by stopping when validation stops improving
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (accuracy), 'min' for metrics to minimize (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop

        Returns:
            True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = metric
            return False

        if self.mode == "max":
            improved = metric > (self.best_value + self.min_delta)
        else:
            improved = metric < (self.best_value - self.min_delta)

        if improved:
            self.best_value = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"No improvement for {self.counter}/{self.patience} epochs")

        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"🛑 Early stopping triggered after {self.counter} epochs without improvement")
            return True

        return False


class TrainingTimer:
    """
    Track training time and estimate completion

    CRITICAL: Helps monitor training progress and estimate ETA
    """
    def __init__(self):
        self.epoch_times = []
        self.start_time = None

    def start_epoch(self):
        """Start timing an epoch"""
        self.start_time = time.time()

    def end_epoch(self, current_epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        End timing an epoch and calculate statistics

        Returns:
            Dictionary with timing statistics
        """
        if self.start_time is None:
            raise RuntimeError("Must call start_epoch() first")

        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)

        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = total_epochs - current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs

        return {
            'epoch_time_seconds': epoch_time,
            'epoch_time_minutes': epoch_time / 60,
            'avg_epoch_time_seconds': avg_epoch_time,
            'eta_seconds': eta_seconds,
            'eta_minutes': eta_seconds / 60,
            'eta_hours': eta_seconds / 3600
        }


