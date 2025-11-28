"""
Shared Common Functions - Consolidated from all services

CRITICAL: Single source of truth for common operations
- load_config: Load YAML configuration files
- cleanup_resources: Clean up model resources
- get_stats: Get model/service statistics
- reset_stats: Reset statistics counters

This eliminates duplicate implementations across:
- training/llm/train_sft.py
- training/vision/train_classifier.py
- training/vision/train_detector.py
- training/gnn/train_gnn.py
- models/vision/classifier.py
- models/vision/detector.py
- models/gnn/inference.py
"""

import yaml
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    CRITICAL: Standardized config loading across all services
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from: {config_path}")
        return config
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {config_path}: {e}")
        raise


def cleanup_resources(model: Optional[torch.nn.Module] = None,
                     device: Optional[torch.device] = None):
    """
    Clean up model resources and free memory

    CRITICAL: Prevents memory leaks in production
    Supports CUDA, MPS (Apple Silicon), and CPU

    Args:
        model: PyTorch model to clean up
        device: Device to clear cache on
    """
    if model is not None:
        try:
            # Move model to CPU to free GPU memory
            model.cpu()
            del model
            logger.info("Model moved to CPU and deleted")
        except Exception as e:
            logger.warning(f"Error cleaning up model: {e}")

    # Clear CUDA cache if using GPU
    if device is not None and device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache: {e}")

    # Clear MPS cache if using Apple Silicon
    if device is not None and device.type == 'mps':
        try:
            torch.mps.empty_cache()
            logger.info("MPS cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing MPS cache: {e}")

    # Also try to clear cache without device check
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Failed to clear CUDA cache: {e}")

    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception as e:
            logger.debug(f"Failed to clear MPS cache: {e}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get appropriate device for model inference/training
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif prefer_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def format_stats(stats: Dict[str, Any]) -> str:
    """
    Format statistics dictionary for logging
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            lines.append(f"  {key}: {value:,}")
        else:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that config contains all required keys
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key paths (e.g., ["model.name", "training.batch_size"])
        
    Returns:
        True if valid, raises ValueError if invalid
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if key not in current:
                missing_keys.append(key_path)
                break
            current = current[key]
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")
    
    return True

