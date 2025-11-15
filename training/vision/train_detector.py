"""
Object Detector Training Script
Train YOLO-based waste detection model
"""

import os
import yaml
from ultralytics import YOLO
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/vision_det.yaml"):
    """Load training configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dataset_yaml(config):
    """Create dataset YAML for YOLO training"""
    dataset_config = {
        'path': str(Path(config["data"]["train_yaml"]).parent.parent),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(config["model"]["classes"])}
    }
    
    # Save dataset config
    dataset_yaml_path = "configs/datasets/waste_dataset.yaml"
    os.makedirs(os.path.dirname(dataset_yaml_path), exist_ok=True)
    
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_config, f)
    
    return dataset_yaml_path


def main():
    """Main training function"""
    # Load config
    config = load_config()
    
    # Create dataset YAML
    dataset_yaml = create_dataset_yaml(config)
    logger.info(f"Dataset config: {dataset_yaml}")
    
    # Initialize model
    model_type = config["model"]["type"]
    logger.info(f"Initializing {model_type} model")
    
    model = YOLO(f"{model_type}.pt")  # Load pretrained model
    
    # Training arguments
    train_args = {
        'data': dataset_yaml,
        'epochs': config["training"]["epochs"],
        'batch': config["training"]["batch_size"],
        'imgsz': config["data"]["img_size"],
        'optimizer': config["training"]["optimizer"],
        'lr0': config["training"]["lr0"],
        'lrf': config["training"]["lrf"],
        'momentum': config["training"]["momentum"],
        'weight_decay': config["training"]["weight_decay"],
        'warmup_epochs': config["training"]["warmup_epochs"],
        'warmup_momentum': config["training"]["warmup_momentum"],
        'warmup_bias_lr': config["training"]["warmup_bias_lr"],
        'box': config["training"]["box"],
        'cls': config["training"]["cls"],
        'dfl': config["training"]["dfl"],
        'dropout': config["training"]["dropout"],
        'amp': config["training"]["amp"],
        'save_period': config["training"]["save_period"],
        'project': config["training"]["output_dir"],
        'name': config["training"]["experiment_name"],
        'verbose': config["training"]["verbose"],
        'plots': config["training"]["plots"],
        'patience': config["training"]["patience"],
        'workers': config["device"]["workers"],
        'device': config["device"]["device"],
        # Augmentation
        'mosaic': config["data"]["augmentations"]["mosaic"],
        'mixup': config["data"]["augmentations"]["mixup"],
        'copy_paste': config["data"]["augmentations"]["copy_paste"],
        'degrees': config["data"]["augmentations"]["degrees"],
        'translate': config["data"]["augmentations"]["translate"],
        'scale': config["data"]["augmentations"]["scale"],
        'shear': config["data"]["augmentations"]["shear"],
        'perspective': config["data"]["augmentations"]["perspective"],
        'flipud': config["data"]["augmentations"]["flipud"],
        'fliplr': config["data"]["augmentations"]["fliplr"],
        'hsv_h': config["data"]["augmentations"]["hsv_h"],
        'hsv_s': config["data"]["augmentations"]["hsv_s"],
        'hsv_v': config["data"]["augmentations"]["hsv_v"],
    }
    
    # Train
    logger.info("Starting training")
    results = model.train(**train_args)
    
    # Validate
    logger.info("Running validation")
    metrics = model.val(
        data=dataset_yaml,
        conf=config["validation"]["conf_thres"],
        iou=config["validation"]["iou_thres"],
        max_det=config["validation"]["max_det"],
        save_json=config["validation"]["save_json"]
    )
    
    # Print results
    logger.info(f"mAP50: {metrics.box.map50:.4f}")
    logger.info(f"mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"Precision: {metrics.box.mp:.4f}")
    logger.info(f"Recall: {metrics.box.mr:.4f}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

