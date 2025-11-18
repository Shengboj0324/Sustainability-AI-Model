"""
Test Training Integration
Verify all training files can load datasets and initialize models
"""

import os
import sys
from pathlib import Path
import logging
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_llm_training_setup():
    """Test LLM training can load config and data"""
    logger.info("=" * 80)
    logger.info("TESTING LLM TRAINING SETUP")
    logger.info("=" * 80)
    
    try:
        # Load config
        config_path = PROJECT_ROOT / "configs" / "llm_sft.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Config loaded successfully")
        
        # Check data files exist
        train_files = config["data"]["train_files"]
        val_files = config["data"]["val_files"]
        
        for file_list, name in [(train_files, "train"), (val_files, "val")]:
            for filepath in file_list:
                full_path = PROJECT_ROOT / filepath
                if not full_path.exists():
                    logger.error(f"❌ Missing {name} file: {filepath}")
                    return False
                logger.info(f"✅ Found {name} file: {filepath}")
        
        # Test loading dataset
        from datasets import load_dataset
        
        train_dataset = load_dataset("json", data_files=train_files, split="train")
        logger.info(f"✅ Loaded train dataset: {len(train_dataset)} examples")
        
        val_dataset = load_dataset("json", data_files=val_files, split="train")
        logger.info(f"✅ Loaded val dataset: {len(val_dataset)} examples")
        
        # Verify data format
        sample = train_dataset[0]
        assert "messages" in sample, "Missing 'messages' field"
        logger.info(f"✅ Data format validated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_training_setup():
    """Test Vision training can load config"""
    logger.info("=" * 80)
    logger.info("TESTING VISION TRAINING SETUP")
    logger.info("=" * 80)
    
    try:
        # Load config
        config_path = PROJECT_ROOT / "configs" / "vision_cls.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Config loaded successfully")
        
        # Check data directories (may not exist yet)
        data_dir = config["data"]["data_dir"]
        logger.info(f"⚠️  Vision data directory: {data_dir} (may not exist yet)")
        
        # Test model creation
        import timm
        model_name = config["model"]["backbone"]
        num_classes = config["model"]["num_classes_item"]
        
        logger.info(f"Testing model creation: {model_name}")
        model = timm.create_model(
            model_name,
            pretrained=False,  # Don't download weights for test
            num_classes=num_classes
        )
        logger.info(f"✅ Model created successfully: {model_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Vision training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gnn_training_setup():
    """Test GNN training can load config and data"""
    logger.info("=" * 80)
    logger.info("TESTING GNN TRAINING SETUP")
    logger.info("=" * 80)
    
    try:
        # Load config
        config_path = PROJECT_ROOT / "configs" / "gnn.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Config loaded successfully")
        
        # Check data files
        graph_file = PROJECT_ROOT / config["data"]["graph_file"]
        features_file = PROJECT_ROOT / config["data"]["node_features_file"]
        
        if not graph_file.exists():
            logger.error(f"❌ Missing graph file: {graph_file}")
            return False
        logger.info(f"✅ Found graph file: {graph_file}")
        
        if not features_file.exists():
            logger.error(f"❌ Missing features file: {features_file}")
            return False
        logger.info(f"✅ Found features file: {features_file}")
        
        # Test loading data
        import pandas as pd
        
        edges_df = pd.read_parquet(graph_file)
        logger.info(f"✅ Loaded graph: {len(edges_df)} edges")
        
        features_df = pd.read_parquet(features_file)
        logger.info(f"✅ Loaded features: {len(features_df)} nodes")
        
        # Test model import
        from models.gnn.inference import UpcyclingGNN
        logger.info("✅ GNN model imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GNN training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    logger.info("🧪 TESTING TRAINING INTEGRATION")
    logger.info("=" * 80)
    
    results = []
    
    # Test LLM
    results.append(("LLM Training", test_llm_training_setup()))
    
    # Test Vision
    results.append(("Vision Training", test_vision_training_setup()))
    
    # Test GNN
    results.append(("GNN Training", test_gnn_training_setup()))
    
    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 80)
    if all_passed:
        logger.info("🎉 ALL TRAINING INTEGRATION TESTS PASSED!")
    else:
        logger.error("❌ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

