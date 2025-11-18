"""
Comprehensive Dataset Validation
Validate all datasets and their integration with training files
"""

import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_llm_datasets():
    """Validate LLM training datasets"""
    logger.info("=" * 80)
    logger.info("VALIDATING LLM DATASETS")
    logger.info("=" * 80)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed" / "llm_sft"
    
    files = [
        "sustainability_qa_train.jsonl",
        "sustainability_qa_val.jsonl",
        "upcycling_qa_train.jsonl",
        "upcycling_qa_val.jsonl",
        "org_routing_train.jsonl",
        "org_routing_val.jsonl"
    ]
    
    total_examples = 0
    for filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            logger.error(f"❌ Missing file: {filepath}")
            continue
        
        # Count lines and validate format
        count = 0
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Validate required fields
                    assert "messages" in data, "Missing 'messages' field"
                    assert isinstance(data["messages"], list), "'messages' must be a list"
                    assert len(data["messages"]) > 0, "'messages' cannot be empty"
                    
                    for msg in data["messages"]:
                        assert "role" in msg, "Message missing 'role'"
                        assert "content" in msg, "Message missing 'content'"
                        assert msg["role"] in ["user", "assistant", "system"], f"Invalid role: {msg['role']}"
                    
                    count += 1
                except Exception as e:
                    logger.error(f"❌ Invalid data in {filename}: {e}")
                    return False
        
        logger.info(f"✅ {filename}: {count} examples")
        total_examples += count
    
    logger.info(f"✅ Total LLM examples: {total_examples}")
    return True


def validate_gnn_datasets():
    """Validate GNN training datasets"""
    logger.info("=" * 80)
    logger.info("VALIDATING GNN DATASETS")
    logger.info("=" * 80)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed" / "gnn"
    
    # Check parquet files
    try:
        import pandas as pd
        
        # Validate graph edges
        graph_file = data_dir / "graph.parquet"
        if not graph_file.exists():
            logger.error(f"❌ Missing file: {graph_file}")
            return False
        
        edges_df = pd.read_parquet(graph_file)
        logger.info(f"✅ Graph edges: {len(edges_df)} edges")
        logger.info(f"   Columns: {list(edges_df.columns)}")
        
        # Validate node features
        features_file = data_dir / "node_features.parquet"
        if not features_file.exists():
            logger.error(f"❌ Missing file: {features_file}")
            return False
        
        features_df = pd.read_parquet(features_file)
        logger.info(f"✅ Node features: {len(features_df)} nodes")
        logger.info(f"   Columns: {list(features_df.columns)}")
        
        return True
        
    except ImportError:
        logger.warning("⚠️  pandas not installed, skipping GNN validation")
        return True
    except Exception as e:
        logger.error(f"❌ GNN validation failed: {e}")
        return False


def validate_raw_datasets():
    """Validate raw JSON datasets"""
    logger.info("=" * 80)
    logger.info("VALIDATING RAW DATASETS")
    logger.info("=" * 80)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    datasets = {
        "llm_training_expanded.json": "list",
        "rag_knowledge_base_expanded.json": "list",
        "gnn_training_expanded.json": "dict",
        "organizations_database.json": "dict",  # Dict with categories
        "sustainability_knowledge_base.json": "dict"
    }
    
    for filename, expected_type in datasets.items():
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning(f"⚠️  Missing file: {filepath}")
            continue
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if expected_type == "list":
                assert isinstance(data, list), f"Expected list, got {type(data)}"
                logger.info(f"✅ {filename}: {len(data)} items")
            elif expected_type == "dict":
                assert isinstance(data, dict), f"Expected dict, got {type(data)}"
                logger.info(f"✅ {filename}: {len(data)} keys")
        
        except Exception as e:
            logger.error(f"❌ {filename}: {e}")
            return False
    
    return True


def main():
    """Main validation function"""
    logger.info("🔍 COMPREHENSIVE DATASET VALIDATION")
    logger.info("=" * 80)
    
    results = []
    
    # Validate raw datasets
    results.append(("Raw Datasets", validate_raw_datasets()))
    
    # Validate LLM datasets
    results.append(("LLM Datasets", validate_llm_datasets()))
    
    # Validate GNN datasets
    results.append(("GNN Datasets", validate_gnn_datasets()))
    
    # Summary
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 80)
    if all_passed:
        logger.info("🎉 ALL DATASETS VALIDATED SUCCESSFULLY!")
    else:
        logger.error("❌ SOME DATASETS FAILED VALIDATION")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

