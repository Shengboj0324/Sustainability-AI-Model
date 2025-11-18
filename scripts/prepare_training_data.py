"""
Prepare Training Data
Convert datasets to proper format for training
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_llm_data_to_chat_format(input_file: str, output_dir: str, train_split: float = 0.9):
    """
    Convert instruction-based dataset to chat format
    
    Input format:
        {"instruction": "...", "input": "...", "output": "...", "category": "..."}
    
    Output format:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    logger.info(f"Converting {input_file} to chat format")
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    
    # Convert to chat format
    chat_data = []
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        category = item.get("category", "general")
        
        # Combine instruction and input
        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
        
        # Create chat format
        chat_item = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": output}
            ],
            "category": category
        }
        chat_data.append(chat_item)
    
    # Shuffle data
    random.seed(42)
    random.shuffle(chat_data)
    
    # Split into train/val
    split_idx = int(len(chat_data) * train_split)
    train_data = chat_data[:split_idx]
    val_data = chat_data[split_idx:]
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train data
    train_file = os.path.join(output_dir, "sustainability_qa_train.jsonl")
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved train data to {train_file}")
    
    # Save val data
    val_file = os.path.join(output_dir, "sustainability_qa_val.jsonl")
    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved val data to {val_file}")
    
    return train_file, val_file


def create_minimal_training_files(output_dir: str):
    """Create minimal training files for other categories"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Upcycling examples
    upcycling_train = [
        {
            "messages": [
                {"role": "user", "content": "How can I upcycle plastic bottles?"},
                {"role": "assistant", "content": "Plastic bottles can be upcycled into planters, bird feeders, organizers, or decorative items. Clean them thoroughly, cut as needed, and decorate. Always ensure sharp edges are smoothed for safety."}
            ],
            "category": "upcycling_ideas"
        },
        {
            "messages": [
                {"role": "user", "content": "What can I make from old t-shirts?"},
                {"role": "assistant", "content": "Old t-shirts can be transformed into reusable shopping bags, cleaning rags, braided rugs, or even yarn for knitting. Cut into strips and braid or weave for various projects."}
            ],
            "category": "upcycling_ideas"
        }
    ]
    
    upcycling_val = [
        {
            "messages": [
                {"role": "user", "content": "Can I upcycle glass jars?"},
                {"role": "assistant", "content": "Yes! Glass jars make excellent storage containers, candle holders, vases, or terrariums. Clean thoroughly and remove labels before use."}
            ],
            "category": "upcycling_ideas"
        }
    ]
    
    # Organization routing examples
    org_train = [
        {
            "messages": [
                {"role": "user", "content": "Where can I donate old clothes?"},
                {"role": "assistant", "content": "You can donate old clothes to Goodwill, Salvation Army, local thrift stores, or clothing banks. Many organizations offer pickup services."}
            ],
            "category": "organization_search"
        }
    ]
    
    org_val = [
        {
            "messages": [
                {"role": "user", "content": "Where can I recycle electronics?"},
                {"role": "assistant", "content": "Electronics can be recycled at Best Buy, Staples, or local e-waste recycling centers. Many manufacturers also offer take-back programs."}
            ],
            "category": "organization_search"
        }
    ]
    
    # Save files
    files = [
        ("upcycling_qa_train.jsonl", upcycling_train),
        ("upcycling_qa_val.jsonl", upcycling_val),
        ("org_routing_train.jsonl", org_train),
        ("org_routing_val.jsonl", org_val)
    ]
    
    for filename, data in files:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Created {filepath} with {len(data)} examples")


def convert_gnn_data_to_parquet(input_file: str, output_dir: str):
    """
    Convert GNN JSON data to parquet format

    Input format:
        {"nodes": [...], "edges": [...]}

    Output format:
        - graph.parquet: edge list
        - node_features.parquet: node features
    """
    logger.info(f"Converting {input_file} to parquet format")

    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    logger.info(f"Loaded {len(nodes)} nodes, {len(edges)} edges")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert edges to DataFrame
    edges_df = pd.DataFrame(edges)
    edges_file = os.path.join(output_dir, "graph.parquet")
    edges_df.to_parquet(edges_file, index=False)
    logger.info(f"Saved edges to {edges_file}")

    # Convert nodes to features DataFrame
    # Create simple one-hot encoding for node types
    node_types = list(set(node["type"] for node in nodes))
    type_to_idx = {t: i for i, t in enumerate(node_types)}

    features = []
    for node in nodes:
        # One-hot encode type
        feature_vec = [0.0] * len(node_types)
        feature_vec[type_to_idx[node["type"]]] = 1.0
        features.append({
            "node_id": node["id"],
            **{f"feature_{i}": v for i, v in enumerate(feature_vec)}
        })

    features_df = pd.DataFrame(features)
    features_file = os.path.join(output_dir, "node_features.parquet")
    features_df.to_parquet(features_file, index=False)
    logger.info(f"Saved node features to {features_file}")

    return edges_file, features_file


def main():
    """Main function"""
    # Paths
    project_root = Path(__file__).parent.parent

    # LLM data - use ultra-expanded if available
    ultra_expanded = project_root / "data" / "llm_training_ultra_expanded.json"
    expanded = project_root / "data" / "llm_training_expanded.json"

    if ultra_expanded.exists():
        llm_input = ultra_expanded
        logger.info(f"Using ultra-expanded dataset with edge cases")
    else:
        llm_input = expanded
        logger.info(f"Using standard expanded dataset")

    llm_output = project_root / "data" / "processed" / "llm_sft"

    # GNN data
    gnn_input = project_root / "data" / "gnn_training_expanded.json"
    gnn_output = project_root / "data" / "processed" / "gnn"

    # Convert LLM dataset
    logger.info("=" * 80)
    logger.info("Converting LLM data...")
    logger.info("=" * 80)
    convert_llm_data_to_chat_format(str(llm_input), str(llm_output))
    create_minimal_training_files(str(llm_output))

    # Convert GNN dataset
    logger.info("=" * 80)
    logger.info("Converting GNN data...")
    logger.info("=" * 80)

    # Import pandas here (only needed for GNN)
    global pd
    import pandas as pd

    convert_gnn_data_to_parquet(str(gnn_input), str(gnn_output))

    logger.info("=" * 80)
    logger.info("✅ ALL TRAINING DATA PREPARATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

