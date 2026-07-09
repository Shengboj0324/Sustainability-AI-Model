from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_pipeline_stress_validate import build_gnn_tables, image_is_decodable
from training.gnn.train_gnn import load_graph_data


def test_gnn_table_generator_matches_training_contract():
    edges, nodes = build_gnn_tables(input_dim=128)
    feature_cols = [c for c in nodes.columns if c.startswith("feature_")]
    relationships = set(edges["relationship"])
    node_types = set(nodes["node_type"])

    assert len(nodes) >= 60
    assert len(edges) >= 170
    assert len(feature_cols) == 128
    assert {"node_id", "node_type", "name"}.issubset(nodes.columns)
    assert {"source", "target", "relationship"}.issubset(edges.columns)
    assert edges[["source", "target"]].max().max() < len(nodes)
    assert {"ItemType", "Material", "Bin", "ProductIdea", "Hazard"}.issubset(node_types)
    assert {"MADE_OF", "GOES_TO", "DISPOSAL_ROUTE", "CAN_BE_UPCYCLED_TO", "HAS_HAZARD", "SIMILAR_TO"}.issubset(relationships)


def test_gnn_trainer_loads_processed_parquet_contract():
    with open("configs/gnn.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data = load_graph_data(config)

    assert data.x.shape[0] >= 60
    assert data.x.shape[1] == 128
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] >= 170


def test_image_decode_helper_rejects_non_image(tmp_path: Path):
    bad = tmp_path / "not-an-image.jpg"
    bad.write_text("this is not an image", encoding="utf-8")

    ok, error = image_is_decodable(bad)

    assert ok is False
    assert error
