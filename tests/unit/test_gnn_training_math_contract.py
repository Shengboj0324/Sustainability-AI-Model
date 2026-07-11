from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.gnn.train_gnn import create_train_val_test_split, negative_sampling
from models.gnn.graph_contract import build_graph_tables
from models.vision import taxonomy as tx


def _split_config():
    return {"data": {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25}}


def test_create_split_stores_train_only_message_passing_edges():
    torch.manual_seed(7)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2, 4, 5],
            [1, 0, 3, 2, 2, 0, 5, 4],
        ],
        dtype=torch.long,
    )
    data = Data(x=torch.randn(6, 4), edge_index=edge_index, num_nodes=6)

    split = create_train_val_test_split(data, _split_config())

    assert split.train_mask.dtype == torch.bool
    assert split.val_mask.dtype == torch.bool
    assert split.test_mask.dtype == torch.bool
    assert split.train_edge_index.shape[1] == int(split.train_mask.sum())
    assert torch.equal(split.train_edge_index, split.edge_index[:, split.train_mask])
    assert not torch.equal(split.train_edge_index, split.edge_index)


def test_negative_sampling_returns_unique_non_edges():
    torch.manual_seed(11)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=torch.long,
    )

    negatives = negative_sampling(edge_index=edge_index, num_nodes=6, num_neg_samples=8)
    original_edges = set(map(tuple, edge_index.t().tolist()))
    sampled_edges = list(map(tuple, negatives.t().tolist()))

    assert negatives.shape == (2, 8)
    assert len(sampled_edges) == len(set(sampled_edges))
    assert not (set(sampled_edges) & original_edges)
    assert all(src != dst for src, dst in sampled_edges)


def test_negative_sampling_zero_samples_returns_empty_edge_index():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    negatives = negative_sampling(edge_index=edge_index, num_nodes=3, num_neg_samples=0)

    assert negatives.shape == (2, 0)


def test_canonical_graph_has_required_taxonomy_relationships_and_finite_features():
    edges, nodes = build_graph_tables(input_dim=128)
    feature_cols = [c for c in nodes.columns if c.startswith("feature_")]
    node_lookup = {
        (row.node_type, row.name): int(row.node_id)
        for row in nodes[["node_id", "node_type", "name"]].itertuples(index=False)
    }
    edge_tuples = set(
        (int(row.source), int(row.target), row.relationship)
        for row in edges[["source", "target", "relationship"]].itertuples(index=False)
    )

    assert nodes["node_id"].tolist() == list(range(len(nodes)))
    assert torch.isfinite(torch.tensor(nodes[feature_cols].to_numpy(dtype="float32"))).all()

    for item, facts in tx.ITEM_FACTS.items():
        item_id = node_lookup[("ItemType", item)]
        material_id = node_lookup[("Material", facts.material)]
        bin_id = node_lookup[("Bin", facts.bin)]
        assert (item_id, material_id, "MADE_OF") in edge_tuples
        assert (material_id, item_id, "reverse_MADE_OF") in edge_tuples
        assert (item_id, bin_id, "DISPOSAL_ROUTE") in edge_tuples
        assert (bin_id, item_id, "reverse_DISPOSAL_ROUTE") in edge_tuples

    upcycling_edges = edges[edges["relationship"] == "CAN_BE_UPCYCLED_TO"]
    assert not upcycling_edges.empty
    assert upcycling_edges["difficulty"].between(0.0, 1.0).all()
    assert upcycling_edges["confidence"].between(0.0, 1.0).all()
