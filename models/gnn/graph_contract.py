"""Canonical GNN graph contract for sustainability relationship reasoning.

The GNN training graph must stay aligned with the vision taxonomy.  This module
builds deterministic node/edge tables used by data validation, training fallback,
and repair tooling.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from models.vision import taxonomy as tx


@dataclass(frozen=True)
class UpcyclingPath:
    material: str
    product: str
    difficulty: float
    confidence: float


UPCYCLING_PATHS: tuple[UpcyclingPath, ...] = (
    UpcyclingPath("PET", "self_watering_planter", 0.35, 0.82),
    UpcyclingPath("HDPE", "storage_scoop", 0.30, 0.78),
    UpcyclingPath("LDPE", "plastic_film_ecobrick", 0.55, 0.62),
    UpcyclingPath("PP", "drawer_organizer", 0.40, 0.72),
    UpcyclingPath("glass", "refill_storage_jar", 0.20, 0.88),
    UpcyclingPath("aluminum", "metal_craft_sheet", 0.60, 0.60),
    UpcyclingPath("steel", "magnetic_tool_holder", 0.50, 0.66),
    UpcyclingPath("paper", "seedling_pots", 0.25, 0.80),
    UpcyclingPath("cardboard", "shipping_organizer", 0.25, 0.84),
    UpcyclingPath("coated_paper", "non_food_craft_material", 0.45, 0.55),
    UpcyclingPath("organic", "compost_feedstock", 0.20, 0.86),
    UpcyclingPath("textile", "repair_or_tote_fabric", 0.45, 0.76),
    UpcyclingPath("mixed", "donation_or_parts_recovery", 0.65, 0.58),
)

HAZARD_RULES: dict[str, tuple[str, ...]] = {
    "aerosol_cans": ("pressurized_container",),
    "plastic_shopping_bags": ("plastic_film_contamination",),
    "plastic_trash_bags": ("plastic_film_contamination",),
    "plastic_straws": ("small_sorting_escape",),
    "paper_cups": ("poly_lined_paper",),
    "tea_bags": ("possible_synthetic_mesh",),
    "styrofoam_cups": ("eps_low_acceptance",),
    "styrofoam_food_containers": ("eps_low_acceptance", "food_contamination_risk"),
}


def _stable_index(text: str, dim: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


def _feature_vector(node_type: str, name: str, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype="float32")
    vector[_stable_index(f"{node_type}:{name}", dim)] = 1.0
    vector[_stable_index(f"name:{name}", dim)] += 0.5

    type_offsets = {
        "ItemType": 0.20,
        "Material": 0.40,
        "Bin": 0.60,
        "ProductIdea": 0.80,
        "Hazard": 1.00,
    }
    vector[_stable_index(f"type:{node_type}", dim)] += type_offsets.get(node_type, 0.1)

    if node_type == "Material" and name in tx.MATERIAL_TO_IDX:
        props = np.asarray(tx.material_property_matrix()[tx.MATERIAL_TO_IDX[name]], dtype="float32")
        width = min(len(props), dim)
        vector[:width] += props[:width]

    return vector


def _add_bidirectional(edges: list[dict], source: int, target: int, relationship: str, **attrs) -> None:
    forward = {"source": source, "target": target, "relationship": relationship, **attrs}
    reverse = {"source": target, "target": source, "relationship": f"reverse_{relationship}", **attrs}
    edges.extend([forward, reverse])


def _node_records(input_dim: int) -> tuple[list[dict], dict[str, int]]:
    records: list[dict] = []
    node_ids: dict[str, int] = {}

    def add(node_type: str, name: str) -> None:
        node_id = len(records)
        node_ids[f"{node_type}:{name}"] = node_id
        records.append({"node_id": node_id, "node_type": node_type, "name": name})

    for name in tx.ITEM_CLASSES:
        add("ItemType", name)
    for name in tx.MATERIAL_CLASSES:
        add("Material", name)
    for name in tx.BIN_CLASSES:
        add("Bin", name)
    for hazard in sorted({h for hazards in HAZARD_RULES.values() for h in hazards}):
        add("Hazard", hazard)
    for product in sorted({path.product for path in UPCYCLING_PATHS}):
        add("ProductIdea", product)

    feature_df = pd.DataFrame(
        [_feature_vector(row["node_type"], row["name"], input_dim) for row in records],
        columns=[f"feature_{i}" for i in range(input_dim)],
    )
    return pd.concat([pd.DataFrame(records), feature_df], axis=1).to_dict("records"), node_ids


def build_graph_tables(input_dim: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build canonical graph parquet tables.

    Returns:
        edge_df, node_df with deterministic zero-indexed node IDs and feature
        columns named ``feature_0..feature_{input_dim-1}``.
    """

    nodes, node_ids = _node_records(input_dim)
    edges: list[dict] = []

    for item in tx.ITEM_CLASSES:
        facts = tx.ITEM_FACTS[item]
        item_id = node_ids[f"ItemType:{item}"]
        material_id = node_ids[f"Material:{facts.material}"]
        bin_id = node_ids[f"Bin:{facts.bin}"]
        _add_bidirectional(edges, item_id, material_id, "MADE_OF")
        _add_bidirectional(edges, item_id, bin_id, "DISPOSAL_ROUTE")
        for hazard in HAZARD_RULES.get(item, ()):
            _add_bidirectional(edges, item_id, node_ids[f"Hazard:{hazard}"], "HAS_HAZARD")

    material_bins: dict[str, set[str]] = {material: set() for material in tx.MATERIAL_CLASSES}
    for facts in tx.ITEM_FACTS.values():
        material_bins[facts.material].add(facts.bin)
    for material, bins in material_bins.items():
        material_id = node_ids[f"Material:{material}"]
        for bin_name in sorted(bins):
            _add_bidirectional(edges, material_id, node_ids[f"Bin:{bin_name}"], "GOES_TO")

    for path in UPCYCLING_PATHS:
        if path.material not in tx.MATERIAL_TO_IDX:
            continue
        _add_bidirectional(
            edges,
            node_ids[f"Material:{path.material}"],
            node_ids[f"ProductIdea:{path.product}"],
            "CAN_BE_UPCYCLED_TO",
            difficulty=path.difficulty,
            confidence=path.confidence,
        )

    for group, items in tx.CONFUSION_GROUPS.items():
        for left, right in _pairwise(items):
            _add_bidirectional(
                edges,
                node_ids[f"ItemType:{left}"],
                node_ids[f"ItemType:{right}"],
                "SIMILAR_TO",
                group=group,
            )

    edge_df = pd.DataFrame(edges)
    node_df = pd.DataFrame(nodes)
    return edge_df, node_df


def _pairwise(items: Iterable[str]) -> Iterable[tuple[str, str]]:
    items = list(items)
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            yield left, right

