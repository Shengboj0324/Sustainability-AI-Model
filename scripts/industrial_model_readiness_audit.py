#!/usr/bin/env python3
"""Industrial model readiness audit for accuracy, data integrity, and 3D claims.

This is intentionally lightweight: it validates deployability contracts without
loading multi-GB checkpoints. Heavy inference accuracy still belongs in
scripts/exhaustive_vision_stress_test.py after model assets are local and warm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
THREE_D_KEYWORDS = (
    "depth",
    "stereo",
    "rgbd",
    "rgb-d",
    "point cloud",
    "pointcloud",
    "mesh",
    "nerf",
    "reconstruction",
)


@dataclass
class GateResult:
    name: str
    passed: bool
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class ReadinessAudit:
    def __init__(self, root: Path, sample_per_split: int, min_images: int):
        self.root = root
        self.sample_per_split = sample_per_split
        self.min_images = min_images
        self.results: list[GateResult] = []

    def add(self, name: str, passed: bool, severity: str, message: str, **details: Any) -> None:
        self.results.append(GateResult(name, passed, severity, message, details))

    def load_yaml(self, path: Path) -> dict[str, Any]:
        if yaml is None:
            self.add("yaml_dependency", False, "error", "PyYAML is unavailable")
            return {}
        if not path.exists():
            self.add("config_exists", False, "error", f"Missing config: {path}")
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def audit_vision_config(self, config_path: Path) -> dict[str, Any]:
        cfg = self.load_yaml(config_path)
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("training", {})

        data_dir = self.root / data_cfg.get("data_dir", "")
        self.add(
            "active_vision_data_dir",
            data_dir.exists(),
            "error",
            "Active vision data directory must exist",
            data_dir=str(data_dir),
        )
        self.add(
            "active_vision_uses_clean_split",
            "vision_cls_clean" in str(data_dir),
            "error",
            "Active vision training must use the clean leakage-free split",
            data_dir=str(data_dir),
        )

        item_classes = model_cfg.get("item_classes", [])
        material_classes = model_cfg.get("material_classes", [])
        self.add(
            "vision_item_class_count",
            len(item_classes) == int(model_cfg.get("num_classes_item", -1)) == 30,
            "error",
            "Vision item class count must match configured output dimension",
            configured=model_cfg.get("num_classes_item"),
            listed=len(item_classes),
        )
        self.add(
            "vision_material_class_count",
            len(material_classes) == int(model_cfg.get("num_classes_material", -1)),
            "error",
            "Vision material class list must match configured output dimension",
            configured=model_cfg.get("num_classes_material"),
            listed=len(material_classes),
        )
        self.add(
            "vision_bin_contract_declared",
            int(model_cfg.get("num_classes_bin", -1)) >= 4,
            "error",
            "Vision bin output dimension must be declared",
            configured=model_cfg.get("num_classes_bin"),
        )

        checkpoint_dir = self.root / train_cfg.get("output_dir", "models/vision/classifier")
        checkpoint = checkpoint_dir / "best_model.pth"
        inference_checkpoint = checkpoint_dir / "inference_model.pth"
        self.audit_checkpoint_file(checkpoint, "vision_classifier_training_checkpoint", warn_over_gb=None)
        self.audit_checkpoint_file(inference_checkpoint, "vision_classifier_inference_checkpoint", warn_over_gb=1.5)
        return cfg

    def audit_checkpoint_file(self, path: Path, gate_name: str, warn_over_gb: float | None) -> None:
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        size_gb = size_bytes / (1024**3)
        self.add(
            gate_name,
            exists and size_bytes > 0,
            "error",
            "Checkpoint file must exist and be non-empty",
            path=str(path),
            size_bytes=size_bytes,
            size_gb=round(size_gb, 3),
        )
        if exists and warn_over_gb is not None:
            self.add(
                f"{gate_name}_deployable_size",
                size_gb <= warn_over_gb,
                "warning",
                "Checkpoint is very large for container/mobile-adjacent deployment; export/quantization is required",
                path=str(path),
                size_gb=round(size_gb, 3),
                threshold_gb=warn_over_gb,
            )

    def iter_images(self, split_dir: Path) -> list[Path]:
        if not split_dir.exists():
            return []
        images: list[Path] = []
        for path in split_dir.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTS and path.is_file():
                images.append(path)
        return sorted(images)

    def audit_image_splits(self, cfg: dict[str, Any]) -> None:
        data_cfg = cfg.get("data", {})
        split_paths = {
            "train": self.root / data_cfg.get("train_dir", ""),
            "val": self.root / data_cfg.get("val_dir", ""),
            "test": self.root / data_cfg.get("test_dir", ""),
        }
        split_images = {name: self.iter_images(path) for name, path in split_paths.items()}
        counts = {name: len(paths) for name, paths in split_images.items()}

        self.add(
            "vision_split_minimum_volume",
            sum(counts.values()) >= self.min_images and all(counts.values()),
            "error",
            "Vision splits must have enough images for meaningful training/evaluation",
            counts=counts,
            min_total_images=self.min_images,
        )

        hashes_by_split: dict[str, dict[str, Path]] = {}
        for split, paths in split_images.items():
            hashes_by_split[split] = {}
            for path in paths:
                digest = self.hash_file(path)
                hashes_by_split[split][digest] = path

        leaks = []
        split_names = list(hashes_by_split)
        for i, left in enumerate(split_names):
            for right in split_names[i + 1 :]:
                overlap = set(hashes_by_split[left]) & set(hashes_by_split[right])
                for digest in sorted(overlap)[:20]:
                    leaks.append(
                        {
                            "hash": digest,
                            left: str(hashes_by_split[left][digest]),
                            right: str(hashes_by_split[right][digest]),
                        }
                    )

        self.add(
            "vision_exact_split_leakage",
            not leaks,
            "error",
            "No exact image hashes may appear in more than one split",
            leak_count=len(leaks),
            sample=leaks[:5],
        )

        decode_failures = []
        for split, paths in split_images.items():
            for path in paths[: self.sample_per_split]:
                try:
                    with Image.open(path) as image:
                        image = ImageOps.exif_transpose(image)
                        image.verify()
                except Exception as exc:
                    decode_failures.append({"split": split, "path": str(path), "error": str(exc)})
        self.add(
            "vision_sample_decode_integrity",
            not decode_failures,
            "error",
            "Sampled images must decode with EXIF handling",
            sample_per_split=self.sample_per_split,
            failure_count=len(decode_failures),
            failures=decode_failures[:10],
        )

    def hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def audit_taxonomy_contract(self, cfg: dict[str, Any]) -> None:
        from models.vision import taxonomy as tx
        from models.vision.classifier import WasteClassifier

        model_cfg = cfg.get("model", {})
        item_classes = model_cfg.get("item_classes", [])
        material_classes = model_cfg.get("material_classes", [])
        serving_derives_canonical_disposal = hasattr(WasteClassifier, "_canonical_material_bin")

        self.add(
            "taxonomy_item_alignment",
            item_classes == tx.ITEM_CLASSES,
            "error",
            "Configured item classes must match canonical taxonomy order",
            configured_count=len(item_classes),
            taxonomy_count=len(tx.ITEM_CLASSES),
        )

        taxonomy_extra_materials = sorted(set(tx.MATERIAL_CLASSES) - set(material_classes))
        self.add(
            "taxonomy_material_dimension_gap",
            not taxonomy_extra_materials or serving_derives_canonical_disposal,
            "warning",
            "Legacy material head gaps must be mitigated by taxonomy-derived serving decisions",
            extra_materials=taxonomy_extra_materials,
            configured_material_count=len(material_classes),
            taxonomy_material_count=len(tx.MATERIAL_CLASSES),
            serving_derives_canonical_disposal=serving_derives_canonical_disposal,
        )

        self.add(
            "taxonomy_bin_dimension_gap",
            int(model_cfg.get("num_classes_bin", 0)) >= len(tx.BIN_CLASSES) or serving_derives_canonical_disposal,
            "warning",
            "Legacy bin head gaps must be mitigated by taxonomy-derived serving decisions",
            configured_bins=model_cfg.get("num_classes_bin"),
            taxonomy_bins=tx.BIN_CLASSES,
            serving_derives_canonical_disposal=serving_derives_canonical_disposal,
        )

    def audit_3d_vision_claims(self) -> None:
        candidate_files = [
            self.root / "models/vision",
            self.root / "services/vision_service",
            self.root / "training/vision",
            self.root / "configs",
        ]
        matches = []
        for base in candidate_files:
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.suffix.lower() not in {".py", ".yaml", ".yml", ".md"} or not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore").lower()
                except Exception:
                    continue
                for keyword in THREE_D_KEYWORDS:
                    if keyword in text:
                        matches.append({"file": str(path), "keyword": keyword})
                        break

        real_3d_files = [
            match
            for match in matches
            if "visualize_gnn" not in match["file"] and "depth_multiple" not in match["keyword"]
        ]
        depth_module = self.root / "models/vision/depth_geometry.py"
        vision_service = self.root / "services/vision_service/server_v2.py"
        gateway_router = self.root / "services/api_gateway/routers/vision.py"
        tests = self.root / "tests/unit/test_depth_geometry.py"
        endpoint_tests = self.root / "tests/unit/test_vision_3d_endpoint_contract.py"
        service_text = vision_service.read_text(encoding="utf-8", errors="ignore") if vision_service.exists() else ""
        gateway_text = gateway_router.read_text(encoding="utf-8", errors="ignore") if gateway_router.exists() else ""
        depth_text = depth_module.read_text(encoding="utf-8", errors="ignore") if depth_module.exists() else ""
        self.add(
            "three_d_depth_geometry_contract",
            depth_module.exists()
            and tests.exists()
            and "/analyze-3d" in service_text
            and "/analyze-3d" in gateway_text,
            "error",
            "Depth/RGB-D geometry ingestion, validation, and API contract must exist",
            depth_module=str(depth_module),
            service_endpoint="/analyze-3d" in service_text,
            gateway_endpoint="/analyze-3d" in gateway_text,
            tests=str(tests),
        )
        self.add(
            "three_d_camera_stabilized_flow_contract",
            depth_module.exists()
            and endpoint_tests.exists()
            and "camera_stabilized_flow" in depth_text
            and "/analyze-3d-flow" in service_text
            and "/analyze-3d-flow" in gateway_text,
            "error",
            "Camera-stabilized 3D flow contract must exist for mobile ego-motion compensation",
            depth_function="camera_stabilized_flow" in depth_text,
            service_endpoint="/analyze-3d-flow" in service_text,
            gateway_endpoint="/analyze-3d-flow" in gateway_text,
            endpoint_tests=str(endpoint_tests),
        )
        self.add(
            "three_d_learned_model_capability",
            False,
            "warning",
            "No trained 3D waste-classification model, labeled RGB-D dataset, or benchmark is present; current 3D support is geometry analysis only",
            detected_keywords=real_3d_files[:20],
            required_external_assets=[
                "calibrated RGB-D/LiDAR samples",
                "camera intrinsics/extrinsics",
                "item/material/bin labels",
                "train/val/test split",
                "3D benchmark metrics",
            ],
        )

    def audit_gnn(self) -> None:
        self.audit_checkpoint_file(self.root / "models/gnn/ckpts/best_model.pth", "gnn_checkpoint", warn_over_gb=0.25)
        canonical_checkpoint = self.root / "models/gnn/ckpts_canonical_smoke/best_model.pth"
        canonical_metrics = self.root / "models/gnn/ckpts_canonical_smoke/training_metrics.json"
        self.audit_checkpoint_file(canonical_checkpoint, "gnn_canonical_checkpoint", warn_over_gb=0.25)
        if canonical_metrics.exists():
            try:
                metrics = json.loads(canonical_metrics.read_text(encoding="utf-8"))
                self.add(
                    "gnn_canonical_training_metrics",
                    metrics.get("nodes") == 72
                    and metrics.get("edges") == 252
                    and float(metrics.get("best_val_acc", 0.0)) >= 0.75
                    and float(metrics.get("test_acc", 0.0)) >= 0.70,
                    "error",
                    "Canonical GNN checkpoint must be trained and benchmarked on the upgraded graph",
                    **metrics,
                )
            except Exception as exc:
                self.add(
                    "gnn_canonical_training_metrics",
                    False,
                    "error",
                    "Could not read canonical GNN training metrics",
                    error=f"{type(exc).__name__}: {exc}",
                )
        else:
            self.add(
                "gnn_canonical_training_metrics",
                False,
                "error",
                "Canonical GNN training metrics must exist",
                path=str(canonical_metrics),
            )
        self.audit_checkpoint_file(self.root / "checkpoints/best_gnn_gatv2.pth", "gnn_gatv2_checkpoint", warn_over_gb=0.25)
        graph = self.root / "data/processed/gnn/graph.parquet"
        features = self.root / "data/processed/gnn/node_features.parquet"
        self.add(
            "gnn_processed_contract_files",
            graph.exists() and features.exists(),
            "error",
            "Processed GNN graph and node feature parquet files must exist",
            graph=str(graph),
            node_features=str(features),
        )
        if graph.exists() and features.exists():
            try:
                import pandas as pd

                edge_df = pd.read_parquet(graph)
                node_df = pd.read_parquet(features)
                node_types = (
                    set(node_df["node_type"].dropna().astype(str))
                    if "node_type" in node_df.columns
                    else set()
                )
                relationships = (
                    set(edge_df["relationship"].dropna().astype(str))
                    if "relationship" in edge_df.columns
                    else set()
                )
                required_node_types = {"ItemType", "Material", "Bin", "ProductIdea", "Hazard"}
                required_relationships = {
                    "MADE_OF",
                    "GOES_TO",
                    "DISPOSAL_ROUTE",
                    "CAN_BE_UPCYCLED_TO",
                    "HAS_HAZARD",
                    "SIMILAR_TO",
                }
                feature_cols = [c for c in node_df.columns if c.startswith("feature_")]
                missing_node_types = sorted(required_node_types - node_types)
                missing_relationships = sorted(required_relationships - relationships)
                self.add(
                    "gnn_canonical_relationship_contract",
                    len(node_df) >= 60
                    and len(edge_df) >= 170
                    and len(feature_cols) == 128
                    and not missing_node_types
                    and not missing_relationships,
                    "error",
                    "GNN graph must expose canonical sustainability relationship types for training and inference",
                    nodes=len(node_df),
                    edges=len(edge_df),
                    feature_dim=len(feature_cols),
                    missing_node_types=missing_node_types,
                    missing_relationships=missing_relationships,
                )
            except Exception as exc:
                self.add(
                    "gnn_canonical_relationship_contract",
                    False,
                    "error",
                    "Could not inspect GNN parquet relationship contract",
                    error=f"{type(exc).__name__}: {exc}",
                )

    def run(self, config_path: Path) -> dict[str, Any]:
        cfg = self.audit_vision_config(config_path)
        self.audit_image_splits(cfg)
        self.audit_taxonomy_contract(cfg)
        self.audit_3d_vision_claims()
        self.audit_gnn()

        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]
        return {
            "summary": {
                "passed": len([r for r in self.results if r.passed]),
                "warnings": len(warnings),
                "errors": len(errors),
                "industrial_ready": not errors and not warnings,
                "staging_ready": not errors,
            },
            "results": [asdict(result) for result in self.results],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vision_cls.yaml")
    parser.add_argument("--sample-per-split", type=int, default=100)
    parser.add_argument("--min-images", type=int, default=10000)
    parser.add_argument("--output", default="outputs/model_readiness/industrial_model_readiness_report.json")
    args = parser.parse_args()

    root = Path.cwd()
    audit = ReadinessAudit(root=root, sample_per_split=args.sample_per_split, min_images=args.min_images)
    report = audit.run(root / args.config)

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report written to {output}")
    return 0 if report["summary"]["staging_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
