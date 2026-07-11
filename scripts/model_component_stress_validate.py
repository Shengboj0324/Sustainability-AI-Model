#!/usr/bin/env python3
"""Bounded neural/model component stress validation.

This complements the exhaustive checkpoint accuracy script. It is designed for
CI and deployment preflight: deterministic, bounded, and honest about warnings.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Check:
    name: str
    passed: bool
    severity: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


class StressReport:
    def __init__(self) -> None:
        self.started = time.time()
        self.checks: list[Check] = []

    def add(self, name: str, passed: bool, severity: str, detail: str, **metrics: Any) -> None:
        self.checks.append(Check(name, passed, severity, detail, metrics))

    def payload(self) -> dict[str, Any]:
        errors = [c for c in self.checks if not c.passed and c.severity == "error"]
        warnings = [c for c in self.checks if not c.passed and c.severity == "warning"]
        return {
            "ok": not errors,
            "summary": {
                "passed": len([c for c in self.checks if c.passed]),
                "errors": len(errors),
                "warnings": len(warnings),
                "duration_ms": round((time.time() - self.started) * 1000, 2),
            },
            "checks": [asdict(c) for c in self.checks],
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def check_vision_architectures(report: StressReport) -> None:
    from models.vision.classifier import MultiHeadClassifier
    from models.vision.physics_informed_classifier import (
        PhysicsInformedWasteClassifier,
        decision_violation_rate,
    )

    torch.manual_seed(42)
    sample = torch.randn(2, 3, 224, 224)
    try:
        with torch.no_grad():
            legacy = MultiHeadClassifier(
                backbone="resnet18",
                num_classes_item=30,
                num_classes_material=15,
                num_classes_bin=4,
                pretrained=False,
            ).eval()
            item, material, bin_logits = legacy(sample)
        finite = torch.isfinite(item).all() and torch.isfinite(material).all() and torch.isfinite(bin_logits).all()
        report.add(
            "vision_legacy_architecture_forward",
            bool(finite) and item.shape == (2, 30) and material.shape == (2, 15) and bin_logits.shape == (2, 4),
            "error",
            "legacy deployed multi-head architecture must run finite forward pass",
            item_shape=tuple(item.shape),
            material_shape=tuple(material.shape),
            bin_shape=tuple(bin_logits.shape),
        )
    except Exception as exc:
        report.add(
            "vision_legacy_architecture_forward",
            False,
            "error",
            "legacy deployed multi-head architecture forward failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        with torch.no_grad():
            physics = PhysicsInformedWasteClassifier(
                backbone="resnet18",
                pretrained=False,
                consistency_mode="soft",
            ).eval()
            heads = physics(sample)
            violation_rate = decision_violation_rate(heads.item_logits)
        finite = (
            torch.isfinite(heads.item_logits).all()
            and torch.isfinite(heads.material_logits).all()
            and torch.isfinite(heads.bin_logits).all()
        )
        report.add(
            "vision_physics_architecture_forward",
            bool(finite)
            and heads.item_logits.shape == (2, 30)
            and heads.material_logits.shape == (2, 16)
            and heads.bin_logits.shape == (2, 6)
            and violation_rate == 0.0,
            "error",
            "physics-informed architecture must run finite forward pass with legal taxonomy decisions",
            item_shape=tuple(heads.item_logits.shape),
            material_shape=tuple(heads.material_logits.shape),
            bin_shape=tuple(heads.bin_logits.shape),
            decision_violation_rate=violation_rate,
        )
    except Exception as exc:
        report.add(
            "vision_physics_architecture_forward",
            False,
            "error",
            "physics-informed architecture forward failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def check_gnn_forward(report: StressReport, config_path: Path) -> None:
    from models.gnn.inference import GATv2Model
    from training.gnn.train_gnn import load_graph_data

    cfg = _load_yaml(config_path)
    try:
        data = load_graph_data(cfg)
        model_cfg = cfg["model"]
        model = GATv2Model(
            in_channels=data.x.shape[1],
            hidden_channels=int(model_cfg["hidden_dim"]),
            out_channels=int(model_cfg["output_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            num_heads=int(model_cfg["num_heads"]),
            dropout=0.0,
            attention_dropout=0.0,
        ).eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        report.add(
            "gnn_canonical_forward",
            out.shape == (data.x.shape[0], int(model_cfg["output_dim"])) and bool(torch.isfinite(out).all()),
            "error",
            "GNN must produce finite embeddings for the canonical processed graph",
            nodes=data.x.shape[0],
            edges=data.edge_index.shape[1],
            input_dim=data.x.shape[1],
            output_shape=tuple(out.shape),
            mean_abs=float(out.abs().mean()),
        )
    except Exception as exc:
        report.add(
            "gnn_canonical_forward",
            False,
            "error",
            "GNN canonical graph forward failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def check_checkpoint_deployability(report: StressReport, config_path: Path, max_gb: float) -> None:
    cfg = _load_yaml(config_path)
    checkpoint_dir = ROOT / cfg.get("training", {}).get("output_dir", "models/vision/classifier")
    checkpoint = checkpoint_dir / "best_model.pth"
    inference_checkpoint = checkpoint_dir / "inference_model.pth"
    exists = checkpoint.exists()
    size_gb = checkpoint.stat().st_size / (1024**3) if exists else 0.0
    report.add(
        "vision_training_checkpoint_present",
        exists and size_gb > 0,
        "error",
        "vision checkpoint must exist and be non-empty",
        path=str(checkpoint),
        size_gb=round(size_gb, 3),
    )
    inference_exists = inference_checkpoint.exists()
    inference_size_gb = inference_checkpoint.stat().st_size / (1024**3) if inference_exists else 0.0
    report.add(
        "vision_inference_checkpoint_mobile_adjacent_size",
        inference_exists and inference_size_gb <= max_gb,
        "error" if not inference_exists else "warning",
        "inference-only vision checkpoint must exist and fit bounded deployment threshold",
        path=str(inference_checkpoint),
        size_gb=round(inference_size_gb, 3),
        threshold_gb=max_gb,
        source_training_size_gb=round(size_gb, 3),
    )


def check_3d_capability(report: StressReport) -> None:
    try:
        from models.vision.depth_geometry import CameraIntrinsics, DepthGeometryAnalyzer

        depth = np.full((4, 5), 1.0, dtype="float32")
        buffer = io.BytesIO()
        np.save(buffer, depth)
        depth_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        result = DepthGeometryAnalyzer().analyze(
            depth_b64=depth_b64,
            depth_format="npy",
            depth_unit_scale=1.0,
            intrinsics=CameraIntrinsics(fx=200.0, fy=200.0, cx=2.0, cy=1.5, width=5, height=4),
        )
        report.add(
            "three_d_depth_geometry_contract",
            result.point_count == 20 and result.model_available is False,
            "error",
            "Depth/RGB-D geometry contract must decode, validate, and project calibrated depth",
            point_count=result.point_count,
            valid_pixel_ratio=result.valid_pixel_ratio,
            model_available=result.model_available,
        )

    except Exception as exc:
        report.add(
            "three_d_depth_geometry_contract",
            False,
            "error",
            "Depth/RGB-D geometry contract failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        from models.vision.depth_geometry import camera_stabilized_flow

        pose_t = np.eye(4, dtype="float64")
        pose_future = np.eye(4, dtype="float64")
        pose_future[0, 3] = 3.0
        points_t = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype="float64")
        points_future = np.array([[-2.0, 0.0, 2.0], [-3.0, 1.0, 3.0]], dtype="float64")
        stabilized_flow = camera_stabilized_flow(points_t, points_future, pose_t, pose_future)
        report.add(
            "three_d_camera_stabilized_flow_invariant",
            bool(np.allclose(stabilized_flow, np.zeros_like(stabilized_flow), rtol=1e-9, atol=1e-9)),
            "error",
            "Camera-stabilized 3D flow must remove ego-motion for static world points",
            max_abs_flow=float(np.abs(stabilized_flow).max()),
        )
    except Exception as exc:
        report.add(
            "three_d_camera_stabilized_flow_invariant",
            False,
            "error",
            "Camera-stabilized 3D flow invariant failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    report.add(
        "three_d_learned_model_capability",
        False,
        "warning",
        "No trained RGB-D/stereo/depth/point-cloud waste classifier or benchmark dataset is configured; current 3D runtime is geometry analysis only",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-config", default="configs/vision_cls.yaml")
    parser.add_argument("--gnn-config", default="configs/gnn.yaml")
    parser.add_argument("--max-vision-checkpoint-gb", type=float, default=1.5)
    parser.add_argument("--output", default="outputs/model_readiness/model_component_stress_report.json")
    args = parser.parse_args()

    report = StressReport()
    check_checkpoint_deployability(report, ROOT / args.vision_config, args.max_vision_checkpoint_gb)
    check_vision_architectures(report)
    check_gnn_forward(report, ROOT / args.gnn_config)
    check_3d_capability(report)

    payload = report.payload()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Model component stress report written to {output}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
