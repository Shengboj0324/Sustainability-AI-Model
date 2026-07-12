#!/usr/bin/env python3
"""Validate calibrated RGB-D/3D manifest readiness before training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.vision.rgbd_dataset import RGBD3DManifestDataset, R2ObjectCache


@dataclass
class ManifestCheck:
    name: str
    passed: bool
    severity: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


class ManifestReport:
    def __init__(self) -> None:
        self.started = time.time()
        self.checks: list[ManifestCheck] = []

    def add(self, name: str, passed: bool, severity: str, detail: str, **metrics: Any) -> None:
        self.checks.append(ManifestCheck(name, passed, severity, detail, metrics))

    def payload(self) -> dict[str, Any]:
        errors = [check for check in self.checks if not check.passed and check.severity == "error"]
        warnings = [check for check in self.checks if not check.passed and check.severity == "warning"]
        return {
            "ok": not errors,
            "summary": {
                "passed": len([check for check in self.checks if check.passed]),
                "errors": len(errors),
                "warnings": len(warnings),
                "duration_ms": round((time.time() - self.started) * 1000, 2),
            },
            "checks": [asdict(check) for check in self.checks],
        }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_manifest(config: dict[str, Any], manifest_override: str | None, max_samples_override: int | None) -> dict[str, Any]:
    report = ManifestReport()
    data_cfg = config["data"]
    manifest_path = ROOT / (manifest_override or data_cfg["manifest_path"])
    max_samples = max_samples_override or int(config.get("validation", {}).get("max_samples_per_split", 64))
    require_boxes = bool(data_cfg.get("require_boxes_3d", True))

    report.add(
        "manifest_file_present",
        manifest_path.exists() and manifest_path.is_file(),
        "error",
        "RGB-D manifest file must exist",
        manifest_path=str(manifest_path),
    )
    if not manifest_path.exists():
        return report.payload()

    cache = R2ObjectCache(
        cache_dir=ROOT / data_cfg.get("cache_dir", "data/cache/r2_3d"),
        endpoint_url=data_cfg.get("r2_endpoint_url"),
        bucket=data_cfg.get("r2_bucket"),
    )

    split_counts: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    for split in data_cfg.get("expected_splits", ["train", "val", "test"]):
        try:
            dataset = RGBD3DManifestDataset(manifest_path, split=split, cache=cache, require_boxes=require_boxes)
            split_counts[split] = len(dataset)
            limit = min(max_samples, len(dataset))
            for idx in range(limit):
                sample = dataset[idx]
                if not sample["boxes_3d"] and require_boxes:
                    raise ValueError("sample has no 3D boxes")
        except Exception as exc:
            failures.append({"split": split, "error": f"{type(exc).__name__}: {exc}"})
            split_counts[split] = 0

    report.add(
        "manifest_split_integrity",
        not failures,
        "error",
        "All configured RGB-D splits decode and validate" if not failures else "RGB-D split validation failed",
        split_counts=split_counts,
        failures=failures[:20],
        max_samples_per_split=max_samples,
    )
    return report.payload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vision_3d_training.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--output", default="outputs/data_pipeline/rgbd_manifest_validation.json")
    args = parser.parse_args()

    payload = validate_manifest(load_yaml(ROOT / args.config), args.manifest, args.max_samples_per_split)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
