#!/usr/bin/env python3
"""Strict data-pipeline stress validation and repair utilities.

This gate is intentionally stricter than the older smoke validators. It checks
the concrete contracts used when training starts:

- vision ImageFolder splits, taxonomy order, full image readability, and
  exact-hash split leakage;
- LLM SFT JSONL files listed in the active config;
- GNN parquet graph/node feature contracts listed in the active config;
- importability of training modules and dependency compatibility.

Repairs are opt-in and non-destructive. Vision repairs create a clean hardlinked
dataset view. GNN repairs regenerate parquet files from the production taxonomy
used by the GNN trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
REPORT_PATH = ROOT / "outputs" / "data_pipeline" / "data_pipeline_stress_report.json"


@dataclass
class Check:
    name: str
    passed: bool
    severity: str
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineReport:
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    checks: List[Check] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0

    def add(self, name: str, passed: bool, severity: str, detail: str, **metrics: Any) -> None:
        check = Check(name=name, passed=passed, severity=severity, detail=detail, metrics=metrics)
        self.checks.append(check)
        if passed:
            self.passed += 1
        elif severity == "warning":
            self.warnings += 1
        else:
            self.failed += 1

    @property
    def ok(self) -> bool:
        return self.failed == 0


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def iter_images(split_dir: Path) -> Iterable[Path]:
    for path in sorted(split_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_is_decodable(path: Path) -> Tuple[bool, Optional[str]]:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.verify()
        return True, None
    except Exception as exc:
        return False, str(exc)


def package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def check_dependency_contract(report: PipelineReport) -> None:
    transformers_ver = package_version("transformers")
    hub_ver = package_version("huggingface-hub")
    datasets_ver = package_version("datasets")
    required = {
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "pillow": package_version("pillow"),
        "pandas": package_version("pandas"),
        "pyarrow": package_version("pyarrow"),
        "albumentations": package_version("albumentations"),
        "pycocotools": package_version("pycocotools"),
    }
    missing = [name for name, installed in required.items() if not installed]

    hub_ok = True
    hub_detail = "compatible"
    if transformers_ver and hub_ver:
        parts = tuple(int(p) for p in hub_ver.split(".")[:2] if p.isdigit())
        hub_ok = bool(parts) and parts < (1, 0)
        if not hub_ok:
            hub_detail = f"transformers {transformers_ver} is incompatible with huggingface-hub {hub_ver}; require <1.0"
    elif transformers_ver and not hub_ver:
        hub_ok = False
        hub_detail = "transformers is installed but huggingface-hub is missing"

    report.add(
        "dependency_contract",
        not missing and hub_ok and bool(datasets_ver),
        "error",
        "training data dependencies are import-compatible" if not missing and hub_ok and datasets_ver else hub_detail,
        transformers=transformers_ver,
        huggingface_hub=hub_ver,
        datasets=datasets_ver,
        missing=missing,
        **required,
    )


def check_training_module_imports(report: PipelineReport) -> None:
    modules = [
        "training.vision.dataset",
        "training.vision.train_classifier",
        "training.vision.train_multihead",
        "training.llm.train_sft",
        "training.gnn.train_gnn",
    ]
    failures: Dict[str, str] = {}
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - reported in JSON
            failures[module] = f"{type(exc).__name__}: {exc}"
    report.add(
        "training_module_imports",
        not failures,
        "error",
        "all training modules import cleanly" if not failures else "training module imports failed",
        failures=failures,
    )


def collect_vision_inventory(vision_cfg: Dict[str, Any]) -> Tuple[Dict[str, CounterLike], Dict[str, List[Dict[str, Any]]]]:
    data_cfg = vision_cfg["data"]
    expected_classes = list(vision_cfg["model"]["item_classes"])
    class_counts: Dict[str, Dict[str, int]] = {}
    hash_index: Dict[str, List[Dict[str, Any]]] = {}

    for split in SPLITS:
        split_dir = ROOT / data_cfg[f"{split}_dir"]
        counts: Dict[str, int] = {}
        for cls in expected_classes:
            class_dir = split_dir / cls
            counts[cls] = sum(1 for _ in iter_images(class_dir)) if class_dir.exists() else 0
        class_counts[split] = counts

        for path in iter_images(split_dir):
            digest = sha1_file(path)
            hash_index.setdefault(digest, []).append(
                {
                    "split": split,
                    "class": path.parent.name,
                    "path": str(path.relative_to(ROOT)),
                    "size": path.stat().st_size,
                }
            )
    return class_counts, hash_index


CounterLike = Dict[str, int]


def check_vision_contract(report: PipelineReport, vision_cfg: Dict[str, Any]) -> Tuple[Dict[str, CounterLike], Dict[str, List[Dict[str, Any]]]]:
    data_cfg = vision_cfg["data"]
    expected_classes = list(vision_cfg["model"]["item_classes"])
    expected_count = int(vision_cfg["model"]["num_classes_item"])
    errors: List[str] = []

    if expected_count != len(expected_classes):
        errors.append(f"num_classes_item={expected_count} but item_classes has {len(expected_classes)} entries")

    for split in SPLITS:
        split_dir = ROOT / data_cfg[f"{split}_dir"]
        if not split_dir.exists():
            errors.append(f"{split} directory missing: {split_dir}")
            continue
        actual_classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if actual_classes != expected_classes:
            errors.append(f"{split} class order/names do not match config exactly")

    class_counts, hash_index = collect_vision_inventory(vision_cfg)
    per_split_totals = {split: sum(counts.values()) for split, counts in class_counts.items()}
    per_split_min = {split: min(counts.values()) if counts else 0 for split, counts in class_counts.items()}
    low_classes = {
        split: {cls: n for cls, n in counts.items() if n < 10}
        for split, counts in class_counts.items()
    }
    if any(low_classes.values()):
        errors.append(f"classes with fewer than 10 images detected: {low_classes}")

    report.add(
        "vision_folder_taxonomy_and_counts",
        not errors,
        "error",
        "vision ImageFolder contract matches config" if not errors else "; ".join(errors[:4]),
        totals=per_split_totals,
        min_per_class=per_split_min,
        expected_classes=len(expected_classes),
    )
    return class_counts, hash_index


def check_vision_image_integrity(report: PipelineReport, vision_cfg: Dict[str, Any], full_scan: bool, sample_size: int) -> None:
    data_cfg = vision_cfg["data"]
    paths: List[Path] = []
    for split in SPLITS:
        paths.extend(iter_images(ROOT / data_cfg[f"{split}_dir"]))
    if not full_scan and len(paths) > sample_size:
        rng = random.Random(42)
        paths = rng.sample(paths, sample_size)

    failures = []
    sizes = []
    start = time.time()
    for path in paths:
        ok, error = image_is_decodable(path)
        if ok:
            with Image.open(path) as image:
                sizes.append(image.size)
        else:
            failures.append({"path": str(path.relative_to(ROOT)), "error": error})
            if len(failures) >= 25:
                break
    duration_ms = (time.time() - start) * 1000

    report.add(
        "vision_image_decode_scan",
        not failures,
        "error",
        "all scanned images decoded" if not failures else "image decode failures detected",
        scanned=len(paths),
        full_scan=full_scan,
        failures=failures,
        duration_ms=round(duration_ms, 2),
        sample_sizes=sizes[:5],
    )


def check_vision_leakage(report: PipelineReport, hash_index: Dict[str, List[Dict[str, Any]]]) -> None:
    cross_split = []
    duplicate_files = 0
    duplicate_hashes = 0
    for digest, entries in hash_index.items():
        if len(entries) <= 1:
            continue
        duplicate_hashes += 1
        duplicate_files += len(entries)
        if len({entry["split"] for entry in entries}) > 1:
            cross_split.append({"sha1": digest, "entries": entries[:8]})

    report.add(
        "vision_exact_hash_leakage",
        not cross_split,
        "error",
        "no exact image hashes appear across train/val/test"
        if not cross_split
        else f"{len(cross_split)} exact hashes leak across splits",
        cross_split_hashes=len(cross_split),
        duplicate_hashes_total=duplicate_hashes,
        duplicate_files_total=duplicate_files,
        samples=cross_split[:10],
    )


def repair_vision_splits(vision_cfg: Dict[str, Any], output_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """Create a clean hardlinked ImageFolder view with one file per hash."""
    expected_classes = list(vision_cfg["model"]["item_classes"])
    _, hash_index = collect_vision_inventory(vision_cfg)
    unique_by_class: Dict[str, List[Dict[str, Any]]] = {cls: [] for cls in expected_classes}
    skipped_corrupt: List[Dict[str, str]] = []

    for digest, entries in sorted(hash_index.items()):
        entries_sorted = sorted(entries, key=lambda e: ({"train": 0, "val": 1, "test": 2}[e["split"]], e["path"]))
        chosen = None
        for candidate in entries_sorted:
            ok, error = image_is_decodable(ROOT / candidate["path"])
            if ok:
                chosen = candidate
                break
            skipped_corrupt.append({"path": candidate["path"], "error": error or "decode failed"})
        if chosen is None:
            continue
        if chosen["class"] in unique_by_class:
            chosen = dict(chosen)
            chosen["sha1"] = digest
            unique_by_class[chosen["class"]].append(chosen)

    if output_dir.exists():
        backup = output_dir.with_name(f"{output_dir.name}_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        output_dir.rename(backup)

    manifest: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data_dir": vision_cfg["data"]["data_dir"],
        "output_dir": str(output_dir.relative_to(ROOT)),
        "seed": seed,
        "classes": {},
        "totals": {split: 0 for split in SPLITS},
        "skipped_corrupt": skipped_corrupt[:100],
        "skipped_corrupt_count": len(skipped_corrupt),
        "split_policy": "one canonical file per sha1, deterministic class-stratified 80/10/10 split",
    }

    rng = random.Random(seed)
    for cls, records in unique_by_class.items():
        records = sorted(records, key=lambda r: (r["sha1"], r["path"]))
        rng.shuffle(records)
        n = len(records)
        if n < 30:
            raise RuntimeError(f"class {cls} has only {n} unique images; refusing to build clean split")
        n_train = max(1, int(math.floor(n * 0.8)))
        n_val = max(1, int(math.floor(n * 0.1)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        split_records = {
            "train": records[:n_train],
            "val": records[n_train:n_train + n_val],
            "test": records[n_train + n_val:],
        }
        manifest["classes"][cls] = {split: len(items) for split, items in split_records.items()}
        for split, items in split_records.items():
            manifest["totals"][split] += len(items)
            for idx, record in enumerate(items):
                src = ROOT / record["path"]
                ext = src.suffix.lower() or ".jpg"
                dst = output_dir / split / cls / f"{record['sha1'][:16]}_{idx:05d}{ext}"
                hardlink_or_copy(src, dst)

    write_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_llm_sft(report: PipelineReport, llm_cfg: Dict[str, Any]) -> None:
    files = [Path(p) for p in llm_cfg["data"]["train_files"] + llm_cfg["data"]["val_files"]]
    errors = []
    totals = {}
    max_length = int(llm_cfg["data"].get("max_length", 2048))
    assistant_missing = 0
    for path in files:
        full = ROOT / path
        if not full.exists():
            errors.append(f"missing {path}")
            continue
        count = 0
        max_chars = 0
        for line_no, line in enumerate(full.open("r", encoding="utf-8"), 1):
            try:
                obj = json.loads(line)
                messages = obj["messages"]
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty list")
                roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
                if "assistant" not in roles:
                    assistant_missing += 1
                for msg in messages:
                    if not isinstance(msg, dict):
                        raise ValueError("message must be an object")
                    if msg.get("role") not in {"system", "user", "assistant"}:
                        raise ValueError(f"invalid role {msg.get('role')!r}")
                    content = msg.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("message content must be non-empty")
                    max_chars = max(max_chars, len(content))
                count += 1
            except Exception as exc:
                errors.append(f"{path}:{line_no}: {exc}")
                if len(errors) >= 20:
                    break
        totals[str(path)] = {"examples": count, "max_message_chars": max_chars}
    total_examples = sum(item["examples"] for item in totals.values())
    if total_examples < 1000:
        errors.append(f"configured SFT dataset is small for production fine-tuning: {total_examples} examples")
    if assistant_missing:
        errors.append(f"{assistant_missing} SFT examples have no assistant message")
    report.add(
        "llm_sft_configured_jsonl_contract",
        not errors,
        "error",
        "configured LLM SFT files are schema-valid" if not errors else "; ".join(errors[:5]),
        files=totals,
        total_examples=total_examples,
        max_length=max_length,
    )


TARGET_CLASSES = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "cardboard_boxes",
    "cardboard_packaging", "clothing", "coffee_grounds", "disposable_plastic_cutlery",
    "eggshells", "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper", "paper_cups",
    "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers",
    "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws",
    "plastic_trash_bags", "plastic_water_bottles", "shoes", "steel_food_cans",
    "styrofoam_cups", "styrofoam_food_containers", "tea_bags",
]
MATERIALS = ["plastic", "paper", "glass", "metal", "organic", "textile", "styrofoam", "mixed"]
BINS = ["recycle", "compost", "landfill", "special", "donate"]


def build_gnn_tables(input_dim: int) -> Tuple[Any, Any]:
    from models.gnn.graph_contract import build_graph_tables

    return build_graph_tables(input_dim)


def repair_gnn_parquet(gnn_cfg: Dict[str, Any]) -> Dict[str, Any]:
    edge_df, node_df = build_gnn_tables(int(gnn_cfg["model"]["input_dim"]))
    graph_path = ROOT / gnn_cfg["data"]["graph_file"]
    feature_path = ROOT / gnn_cfg["data"]["node_features_file"]
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    edge_df.to_parquet(graph_path, index=False)
    node_df.to_parquet(feature_path, index=False)
    return {
        "graph_file": str(graph_path.relative_to(ROOT)),
        "node_features_file": str(feature_path.relative_to(ROOT)),
        "nodes": len(node_df),
        "edges": len(edge_df),
        "feature_dim": int(gnn_cfg["model"]["input_dim"]),
    }


def check_gnn_parquet(report: PipelineReport, gnn_cfg: Dict[str, Any]) -> None:
    try:
        import pandas as pd
    except Exception as exc:
        report.add("gnn_parquet_contract", False, "error", f"pandas unavailable: {exc}")
        return
    graph_path = ROOT / gnn_cfg["data"]["graph_file"]
    feature_path = ROOT / gnn_cfg["data"]["node_features_file"]
    errors = []
    if not graph_path.exists():
        errors.append(f"missing graph file {graph_path}")
    if not feature_path.exists():
        errors.append(f"missing node feature file {feature_path}")
    if errors:
        report.add("gnn_parquet_contract", False, "error", "; ".join(errors))
        return
    edge_df = pd.read_parquet(graph_path)
    node_df = pd.read_parquet(feature_path)
    feature_cols = [c for c in node_df.columns if c.startswith("feature_")]
    input_dim = int(gnn_cfg["model"]["input_dim"])
    node_count = len(node_df)
    if len(feature_cols) != input_dim:
        errors.append(f"feature dimension {len(feature_cols)} does not match config input_dim {input_dim}")
    for col in ["source", "target", "relationship"]:
        if col not in edge_df.columns:
            errors.append(f"graph missing required column {col}")
    for col in ["node_id", "node_type", "name"]:
        if col not in node_df.columns:
            errors.append(f"node features missing required column {col}")
    if not errors and not edge_df.empty:
        max_node = max(int(edge_df["source"].max()), int(edge_df["target"].max()))
        if max_node >= node_count:
            errors.append(f"edge references node {max_node} but only {node_count} nodes exist")
    node_types = set(node_df["node_type"].dropna().astype(str)) if "node_type" in node_df.columns else set()
    relationships = set(edge_df["relationship"].dropna().astype(str)) if "relationship" in edge_df.columns else set()
    required_node_types = {"ItemType", "Material", "Bin", "ProductIdea", "Hazard"}
    required_relationships = {"MADE_OF", "GOES_TO", "DISPOSAL_ROUTE", "CAN_BE_UPCYCLED_TO", "HAS_HAZARD", "SIMILAR_TO"}
    missing_node_types = sorted(required_node_types - node_types)
    missing_relationships = sorted(required_relationships - relationships)
    if missing_node_types:
        errors.append(f"GNN graph missing node types: {missing_node_types}")
    if missing_relationships:
        errors.append(f"GNN graph missing relationship types: {missing_relationships}")
    if node_count < 60:
        errors.append(f"node count {node_count} is below canonical taxonomy graph contract")
    if len(edge_df) < 170:
        errors.append(f"edge count {len(edge_df)} is below expected production relationship coverage")
    report.add(
        "gnn_parquet_contract",
        not errors,
        "error",
        "GNN parquet files match graph training contract" if not errors else "; ".join(errors[:5]),
        nodes=node_count,
        edges=len(edge_df),
        feature_dim=len(feature_cols),
        graph_file=str(graph_path.relative_to(ROOT)),
        node_features_file=str(feature_path.relative_to(ROOT)),
    )


def apply_config_overrides(vision_cfg_path: Path, output_dir: Path, source_dir: Optional[Path] = None) -> None:
    cfg = load_yaml(vision_cfg_path)
    rel = str(output_dir.relative_to(ROOT))
    if source_dir is not None:
        cfg["data"]["source_data_dir"] = str(source_dir.relative_to(ROOT))
    cfg["data"]["data_dir"] = rel
    for split in SPLITS:
        cfg["data"][f"{split}_dir"] = f"{rel}/{split}"
    with vision_cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-config", default="configs/vision_cls.yaml")
    parser.add_argument("--llm-config", default="configs/llm_sft.yaml")
    parser.add_argument("--gnn-config", default="configs/gnn.yaml")
    parser.add_argument("--full-image-scan", action="store_true", help="Decode every image instead of a deterministic sample")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--vision-source-dir", default=None, help="Source ImageFolder root used when rebuilding clean splits")
    parser.add_argument("--repair-vision-splits", action="store_true", help="Build clean hardlinked vision split view")
    parser.add_argument("--repair-gnn", action="store_true", help="Regenerate GNN parquet files from production taxonomy")
    parser.add_argument("--update-vision-config", action="store_true", help="Point vision config to the repaired clean dataset")
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()

    start = time.time()
    report = PipelineReport()
    vision_cfg_path = ROOT / args.vision_config
    vision_cfg = load_yaml(vision_cfg_path)
    llm_cfg = load_yaml(ROOT / args.llm_config)
    gnn_cfg = load_yaml(ROOT / args.gnn_config)

    if args.repair_vision_splits:
        clean_dir = ROOT / "data" / "processed" / "vision_cls_clean"
        repair_source_dir = ROOT / args.vision_source_dir if args.vision_source_dir else ROOT / vision_cfg["data"].get("source_data_dir", vision_cfg["data"]["data_dir"])
        source_cfg = json.loads(json.dumps(vision_cfg))
        source_cfg["data"]["data_dir"] = str(repair_source_dir.relative_to(ROOT))
        for split in SPLITS:
            source_cfg["data"][f"{split}_dir"] = f"{repair_source_dir.relative_to(ROOT)}/{split}"
        if repair_source_dir.resolve() == clean_dir.resolve():
            raise RuntimeError(
                "Refusing to repair vision splits from the same directory as the clean output. "
                "Pass --vision-source-dir data/processed/vision_cls or set data.source_data_dir."
            )
        manifest = repair_vision_splits(source_cfg, clean_dir)
        report.add("vision_clean_split_repair", True, "info", "clean hardlinked vision dataset created", **manifest["totals"])
        if args.update_vision_config:
            apply_config_overrides(vision_cfg_path, clean_dir, repair_source_dir)
            vision_cfg = load_yaml(vision_cfg_path)
            report.add("vision_config_updated", True, "info", "vision config now points to clean dataset", data_dir=vision_cfg["data"]["data_dir"])

    if args.repair_gnn:
        repair_meta = repair_gnn_parquet(gnn_cfg)
        report.add("gnn_parquet_repair", True, "info", "GNN parquet files regenerated from production taxonomy", **repair_meta)

    check_dependency_contract(report)
    check_training_module_imports(report)
    class_counts, hash_index = check_vision_contract(report, vision_cfg)
    check_vision_leakage(report, hash_index)
    check_vision_image_integrity(report, vision_cfg, args.full_image_scan, args.sample_size)
    validate_llm_sft(report, llm_cfg)
    check_gnn_parquet(report, gnn_cfg)

    report.duration_ms = round((time.time() - start) * 1000, 2)
    payload = asdict(report)
    payload["ok"] = report.ok
    write_json(ROOT / args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Data pipeline report written to {ROOT / args.output}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
