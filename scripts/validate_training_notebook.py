#!/usr/bin/env python3
"""Validate the training notebook contract.

The root training notebook should be a thin, reproducible runbook over the
maintained training scripts. It must not become a second copy of the training
system with stale paths, embedded credentials, or historical outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Sustainability_AI_Model_Training.ipynb"


REQUIRED_SNIPPETS = [
    "scripts/data_pipeline_stress_validate.py",
    "scripts/validate_vision_data.py",
    "scripts/validate_all_datasets.py",
    "scripts/train_unified_pipeline.py",
    "data/processed/vision_cls_clean",
    "huggingface-hub>=0.34.0,<1.0",
]

FORBIDDEN_SNIPPETS = [
    "kaggle_credentials",
    "json.dump(kaggle_credentials",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "def train_vision_model(",
    "class UnifiedWasteDataset",
    "def generate_structured_knowledge_graph(",
    "best_model_epoch",
    "Epoch 20/20",
    "Val Acc 94",
    "ALL VALIDATION TESTS PASSED - READY TO TRAIN",
]


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _all_sources(cells: Iterable[dict]) -> str:
    return "\n".join(_cell_source(cell) for cell in cells)


def validate_notebook(path: Path = NOTEBOOK) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Notebook not found: {path}"]

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Notebook is not valid JSON: {exc}"]

    cells = notebook.get("cells", [])
    if not cells:
        errors.append("Notebook has no cells")
        return errors

    text = _all_sources(cells)
    for snippet in REQUIRED_SNIPPETS:
        if snippet not in text:
            errors.append(f"Missing required snippet: {snippet}")

    for snippet in FORBIDDEN_SNIPPETS:
        if snippet in text:
            errors.append(f"Forbidden stale/unsafe snippet present: {snippet}")

    for idx, cell in enumerate(cells, start=1):
        if cell.get("execution_count") is not None:
            errors.append(f"Cell {idx} has execution_count; clear notebook before committing")
        if cell.get("outputs"):
            errors.append(f"Cell {idx} has outputs; clear notebook before committing")

    kernel_name = notebook.get("metadata", {}).get("kernelspec", {}).get("name")
    if kernel_name != "python3":
        errors.append(f"Unexpected kernelspec name: {kernel_name!r}")

    return errors


def main() -> int:
    errors = validate_notebook()
    if errors:
        print("TRAINING NOTEBOOK VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("TRAINING NOTEBOOK VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
