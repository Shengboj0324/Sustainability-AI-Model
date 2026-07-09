#!/usr/bin/env python3
"""Export a deployment-sized inference-only vision checkpoint.

The training checkpoint includes optimizer and scheduler state, which is useful
for resuming training but wasteful for serving. This script preserves the model
weights and serving metadata only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _size_gb(path: Path) -> float:
    return path.stat().st_size / (1024**3)


def export_checkpoint(source: Path, output: Path, metadata_output: Path) -> dict[str, Any]:
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{source} does not contain model_state_dict")

    payload = {
        "format": "releaf_vision_inference_checkpoint_v1",
        "model_state_dict": ckpt["model_state_dict"],
        "class_names": ckpt.get("class_names") or ckpt.get("config", {}).get("class_names"),
        "metrics": ckpt.get("metrics"),
        "source_epoch": ckpt.get("epoch"),
        "source_checkpoint": str(source),
        "training_state_removed": True,
        "removed_keys": sorted(k for k in ckpt.keys() if k not in {"model_state_dict", "class_names", "metrics", "config", "epoch"}),
    }
    if "config" in ckpt:
        payload["config"] = ckpt["config"]

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    metadata = {
        "source": str(source),
        "output": str(output),
        "metadata_output": str(metadata_output),
        "source_size_gb": round(_size_gb(source), 3),
        "output_size_gb": round(_size_gb(output), 3),
        "reduction_gb": round(_size_gb(source) - _size_gb(output), 3),
        "training_state_removed": True,
        "metrics": payload.get("metrics"),
        "source_epoch": payload.get("source_epoch"),
        "class_count": len(payload.get("class_names") or []),
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="models/vision/classifier/best_model.pth")
    parser.add_argument("--output", default="models/vision/classifier/inference_model.pth")
    parser.add_argument("--metadata-output", default="models/vision/classifier/inference_model_metadata.json")
    args = parser.parse_args()

    metadata = export_checkpoint(Path(args.source), Path(args.output), Path(args.metadata_output))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
