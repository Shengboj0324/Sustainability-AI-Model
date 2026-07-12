import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.r2_3d_dataset_mirror import build_objectron_plan, command_preflight, load_config
from scripts.validate_rgbd_manifest import validate_manifest
from training.vision.rgbd_dataset import RGBD3DManifestDataset, parse_manifest_sample, rgbd_collate


def _write_sample_manifest(tmp_path: Path, pose=None, include_boxes=True) -> Path:
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    Image.new("RGB", (4, 3), color=(20, 40, 60)).save(rgb_path)
    np.save(depth_path, np.full((3, 4), 1.25, dtype="float32"))

    payload = {
        "dataset": "unit_rgbd",
        "split": "train",
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.5, "cy": 1.0, "width": 4, "height": 3},
        "pose": pose
        or [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    if include_boxes:
        payload["boxes_3d"] = [
            {"label": "bottle", "center_m": [0.0, 0.0, 1.0], "size_m": [0.1, 0.1, 0.3], "yaw_rad": 0.0}
        ]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return manifest_path


def test_rgbd_manifest_dataset_loads_calibrated_sample(tmp_path):
    manifest_path = _write_sample_manifest(tmp_path)

    dataset = RGBD3DManifestDataset(manifest_path, split="train")
    sample = dataset[0]
    batch = rgbd_collate([sample])

    assert len(dataset) == 1
    assert sample["depth_m"].shape == (3, 4)
    assert sample["intrinsics"].width == 4
    assert sample["boxes_3d"][0]["label"] == "bottle"
    assert batch["pose"].shape == (1, 4, 4)


def test_rgbd_manifest_rejects_missing_3d_boxes(tmp_path):
    manifest_path = _write_sample_manifest(tmp_path, include_boxes=False)

    with pytest.raises(ValueError, match="boxes_3d"):
        RGBD3DManifestDataset(manifest_path, split="train")


def test_rgbd_manifest_rejects_non_rigid_pose(tmp_path):
    bad_pose = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    manifest_path = _write_sample_manifest(tmp_path, pose=bad_pose)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="orthonormal"):
        parse_manifest_sample(payload)


def test_r2_preflight_fails_closed_without_credentials(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    config = load_config(PROJECT_ROOT / "configs" / "vision_3d_datasets.yaml")

    payload = command_preflight(config, require_credentials=False)

    assert payload["credentials_present"] is False
    assert "Missing R2 credentials" in payload["credential_error"]


def test_objectron_plan_uses_selected_relevant_classes(monkeypatch):
    config = load_config(PROJECT_ROOT / "configs" / "vision_3d_datasets.yaml")

    def fake_text(url, timeout):
        assert "cup_annotations_test" in url
        return "cup/batch-1/2\n"

    monkeypatch.setattr("scripts.r2_3d_dataset_mirror.http_text", fake_text)
    rows = build_objectron_plan(
        config=config,
        classes=["cup"],
        splits=["test"],
        include_videos=True,
        include_records=False,
        max_items_per_class_split=1,
        timeout=1.0,
    )

    keys = {row.r2_key for row in rows}
    assert "external/3d_vision/objectron/v1/index/cup_test.txt" in keys
    assert "external/3d_vision/objectron/annotations/cup/batch-1/2.pbdata" in keys
    assert "external/3d_vision/objectron/videos/cup/batch-1/2/geometry.pbdata" in keys
    assert "external/3d_vision/objectron/videos/cup/batch-1/2/video.MOV" in keys


def test_r2_preflight_cli_reports_missing_credentials(monkeypatch):
    env = os.environ.copy()
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)

    result = subprocess.run(
        [sys.executable, "scripts/r2_3d_dataset_mirror.py", "preflight"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["credentials_present"] is False


def test_r2_execute_cli_fails_closed_without_traceback(tmp_path):
    plan = tmp_path / "plan.jsonl"
    plan.write_text(
        json.dumps(
            {
                "dataset": "objectron_relevant_classes",
                "source_url": "https://storage.googleapis.com/objectron/v1/index/cup_annotations_test",
                "r2_key": "external/3d_vision/objectron/v1/index/cup_test.txt",
                "asset_type": "index",
                "license_summary": "C-UDA",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)

    result = subprocess.run(
        [sys.executable, "scripts/r2_3d_dataset_mirror.py", "upload-url-list", "--plan", str(plan), "--execute"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "Missing R2 credentials" in payload["error"]
    assert "Traceback" not in result.stderr


def test_rgbd_manifest_validator_reports_split_failure(tmp_path):
    manifest_path = _write_sample_manifest(tmp_path)
    config = {
        "data": {
            "manifest_path": str(manifest_path),
            "cache_dir": str(tmp_path / "cache"),
            "require_boxes_3d": True,
            "expected_splits": ["train", "val"],
        },
        "validation": {"max_samples_per_split": 2},
    }

    payload = validate_manifest(config, manifest_override=str(manifest_path), max_samples_override=2)

    assert payload["ok"] is False
    failures = payload["checks"][1]["metrics"]["failures"]
    assert failures[0]["split"] == "val"
