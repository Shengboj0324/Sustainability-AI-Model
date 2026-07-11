import json
import subprocess
import sys
from pathlib import Path

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.industrial_model_readiness_audit import ReadinessAudit


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def _minimal_cfg(root: Path) -> dict:
    return {
        "data": {
            "data_dir": str(root / "data/processed/vision_cls_clean"),
            "train_dir": str(root / "data/processed/vision_cls_clean/train"),
            "val_dir": str(root / "data/processed/vision_cls_clean/val"),
            "test_dir": str(root / "data/processed/vision_cls_clean/test"),
        },
        "model": {
            "num_classes_item": 30,
            "num_classes_material": 15,
            "num_classes_bin": 4,
            "item_classes": [],
            "material_classes": ["plastic"] * 15,
        },
        "training": {"output_dir": "models/vision/classifier"},
    }


def test_image_split_audit_detects_exact_cross_split_leakage(tmp_path: Path):
    cfg = _minimal_cfg(tmp_path)
    train_duplicate = Path(cfg["data"]["train_dir"]) / "bottle" / "dup.png"
    val_duplicate = Path(cfg["data"]["val_dir"]) / "bottle" / "dup.png"
    _write_image(train_duplicate, (20, 80, 140))
    _write_image(val_duplicate, (20, 80, 140))
    _write_image(Path(cfg["data"]["test_dir"]) / "bottle" / "unique.png", (140, 80, 20))

    audit = ReadinessAudit(root=tmp_path, sample_per_split=10, min_images=1)
    audit.audit_image_splits(cfg)

    leakage_gate = next(result for result in audit.results if result.name == "vision_exact_split_leakage")
    assert leakage_gate.passed is False
    assert leakage_gate.severity == "error"
    assert leakage_gate.details["leak_count"] == 1


def test_image_split_audit_accepts_clean_decodable_sample(tmp_path: Path):
    cfg = _minimal_cfg(tmp_path)
    _write_image(Path(cfg["data"]["train_dir"]) / "bottle" / "train.png", (20, 80, 140))
    _write_image(Path(cfg["data"]["val_dir"]) / "bottle" / "val.png", (30, 90, 150))
    _write_image(Path(cfg["data"]["test_dir"]) / "bottle" / "test.png", (40, 100, 160))

    audit = ReadinessAudit(root=tmp_path, sample_per_split=10, min_images=1)
    audit.audit_image_splits(cfg)

    gates = {result.name: result for result in audit.results}
    assert gates["vision_exact_split_leakage"].passed is True
    assert gates["vision_sample_decode_integrity"].passed is True


def test_3d_vision_gates_separate_geometry_contract_from_learned_model(tmp_path: Path):
    (tmp_path / "models/vision").mkdir(parents=True)
    (tmp_path / "services/vision_service").mkdir(parents=True)
    (tmp_path / "training/vision").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)

    audit = ReadinessAudit(root=tmp_path, sample_per_split=1, min_images=1)
    audit.audit_3d_vision_claims()

    gates = {result.name: result for result in audit.results}
    assert gates["three_d_depth_geometry_contract"].passed is False
    assert gates["three_d_depth_geometry_contract"].severity == "error"
    assert gates["three_d_learned_model_capability"].passed is False
    assert gates["three_d_learned_model_capability"].severity == "warning"
    assert "geometry analysis only" in gates["three_d_learned_model_capability"].message


def test_industrial_readiness_script_runs_from_file_entrypoint(tmp_path: Path):
    cfg = _minimal_cfg(tmp_path)
    _write_image(Path(cfg["data"]["train_dir"]) / "bottle" / "train.png", (20, 80, 140))
    _write_image(Path(cfg["data"]["val_dir"]) / "bottle" / "val.png", (30, 90, 150))
    _write_image(Path(cfg["data"]["test_dir"]) / "bottle" / "test.png", (40, 100, 160))

    checkpoint_dir = tmp_path / "models/vision/classifier"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best_model.pth").write_bytes(b"test-checkpoint")
    cfg["training"]["output_dir"] = str(checkpoint_dir)

    config_path = tmp_path / "vision_cls.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/industrial_model_readiness_audit.py"),
            "--config",
            str(config_path),
            "--sample-per-split",
            "1",
            "--min-images",
            "1",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode in {0, 1}
    assert output.exists(), result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "summary" in report
    assert any(entry["name"] == "three_d_depth_geometry_contract" for entry in report["results"])
    assert any(entry["name"] == "three_d_learned_model_capability" for entry in report["results"])
