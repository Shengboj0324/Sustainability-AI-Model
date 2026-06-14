#!/usr/bin/env python3
"""Industrial readiness validation for ReleAF AI.

The default mode uses only the Python standard library so it can run inside a
plain Linux container without installing the full ML stack. Use --runtime to add
local FastAPI/TestClient checks when project dependencies are available.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class Validator:
    results: List[CheckResult] = field(default_factory=list)

    def check(self, name: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn()
            self.results.append(CheckResult(name, True, detail))
        except Exception as exc:
            self.results.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))

    def require(self, condition: bool, message: str) -> str:
        if not condition:
            raise AssertionError(message)
        return message

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if not result.passed)

    def print_report(self) -> None:
        print("# ReleAF Industrial Validation")
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status}: {result.name} - {result.detail}")
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed")


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def command_to_file(module: str) -> Path:
    parts = module.split(".")
    return ROOT.joinpath(*parts).with_suffix(".py")


def check_compose_dockerfiles() -> str:
    compose = read("docker-compose.yml")
    paths = re.findall(r"dockerfile:\s*(\S+)", compose)
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        raise AssertionError(f"missing Dockerfiles: {missing}")
    return f"{len(paths)} Dockerfile references exist"


def check_compose_commands() -> str:
    compose = read("docker-compose.yml")
    commands = re.findall(r"uvicorn\s+([\w.]+):app", compose)
    missing = []
    for module in commands:
        if not command_to_file(module).exists():
            missing.append(module)
    if missing:
        raise AssertionError(f"uvicorn modules missing: {missing}")
    return f"{len(commands)} uvicorn app modules exist"


def check_required_secret_placeholders() -> str:
    compose = read("docker-compose.yml")
    required = [
        "POSTGRES_PASSWORD=${POSTGRES_PASSWORD:?",
        "NEO4J_PASSWORD=${NEO4J_PASSWORD:?",
    ]
    missing = [item for item in required if item not in compose]
    if missing:
        raise AssertionError(f"compose does not require secrets: {missing}")
    return "Compose requires Postgres and Neo4j passwords externally"


def check_no_tracked_kaggle_secret() -> str:
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "kaggle.json"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        tracked = ""
    deleted = False
    if tracked:
        try:
            deleted = bool(subprocess.check_output(
                ["git", "ls-files", "--deleted", "kaggle.json"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip())
        except Exception:
            deleted = False
    if tracked and not deleted:
        raise AssertionError("kaggle.json is still tracked and present")
    template = ROOT / "kaggle.json.template"
    if not template.exists():
        raise AssertionError("kaggle.json.template missing")
    return "No tracked kaggle.json; template present"


def check_model_artifacts() -> str:
    required = {
        "vision": ROOT / "models/vision/classifier/best_model.pth",
        "vision_physics": ROOT / "models/vision/classifier_physics/best_model.pth",
        "gnn": ROOT / "models/gnn/ckpts/best_model.pth",
        "metadata": ROOT / "deployment_package/model_metadata.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise AssertionError(f"missing model artifacts: {missing}")
    small = [name for name, path in required.items() if path.suffix == ".pth" and path.stat().st_size < 1_000_000]
    if small:
        raise AssertionError(f"unexpectedly small checkpoints: {small}")
    return "Vision, physics-informed vision, GNN checkpoints and metadata are present"


def check_model_metadata_quality() -> str:
    path = ROOT / "deployment_package/model_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    vision = metadata.get("vision_model", {})
    accuracy = float(vision.get("validation_accuracy", 0))
    if accuracy < 90:
        raise AssertionError(f"vision validation accuracy below gate: {accuracy}")
    if vision.get("num_classes_item") != 30:
        raise AssertionError("vision metadata does not expose 30 item classes")
    return f"Vision metadata accuracy gate met: {accuracy:.2f}%"


def check_local_rag_corpus() -> str:
    paths = [
        ROOT / "data/knowledge_corpus/sustainability_knowledge.jsonl",
        ROOT / "data/sustainability_knowledge_base.json",
    ]
    existing = [path for path in paths if path.exists() and path.stat().st_size > 100]
    if not existing:
        raise AssertionError("no local RAG corpus available for degraded retrieval")
    return f"{len(existing)} local RAG corpora available"


def check_shared_contract_files() -> str:
    required = [
        ROOT / "services/shared/schemas.py",
        ROOT / "services/orchestrator/engine.py",
        ROOT / "tests/unit/test_multimodal_orchestrator_engine.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"missing contract/orchestration files: {missing}")
    return "Shared schemas, orchestrator engine, and route tests are present"


def check_runtime_orchestrator_and_rag() -> str:
    sys.path.insert(0, str(ROOT))
    os.environ["ORCHESTRATOR_EXECUTION_MODE"] = "deterministic_test"

    from fastapi.testclient import TestClient
    from PIL import Image

    from services.orchestrator.main import app as orchestrator_app
    from services.rag_service import server as rag_server

    image = Image.new("RGB", (96, 96), color=(40, 120, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    with TestClient(orchestrator_app) as client:
        response = client.post(
            "/orchestrate",
            json={
                "messages": [{"role": "user", "content": "Can I recycle this plastic bottle?"}],
                "image": image_b64,
            },
        )
    if response.status_code != 200:
        raise AssertionError(f"orchestrator status {response.status_code}: {response.text[:300]}")
    payload = response.json()
    if not payload.get("citations"):
        raise AssertionError("orchestrator response missing citations")

    rag_server.rag_service.embedding_model = None
    rag_server.rag_service.qdrant_client = None
    with TestClient(rag_server.app) as client:
        rag_response = client.post(
            "/retrieve",
            json={"query": "battery hazardous disposal", "top_k": 2, "mode": "hybrid"},
        )
    if rag_response.status_code != 200:
        raise AssertionError(f"rag status {rag_response.status_code}: {rag_response.text[:300]}")
    rag_payload = rag_response.json()
    if not rag_payload["metadata"].get("degraded"):
        raise AssertionError("RAG fallback did not disclose degraded mode")
    return "Runtime orchestrator and degraded RAG API checks passed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true", help="Run FastAPI runtime checks")
    args = parser.parse_args()

    validator = Validator()
    validator.check("compose dockerfiles", check_compose_dockerfiles)
    validator.check("compose uvicorn modules", check_compose_commands)
    validator.check("required secret placeholders", check_required_secret_placeholders)
    validator.check("no tracked Kaggle secret", check_no_tracked_kaggle_secret)
    validator.check("model artifacts", check_model_artifacts)
    validator.check("model metadata quality", check_model_metadata_quality)
    validator.check("local RAG corpus", check_local_rag_corpus)
    validator.check("shared contracts", check_shared_contract_files)
    if args.runtime:
        validator.check("runtime orchestrator and RAG APIs", check_runtime_orchestrator_and_rag)

    validator.print_report()
    return 0 if validator.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
