#!/usr/bin/env python3
"""Stress test recently added multimodal/RAG/deployment paths.

This script is intentionally dependency-light beyond the app's FastAPI/Pillow
test dependencies. It exercises concurrency, malformed inputs, long inputs,
endpoint envelopes, degraded RAG retrieval, and deployment validation commands.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ORCHESTRATOR_EXECUTION_MODE", "deterministic_test")


@dataclass
class StressResult:
    name: str
    passed: bool
    duration_ms: float
    checks: int
    detail: str


def _now() -> float:
    return time.perf_counter()


def _png_b64(size=(96, 96), color=(30, 120, 90)) -> str:
    from PIL import Image

    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def stress_rag_lexical_direct() -> StressResult:
    start = _now()
    checks = 0

    from services.rag_service import server as rag_server
    from services.rag_service.server import RetrievalMode

    rag_server.rag_service.embedding_model = None
    rag_server.rag_service.qdrant_client = None

    queries = [
        "battery hazardous disposal",
        "PET plastic water bottle recycling",
        "motor oil household hazardous waste",
        "compost cardboard food residue",
        "textile donation worn clothing",
        "styrofoam food container recycle",
        "glass jar recycling clean dry",
        "lithium battery fire risk",
        "upcycle cardboard box",
        "local recycling rules vary",
    ]
    queries += [f"{query} case {idx}" for idx, query in enumerate(queries * 8)]

    async def run_query(query: str):
        return await rag_server.rag_service.retrieve(
            query=query,
            top_k=5,
            mode=RetrievalMode.HYBRID,
            rerank=True,
        )

    batches = [queries[i:i + 20] for i in range(0, len(queries), 20)]
    total_docs = 0
    for batch in batches:
        results = await asyncio.gather(*(run_query(query) for query in batch))
        for docs in results:
            checks += 1
            _assert(len(docs) > 0, "lexical fallback returned no docs")
            _assert(len(docs) <= 5, "top_k was not respected")
            for doc in docs:
                total_docs += 1
                _assert(0 <= doc.score <= 1, "doc score outside [0,1]")
                _assert(bool(doc.doc_id), "doc missing doc_id")
                _assert(bool(doc.content), "doc missing content")
                _assert(doc.metadata.get("retrieval_backend") == "local_lexical", "wrong retrieval backend")
                _assert(doc.lineage is not None and bool(doc.lineage.original_source), "missing lineage")
                _assert(doc.trust_indicators is not None and doc.trust_indicators.trust_score > 0, "missing trust score")

    return StressResult(
        name="rag_lexical_direct_concurrency",
        passed=True,
        duration_ms=round((_now() - start) * 1000, 2),
        checks=checks,
        detail=f"{len(queries)} queries, {total_docs} docs validated",
    )


def stress_rag_endpoint() -> StressResult:
    start = _now()
    checks = 0

    from fastapi.testclient import TestClient
    from services.rag_service import server as rag_server

    rag_server.rag_service.embedding_model = None
    rag_server.rag_service.qdrant_client = None

    payloads = [
        {"query": "battery hazardous disposal", "top_k": 3, "mode": "hybrid"},
        {"query": "PET plastic bottle recycling", "top_k": 2, "mode": "hybrid"},
        {"query": "x" * 1500, "top_k": 1, "mode": "hybrid"},
        {"query": "unknownzzzzzzzzzzzz material", "top_k": 5, "mode": "hybrid"},
    ]

    with TestClient(rag_server.app) as client:
        for _ in range(20):
            for payload in payloads:
                response = client.post("/retrieve", json=payload)
                checks += 1
                _assert(response.status_code == 200, f"RAG endpoint failed: {response.text[:200]}")
                body = response.json()
                _assert(body["metadata"]["degraded"] is True, "RAG endpoint did not disclose degraded mode")
                _assert(body["metadata"]["retrieval_backend"] == "local_lexical_fallback", "bad endpoint backend")
                _assert(body["query"], "sanitized query missing")
                _assert(body["num_results"] == len(body["documents"]), "num_results mismatch")
                for doc in body["documents"]:
                    _assert("lineage" in doc, "endpoint doc missing lineage")

    return StressResult(
        name="rag_endpoint_degraded_envelope",
        passed=True,
        duration_ms=round((_now() - start) * 1000, 2),
        checks=checks,
        detail=f"{checks} endpoint requests validated",
    )


async def stress_orchestrator_engine() -> StressResult:
    start = _now()
    checks = 0

    from services.orchestrator.engine import MultimodalOrchestratorEngine
    from services.shared.schemas import ChatMessage, MultimodalRequest

    engine = MultimodalOrchestratorEngine()
    good_image = _png_b64()
    tiny_image = _png_b64(size=(12, 12))
    bad_image = base64.b64encode(b"not an image").decode("ascii")
    prompts = [
        "Can I recycle this plastic bottle?",
        "Is this lithium battery safe to throw away?",
        "Give me upcycling ideas for a clean cardboard box",
        "Where can I donate clothing near me?",
        "What material is a glass jar made of?",
        "Why do local recycling rules vary?",
        "How do I dispose of motor oil?",
        "Can greasy pizza box go in recycling?",
    ]

    requests: List[MultimodalRequest] = []
    for idx in range(160):
        prompt = prompts[idx % len(prompts)]
        image = None
        if idx % 4 == 0:
            image = good_image
        elif idx % 17 == 0:
            image = tiny_image
        elif idx % 29 == 0:
            image = bad_image
        requests.append(
            MultimodalRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                image=image,
            )
        )
    requests.append(MultimodalRequest(messages=[]))

    async def run_request(request: MultimodalRequest):
        return await engine.handle(request)

    for i in range(0, len(requests), 25):
        answers = await asyncio.gather(*(run_request(request) for request in requests[i:i + 25]))
        for answer in answers:
            checks += 1
            _assert(answer.answer_text, "orchestrator returned empty answer")
            _assert(0 <= answer.confidence.score <= 1, "confidence outside [0,1]")
            _assert(answer.metadata.get("mode") == "deterministic_test", "mode missing")
            _assert(any(w.code == "DETERMINISTIC_TEST_MODE" for w in answer.warnings), "mode warning missing")
            if not answer.errors:
                _assert(answer.metadata.get("request_type"), "request_type missing")

    return StressResult(
        name="orchestrator_engine_concurrency_and_malformed_inputs",
        passed=True,
        duration_ms=round((_now() - start) * 1000, 2),
        checks=checks,
        detail=f"{checks} orchestrator engine responses validated",
    )


def stress_orchestrator_endpoint() -> StressResult:
    start = _now()
    checks = 0

    from fastapi.testclient import TestClient
    from services.orchestrator.main import app

    good_image = _png_b64()
    payloads: List[Dict[str, Any]] = [
        {"messages": [{"role": "user", "content": "Can I recycle this plastic bottle?"}], "image": good_image},
        {"messages": [{"role": "user", "content": "Is this lithium battery hazardous?"}], "image": good_image},
        {"messages": [{"role": "user", "content": "Give me upcycling ideas for cardboard"}]},
        {"messages": [{"role": "user", "content": "Where can I recycle motor oil near me?"}]},
        {"messages": []},
    ]

    with TestClient(app) as client:
        for _ in range(12):
            for payload in payloads:
                response = client.post("/orchestrate", json=payload)
                checks += 1
                _assert(response.status_code == 200, f"orchestrator endpoint failed: {response.text[:200]}")
                body = response.json()
                _assert(body["response"], "empty endpoint response")
                _assert(0 <= body["confidence_score"] <= 1, "endpoint confidence invalid")
                _assert(body["metadata"].get("mode") == "deterministic_test", "endpoint mode missing")
                _assert(body["warnings"], "endpoint warnings missing")

    return StressResult(
        name="orchestrator_fastapi_envelope_stress",
        passed=True,
        duration_ms=round((_now() - start) * 1000, 2),
        checks=checks,
        detail=f"{checks} endpoint responses validated",
    )


def stress_validation_commands() -> StressResult:
    start = _now()
    checks = 0

    commands = [
        ["python", "scripts/industrial_e2e_validate.py"],
        ["python", "scripts/industrial_e2e_validate.py", "--runtime"],
    ]
    env = os.environ.copy()
    env["POSTGRES_PASSWORD"] = "stress-validation-only"
    env["NEO4J_PASSWORD"] = "stress-validation-only"

    compose = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    checks += 1
    _assert(compose.returncode == 0, f"docker compose config failed: {compose.stderr[:500]}")

    for command in commands:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        checks += 1
        _assert(proc.returncode == 0, f"{' '.join(command)} failed:\n{proc.stdout[-1500:]}")
        summary = proc.stdout.split("SUMMARY:")[-1].strip()
        _assert("0 failed" in summary, f"validator reported failure:\n{proc.stdout[-1500:]}")

    return StressResult(
        name="deployment_validation_commands",
        passed=True,
        duration_ms=round((_now() - start) * 1000, 2),
        checks=checks,
        detail="compose config and industrial validators passed",
    )


async def main() -> int:
    random.seed(42)
    tests = [
        stress_rag_lexical_direct,
        stress_orchestrator_engine,
    ]
    sync_tests = [
        stress_rag_endpoint,
        stress_orchestrator_endpoint,
        stress_validation_commands,
    ]

    results: List[StressResult] = []
    failures: List[str] = []

    for test in tests:
        try:
            results.append(await test())
        except Exception as exc:
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            results.append(StressResult(test.__name__, False, 0.0, 0, str(exc)))

    for test in sync_tests:
        try:
            results.append(test())
        except Exception as exc:
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            results.append(StressResult(test.__name__, False, 0.0, 0, str(exc)))

    report = {
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "total_checks": sum(r.checks for r in results),
        "results": [asdict(r) for r in results],
        "failures": failures,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output_dir = ROOT / "outputs" / "stress"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "new_multimodal_paths_stress_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Stress report written to {output_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
