import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.orchestrator.engine import (
    MultimodalOrchestratorEngine,
    classify_request_type,
    classify_task_type,
)
from services.shared.schemas import ChatMessage, MultimodalRequest, RequestType, ServiceMode, TaskType


def _png_b64(size=(96, 96), color=(20, 140, 90)) -> str:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_request_and_task_classification_text_only():
    request = MultimodalRequest(
        messages=[ChatMessage(role="user", content="Can I recycle a clean plastic bottle?")]
    )

    assert classify_request_type(request) == RequestType.TEXT_ONLY
    assert classify_task_type(request)[0] == TaskType.BIN_DECISION


def test_request_and_task_classification_multimodal_safety():
    request = MultimodalRequest(
        messages=[ChatMessage(role="user", content="Is this lithium battery safe to throw away?")],
        image=_png_b64(),
    )

    assert classify_request_type(request) == RequestType.MULTIMODAL
    assert classify_task_type(request)[0] == TaskType.SAFETY_CHECK


@pytest.mark.asyncio
async def test_text_only_flow_preserves_rag_citations_and_schema():
    engine = MultimodalOrchestratorEngine()
    request = MultimodalRequest(
        messages=[ChatMessage(role="user", content="Why do local recycling rules vary?")]
    )

    answer = await engine.handle(request)

    assert answer.answer_text
    assert answer.confidence.score > 0.4
    assert answer.metadata["request_type"] == RequestType.TEXT_ONLY.value
    assert answer.metadata["task_type"] == TaskType.THEORY_QA.value
    assert answer.citations
    assert answer.sources[0].doc_id
    assert answer.sources[0].provenance["ingestion_mode"] == "deterministic_test_fixture"
    assert any(w.code == "DETERMINISTIC_TEST_MODE" for w in answer.warnings)


@pytest.mark.asyncio
async def test_image_only_flow_returns_structured_vision_metadata():
    engine = MultimodalOrchestratorEngine()
    request = MultimodalRequest(image=_png_b64(), context={"vision_hint": "plastic bottle"})

    answer = await engine.handle(request)

    assert answer.metadata["request_type"] == RequestType.IMAGE_ONLY.value
    assert answer.metadata["task_type"] == TaskType.BIN_DECISION.value
    assert answer.metadata["evidence_counts"]["vision_objects"] == 1
    assert "recommended bin" in answer.answer_text.lower()
    assert "deterministic_test" == answer.metadata["mode"]
    assert answer.citations


@pytest.mark.asyncio
async def test_multimodal_safety_flow_avoids_unsafe_disposal_claim():
    engine = MultimodalOrchestratorEngine()
    request = MultimodalRequest(
        messages=[ChatMessage(role="user", content="Can I put this lithium battery in recycling?")],
        image=_png_b64(),
    )

    answer = await engine.handle(request)

    assert answer.metadata["request_type"] == RequestType.MULTIMODAL.value
    assert answer.metadata["task_type"] == TaskType.SAFETY_CHECK.value
    assert "approved drop-off" in answer.answer_text
    assert "curbside" in answer.answer_text
    assert any(c.id == "releaf-battery-safety" for c in answer.citations)


@pytest.mark.asyncio
async def test_upcycling_flow_uses_kg_relationships():
    engine = MultimodalOrchestratorEngine()
    request = MultimodalRequest(
        messages=[ChatMessage(role="user", content="Give me upcycling ideas for a clean cardboard box")]
    )

    answer = await engine.handle(request)

    assert answer.metadata["task_type"] == TaskType.UPCYCLING_IDEA.value
    assert answer.metadata["evidence_counts"]["kg_results"] >= 1
    assert "drawer divider" in answer.answer_text
    assert answer.citations


@pytest.mark.asyncio
async def test_org_search_discloses_missing_location():
    engine = MultimodalOrchestratorEngine()
    request = MultimodalRequest(
        messages=[ChatMessage(role="user", content="Where can I donate or recycle plastic near me?")]
    )

    answer = await engine.handle(request)

    assert answer.metadata["task_type"] == TaskType.ORG_SEARCH.value
    assert "Verify local availability" in answer.answer_text
    assert any(w.code == "LOCATION_REQUIRED_FOR_LOCAL_RESULTS" for w in answer.warnings)


@pytest.mark.asyncio
async def test_corrupted_image_is_honest_low_confidence_warning():
    engine = MultimodalOrchestratorEngine()
    request = MultimodalRequest(
        messages=[ChatMessage(role="user", content="Can I recycle this?")],
        image=base64.b64encode(b"not-an-image").decode("ascii"),
    )

    answer = await engine.handle(request)

    assert any(w.code == "IMAGE_DECODE_FAILED" for w in answer.warnings)
    assert answer.confidence.score < 0.75
    assert answer.metadata["mode"] == ServiceMode.DETERMINISTIC_TEST.value


def test_fastapi_orchestrate_endpoint_uses_deterministic_contract(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_EXECUTION_MODE", "deterministic_test")

    from fastapi.testclient import TestClient
    from services.orchestrator.main import app

    with TestClient(app) as client:
        response = client.post(
            "/orchestrate",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Can I recycle this plastic bottle?",
                    }
                ],
                "image": _png_b64(),
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"]
    assert payload["confidence_score"] > 0
    assert payload["metadata"]["request_type"] == RequestType.MULTIMODAL.value
    assert payload["metadata"]["task_type"] == TaskType.BIN_DECISION.value
    assert payload["metadata"]["service_modes"] == ["deterministic_test"]
    assert payload["citations"]
    assert any("DETERMINISTIC_TEST_MODE" in warning for warning in payload["warnings"])
