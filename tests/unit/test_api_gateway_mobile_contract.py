import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("ENV", "production")
os.environ.setdefault("VALID_API_KEYS", "mobile-test-key")
os.environ.setdefault("ORCHESTRATOR_URL", "http://orchestrator:8000")
os.environ.setdefault("VISION_SERVICE_URL", "http://vision-service:8001")
os.environ.setdefault("LLM_SERVICE_URL", "http://llm-service:8002")
os.environ.setdefault("RAG_SERVICE_URL", "http://rag-service:8003")
os.environ.setdefault("KG_SERVICE_URL", "http://kg-service:8004")
os.environ.setdefault("ORG_SEARCH_SERVICE_URL", "http://org-search-service:8005")
os.environ.setdefault(
    "CORS_ORIGINS",
    "http://localhost:3000,capacitor://localhost,ionic://localhost",
)

from services.api_gateway.main import app  # noqa: E402
from services.api_gateway.schemas import ChatRequest, Location  # noqa: E402
from services.api_gateway.routers import chat as chat_router  # noqa: E402
from services.api_gateway import main as gateway_main  # noqa: E402


class _FakeHTTPResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.elapsed = timedelta(milliseconds=7)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected fake HTTP error: {self.status_code}")


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        assert url == "http://orchestrator:8000/orchestrate"
        assert json["image"] == "ZmFrZS1pbWFnZQ=="
        assert json["location"] == {"lat": 37.7749, "lon": -122.4194}
        return _FakeHTTPResponse(
            {
                "response": "Recycle clean PET bottles where accepted locally.",
                "sources": [
                    {
                        "doc_id": "local-rule-1",
                        "title": "Municipal Recycling Guide",
                        "snippet": "Clean PET bottles are accepted.",
                    }
                ],
                "suggestions": ["Check your hauler's accepted materials list."],
                "processing_time_ms": 42.5,
                "confidence_score": 0.78,
                "confidence_level": "medium",
                "warnings": ["Local rules vary."],
                "citations": [{"doc_id": "local-rule-1", "title": "Municipal Recycling Guide"}],
                "fallback_used": False,
                "partial_answer": False,
                "response_id": "resp_mobile_contract",
                "metadata": {"request_type": "MULTIMODAL", "task_type": "BIN_DECISION"},
            }
        )

    async def get(self, url):
        assert url.startswith("http://")
        return _FakeHTTPResponse({"status": "healthy", "service": url})


def test_chat_request_accepts_mobile_aliases():
    request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Can I recycle this bottle?"}],
            "image_b64": "ZmFrZS1pbWFnZQ==",
            "location": {"latitude": 37.7749, "longitude": -122.4194},
        }
    )

    assert request.image == "ZmFrZS1pbWFnZQ=="
    assert request.location == Location(lat=37.7749, lon=-122.4194)


def test_chat_request_rejects_conflicting_image_inputs():
    with pytest.raises(ValueError):
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Analyze this"}],
                "image": "ZmFrZQ==",
                "image_url": "https://example.com/image.png",
            }
        )


def test_gateway_chat_preserves_orchestrator_intelligence(monkeypatch):
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", _FakeAsyncClient)
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat/",
        json={
            "messages": [{"role": "user", "content": "Can I recycle this bottle?"}],
            "image_b64": "ZmFrZS1pbWFnZQ==",
            "location": {"latitude": 37.7749, "longitude": -122.4194},
        },
        headers={
            "Origin": "capacitor://localhost",
            "User-Agent": "ReleAF-iOS-SDK/1.0",
            "X-API-Key": "mobile-test-key",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"].startswith("Recycle clean PET")
    assert payload["processing_time_ms"] == 42.5
    assert payload["confidence_score"] == 0.78
    assert payload["confidence_level"] == "medium"
    assert payload["sources"][0]["doc_id"] == "local-rule-1"
    assert payload["citations"][0]["doc_id"] == "local-rule-1"
    assert payload["warnings"] == ["Local rules vary."]
    assert payload["metadata"]["request_type"] == "MULTIMODAL"
    assert payload["metadata"]["gateway_route"] == "/api/v1/chat"
    assert response.headers["x-request-id"]


def test_gateway_cors_preflight_for_ios_origin(monkeypatch):
    monkeypatch.setattr(gateway_main.httpx, "AsyncClient", _FakeAsyncClient)
    client = TestClient(app)

    response = client.options(
        "/api/v1/chat/",
        headers={
            "Origin": "capacitor://localhost",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,X-API-Key,X-Request-ID",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "capacitor://localhost"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_ios_health_is_dependency_aware(monkeypatch):
    monkeypatch.setattr(gateway_main.httpx, "AsyncClient", _FakeAsyncClient)
    client = TestClient(app)

    response = client.get("/health/ios")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["features"]["chat"] is True
    assert payload["features"]["image_analysis"] is True
    assert payload["services"]["orchestrator"]["url"] == "http://orchestrator:8000/health"
