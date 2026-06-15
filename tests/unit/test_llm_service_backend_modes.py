import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.common.health_checks import HealthChecker, HealthCheckResult, HealthStatus
from services.llm_service.server_v2 import LLMServiceV2


@pytest.mark.asyncio
async def test_llm_disable_local_starts_degraded_without_backend(monkeypatch):
    monkeypatch.setenv("LLM_DISABLE_LOCAL", "true")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    async def fail_if_called(self):
        raise AssertionError("local model loading should be skipped")

    monkeypatch.setattr(LLMServiceV2, "_try_load_local_model", fail_if_called)

    service = LLMServiceV2()
    await service.initialize()

    assert service.model is None
    assert service.tokenizer is None
    assert not service.has_backend()


@pytest.mark.asyncio
async def test_llm_explicit_api_backend_skips_local_model(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "api")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")

    async def fail_if_called(self):
        raise AssertionError("local model loading should be skipped in explicit API mode")

    monkeypatch.setattr(LLMServiceV2, "_try_load_local_model", fail_if_called)

    service = LLMServiceV2()
    await service.initialize()

    assert service.model is None
    assert service.tokenizer is None
    assert service.has_backend()
    assert service._use_api_backend is True
    assert service.openai_backend.model == "test-model"


def test_health_check_result_serializes_json_safe_enums():
    result = HealthCheckResult(
        status=HealthStatus.UNHEALTHY,
        message="No backend",
        details={"llm_backend": {"status": HealthStatus.UNHEALTHY}},
    )

    payload = json.loads(result.model_dump_json())

    assert payload["status"] == "unhealthy"
    assert payload["details"]["llm_backend"]["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_not_ready_health_checker_does_not_report_healthy_without_checks():
    checker = HealthChecker(service_name="rag_service")

    result = await checker.check_health()

    assert result.status == HealthStatus.UNHEALTHY
    assert result.message == "Service not ready"
    assert result.details["ready"] is False
