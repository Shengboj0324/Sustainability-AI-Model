#!/usr/bin/env python3
"""Validate website/iOS deployment behavior at the public API gateway port."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from PIL import Image


@dataclass
class CheckResult:
    name: str
    ok: bool
    status_code: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] | None = None


def _tiny_png_b64() -> str:
    image = Image.new("RGB", (64, 64), color=(30, 144, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _request(client: httpx.Client, method: str, url: str, **kwargs: Any) -> tuple[httpx.Response, float]:
    start = time.perf_counter()
    response = client.request(method, url, **kwargs)
    return response, (time.perf_counter() - start) * 1000


def _record_http(checks: list[CheckResult], name: str, response: httpx.Response, latency_ms: float, ok: bool, details: dict[str, Any] | None = None) -> None:
    checks.append(
        CheckResult(
            name=name,
            ok=ok,
            status_code=response.status_code,
            latency_ms=latency_ms,
            details=details,
        )
    )


def _summarize_json(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    if endpoint == "/openapi.json":
        return {
            "openapi": payload.get("openapi"),
            "title": payload.get("info", {}).get("title"),
            "version": payload.get("info", {}).get("version"),
            "path_count": len(payload.get("paths", {})),
            "schema_count": len(payload.get("components", {}).get("schemas", {})),
            "has_chat_contract": "/api/v1/chat/" in payload.get("paths", {}),
            "has_vision_contract": "/api/v1/vision/analyze" in payload.get("paths", {}),
            "has_org_search_contract": "/api/v1/organizations/search" in payload.get("paths", {}),
        }
    return payload


def validate_gateway(
    base_url: str,
    origin: str,
    api_key: str | None,
    timeout: float,
    allow_degraded_readiness: bool,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": origin,
        "User-Agent": "ReleAF-iOS-SDK/DeploymentValidator",
        "X-Request-ID": "mobile-deployment-validator",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    checks: list[CheckResult] = []
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for endpoint in ["/", "/health", "/health/ios", "/health/live", "/health/ready", "/openapi.json"]:
            try:
                response, latency = _request(client, "GET", f"{base_url}{endpoint}", headers=headers)
                expected_ok = response.status_code == 200
                if endpoint == "/health/ready":
                    expected_ok = response.status_code == 200 and response.json().get("status") == "ready"
                    if allow_degraded_readiness:
                        expected_ok = response.status_code == 200
                details = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
                _record_http(
                    checks,
                    f"GET {endpoint}",
                    response,
                    latency,
                    expected_ok,
                    _summarize_json(endpoint, details) if isinstance(details, dict) else None,
                )
            except Exception as exc:
                checks.append(CheckResult(name=f"GET {endpoint}", ok=False, error=str(exc)))

        try:
            response, latency = _request(
                client,
                "OPTIONS",
                f"{base_url}/api/v1/chat/",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type,X-API-Key,X-Request-ID",
                },
            )
            cors_ok = (
                response.status_code in (200, 204)
                and response.headers.get("access-control-allow-origin") == origin
                and "POST" in response.headers.get("access-control-allow-methods", "")
            )
            _record_http(
                checks,
                "OPTIONS /api/v1/chat/",
                response,
                latency,
                cors_ok,
                {"allow_origin": response.headers.get("access-control-allow-origin")},
            )
        except Exception as exc:
            checks.append(CheckResult(name="OPTIONS /api/v1/chat/", ok=False, error=str(exc)))

        chat_payloads = {
            "text_chat": {
                "messages": [{"role": "user", "content": "Can I recycle a clean PET water bottle?"}],
                "location": {"latitude": 37.7749, "longitude": -122.4194},
            },
            "multimodal_chat": {
                "messages": [{"role": "user", "content": "Can I recycle this item?"}],
                "image_b64": _tiny_png_b64(),
                "location": {"latitude": 37.7749, "longitude": -122.4194},
            },
        }
        required_response_fields = {
            "response",
            "processing_time_ms",
            "confidence_score",
            "confidence_level",
            "sources",
            "suggestions",
            "warnings",
            "citations",
            "fallback_used",
            "partial_answer",
            "metadata",
        }
        for name, payload in chat_payloads.items():
            try:
                response, latency = _request(
                    client,
                    "POST",
                    f"{base_url}/api/v1/chat/",
                    headers=headers,
                    json=payload,
                )
                body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                missing = sorted(required_response_fields - set(body))
                ok = response.status_code == 200 and not missing and bool(body.get("response"))
                _record_http(
                    checks,
                    f"POST /api/v1/chat/ {name}",
                    response,
                    latency,
                    ok,
                    {
                        "missing_fields": missing,
                        "confidence_score": body.get("confidence_score"),
                        "confidence_level": body.get("confidence_level"),
                        "fallback_used": body.get("fallback_used"),
                        "partial_answer": body.get("partial_answer"),
                        "response_preview": str(body.get("response", ""))[:160],
                    },
                )
            except Exception as exc:
                checks.append(CheckResult(name=f"POST /api/v1/chat/ {name}", ok=False, error=str(exc)))

    failures = [check for check in checks if not check.ok]
    return {
        "base_url": base_url,
        "origin": origin,
        "total_checks": len(checks),
        "passed": len(checks) - len(failures),
        "failed": len(failures),
        "ok": not failures,
        "checks": [asdict(check) for check in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080", help="Public API gateway URL")
    parser.add_argument("--origin", default="capacitor://localhost", help="Mobile/web Origin header to validate")
    parser.add_argument("--api-key", default=None, help="Optional API key for production-authenticated gateways")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--allow-degraded-readiness",
        action="store_true",
        help="Do not fail solely because /health/ready reports not_ready; useful for functional degraded-mode validation.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    report = validate_gateway(
        args.base_url,
        args.origin,
        args.api_key,
        args.timeout,
        args.allow_degraded_readiness,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
