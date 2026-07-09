#!/usr/bin/env python3
"""Stress test the public gateway routes used by website and iOS clients."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from PIL import Image


@dataclass
class StressResult:
    index: int
    route: str
    status_code: int | None
    ok: bool
    latency_ms: float
    confidence_score: float | None = None
    confidence_level: str | None = None
    error: str | None = None


def _png_b64() -> str:
    image = Image.new("RGB", (64, 64), color=(50, 205, 50))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _payload(index: int, image_b64: str) -> tuple[str, dict[str, Any]]:
    if index % 4 == 0:
        return (
            "multimodal",
            {
                "messages": [{"role": "user", "content": "Can I recycle this item?"}],
                "image_b64": image_b64,
                "location": {"latitude": 37.7749, "longitude": -122.4194},
            },
        )
    if index % 4 == 1:
        return (
            "safety",
            {
                "messages": [{"role": "user", "content": "Is a lithium battery safe to put in curbside recycling?"}],
                "location": {"latitude": 37.7749, "longitude": -122.4194},
            },
        )
    if index % 4 == 2:
        return (
            "org_search_intent",
            {
                "messages": [{"role": "user", "content": "Find electronics recycling near San Francisco."}],
                "location": {"latitude": 37.7749, "longitude": -122.4194},
            },
        )
    return (
        "text_recycling",
        {
            "messages": [{"role": "user", "content": "Can I recycle a clean PET water bottle?"}],
            "location": {"latitude": 37.7749, "longitude": -122.4194},
        },
    )


async def _send(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    index: int,
    image_b64: str,
) -> StressResult:
    route, payload = _payload(index, image_b64)
    start = time.perf_counter()
    try:
        response = await client.post(f"{base_url}/api/v1/chat/", headers=headers, json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        required = {"response", "processing_time_ms", "confidence_score", "confidence_level", "metadata"}
        ok = response.status_code == 200 and required.issubset(body) and bool(body.get("response"))
        return StressResult(
            index=index,
            route=route,
            status_code=response.status_code,
            ok=ok,
            latency_ms=latency_ms,
            confidence_score=body.get("confidence_score"),
            confidence_level=body.get("confidence_level"),
            error=None if ok else f"missing={sorted(required - set(body))}",
        )
    except Exception as exc:
        return StressResult(
            index=index,
            route=route,
            status_code=None,
            ok=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            error=str(exc),
        )


async def run_stress(base_url: str, origin: str, api_key: str | None, requests: int, concurrency: int, timeout: float) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    headers = {
        "Origin": origin,
        "User-Agent": "ReleAF-iOS-SDK/GatewayStress",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    image_b64 = _png_b64()
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def guarded(index: int) -> StressResult:
            async with semaphore:
                return await _send(client, base_url, headers, index, image_b64)

        results = await asyncio.gather(*(guarded(index) for index in range(requests)))

    latencies = [result.latency_ms for result in results]
    failures = [result for result in results if not result.ok]
    by_route: dict[str, dict[str, int]] = {}
    for result in results:
        route_stats = by_route.setdefault(result.route, {"ok": 0, "failed": 0})
        route_stats["ok" if result.ok else "failed"] += 1

    return {
        "base_url": base_url,
        "requests": requests,
        "concurrency": concurrency,
        "ok": len(failures) == 0,
        "successes": requests - len(failures),
        "failures": len(failures),
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "p50": statistics.median(latencies) if latencies else None,
            "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=None),
            "max": max(latencies) if latencies else None,
        },
        "by_route": by_route,
        "sample_failures": [asdict(result) for result in failures[:10]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--origin", default="capacitor://localhost")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = asyncio.run(
        run_stress(args.base_url, args.origin, args.api_key, args.requests, args.concurrency, args.timeout)
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
