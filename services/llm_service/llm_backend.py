"""
LLM Backend — OpenAI-Compatible API Client

CRITICAL: This module provides the actual LLM inference capability.
When the local Llama model is unavailable (TRANSFORMERS_AVAILABLE=False),
this backend calls any OpenAI-compatible API (OpenAI, Anthropic via proxy,
Ollama, vLLM, LM Studio, etc.)

Environment variables:
    LLM_API_BASE_URL: API base URL (default: https://api.openai.com/v1)
    LLM_API_KEY: API key (required)
    LLM_MODEL_NAME: Model name (default: gpt-4o-mini)
    LLM_API_TIMEOUT: Request timeout in seconds (default: 120)
    LLM_MAX_RETRIES: Max retries on failure (default: 2)
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available — OpenAI API backend disabled. pip install httpx")


class OpenAIBackend:
    """
    Async client for OpenAI-compatible chat completion APIs.

    Supports: OpenAI, Azure OpenAI, Ollama, vLLM, LM Studio, text-generation-inference
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.base_url = (base_url or os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.timeout = timeout or float(os.getenv("LLM_API_TIMEOUT", "120"))
        self.max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "2"))

        self._client: Optional["httpx.AsyncClient"] = None
        self._available = HTTPX_AVAILABLE and bool(self.api_key)

        if not HTTPX_AVAILABLE:
            logger.error("httpx not installed — OpenAI backend unavailable")
        elif not self.api_key:
            logger.warning("LLM_API_KEY not set — OpenAI backend unavailable")
        else:
            logger.info(
                f"OpenAI backend configured: model={self.model}, "
                f"base_url={self.base_url}"
            )

    @property
    def available(self) -> bool:
        return self._available

    async def _get_client(self) -> "httpx.AsyncClient":
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Tuple[str, int, int]:
        """
        Call the chat completion API.

        Args:
            messages: List of {role, content} message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences

        Returns:
            (response_text, prompt_tokens, completion_tokens)

        Raises:
            RuntimeError: If backend is unavailable or API call fails after retries
        """
        if not self._available:
            raise RuntimeError("OpenAI backend unavailable (missing httpx or API key)")

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                resp = await client.post("/chat/completions", json=payload)

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"Rate limited, retrying in {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                resp.raise_for_status()
                data = resp.json()

                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                return text, prompt_tokens, completion_tokens

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(f"API call failed (attempt {attempt+1}): {e}, retrying in {wait}s")
                    await asyncio.sleep(wait)

        raise RuntimeError(f"OpenAI API call failed after {self.max_retries+1} attempts: {last_error}")

    async def simple_completion(self, prompt: str, max_tokens: int = 256) -> str:
        """Simple single-prompt completion. Used for classification/extraction."""
        messages = [{"role": "user", "content": prompt}]
        text, _, _ = await self.chat_completion(messages, max_tokens=max_tokens, temperature=0.1)
        return text

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
