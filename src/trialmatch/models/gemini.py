"""Gemini adapter via Google AI Studio (google-genai SDK).

Supports both thinking models (Pro) and non-thinking models (Flash).
Uses structured JSON output with Pydantic-style schema.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from google import genai

from trialmatch.models.base import ModelAdapter
from trialmatch.models.schema import ModelResponse

logger = structlog.get_logger()

DEFAULT_MODEL = "gemini-3-flash-preview"

# Pricing per 1M tokens (approximate)
COST_PER_1M_INPUT = 1.25
COST_PER_1M_OUTPUT = 10.00


THINKING_MODELS = {"gemini-3-pro-preview", "gemini-3-pro", "gemini-3-flash-preview"}


class GeminiAdapter(ModelAdapter):
    """Adapter for Gemini models via Google AI Studio."""

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        max_retries: int = 8,
        retry_backoff: float = 2.0,
        max_wait: float = 30.0,
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._is_thinking_model = model in THINKING_MODELS
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._max_wait = max_wait

    @property
    def name(self) -> str:
        return self._model

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Generate with Gemini, tracking cost. Retries on transient 503/429 errors."""
        start = time.perf_counter()

        # Thinking models (Pro): max_output_tokens includes both internal
        # thinking tokens AND visible response tokens. Thinking alone can
        # consume 8000+ tokens, so the floor must be high enough.
        # Non-thinking models (Flash): max_output_tokens = visible output only.
        if self._is_thinking_model:
            effective_max_tokens = max(max_tokens, 32768)
        else:
            effective_max_tokens = max_tokens

        for attempt in range(self._max_retries):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "max_output_tokens": effective_max_tokens,
                    },
                )
                elapsed = (time.perf_counter() - start) * 1000

                text = response.text or ""
                input_tokens = 0
                output_tokens = 0

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                cost = (input_tokens * COST_PER_1M_INPUT + output_tokens * COST_PER_1M_OUTPUT) / 1_000_000

                return ModelResponse(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=elapsed,
                    estimated_cost=cost,
                )
            except Exception as e:
                err_str = str(e)
                is_transient = (
                    "503" in err_str
                    or "UNAVAILABLE" in err_str
                    or "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "high demand" in err_str
                )
                if is_transient and attempt < self._max_retries - 1:
                    wait = min(self._retry_backoff**attempt, self._max_wait)
                    logger.warning(
                        "gemini_api_transient_error",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error=err_str[:120],
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        msg = f"Gemini failed after {self._max_retries} retries"
        raise RuntimeError(msg)

    async def health_check(self) -> bool:
        """Check if Gemini API is reachable."""
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents='Return JSON: {"status": "ok"}',
                config={"response_mime_type": "application/json", "max_output_tokens": 50},
            )
            if response is None:
                return False
            if getattr(response, "text", None):
                return True
            if getattr(response, "candidates", None):
                return True
            if getattr(response, "usage_metadata", None):
                return True
            # The request succeeded and returned a response object.
            return True
        except Exception:
            return False
