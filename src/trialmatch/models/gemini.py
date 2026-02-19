"""Gemini 3 Pro adapter via Google AI Studio (google-genai SDK).

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

DEFAULT_MODEL = "gemini-3-pro-preview"

# Pricing per 1M tokens (approximate)
COST_PER_1M_INPUT = 1.25
COST_PER_1M_OUTPUT = 10.00


class GeminiAdapter(ModelAdapter):
    """Adapter for Gemini 3 Pro via Google AI Studio."""

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return self._model

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Generate with Gemini, tracking cost."""
        start = time.perf_counter()

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "max_output_tokens": max_tokens,
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

    async def health_check(self) -> bool:
        """Check if Gemini API is reachable."""
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents='Return JSON: {"status": "ok"}',
                config={"response_mime_type": "application/json", "max_output_tokens": 50},
            )
            return response.text is not None
        except Exception:
            return False
