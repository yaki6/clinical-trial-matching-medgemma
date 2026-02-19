"""MedGemma adapter for HuggingFace Inference Endpoint.

Uses the Gemma chat template (no /v1/chat/completions — raw text generation).
Handles cold-start 503 with exponential backoff.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from huggingface_hub import InferenceClient

from trialmatch.models.base import ModelAdapter
from trialmatch.models.schema import ModelResponse

logger = structlog.get_logger()

ENDPOINT_URL = "https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud"


def format_gemma_prompt(system: str, user: str) -> str:
    """Format prompt using Gemma chat template.

    Gemma folds system prompt into first user turn.
    """
    return f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"


class MedGemmaAdapter(ModelAdapter):
    """Adapter for MedGemma 1.5 4B on HF Inference Endpoint."""

    def __init__(
        self,
        hf_token: str = "",
        endpoint_url: str = ENDPOINT_URL,
        max_retries: int = 5,
        retry_backoff: float = 2.0,
        max_wait: float = 60.0,
    ):
        self._client = InferenceClient(model=endpoint_url, token=hf_token or None)
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._max_wait = max_wait

    @property
    def name(self) -> str:
        return "medgemma-1.5-4b"

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Generate text with retry for cold-start 503."""
        formatted = format_gemma_prompt("", prompt)
        start = time.perf_counter()

        for attempt in range(self._max_retries):
            try:
                result = await asyncio.to_thread(
                    self._client.text_generation,
                    formatted,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                )
                elapsed = (time.perf_counter() - start) * 1000

                # Estimate tokens (rough: 4 chars per token)
                input_tokens = len(formatted) // 4
                output_tokens = len(result) // 4
                cost = input_tokens * 0.00001 + output_tokens * 0.00003

                return ModelResponse(
                    text=result,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=elapsed,
                    estimated_cost=cost,
                    token_count_estimated=True,
                )
            except Exception as e:
                if "503" in str(e) or "Service Unavailable" in str(e):
                    wait = min(self._retry_backoff**attempt, self._max_wait)
                    logger.warning(
                        "medgemma_cold_start",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        msg = f"MedGemma failed after {self._max_retries} retries"
        raise RuntimeError(msg)

    async def health_check(self) -> bool:
        """Check if endpoint is reachable."""
        try:
            await asyncio.to_thread(
                self._client.text_generation,
                "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n",
                max_new_tokens=5,
            )
            return True
        except Exception:
            return False
