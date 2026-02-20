"""MedGemma adapter for HuggingFace Inference Endpoint.

Uses the OpenAI-compatible /v1/chat/completions endpoint exposed by TGI.
Server-side chat template application ensures correct Gemma 3 turn tokens.
Handles cold-start via X-Scale-Up-Timeout header with fallback retry.
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

# HF Inference Endpoint cost (amortized per-token from compute-hour pricing)
COST_PER_1M_INPUT = 0.10
COST_PER_1M_OUTPUT = 0.30


def format_gemma_prompt(system: str, user: str) -> str:
    """Format prompt using Gemma chat template.

    Deprecated: MedGemmaAdapter now uses chat_completion() which applies the
    Gemma chat template server-side via TGI. Retained for external callers
    (e.g., standalone scripts, smoke tests that bypass the adapter).
    """
    return f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"


class MedGemmaAdapter(ModelAdapter):
    """Adapter for MedGemma on HF Inference Endpoint.

    Uses /v1/chat/completions (OpenAI-compatible) for:
    - Correct server-side chat template application (Gemma 3 turn tokens)
    - Exact token counts from response.usage (no heuristic estimation)
    """

    def __init__(
        self,
        hf_token: str = "",
        endpoint_url: str = ENDPOINT_URL,
        model_name: str = "medgemma-1.5-4b",
        max_retries: int = 5,  # server-side scale-up handles cold starts
        retry_backoff: float = 2.0,
        max_wait: float = 30.0,
    ):
        self._client = InferenceClient(
            model=endpoint_url,
            token=hf_token or None,
            headers={"X-Scale-Up-Timeout": "300"},  # HF proxy holds conn during GPU warmup
        )
        self._model_name = model_name
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._max_wait = max_wait

    @property
    def name(self) -> str:
        return self._model_name

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Generate via /v1/chat/completions with retry for transient errors.

        The prompt string is wrapped in a single user message. TGI applies
        the Gemma 3 chat template server-side, ensuring correct turn tokens.
        """
        messages = [{"role": "user", "content": prompt}]
        start = time.perf_counter()

        for attempt in range(self._max_retries):
            try:
                response = await asyncio.to_thread(
                    self._client.chat_completion,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                elapsed = (time.perf_counter() - start) * 1000

                result_text = response.choices[0].message.content or ""
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = (
                    input_tokens * COST_PER_1M_INPUT + output_tokens * COST_PER_1M_OUTPUT
                ) / 1_000_000

                return ModelResponse(
                    text=result_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=elapsed,
                    estimated_cost=cost,
                    token_count_estimated=False,
                )
            except Exception as e:
                err_str = str(e)
                is_transient = (
                    "503" in err_str
                    or "Service Unavailable" in err_str
                    or "workload is not stopped" in err_str
                )
                if is_transient and attempt < self._max_retries - 1:
                    wait = min(self._retry_backoff**attempt, self._max_wait)
                    logger.warning(
                        "medgemma_endpoint_not_ready",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error=err_str[:120],
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        msg = f"MedGemma failed after {self._max_retries} retries"
        raise RuntimeError(msg)

    async def health_check(self) -> bool:
        """Check if endpoint is reachable via /v1/chat/completions."""
        try:
            await asyncio.to_thread(
                self._client.chat_completion,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False
