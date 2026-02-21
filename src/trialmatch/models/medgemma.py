"""MedGemma adapter for HuggingFace Inference Endpoint.

Supports two backend modes:
- TGI (4B): text_generation with Gemma chat template tags
- vLLM (27B): OpenAI-compatible chat_completion API

Handles cold-start 503 with exponential backoff.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog
from huggingface_hub import InferenceClient

if TYPE_CHECKING:
    from pathlib import Path

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
    """Adapter for MedGemma on HF Inference Endpoint.

    For TGI backends (4B), uses text_generation with manual Gemma template.
    For vLLM backends (27B), uses chat_completion (OpenAI-compatible API).
    """

    def __init__(
        self,
        hf_token: str = "",
        endpoint_url: str = ENDPOINT_URL,
        model_name: str = "medgemma-1.5-4b",
        max_retries: int = 8,
        retry_backoff: float = 2.0,
        max_wait: float = 60.0,
        use_chat_api: bool = False,
    ):
        self._endpoint_url = endpoint_url
        self._hf_token = hf_token or ""
        self._client = InferenceClient(
            model=endpoint_url,
            token=hf_token or None,
            headers={"X-Scale-Up-Timeout": "300"},
        )
        self._model_name = model_name
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._max_wait = max_wait
        self._use_chat_api = use_chat_api

    @property
    def name(self) -> str:
        return self._model_name

    def _call_text_generation(self, prompt: str, max_tokens: int) -> tuple[str, int]:
        """TGI backend: text_generation with Gemma template tags.

        Returns (text, estimated_input_tokens) — chars // 4 heuristic.
        """
        formatted = format_gemma_prompt("", prompt)
        result = self._client.text_generation(formatted, max_new_tokens=max_tokens)
        return result, len(formatted) // 4

    def _call_chat_completion(self, prompt: str, max_tokens: int) -> tuple[str, int]:
        """vLLM backend: OpenAI-compatible chat_completion API.

        Returns (text, actual_input_tokens) from API usage stats.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat_completion(messages=messages, max_tokens=max_tokens)
        text = response.choices[0].message.content or ""
        input_tokens = getattr(response.usage, "prompt_tokens", len(prompt) // 4)
        return text, input_tokens

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Generate text with retry for cold-start 503."""
        start = time.perf_counter()

        for attempt in range(self._max_retries):
            try:
                if self._use_chat_api:
                    result, prompt_len = await asyncio.to_thread(
                        self._call_chat_completion, prompt, max_tokens
                    )
                else:
                    result, prompt_len = await asyncio.to_thread(
                        self._call_text_generation, prompt, max_tokens
                    )

                elapsed = (time.perf_counter() - start) * 1000
                input_tokens = prompt_len  # already estimated (TGI) or actual (vLLM)
                output_tokens = len(result) // 4
                cost = input_tokens * 0.00001 + output_tokens * 0.00003

                return ModelResponse(
                    text=result,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=elapsed,
                    estimated_cost=cost,
                    token_count_estimated=not self._use_chat_api,
                )
            except Exception as e:
                err_str = str(e)
                is_transient = (
                    "503" in err_str
                    or "Service Unavailable" in err_str
                    or "workload is not stopped" in err_str
                    or "CUDA" in err_str
                    or "CUBLAS" in err_str
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

    def _call_image_generation(
        self, prompt: str, image_path: Path, max_tokens: int
    ) -> tuple[str, int]:
        """Raw HTTP POST for multimodal: image + text.

        MUST use raw HTTP POST (not InferenceClient methods).
        MUST use plain text prompt (NOT Gemma template).
        Returns (stripped_text, estimated_input_tokens).
        """
        import base64

        import requests

        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode()

        resp = requests.post(
            self._endpoint_url,
            headers={
                "Authorization": f"Bearer {self._hf_token}",
                "Content-Type": "application/json",
            },
            json={
                "inputs": {"text": prompt, "image": b64_image},
                "parameters": {"max_new_tokens": max_tokens},
            },
            timeout=60,
        )
        resp.raise_for_status()

        data = resp.json()
        input_text = data[0].get("input_text", prompt)
        generated_text = data[0]["generated_text"]
        # Strip prompt echo
        cleaned = generated_text.replace(input_text, "", 1).strip()

        return cleaned, len(prompt) // 4

    async def generate_with_image(
        self, prompt: str, image_path: Path, max_tokens: int = 512
    ) -> ModelResponse:
        """Multimodal generation: image + text -> findings.

        Uses raw HTTP POST with base64-encoded image.
        Plain text prompt (no Gemma template) — template breaks multimodal.
        Retries on 503 cold-start, same pattern as generate().
        """
        from pathlib import Path as _Path

        image_path = _Path(image_path)
        start = time.perf_counter()

        for attempt in range(self._max_retries):
            try:
                result, prompt_len = await asyncio.to_thread(
                    self._call_image_generation, prompt, image_path, max_tokens
                )

                elapsed = (time.perf_counter() - start) * 1000
                input_tokens = prompt_len
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
                err_str = str(e)
                is_transient = (
                    "503" in err_str
                    or "Service Unavailable" in err_str
                    or "workload is not stopped" in err_str
                    or "CUDA" in err_str
                    or "CUBLAS" in err_str
                )
                if is_transient and attempt < self._max_retries - 1:
                    wait = min(self._retry_backoff**attempt, self._max_wait)
                    logger.warning(
                        "medgemma_image_endpoint_not_ready",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error=err_str[:120],
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        msg = f"MedGemma image generation failed after {self._max_retries} retries"
        raise RuntimeError(msg)

    async def health_check(self) -> bool:
        """Check if endpoint is reachable."""
        try:
            if self._use_chat_api:
                await asyncio.to_thread(
                    self._call_chat_completion, "hi", 5
                )
            else:
                await asyncio.to_thread(
                    self._client.text_generation,
                    "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n",
                    max_new_tokens=5,
                )
            return True
        except Exception:
            return False
