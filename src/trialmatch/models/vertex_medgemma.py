"""Vertex AI Model Garden adapter for MedGemma.

Uses google-auth ADC + httpx REST calls — avoids heavy google-cloud-aiplatform SDK.
Supports chatCompletions request format (vLLM on Vertex).
"""

from __future__ import annotations

import asyncio
import time

import google.auth
import google.auth.transport.requests
import httpx
import structlog

from trialmatch.models.base import ModelAdapter
from trialmatch.models.schema import ModelResponse

logger = structlog.get_logger()


class VertexMedGemmaAdapter(ModelAdapter):
    """Adapter for MedGemma deployed on Vertex AI Model Garden.

    Uses ADC credentials and direct REST calls via httpx to avoid
    the heavy google-cloud-aiplatform SDK dependency.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        endpoint_id: str,
        model_name: str = "medgemma-4b-vertex",
        max_retries: int = 5,
        retry_backoff: float = 2.0,
        max_wait: float = 30.0,
        gpu_hourly_rate: float = 1.15,  # L4 default
    ):
        self._project_id = project_id
        self._region = region
        self._endpoint_id = endpoint_id
        self._model_name = model_name
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._max_wait = max_wait
        self._gpu_hourly_rate = gpu_hourly_rate
        self._credentials = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def _predict_url(self) -> str:
        """Vertex AI prediction endpoint URL."""
        return (
            f"https://{self._region}-aiplatform.googleapis.com/v1/"
            f"projects/{self._project_id}/locations/{self._region}/"
            f"endpoints/{self._endpoint_id}:predict"
        )

    def _get_auth_token(self) -> str:
        """Get a fresh ADC access token, refreshing if needed."""
        if self._credentials is None:
            self._credentials, _ = google.auth.default()
        self._credentials.refresh(google.auth.transport.requests.Request())
        return self._credentials.token

    def _estimate_cost(self, latency_ms: float) -> float:
        """Estimate cost based on GPU-hour rate and request latency."""
        return (latency_ms / 3_600_000) * self._gpu_hourly_rate

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Generate text via Vertex AI endpoint with retry for transient errors."""
        start = time.perf_counter()

        token = self._get_auth_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                }
            ]
        }

        for attempt in range(self._max_retries):
            response = await asyncio.to_thread(
                httpx.post,
                self._predict_url,
                json=payload,
                headers=headers,
                timeout=120.0,
            )

            # Check for transient errors that should trigger retry
            is_transient = response.status_code in (503, 429) or any(
                keyword in response.text
                for keyword in ("SERVICE_UNAVAILABLE", "RESOURCE_EXHAUSTED")
            )

            if is_transient:
                if attempt < self._max_retries - 1:
                    wait = min(self._retry_backoff ** attempt, self._max_wait)
                    logger.warning(
                        "vertex_medgemma_transient_error",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        status_code=response.status_code,
                        error=response.text[:120],
                    )
                    await asyncio.sleep(wait)
                    continue
                # Last attempt exhausted
                raise RuntimeError(
                    f"Vertex MedGemma failed after {self._max_retries} retries"
                )

            # Non-transient HTTP errors: raise immediately
            response.raise_for_status()

            # Parse successful response
            elapsed = (time.perf_counter() - start) * 1000
            data = response.json()
            prediction = data["predictions"][0]

            # Try chatCompletions format first
            token_count_estimated = False
            if isinstance(prediction, dict) and "choices" in prediction:
                text = prediction["choices"][0]["message"]["content"]
                usage = prediction.get("usage", {})
                input_tokens = usage.get("prompt_tokens", len(text) // 4)
                output_tokens = usage.get("completion_tokens", len(text) // 4)
            else:
                # Fallback: raw text prediction
                text = str(prediction)
                input_tokens = len(text) // 4
                output_tokens = len(text) // 4
                token_count_estimated = True

            return ModelResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=elapsed,
                estimated_cost=self._estimate_cost(elapsed),
                token_count_estimated=token_count_estimated,
            )

        raise RuntimeError(
            f"Vertex MedGemma failed after {self._max_retries} retries"
        )

    async def health_check(self) -> bool:
        """Check if endpoint is reachable with a minimal generation."""
        try:
            await self.generate("hi", max_tokens=5)
            return True
        except Exception:
            return False
