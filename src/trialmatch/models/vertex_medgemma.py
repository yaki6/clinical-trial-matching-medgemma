"""Vertex AI Model Garden adapter for MedGemma.

Uses google-auth ADC + httpx REST calls — avoids heavy google-cloud-aiplatform SDK.
Supports chatCompletions request format (vLLM on Vertex).
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
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
        dedicated_endpoint_dns: str | None = None,
    ):
        self._project_id = project_id
        self._region = region
        self._endpoint_id = endpoint_id
        self._model_name = model_name
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._max_wait = max_wait
        self._gpu_hourly_rate = gpu_hourly_rate
        self._dedicated_endpoint_dns = dedicated_endpoint_dns
        self._credentials = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def _predict_url(self) -> str:
        """Vertex AI prediction endpoint URL.

        Dedicated endpoints (Model Garden) require their own DNS domain.
        Shared endpoints use the regional aiplatform.googleapis.com domain.
        """
        if self._dedicated_endpoint_dns:
            host = self._dedicated_endpoint_dns
        else:
            host = f"{self._region}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1/"
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

    async def _post_with_retry(self, payload: dict) -> dict:
        """POST prediction payload to Vertex with retry on transient errors."""
        token = self._get_auth_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
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
            return response.json()

        raise RuntimeError(
            f"Vertex MedGemma failed after {self._max_retries} retries"
        )

    @staticmethod
    def _extract_text_and_usage(data: dict) -> tuple[str, int, int, bool]:
        """Parse Vertex prediction response into text and token counts.

        Vertex returns predictions in nested formats depending on endpoint type:
        - Dedicated: {"predictions": [{"choices": [...], "usage": {...}}]}
        - Shared (chatCompletions): {"predictions": [[{"message": {...}, ...}]]}
        The second format wraps choices in an extra list layer.
        """
        predictions = data["predictions"]

        # Normalize: get the first prediction element.
        if isinstance(predictions, list):
            prediction = predictions[0]
        else:
            prediction = predictions

        # Unwrap nested list: predictions[0] may itself be a list of choices.
        # e.g. [[{"index": 0, "message": {"content": "..."}, ...}]]
        if isinstance(prediction, list) and prediction:
            # This is a list of choice objects — extract content directly.
            choice = prediction[0]
            if isinstance(choice, dict) and "message" in choice:
                text = choice["message"].get("content", "")
                token_count_estimated = True
                input_tokens = len(text) // 4
                output_tokens = len(text) // 4
                return text, input_tokens, output_tokens, token_count_estimated

        # Standard chatCompletions format: {"choices": [...], "usage": {...}}
        token_count_estimated = False
        if isinstance(prediction, dict) and "choices" in prediction:
            text = prediction["choices"][0]["message"]["content"]
            usage = prediction.get("usage", {})
            input_tokens = usage.get("prompt_tokens", len(text) // 4)
            output_tokens = usage.get("completion_tokens", len(text) // 4)
            return text, input_tokens, output_tokens, token_count_estimated

        # Fallback: raw text prediction.
        text = str(prediction)
        input_tokens = len(text) // 4
        output_tokens = len(text) // 4
        token_count_estimated = True
        return text, input_tokens, output_tokens, token_count_estimated

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Generate text via Vertex AI endpoint with retry for transient errors."""
        start = time.perf_counter()
        payload = {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                }
            ]
        }
        data = await self._post_with_retry(payload)
        elapsed = (time.perf_counter() - start) * 1000
        text, input_tokens, output_tokens, token_count_estimated = self._extract_text_and_usage(data)

        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
            estimated_cost=self._estimate_cost(elapsed),
            token_count_estimated=token_count_estimated,
        )

    @staticmethod
    def _preprocess_image(image_path) -> str:
        """Preprocess image for SigLIP encoder: RGB, square pad, 8-bit normalize.

        MedGemma's SigLIP encoder expects 3-channel RGB input at 896x896.
        Medical images (CT/X-ray PNGs) may be grayscale, 16-bit, or non-square.
        """
        import io

        from PIL import Image as _PILImage

        img = _PILImage.open(image_path)

        # Normalize 16-bit to 8-bit (CT exports from DICOM)
        if img.mode in ("I", "I;16", "I;16B"):
            import numpy as np
            arr = np.array(img, dtype=np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            img = _PILImage.fromarray(arr.astype(np.uint8), mode="L")

        # Convert to RGB (grayscale, RGBA, palette → RGB)
        img = img.convert("RGB")

        # Pad to square (SigLIP expects square input)
        w, h = img.size
        if w != h:
            max_dim = max(w, h)
            square = _PILImage.new("RGB", (max_dim, max_dim), (0, 0, 0))
            square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
            img = square

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    async def generate_with_image(
        self,
        prompt: str,
        image_path,
        max_tokens: int = 512,
        system_message: str | None = None,
    ) -> ModelResponse:
        """Generate text from image + prompt via Vertex chatCompletions payload."""
        from pathlib import Path as _Path

        start = time.perf_counter()
        image_path = _Path(image_path)

        # Preprocess: RGB, square pad, 8-bit normalize for SigLIP encoder
        image_b64 = self._preprocess_image(image_path)

        messages = []
        # System message (multimodal requires content block array format)
        if system_message:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            })
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        })

        payload = {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                }
            ]
        }
        data = await self._post_with_retry(payload)
        elapsed = (time.perf_counter() - start) * 1000
        text, input_tokens, output_tokens, token_count_estimated = self._extract_text_and_usage(data)

        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
            estimated_cost=self._estimate_cost(elapsed),
            token_count_estimated=token_count_estimated,
        )

    async def health_check(self) -> bool:
        """Check if endpoint is reachable with a minimal generation."""
        try:
            await self.generate("hi", max_tokens=5)
            return True
        except Exception:
            return False
