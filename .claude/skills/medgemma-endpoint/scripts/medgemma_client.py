"""Reusable MedGemma HF Inference Endpoint client.

Model: google/medgemma-1-5-4b-it-hae (multimodal, image-text-to-text)
Endpoint: HuggingFace Inference Endpoint (default inference image, NOT TGI)

Usage:
    client = MedGemmaClient(endpoint_url="https://...", hf_token="hf_...")
    result = await client.generate(messages=[...], max_tokens=2048)
    health = await client.health_check()
"""

import asyncio
import base64
import json
import re
import time
from pathlib import Path

from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError


class MedGemmaClient:
    """Client for MedGemma HF Inference Endpoint.

    Key constraints:
    - Endpoint uses default HF inference image (NOT TGI)
    - /v1/chat/completions is NOT available (returns 404)
    - Must use text_generation() with manual Gemma chat template
    - Supports multimodal input (text extracted from images separately)
    """

    # Default retry settings for cold-start 503 errors
    DEFAULT_MAX_RETRIES = 6
    DEFAULT_RETRY_BACKOFF = 2.0  # exponential base
    DEFAULT_MAX_WAIT = 60.0  # max seconds per retry wait
    DEFAULT_COLD_START_TIMEOUT = 60.0  # total retry budget

    def __init__(
        self,
        endpoint_url: str,
        hf_token: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        max_wait: float = DEFAULT_MAX_WAIT,
        cold_start_timeout: float = DEFAULT_COLD_START_TIMEOUT,
    ):
        self.endpoint_url = endpoint_url
        self.hf_token = hf_token
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_wait = max_wait
        self.cold_start_timeout = cold_start_timeout
        self._client = InferenceClient(model=endpoint_url, token=hf_token)

    async def health_check(self) -> dict:
        """Quick ping to check endpoint readiness."""
        try:
            messages = [
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": "Respond with OK if you are ready."},
            ]
            result = await self._call_with_retry(messages, max_tokens=10)
            return {"status": "ready", "response": result[:50]}
        except Exception as e:
            return {"status": "unavailable", "error": str(e)[:200]}

    async def warm_up(self) -> None:
        """Send lightweight ping to prevent cold-start delays."""
        await self.health_check()

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> str:
        """Generate text from chat messages.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            max_tokens: Maximum new tokens to generate.

        Returns:
            Generated text string.
        """
        return await self._call_with_retry(messages, max_tokens)

    @staticmethod
    def format_gemma_prompt(messages: list[dict]) -> str:
        """Convert chat messages to Gemma chat template.

        CRITICAL: The HF endpoint uses default inference image (not TGI),
        so /v1/chat/completions returns 404. Must format manually and
        call text_generation() on the root endpoint.

        Template: <start_of_turn>role\\ncontent<end_of_turn>
        Gemma folds system prompt into the first user turn.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                text_parts = [p["text"] for p in content if p.get("type") == "text"]
                content = "\n".join(text_parts)
            if role == "system":
                parts.append(f"<start_of_turn>user\n{content}")
                continue
            if role == "user":
                if parts and parts[-1].startswith("<start_of_turn>user"):
                    parts[-1] += f"\n\n{content}<end_of_turn>"
                else:
                    parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role in ("model", "assistant"):
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)

    @staticmethod
    def encode_image(path: str | Path) -> dict:
        """Encode an image file as base64 data URL.

        Returns dict with type/image_url for multimodal message content.
        """
        path = Path(path)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    @staticmethod
    def parse_json(raw_text: str) -> dict:
        """Parse JSON from MedGemma output, handling markdown-wrapped responses."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            raise ValueError(f"Could not parse JSON: {raw_text[:200]}")

    async def _call_with_retry(self, messages: list[dict], max_tokens: int) -> str:
        """Call HF endpoint with retry logic for cold-start 503 errors.

        Retry strategy:
        - 4XX errors: fail immediately (client error)
        - 503 errors: exponential backoff (cold-start)
        - Total budget capped at cold_start_timeout seconds
        - Per-retry wait capped at max_wait seconds
        """
        prompt = self.format_gemma_prompt(messages)
        start = time.monotonic()
        last_error = None

        for attempt in range(self.max_retries):
            if time.monotonic() - start > self.cold_start_timeout:
                break
            try:
                content = await asyncio.to_thread(
                    self._client.text_generation,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                )
                if not content:
                    raise ValueError("MedGemma returned empty content")
                return content
            except HfHubHTTPError as e:
                last_error = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status and 400 <= status < 500:
                    raise
                if status == 503 and attempt < self.max_retries - 1:
                    wait = min(self.retry_backoff**attempt, self.max_wait)
                    await asyncio.sleep(wait)
                    continue
                raise

        raise last_error or RuntimeError("MedGemma retry budget exhausted")
