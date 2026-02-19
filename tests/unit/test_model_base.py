"""Tests for model adapter base protocol."""

import asyncio

from trialmatch.models.base import ModelAdapter
from trialmatch.models.schema import ModelResponse


class FakeAdapter(ModelAdapter):
    @property
    def name(self) -> str:
        return "fake"

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        return ModelResponse(
            text='{"verdict": "MET", "reasoning": "test"}',
            input_tokens=len(prompt.split()),
            output_tokens=10,
            latency_ms=100.0,
            estimated_cost=0.0,
        )

    async def health_check(self) -> bool:
        return True


def test_fake_adapter_implements_protocol():
    adapter = FakeAdapter()
    assert adapter.name == "fake"


def test_fake_adapter_generate():
    adapter = FakeAdapter()
    result = asyncio.run(adapter.generate("test prompt"))
    assert isinstance(result, ModelResponse)
    assert result.input_tokens > 0


def test_fake_adapter_health_check():
    adapter = FakeAdapter()
    assert asyncio.run(adapter.health_check()) is True
