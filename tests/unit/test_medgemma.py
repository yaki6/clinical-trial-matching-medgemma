"""Tests for MedGemma adapter."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trialmatch.models.medgemma import MedGemmaAdapter, format_gemma_prompt
from trialmatch.models.schema import ModelResponse


def test_format_gemma_prompt():
    result = format_gemma_prompt("system msg", "user msg")
    assert "<start_of_turn>user" in result
    assert "system msg" in result
    assert "user msg" in result
    assert "<start_of_turn>model" in result


def test_medgemma_adapter_name():
    adapter = MedGemmaAdapter(hf_token="fake")
    assert adapter.name == "medgemma-1.5-4b"


@patch("trialmatch.models.medgemma.InferenceClient")
def test_medgemma_generate(mock_client_cls):
    mock_instance = MagicMock()
    mock_instance.text_generation.return_value = '{"verdict": "MET", "reasoning": "ok"}'
    mock_client_cls.return_value = mock_instance

    adapter = MedGemmaAdapter(hf_token="fake")
    adapter._client = mock_instance

    result = asyncio.run(adapter.generate("test prompt"))
    assert isinstance(result, ModelResponse)
    assert "MET" in result.text
    assert result.latency_ms > 0


@patch("trialmatch.models.medgemma.InferenceClient")
def test_medgemma_health_check(mock_client_cls):
    mock_instance = MagicMock()
    mock_instance.text_generation.return_value = "ok"
    mock_client_cls.return_value = mock_instance

    adapter = MedGemmaAdapter(hf_token="fake")
    adapter._client = mock_instance

    assert asyncio.run(adapter.health_check()) is True


def test_medgemma_sets_scale_up_timeout_header():
    """X-Scale-Up-Timeout header must be passed to InferenceClient."""
    with patch("trialmatch.models.medgemma.InferenceClient") as mock_cls:
        MedGemmaAdapter(hf_token="fake")
        call_kwargs = mock_cls.call_args[1]  # keyword args
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Scale-Up-Timeout"] == "300"


def test_medgemma_default_max_retries_is_8():
    """Default retries set to 8 for cold-start tolerance with X-Scale-Up-Timeout."""
    adapter = MedGemmaAdapter(hf_token="fake")
    assert adapter._max_retries == 8


@patch("trialmatch.models.medgemma.InferenceClient")
def test_medgemma_generate_cuda_kernel_fault_fails_fast(mock_client_cls):
    """CUDA kernel faults should fail immediately with actionable error."""
    mock_instance = MagicMock()
    mock_instance.text_generation.side_effect = RuntimeError("CUDA error: misaligned address")
    mock_client_cls.return_value = mock_instance

    adapter = MedGemmaAdapter(hf_token="fake")
    adapter._client = mock_instance

    with pytest.raises(RuntimeError, match="kernel fault"):
        asyncio.run(adapter.generate("test prompt", max_tokens=32))

    # Must not spin through full retry budget on non-recoverable kernel faults.
    assert mock_instance.text_generation.call_count == 1


@patch("trialmatch.models.medgemma.InferenceClient")
def test_medgemma_generate_with_image_cuda_kernel_fault_fails_fast(mock_client_cls):
    """Image path should also fail fast on CUDA kernel faults."""
    mock_client_cls.return_value = MagicMock()

    adapter = MedGemmaAdapter(hf_token="fake")
    adapter._call_image_generation = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("CUDA error: misaligned address")
    )

    with pytest.raises(RuntimeError, match="kernel fault"):
        asyncio.run(
            adapter.generate_with_image(
                prompt="Describe image",
                image_path=Path("dummy.png"),
                max_tokens=32,
            )
        )

    assert adapter._call_image_generation.call_count == 1  # type: ignore[attr-defined]
