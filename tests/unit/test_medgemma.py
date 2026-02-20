"""Tests for MedGemma adapter."""

import asyncio
from unittest.mock import MagicMock, patch

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


def test_medgemma_default_max_retries_is_5():
    """Default retries reduced from 12 to 5 (server handles cold start)."""
    adapter = MedGemmaAdapter(hf_token="fake")
    assert adapter._max_retries == 5
