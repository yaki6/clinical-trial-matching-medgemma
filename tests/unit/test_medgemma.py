"""Tests for MedGemma adapter."""

import asyncio
from unittest.mock import MagicMock, patch

from trialmatch.models.medgemma import MedGemmaAdapter, format_gemma_prompt
from trialmatch.models.schema import ModelResponse


def _mock_chat_response(
    content: str = '{"verdict": "MET", "reasoning": "ok"}',
    prompt_tokens: int = 150,
    completion_tokens: int = 30,
):
    """Create a mock ChatCompletionOutput matching HF InferenceClient structure."""
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens

    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


def test_format_gemma_prompt():
    """Deprecated format_gemma_prompt still works for external callers."""
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
    """generate() uses chat_completion and returns exact token counts."""
    mock_instance = MagicMock()
    mock_instance.chat_completion.return_value = _mock_chat_response()
    mock_client_cls.return_value = mock_instance

    adapter = MedGemmaAdapter(hf_token="fake")
    adapter._client = mock_instance

    result = asyncio.run(adapter.generate("test prompt"))
    assert isinstance(result, ModelResponse)
    assert "MET" in result.text
    assert result.latency_ms > 0
    assert result.input_tokens == 150
    assert result.output_tokens == 30
    assert result.token_count_estimated is False


@patch("trialmatch.models.medgemma.InferenceClient")
def test_medgemma_health_check(mock_client_cls):
    """health_check uses chat_completion endpoint."""
    mock_instance = MagicMock()
    mock_instance.chat_completion.return_value = _mock_chat_response(content="ok")
    mock_client_cls.return_value = mock_instance

    adapter = MedGemmaAdapter(hf_token="fake")
    adapter._client = mock_instance

    assert asyncio.run(adapter.health_check()) is True


def test_medgemma_sets_scale_up_timeout_header():
    """X-Scale-Up-Timeout header must be passed to InferenceClient."""
    with patch("trialmatch.models.medgemma.InferenceClient") as mock_cls:
        MedGemmaAdapter(hf_token="fake")
        call_kwargs = mock_cls.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Scale-Up-Timeout"] == "300"


def test_medgemma_default_max_retries_is_5():
    """Default retries reduced from 12 to 5 (server handles cold start)."""
    adapter = MedGemmaAdapter(hf_token="fake")
    assert adapter._max_retries == 5


@patch("trialmatch.models.medgemma.InferenceClient")
def test_medgemma_returns_exact_token_counts(mock_client_cls):
    """chat_completion provides exact token counts, not len//4 estimates."""
    mock_instance = MagicMock()
    mock_instance.chat_completion.return_value = _mock_chat_response(
        content='{"label": "included"}',
        prompt_tokens=200,
        completion_tokens=50,
    )
    mock_client_cls.return_value = mock_instance

    adapter = MedGemmaAdapter(hf_token="fake")
    adapter._client = mock_instance

    result = asyncio.run(adapter.generate("test"))
    assert result.token_count_estimated is False
    assert result.input_tokens == 200
    assert result.output_tokens == 50
