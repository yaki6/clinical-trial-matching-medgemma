"""Tests for Gemini adapter."""

import asyncio
from unittest.mock import MagicMock, patch

from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.schema import ModelResponse


def test_gemini_adapter_name():
    with patch("trialmatch.models.gemini.genai.Client"):
        adapter = GeminiAdapter(api_key="fake")
        assert adapter.name == "gemini-3-pro-preview"


@patch("trialmatch.models.gemini.genai.Client")
def test_gemini_generate(mock_client_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"verdict": "NOT_MET", "reasoning": "excluded"}'
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 150
    mock_response.usage_metadata.candidates_token_count = 30
    mock_client.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client

    adapter = GeminiAdapter(api_key="fake")
    result = asyncio.run(adapter.generate("test prompt"))

    assert isinstance(result, ModelResponse)
    assert "NOT_MET" in result.text
    assert result.input_tokens == 150
    assert result.output_tokens == 30
    assert result.estimated_cost > 0


@patch("trialmatch.models.gemini.genai.Client")
def test_gemini_health_check(mock_client_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_client.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client

    adapter = GeminiAdapter(api_key="fake")
    assert asyncio.run(adapter.health_check()) is True
