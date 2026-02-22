"""Unit tests for VertexMedGemmaAdapter.

All tests mock HTTP calls — no live Vertex AI calls.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from trialmatch.models.vertex_medgemma import VertexMedGemmaAdapter

OK_JSON = {
    "predictions": [
        {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    ]
}

MINI_JSON = {
    "predictions": [
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    ]
}


@pytest.fixture
def adapter():
    """Create adapter with test config."""
    return VertexMedGemmaAdapter(
        project_id="test-project",
        region="us-central1",
        endpoint_id="1234567890",
        model_name="medgemma-4b-vertex",
        max_retries=3,
        retry_backoff=0.01,
        max_wait=0.05,
    )


def test_name_property(adapter):
    """Adapter name matches configured model_name."""
    assert adapter.name == "medgemma-4b-vertex"


def test_predict_url(adapter):
    """Predict URL is constructed correctly from project/region/endpoint."""
    expected = (
        "https://us-central1-aiplatform.googleapis.com/v1/"
        "projects/test-project/locations/us-central1/"
        "endpoints/1234567890:predict"
    )
    assert adapter._predict_url == expected


def test_predict_url_dedicated_endpoint():
    """Dedicated endpoint DNS overrides the shared domain."""
    a = VertexMedGemmaAdapter(
        project_id="p",
        region="us-central1",
        endpoint_id="ep-123",
        dedicated_endpoint_dns="ep-123.us-central1-12345.prediction.vertexai.goog",
    )
    expected = (
        "https://ep-123.us-central1-12345.prediction.vertexai.goog/v1/"
        "projects/p/locations/us-central1/"
        "endpoints/ep-123:predict"
    )
    assert a._predict_url == expected


@pytest.mark.asyncio
async def test_generate_chat_completions_format(adapter):
    """Request payload uses chatCompletions format and response is parsed correctly."""
    mock_response = httpx.Response(
        200,
        json={
            "predictions": [
                {
                    "choices": [
                        {"message": {"content": "The patient meets criterion."}}
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                }
            ]
        },
        request=httpx.Request("POST", "https://example.com"),
    )

    with (
        patch.object(adapter, "_get_auth_token", return_value="fake-token"),
        patch("httpx.post", return_value=mock_response) as mock_post,
    ):
        result = await adapter.generate("Evaluate this criterion", max_tokens=512)

    assert result.text == "The patient meets criterion."
    assert result.input_tokens == 50
    assert result.output_tokens == 20
    assert result.latency_ms > 0
    assert result.token_count_estimated is False

    # Verify request payload structure
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert "instances" in payload
    instance = payload["instances"][0]
    assert instance["@requestFormat"] == "chatCompletions"
    assert instance["messages"][0]["role"] == "user"
    assert instance["messages"][0]["content"] == "Evaluate this criterion"
    assert instance["max_tokens"] == 512


@pytest.mark.asyncio
async def test_generate_with_image_chat_completions_format(adapter, tmp_path):
    """Multimodal payload includes image_url content for Vertex chatCompletions."""
    image_path = tmp_path / "ct.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    mock_response = httpx.Response(
        200,
        json={
            "predictions": [
                {
                    "choices": [{"message": {"content": "Image findings"}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 12},
                }
            ]
        },
        request=httpx.Request("POST", "https://example.com"),
    )

    with (
        patch.object(adapter, "_get_auth_token", return_value="fake-token"),
        patch("httpx.post", return_value=mock_response) as mock_post,
    ):
        result = await adapter.generate_with_image("Analyze image", image_path, max_tokens=64)

    assert result.text == "Image findings"
    assert result.input_tokens == 30
    assert result.output_tokens == 12

    payload = mock_post.call_args.kwargs["json"]
    instance = payload["instances"][0]
    assert instance["@requestFormat"] == "chatCompletions"
    content = instance["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Analyze image"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert instance["max_tokens"] == 64


@pytest.mark.asyncio
async def test_generate_fallback_on_no_choices(adapter):
    """Falls back to raw text extraction when chatCompletions format not in response."""
    mock_response = httpx.Response(
        200,
        json={"predictions": ["Raw text response from model"]},
        request=httpx.Request("POST", "https://example.com"),
    )

    with (
        patch.object(adapter, "_get_auth_token", return_value="fake-token"),
        patch("httpx.post", return_value=mock_response),
    ):
        result = await adapter.generate("Test prompt")

    assert result.text == "Raw text response from model"
    assert result.token_count_estimated is True


@pytest.mark.asyncio
async def test_generate_retry_on_503(adapter):
    """Transient 503 triggers retry, eventual success."""
    error_response = httpx.Response(
        503,
        text="Service Unavailable",
        request=httpx.Request("POST", "https://example.com"),
    )
    ok_response = httpx.Response(
        200,
        json=OK_JSON,
        request=httpx.Request("POST", "https://example.com"),
    )

    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return error_response
        return ok_response

    with (
        patch.object(adapter, "_get_auth_token", return_value="fake-token"),
        patch("httpx.post", side_effect=mock_post),
    ):
        result = await adapter.generate("Test")

    assert result.text == "OK"
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_retry_on_429(adapter):
    """Rate limit 429 triggers retry."""
    error_response = httpx.Response(
        429,
        text="RESOURCE_EXHAUSTED",
        request=httpx.Request("POST", "https://example.com"),
    )
    ok_response = httpx.Response(
        200,
        json=OK_JSON,
        request=httpx.Request("POST", "https://example.com"),
    )

    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return error_response
        return ok_response

    with (
        patch.object(adapter, "_get_auth_token", return_value="fake-token"),
        patch("httpx.post", side_effect=mock_post),
    ):
        result = await adapter.generate("Test")

    assert result.text == "OK"
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_retry_exhausted_raises(adapter):
    """All retries exhausted raises RuntimeError."""
    error_response = httpx.Response(
        503,
        text="Service Unavailable",
        request=httpx.Request("POST", "https://example.com"),
    )

    with (
        patch.object(adapter, "_get_auth_token", return_value="fake-token"),
        patch("httpx.post", return_value=error_response),
        pytest.raises(RuntimeError, match="failed after 3 retries"),
    ):
        await adapter.generate("Test")


@pytest.mark.asyncio
async def test_generate_non_transient_error_raises_immediately(adapter):
    """Non-transient errors (400, 404) raise immediately without retry."""
    error_response = httpx.Response(
        400,
        text="Bad Request",
        request=httpx.Request("POST", "https://example.com"),
    )

    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return error_response

    with (
        patch.object(adapter, "_get_auth_token", return_value="fake-token"),
        patch("httpx.post", side_effect=mock_post),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await adapter.generate("Test")

    assert call_count == 1


@pytest.mark.asyncio
async def test_health_check_success(adapter):
    """Health check returns True on successful response."""
    mock_response = httpx.Response(
        200,
        json=MINI_JSON,
        request=httpx.Request("POST", "https://example.com"),
    )

    with (
        patch.object(adapter, "_get_auth_token", return_value="fake-token"),
        patch("httpx.post", return_value=mock_response),
    ):
        assert await adapter.health_check() is True


@pytest.mark.asyncio
async def test_health_check_failure(adapter):
    """Health check returns False on failure."""
    with (
        patch.object(adapter, "_get_auth_token", return_value="fake-token"),
        patch("httpx.post", side_effect=Exception("Connection refused")),
    ):
        assert await adapter.health_check() is False


def test_cost_estimation(adapter):
    """GPU-hour cost estimation from latency."""
    # 1000ms at default $1.15/hr L4 rate
    cost = adapter._estimate_cost(latency_ms=1000.0)
    expected = (1000.0 / 3_600_000) * 1.15
    assert abs(cost - expected) < 1e-8

    # Custom hourly rate
    adapter_custom = VertexMedGemmaAdapter(
        project_id="p",
        region="r",
        endpoint_id="e",
        gpu_hourly_rate=4.22,
    )
    cost = adapter_custom._estimate_cost(latency_ms=3600000.0)
    assert abs(cost - 4.22) < 1e-8


@pytest.mark.asyncio
async def test_auth_credentials_used(adapter):
    """Auth token is attached to request headers."""
    mock_response = httpx.Response(
        200,
        json=MINI_JSON,
        request=httpx.Request("POST", "https://example.com"),
    )

    with (
        patch.object(adapter, "_get_auth_token", return_value="test-bearer-token"),
        patch("httpx.post", return_value=mock_response) as mock_post,
    ):
        await adapter.generate("Test")

    call_args = mock_post.call_args
    headers = call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-bearer-token"
