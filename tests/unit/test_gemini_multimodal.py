"""Unit tests for GeminiAdapter.generate_with_image() multimodal method."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.schema import ModelResponse


def _make_1x1_png(path: Path) -> None:
    """Write a minimal valid 1x1 red PNG to *path*."""
    # Minimal PNG: 8-byte signature + IHDR + IDAT + IEND
    import zlib

    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR: 1x1, 8-bit RGBA
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    # IDAT: single row, filter byte 0 + RGBA (red pixel)
    raw_row = b"\x00\xff\x00\x00\xff"  # filter=0, R=255, G=0, B=0, A=255
    compressed = zlib.compress(raw_row)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF)
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc

    # IEND
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    iend = struct.pack(">I", 0) + b"IEND" + iend_crc

    path.write_bytes(signature + ihdr + idat + iend)


def _mock_response(text: str = '{"findings": "normal"}') -> SimpleNamespace:
    """Build a fake Gemini response object."""
    usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=50)
    return SimpleNamespace(text=text, usage_metadata=usage)


@pytest.fixture()
def tmp_png(tmp_path: Path) -> Path:
    """Create a temporary 1x1 PNG file and return its path."""
    p = tmp_path / "test_image.png"
    _make_1x1_png(p)
    return p


@pytest.fixture()
def tmp_jpg(tmp_path: Path) -> Path:
    """Create a temporary JPEG file (minimal bytes, not a real JPEG)."""
    p = tmp_path / "test_image.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)  # minimal JPEG header
    return p


@pytest.mark.asyncio
async def test_generate_with_image_contents_order(tmp_png: Path) -> None:
    """Image part must come FIRST, text part SECOND in the contents list."""
    mock_generate = MagicMock(return_value=_mock_response())

    with patch("trialmatch.models.gemini.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content = mock_generate
        adapter = GeminiAdapter(api_key="test-key", model="gemini-3-flash-preview")

        result = await adapter.generate_with_image(
            prompt="Describe findings", image_path=tmp_png, max_tokens=512
        )

    # Verify generate_content was called exactly once
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args

    # Contents are the second positional or keyword arg
    contents = call_kwargs.kwargs.get("contents") or call_kwargs.args[0]

    # Must have exactly 2 parts
    assert len(contents) == 2, f"Expected 2 content parts, got {len(contents)}"

    # First part: image (has inline_data)
    image_part = contents[0]
    assert hasattr(image_part, "inline_data") and image_part.inline_data is not None, (
        "First part must be an image with inline_data"
    )
    assert image_part.inline_data.mime_type == "image/png"

    # Second part: text
    text_part = contents[1]
    assert hasattr(text_part, "text") and text_part.text == "Describe findings", (
        "Second part must be the text prompt"
    )


@pytest.mark.asyncio
async def test_generate_with_image_no_json_response_mime(tmp_png: Path) -> None:
    """Config must NOT include response_mime_type: application/json for multimodal."""
    mock_generate = MagicMock(return_value=_mock_response())

    with patch("trialmatch.models.gemini.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content = mock_generate
        adapter = GeminiAdapter(api_key="test-key", model="gemini-3-flash-preview")

        await adapter.generate_with_image(
            prompt="Describe findings", image_path=tmp_png
        )

    call_kwargs = mock_generate.call_args
    config = call_kwargs.kwargs.get("config", {})

    # Config should NOT have response_mime_type set to application/json
    if isinstance(config, dict):
        assert config.get("response_mime_type") != "application/json", (
            "response_mime_type must not be 'application/json' for multimodal"
        )
    else:
        # If config is a GenerateContentConfig object
        mime = getattr(config, "response_mime_type", None)
        assert mime != "application/json"


@pytest.mark.asyncio
async def test_generate_with_image_returns_model_response(tmp_png: Path) -> None:
    """Must return a valid ModelResponse with cost and latency populated."""
    mock_generate = MagicMock(return_value=_mock_response("Chest X-ray shows normal lung fields."))

    with patch("trialmatch.models.gemini.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content = mock_generate
        adapter = GeminiAdapter(api_key="test-key", model="gemini-3-flash-preview")

        result = await adapter.generate_with_image(
            prompt="Describe findings", image_path=tmp_png, max_tokens=1024
        )

    assert isinstance(result, ModelResponse)
    assert result.text == "Chest X-ray shows normal lung fields."
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.latency_ms > 0
    assert result.estimated_cost > 0


@pytest.mark.asyncio
async def test_generate_with_image_jpeg_mime(tmp_jpg: Path) -> None:
    """JPEG files should have mime_type image/jpeg."""
    mock_generate = MagicMock(return_value=_mock_response())

    with patch("trialmatch.models.gemini.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content = mock_generate
        adapter = GeminiAdapter(api_key="test-key", model="gemini-3-flash-preview")

        await adapter.generate_with_image(
            prompt="Analyze", image_path=tmp_jpg
        )

    contents = mock_generate.call_args.kwargs["contents"]
    assert contents[0].inline_data.mime_type == "image/jpeg"


@pytest.mark.asyncio
async def test_generate_with_image_thinking_model_token_floor(tmp_png: Path) -> None:
    """Thinking models should use effective_max_tokens = max(max_tokens, 32768)."""
    mock_generate = MagicMock(return_value=_mock_response())

    with patch("trialmatch.models.gemini.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content = mock_generate
        adapter = GeminiAdapter(api_key="test-key", model="gemini-3-pro-preview")

        await adapter.generate_with_image(
            prompt="Findings", image_path=tmp_png, max_tokens=512
        )

    config = mock_generate.call_args.kwargs["config"]
    if isinstance(config, dict):
        assert config["max_output_tokens"] == 32768
    else:
        assert config.max_output_tokens == 32768


@pytest.mark.asyncio
async def test_generate_with_image_retries_on_transient_error(tmp_png: Path) -> None:
    """Should retry on 503/429 transient errors, then succeed."""
    good_response = _mock_response("OK")
    mock_generate = MagicMock(
        side_effect=[Exception("503 Service Unavailable"), good_response]
    )

    with patch("trialmatch.models.gemini.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content = mock_generate
        adapter = GeminiAdapter(
            api_key="test-key",
            model="gemini-3-flash-preview",
            max_retries=3,
            retry_backoff=0.01,  # fast retry for tests
            max_wait=0.01,
        )

        result = await adapter.generate_with_image(
            prompt="Retry test", image_path=tmp_png
        )

    assert result.text == "OK"
    assert mock_generate.call_count == 2
