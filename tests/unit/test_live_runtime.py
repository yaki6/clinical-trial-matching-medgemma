"""Unit tests for live runtime helper utilities."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trialmatch.live_runtime import (
    GEMINI_PRO_MODEL_ID,
    VALIDATE_MODE_GEMINI_SINGLE,
    VALIDATE_MODE_MEDGEMMA_SINGLE,
    VALIDATE_MODE_TWO_STAGE,
    HealthCheckResult,
    check_ctgov_health,
    check_model_health,
    create_imaging_adapter,
    create_prescreen_adapters,
    create_validate_adapters,
    run_live_preflight,
)


def test_create_prescreen_adapters_uses_gemini_pro_preview():
    with (
        patch.dict(
            os.environ,
            {
                "TRIALMATCH_FORCE_HF_MEDGEMMA": "",
                "GCP_PROJECT_ID": "",
                "VERTEX_ENDPOINT_ID_27B": "",
            },
            clear=False,
        ),
        patch("trialmatch.live_runtime.GeminiAdapter") as mock_gemini_cls,
        patch("trialmatch.live_runtime.MedGemmaAdapter") as mock_medgemma_cls,
    ):
        gemini_instance = MagicMock()
        medgemma_instance = MagicMock()
        mock_gemini_cls.return_value = gemini_instance
        mock_medgemma_cls.return_value = medgemma_instance

        gemini, medgemma = create_prescreen_adapters(api_key="key", hf_token="token")

    mock_gemini_cls.assert_called_once_with(api_key="key", model=GEMINI_PRO_MODEL_ID)
    mock_medgemma_cls.assert_called_once_with(hf_token="token")
    assert gemini is gemini_instance
    assert medgemma is medgemma_instance


def test_create_validate_adapters_two_stage_uses_gemini_pro_preview():
    with (
        patch.dict(
            os.environ,
            {
                "TRIALMATCH_FORCE_HF_MEDGEMMA": "",
                "GCP_PROJECT_ID": "",
                "VERTEX_ENDPOINT_ID_27B": "",
            },
            clear=False,
        ),
        patch("trialmatch.live_runtime.GeminiAdapter") as mock_gemini_cls,
        patch("trialmatch.live_runtime.MedGemmaAdapter") as mock_medgemma_cls,
    ):
        gemini_instance = MagicMock()
        medgemma_instance = MagicMock()
        mock_gemini_cls.return_value = gemini_instance
        mock_medgemma_cls.return_value = medgemma_instance

        reasoning, labeling = create_validate_adapters(
            VALIDATE_MODE_TWO_STAGE,
            api_key="key",
            hf_token="token",
        )

    mock_gemini_cls.assert_called_once_with(api_key="key", model=GEMINI_PRO_MODEL_ID)
    mock_medgemma_cls.assert_called_once_with(hf_token="token")
    assert reasoning is medgemma_instance
    assert labeling is gemini_instance


def test_create_validate_adapters_gemini_single_uses_gemini_pro_preview():
    with patch("trialmatch.live_runtime.GeminiAdapter") as mock_gemini_cls:
        gemini_instance = MagicMock()
        mock_gemini_cls.return_value = gemini_instance

        reasoning, labeling = create_validate_adapters(
            VALIDATE_MODE_GEMINI_SINGLE,
            api_key="key",
        )

    mock_gemini_cls.assert_called_once_with(api_key="key", model=GEMINI_PRO_MODEL_ID)
    assert reasoning is gemini_instance
    assert labeling is None


def test_create_validate_adapters_medgemma_single():
    with (
        patch.dict(
            os.environ,
            {
                "TRIALMATCH_FORCE_HF_MEDGEMMA": "",
                "GCP_PROJECT_ID": "",
                "VERTEX_ENDPOINT_ID_27B": "",
            },
            clear=False,
        ),
        patch("trialmatch.live_runtime.MedGemmaAdapter") as mock_medgemma_cls,
    ):
        medgemma_instance = MagicMock()
        mock_medgemma_cls.return_value = medgemma_instance

        reasoning, labeling = create_validate_adapters(
            VALIDATE_MODE_MEDGEMMA_SINGLE,
            hf_token="token",
        )

    mock_medgemma_cls.assert_called_once_with(hf_token="token")
    assert reasoning is medgemma_instance
    assert labeling is None


def test_create_reasoning_adapter_prefers_vertex_27b_when_configured():
    with (
        patch.dict(
            os.environ,
            {
                "TRIALMATCH_FORCE_HF_MEDGEMMA": "",
                "GCP_PROJECT_ID": "proj-123",
                "GCP_REGION": "us-central1",
                "VERTEX_ENDPOINT_ID_27B": "ep-27b",
                "VERTEX_DEDICATED_DNS_27B": "ep-27b.vertex.test",
            },
            clear=False,
        ),
        patch("trialmatch.live_runtime.VertexMedGemmaAdapter") as mock_vertex_cls,
    ):
        vertex_instance = MagicMock()
        mock_vertex_cls.return_value = vertex_instance

        adapter = create_validate_adapters(
            VALIDATE_MODE_MEDGEMMA_SINGLE,
            hf_token="",
        )[0]

    mock_vertex_cls.assert_called_once_with(
        project_id="proj-123",
        region="us-central1",
        endpoint_id="ep-27b",
        model_name="medgemma-27b-vertex",
        dedicated_endpoint_dns="ep-27b.vertex.test",
        gpu_hourly_rate=2.30,
    )
    assert adapter is vertex_instance


def test_create_reasoning_adapter_requires_hf_token_when_no_vertex():
    with patch.dict(
        os.environ,
        {
            "TRIALMATCH_FORCE_HF_MEDGEMMA": "",
            "GCP_PROJECT_ID": "",
            "VERTEX_ENDPOINT_ID_27B": "",
        },
        clear=False,
    ):
        with pytest.raises(ValueError, match="HF_TOKEN not set"):
            create_validate_adapters(
                VALIDATE_MODE_MEDGEMMA_SINGLE,
                hf_token="",
            )


def test_create_imaging_adapter_prefers_vertex_4b_when_configured():
    with (
        patch.dict(
            os.environ,
            {
                "TRIALMATCH_FORCE_HF_MEDGEMMA": "",
                "GCP_PROJECT_ID": "proj-123",
                "GCP_REGION": "us-central1",
                "VERTEX_ENDPOINT_ID": "ep-4b",
                "VERTEX_DEDICATED_DNS": "ep-4b.vertex.test",
            },
            clear=False,
        ),
        patch("trialmatch.live_runtime.VertexMedGemmaAdapter") as mock_vertex_cls,
    ):
        vertex_instance = MagicMock()
        mock_vertex_cls.return_value = vertex_instance
        adapter = create_imaging_adapter(hf_token="")

    mock_vertex_cls.assert_called_once_with(
        project_id="proj-123",
        region="us-central1",
        endpoint_id="ep-4b",
        model_name="medgemma-4b-vertex",
        dedicated_endpoint_dns="ep-4b.vertex.test",
        gpu_hourly_rate=1.15,
    )
    assert adapter is vertex_instance


def test_create_validate_adapters_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unsupported VALIDATE mode"):
        create_validate_adapters("invalid mode", api_key="key", hf_token="token")


def test_check_model_health_success():
    adapter = MagicMock()
    adapter.name = "gemini-3-pro-preview"
    adapter.health_check = AsyncMock(return_value=True)

    result = asyncio.run(check_model_health("Gemini", adapter))

    assert result.name == "Gemini"
    assert result.ok is True
    assert result.detail == "ok"
    assert result.latency_ms >= 0


def test_check_model_health_failure_detail_without_raise():
    adapter = MagicMock()
    adapter.name = "medgemma-4b"
    adapter.health_check = AsyncMock(side_effect=RuntimeError("endpoint unavailable"))

    result = asyncio.run(check_model_health("MedGemma", adapter))

    assert result.name == "MedGemma"
    assert result.ok is False
    assert "endpoint unavailable" in result.detail


def test_check_ctgov_health_success():
    mock_client = MagicMock()
    mock_client.search = AsyncMock(return_value={"studies": [{}]})
    mock_client.aclose = AsyncMock()

    with patch("trialmatch.live_runtime.CTGovClient", return_value=mock_client):
        result = asyncio.run(check_ctgov_health())

    assert result.name == "CT.gov API"
    assert result.ok is True
    assert "ok" in result.detail
    mock_client.search.assert_awaited_once()
    mock_client.aclose.assert_awaited_once()


def test_check_ctgov_health_failure_detail_without_raise():
    mock_client = MagicMock()
    mock_client.search = AsyncMock(side_effect=RuntimeError("ctgov timeout"))
    mock_client.aclose = AsyncMock()

    with patch("trialmatch.live_runtime.CTGovClient", return_value=mock_client):
        result = asyncio.run(check_ctgov_health())

    assert result.ok is False
    assert "ctgov timeout" in result.detail
    mock_client.aclose.assert_awaited_once()


def test_run_live_preflight_reports_failures_without_raising():
    async def _fake_model_check(name: str, _adapter):
        if name == "Gemini":
            return HealthCheckResult(name=name, ok=True, latency_ms=10.0, detail="ok")
        return HealthCheckResult(name=name, ok=False, latency_ms=20.0, detail="failed")

    async def _fake_ctgov_check():
        return HealthCheckResult(name="CT.gov API", ok=False, latency_ms=30.0, detail="503")

    with (
        patch(
            "trialmatch.live_runtime.check_model_health",
            new=AsyncMock(side_effect=_fake_model_check),
        ),
        patch(
            "trialmatch.live_runtime.check_ctgov_health",
            new=AsyncMock(side_effect=_fake_ctgov_check),
        ),
    ):
        results = asyncio.run(
            run_live_preflight(
                gemini_adapter=MagicMock(),
                medgemma_adapter=MagicMock(),
                include_ctgov=True,
            )
        )

    assert len(results) == 3
    assert [r.name for r in results] == ["Gemini", "MedGemma", "CT.gov API"]
    assert [r.ok for r in results] == [True, False, False]
    assert all(isinstance(r, HealthCheckResult) for r in results)
