"""Runtime helpers for live Streamlit pipeline execution.

This module centralizes:
1. Adapter factories for PRESCREEN/VALIDATE runtime wiring.
2. Health preflight checks for model endpoints and CT.gov connectivity.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.medgemma import MedGemmaAdapter
from trialmatch.models.vertex_medgemma import VertexMedGemmaAdapter
from trialmatch.prescreen.ctgov_client import CTGovClient

if TYPE_CHECKING:
    from trialmatch.models.base import ModelAdapter

logger = structlog.get_logger()

GEMINI_PRO_MODEL_ID = "gemini-3-pro-preview"

VALIDATE_MODE_TWO_STAGE = "Two-Stage (MedGemma → Gemini)"
VALIDATE_MODE_GEMINI_SINGLE = "Gemini 3 Pro (single)"
VALIDATE_MODE_MEDGEMMA_SINGLE = "MedGemma 4B (single)"

_FORCE_HF_MEDGEMMA_ENV = "TRIALMATCH_FORCE_HF_MEDGEMMA"
_VERTEX_PROJECT_ENV = "GCP_PROJECT_ID"
_VERTEX_REGION_ENV = "GCP_REGION"
_VERTEX_REASONING_ENDPOINT_ENV = "VERTEX_ENDPOINT_ID_27B"
_VERTEX_REASONING_DNS_ENV = "VERTEX_DEDICATED_DNS_27B"
_VERTEX_IMAGING_ENDPOINT_ENV = "VERTEX_ENDPOINT_ID"
_VERTEX_IMAGING_DNS_ENV = "VERTEX_DEDICATED_DNS"


@dataclass(slots=True)
class HealthCheckResult:
    """Result of one live preflight check."""

    name: str
    ok: bool
    latency_ms: float
    detail: str


def _is_truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _force_hf_medgemma() -> bool:
    return _is_truthy(os.environ.get(_FORCE_HF_MEDGEMMA_ENV, ""))


def _vertex_reasoning_configured() -> bool:
    return bool(
        os.environ.get(_VERTEX_PROJECT_ENV, "") and
        os.environ.get(_VERTEX_REASONING_ENDPOINT_ENV, "")
    )


def _vertex_imaging_configured() -> bool:
    return bool(
        os.environ.get(_VERTEX_PROJECT_ENV, "") and
        os.environ.get(_VERTEX_IMAGING_ENDPOINT_ENV, "")
    )


def should_use_vertex_reasoning() -> bool:
    """Return True when MedGemma reasoning should use Vertex 27B."""
    return _vertex_reasoning_configured() and not _force_hf_medgemma()


def should_use_vertex_imaging() -> bool:
    """Return True when MedGemma image tasks should use Vertex 4B."""
    return _vertex_imaging_configured() and not _force_hf_medgemma()


def create_reasoning_adapter(hf_token: str = "") -> ModelAdapter:
    """Create MedGemma adapter for medical reasoning tasks."""
    if should_use_vertex_reasoning():
        project_id = os.environ.get(_VERTEX_PROJECT_ENV, "")
        region = os.environ.get(_VERTEX_REGION_ENV, "us-central1")
        endpoint_id = os.environ.get(_VERTEX_REASONING_ENDPOINT_ENV, "")
        dedicated_dns = os.environ.get(_VERTEX_REASONING_DNS_ENV, "") or None
        return VertexMedGemmaAdapter(
            project_id=project_id,
            region=region,
            endpoint_id=endpoint_id,
            model_name="medgemma-27b-vertex",
            dedicated_endpoint_dns=dedicated_dns,
            gpu_hourly_rate=2.30,
        )

    if not hf_token:
        msg = (
            "HF_TOKEN not set. Required for MedGemma reasoning when Vertex 27B "
            "is not configured."
        )
        raise ValueError(msg)
    return MedGemmaAdapter(hf_token=hf_token)


def create_imaging_adapter(hf_token: str = "") -> ModelAdapter:
    """Create MedGemma adapter for imaging tasks."""
    if should_use_vertex_imaging():
        project_id = os.environ.get(_VERTEX_PROJECT_ENV, "")
        region = os.environ.get(_VERTEX_REGION_ENV, "us-central1")
        endpoint_id = os.environ.get(_VERTEX_IMAGING_ENDPOINT_ENV, "")
        dedicated_dns = os.environ.get(_VERTEX_IMAGING_DNS_ENV, "") or None
        return VertexMedGemmaAdapter(
            project_id=project_id,
            region=region,
            endpoint_id=endpoint_id,
            model_name="medgemma-4b-vertex",
            dedicated_endpoint_dns=dedicated_dns,
            gpu_hourly_rate=1.15,
        )

    if not hf_token:
        msg = (
            "HF_TOKEN not set. Required for MedGemma imaging when Vertex 4B "
            "is not configured."
        )
        raise ValueError(msg)
    return MedGemmaAdapter(hf_token=hf_token)


def create_prescreen_adapters(api_key: str, hf_token: str) -> tuple[GeminiAdapter, ModelAdapter]:
    """Create runtime adapters for PRESCREEN."""
    gemini = GeminiAdapter(api_key=api_key, model=GEMINI_PRO_MODEL_ID)
    medgemma = create_reasoning_adapter(hf_token=hf_token)
    return gemini, medgemma


def create_validate_adapters(
    validate_mode: str,
    api_key: str = "",
    hf_token: str = "",
) -> tuple[ModelAdapter, ModelAdapter | None]:
    """Create VALIDATE adapters based on selected mode.

    Returns:
        (reasoning_adapter, labeling_adapter)
        - Two-stage: (MedGemma, Gemini)
        - Single-stage Gemini: (Gemini, None)
        - Single-stage MedGemma: (MedGemma, None)
    """
    if validate_mode == VALIDATE_MODE_TWO_STAGE:
        return (
            create_reasoning_adapter(hf_token=hf_token),
            GeminiAdapter(api_key=api_key, model=GEMINI_PRO_MODEL_ID),
        )

    if validate_mode == VALIDATE_MODE_GEMINI_SINGLE:
        return GeminiAdapter(api_key=api_key, model=GEMINI_PRO_MODEL_ID), None

    if validate_mode == VALIDATE_MODE_MEDGEMMA_SINGLE:
        return create_reasoning_adapter(hf_token=hf_token), None

    msg = f"Unsupported VALIDATE mode: {validate_mode}"
    raise ValueError(msg)


def _truncate_detail(raw: str, max_len: int = 200) -> str:
    if len(raw) <= max_len:
        return raw
    return raw[:max_len].rstrip() + "..."


async def check_model_health(name: str, adapter: ModelAdapter) -> HealthCheckResult:
    """Run model adapter health check without raising."""
    start = time.perf_counter()
    try:
        ok = await adapter.health_check()
        latency_ms = (time.perf_counter() - start) * 1000
        detail = "ok" if ok else "health check returned False"
        event = "live_preflight_model_ok" if ok else "live_preflight_model_failed"
        log_fn = logger.info if ok else logger.warning
        log_fn(
            event,
            check=name,
            model=adapter.name,
            latency_ms=f"{latency_ms:.0f}",
            detail=detail,
        )
        return HealthCheckResult(name=name, ok=ok, latency_ms=latency_ms, detail=detail)
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        detail = _truncate_detail(f"{type(exc).__name__}: {exc}")
        logger.warning(
            "live_preflight_model_exception",
            check=name,
            model=adapter.name,
            latency_ms=f"{latency_ms:.0f}",
            error=detail,
        )
        return HealthCheckResult(name=name, ok=False, latency_ms=latency_ms, detail=detail)


async def check_ctgov_health() -> HealthCheckResult:
    """Run a minimal CT.gov API connectivity check without raising."""
    start = time.perf_counter()
    client = CTGovClient(timeout_seconds=15.0, max_retries=1)
    try:
        raw = await client.search(
            condition="cancer",
            status=["RECRUITING"],
            page_size=1,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if isinstance(raw, dict):
            studies = raw.get("studies") or []
            detail = f"ok ({len(studies)} studies)"
            logger.info(
                "live_preflight_ctgov_ok",
                latency_ms=f"{latency_ms:.0f}",
                studies=len(studies),
            )
            return HealthCheckResult(
                name="CT.gov API",
                ok=True,
                latency_ms=latency_ms,
                detail=detail,
            )

        detail = "unexpected response type"
        logger.warning(
            "live_preflight_ctgov_failed",
            latency_ms=f"{latency_ms:.0f}",
            detail=detail,
        )
        return HealthCheckResult(
            name="CT.gov API",
            ok=False,
            latency_ms=latency_ms,
            detail=detail,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        detail = _truncate_detail(f"{type(exc).__name__}: {exc}")
        logger.warning(
            "live_preflight_ctgov_exception",
            latency_ms=f"{latency_ms:.0f}",
            error=detail,
        )
        return HealthCheckResult(
            name="CT.gov API",
            ok=False,
            latency_ms=latency_ms,
            detail=detail,
        )
    finally:
        await client.aclose()


async def run_live_preflight(
    gemini_adapter: ModelAdapter,
    medgemma_adapter: ModelAdapter,
    include_ctgov: bool = True,
) -> list[HealthCheckResult]:
    """Run all configured live preflight checks.

    This function never raises for individual failed checks.
    """
    coroutines = [
        check_model_health("Gemini", gemini_adapter),
        check_model_health("MedGemma", medgemma_adapter),
    ]
    if include_ctgov:
        coroutines.append(check_ctgov_health())

    raw_results = await asyncio.gather(*coroutines, return_exceptions=True)
    results: list[HealthCheckResult] = []
    for idx, item in enumerate(raw_results):
        if isinstance(item, Exception):
            detail = _truncate_detail(f"{type(item).__name__}: {item}")
            logger.warning(
                "live_preflight_unexpected_exception",
                check_index=idx,
                error=detail,
            )
            results.append(
                HealthCheckResult(
                    name=f"check-{idx}",
                    ok=False,
                    latency_ms=0.0,
                    detail=detail,
                )
            )
            continue
        results.append(item)

    failed_checks = [r.name for r in results if not r.ok]
    logger.info(
        "live_preflight_complete",
        total_checks=len(results),
        failed_checks=failed_checks,
    )
    if failed_checks:
        logger.warning(
            "live_preflight_failures_detected",
            failed_checks=failed_checks,
        )

    return results


def failed_preflight_checks(results: list[HealthCheckResult]) -> list[HealthCheckResult]:
    """Return preflight checks that failed."""
    return [r for r in results if not r.ok]
