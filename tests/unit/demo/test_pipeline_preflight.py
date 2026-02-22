"""Unit tests for preflight gating helper behavior."""

from __future__ import annotations

from trialmatch.live_runtime import HealthCheckResult, failed_preflight_checks


def test_failed_preflight_checks_returns_only_failed():
    results = [
        HealthCheckResult(name="Gemini", ok=True, latency_ms=10.0, detail="ok"),
        HealthCheckResult(name="MedGemma", ok=False, latency_ms=20.0, detail="401"),
        HealthCheckResult(name="CT.gov API", ok=False, latency_ms=30.0, detail="timeout"),
    ]

    failed = failed_preflight_checks(results)
    assert [r.name for r in failed] == ["MedGemma", "CT.gov API"]


def test_failed_preflight_checks_empty_when_all_ok():
    results = [
        HealthCheckResult(name="Gemini", ok=True, latency_ms=10.0, detail="ok"),
        HealthCheckResult(name="MedGemma", ok=True, latency_ms=20.0, detail="ok"),
    ]

    assert failed_preflight_checks(results) == []
