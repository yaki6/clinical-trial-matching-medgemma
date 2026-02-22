"""Unit tests for demo/cache_manager.py cache consistency contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from trialmatch.prescreen.schema import PresearchResult, TrialCandidate

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_MANAGER_PATH = REPO_ROOT / "demo" / "cache_manager.py"


def _load_cache_manager_module():
    spec = importlib.util.spec_from_file_location("demo_cache_manager", CACHE_MANAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_prescreen(topic_id: str, nct_ids: list[str]) -> PresearchResult:
    return PresearchResult(
        topic_id=topic_id,
        ingest_source="gold",
        candidates=[
            TrialCandidate(
                nct_id=nct_id,
                title=f"Trial {nct_id}",
                brief_title=f"Trial {nct_id}",
                status="RECRUITING",
                phase=["Phase 2"],
                conditions=["NSCLC"],
                interventions=[],
                sponsor="Demo Sponsor",
                enrollment=100,
                study_type="INTERVENTIONAL",
                found_by_queries=["nsclc"],
            )
            for nct_id in nct_ids
        ],
        agent_reasoning="cached",
        tool_call_trace=[],
        total_api_calls=1,
        total_unique_nct_ids=len(nct_ids),
        gemini_input_tokens=0,
        gemini_output_tokens=0,
        gemini_estimated_cost=0.0,
        medgemma_calls=0,
        medgemma_estimated_cost=0.0,
        latency_ms=0.0,
    )


def test_validate_cached_run_valid_subset(monkeypatch, tmp_path):
    cache_manager = _load_cache_manager_module()
    monkeypatch.setattr(cache_manager, "CACHED_RUNS_DIR", tmp_path)

    topic_id = "mpx1016"
    prescreen_ids = ["NCT001", "NCT002", "NCT003"]
    validate_data = {
        "NCT001": {"verdict": "ELIGIBLE", "mode": "two_stage", "criteria": []},
        "NCT003": {"verdict": "UNCERTAIN", "mode": "two_stage", "criteria": []},
    }

    cache_manager.save_ingest_result(topic_id, "note", {"age": "43"})
    cache_manager.save_prescreen_result(topic_id, _build_prescreen(topic_id, prescreen_ids))
    cache_manager.save_validate_results(topic_id, validate_data)
    cache_manager.save_cached_manifest(
        topic_id=topic_id,
        prescreen_trial_ids=prescreen_ids,
        validated_trial_ids=list(validate_data.keys()),
        validate_mode="two_stage",
        generated_at="2026-02-22T00:00:00+00:00",
    )

    report = cache_manager.validate_cached_run(topic_id)
    assert report.valid is True
    assert report.errors == []
    assert report.prescreen_candidate_count == 3
    assert report.validated_trial_count == 2


def test_validate_cached_run_rejects_orphan_validate_ids(monkeypatch, tmp_path):
    cache_manager = _load_cache_manager_module()
    monkeypatch.setattr(cache_manager, "CACHED_RUNS_DIR", tmp_path)

    topic_id = "mpx1016"
    prescreen_ids = ["NCT001", "NCT002"]
    validate_data = {
        "NCT001": {"verdict": "ELIGIBLE", "mode": "two_stage", "criteria": []},
        "NCT999": {"verdict": "EXCLUDED", "mode": "two_stage", "criteria": []},
    }

    cache_manager.save_ingest_result(topic_id, "note", {"age": "43"})
    cache_manager.save_prescreen_result(topic_id, _build_prescreen(topic_id, prescreen_ids))
    cache_manager.save_validate_results(topic_id, validate_data)
    cache_manager.save_cached_manifest(
        topic_id=topic_id,
        prescreen_trial_ids=prescreen_ids,
        validated_trial_ids=list(validate_data.keys()),
        validate_mode="two_stage",
    )

    report = cache_manager.validate_cached_run(topic_id)
    assert report.valid is False
    assert any("not present in prescreen_result.json" in err for err in report.errors)

    # Manifest is still generated/readable even when IDs are inconsistent.
    manifest_path = tmp_path / topic_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["topic_id"] == topic_id
