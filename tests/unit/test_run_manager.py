"""Tests for run manager."""

import json

from trialmatch.models.schema import CriterionResult, CriterionVerdict, ModelResponse, RunResult
from trialmatch.tracing.run_manager import RunManager


def test_generate_run_id():
    mgr = RunManager()
    run_id = mgr.generate_run_id("medgemma")
    assert "phase0-medgemma-" in run_id


def test_save_run(tmp_path):
    mgr = RunManager(runs_dir=tmp_path)

    mr = ModelResponse(
        text="t", input_tokens=100, output_tokens=20, latency_ms=500, estimated_cost=0.01
    )
    result = CriterionResult(
        verdict=CriterionVerdict.MET,
        reasoning="meets",
        evidence_sentences=[0],
        model_response=mr,
    )
    run_result = RunResult(
        run_id="test-run-001",
        model_name="medgemma",
        results=[result],
        metrics={"accuracy": 0.85},
    )

    run_dir = mgr.save_run(run_result, config={"n_pairs": 20})

    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "cost_summary.json").exists()

    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert metrics["accuracy"] == 0.85

    with open(run_dir / "cost_summary.json") as f:
        cost = json.load(f)
    assert cost["total_pairs"] == 1
    assert cost["total_cost_usd"] == 0.01
