"""Tests for run manager."""

import json

import pytest

from trialmatch.models.schema import (
    CriterionAnnotation,
    CriterionResult,
    CriterionVerdict,
    ModelResponse,
    RunResult,
)
from trialmatch.tracing.run_manager import RunManager


def _make_result(verdict=CriterionVerdict.MET, cost=0.01):
    mr = ModelResponse(
        text="t", input_tokens=100, output_tokens=20, latency_ms=500, estimated_cost=cost
    )
    return CriterionResult(
        verdict=verdict,
        reasoning="meets",
        evidence_sentences=[0],
        model_response=mr,
    )


def _make_annotation(patient_id="pat-001", trial_id="NCT001", expert=CriterionVerdict.MET):
    return CriterionAnnotation(
        annotation_id=1,
        patient_id=patient_id,
        note="Patient note",
        trial_id=trial_id,
        trial_title="Test Trial",
        criterion_type="inclusion",
        criterion_text="Age >= 18",
        expert_label=expert,
        expert_label_raw="included",
        expert_sentences=[0],
        gpt4_label=CriterionVerdict.MET,
        gpt4_label_raw="included",
        gpt4_explanation="Patient is 30.",
        explanation_correctness="correct",
    )


def test_generate_run_id():
    mgr = RunManager()
    run_id = mgr.generate_run_id("medgemma")
    assert "phase0-medgemma-" in run_id


def test_save_run(tmp_path):
    mgr = RunManager(runs_dir=tmp_path)
    run_result = RunResult(
        run_id="test-run-001",
        model_name="medgemma",
        results=[_make_result()],
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


def test_save_run_enriches_results_with_annotations(tmp_path):
    """results.json contains identity fields when annotations are passed."""
    mgr = RunManager(runs_dir=tmp_path)
    ann = _make_annotation(patient_id="sigir-001", trial_id="NCT999")
    result = _make_result(verdict=CriterionVerdict.MET)
    run_result = RunResult(
        run_id="test-enriched",
        model_name="gemini",
        results=[result],
        metrics={},
    )

    run_dir = mgr.save_run(run_result, annotations=[ann])

    with open(run_dir / "results.json") as f:
        rows = json.load(f)

    row = rows[0]
    assert row["patient_id"] == "sigir-001"
    assert row["trial_id"] == "NCT999"
    assert row["criterion_text"] == "Age >= 18"
    assert row["expert_label"] == "MET"
    assert row["gpt4_label"] == "MET"
    assert row["model_verdict"] == "MET"
    assert row["correct"] is True


def test_save_run_correct_false_when_wrong(tmp_path):
    """correct=False when model verdict differs from expert label."""
    mgr = RunManager(runs_dir=tmp_path)
    ann = _make_annotation(expert=CriterionVerdict.NOT_MET)
    result = _make_result(verdict=CriterionVerdict.MET)  # wrong prediction
    run_result = RunResult(
        run_id="test-wrong",
        model_name="gemini",
        results=[result],
        metrics={},
    )

    run_dir = mgr.save_run(run_result, annotations=[ann])

    with open(run_dir / "results.json") as f:
        rows = json.load(f)
    assert rows[0]["correct"] is False


def test_save_run_generates_audit_table(tmp_path):
    """audit_table.md is generated when annotations are provided."""
    mgr = RunManager(runs_dir=tmp_path)
    ann = _make_annotation()
    run_result = RunResult(
        run_id="test-audit",
        model_name="gemini",
        results=[_make_result()],
        metrics={},
    )

    run_dir = mgr.save_run(run_result, annotations=[ann])

    assert (run_dir / "audit_table.md").exists()
    content = (run_dir / "audit_table.md").read_text()
    assert "pat-001" in content
    assert "NCT001" in content
    assert "✓" in content


def test_save_run_no_audit_table_without_annotations(tmp_path):
    """audit_table.md is NOT generated when no annotations provided."""
    mgr = RunManager(runs_dir=tmp_path)
    run_result = RunResult(
        run_id="test-no-audit",
        model_name="gemini",
        results=[_make_result()],
        metrics={},
    )

    run_dir = mgr.save_run(run_result)
    assert not (run_dir / "audit_table.md").exists()
