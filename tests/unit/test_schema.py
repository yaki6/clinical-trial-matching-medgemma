"""Tests for domain models."""

from trialmatch.models.schema import (
    CriterionAnnotation,
    CriterionResult,
    CriterionVerdict,
    ModelResponse,
    Phase0Sample,
    RunResult,
)


def test_criterion_verdict_values():
    assert CriterionVerdict.MET == "MET"
    assert CriterionVerdict.NOT_MET == "NOT_MET"
    assert CriterionVerdict.UNKNOWN == "UNKNOWN"


def test_criterion_annotation_creation():
    a = CriterionAnnotation(
        annotation_id=1,
        patient_id="P1",
        note="45-year-old male with lung cancer",
        trial_id="NCT001",
        trial_title="Test Trial",
        criterion_type="inclusion",
        criterion_text="Age >= 18",
        expert_label=CriterionVerdict.MET,
        expert_label_raw="included",
        expert_sentences=[0, 1],
        gpt4_label=CriterionVerdict.MET,
        gpt4_label_raw="included",
        gpt4_explanation="Patient is 45, meeting age criterion.",
        explanation_correctness="Correct",
    )
    assert a.criterion_type == "inclusion"
    assert a.expert_label == CriterionVerdict.MET


def test_model_response():
    r = ModelResponse(
        text='{"verdict": "MET"}',
        input_tokens=100,
        output_tokens=50,
        latency_ms=1200.0,
        estimated_cost=0.01,
    )
    assert r.input_tokens == 100


def test_criterion_result():
    mr = ModelResponse(text="test", input_tokens=0, output_tokens=0, latency_ms=0, estimated_cost=0)
    cr = CriterionResult(
        verdict=CriterionVerdict.MET,
        reasoning="Patient meets criterion",
        evidence_sentences=[0, 2],
        model_response=mr,
    )
    assert cr.verdict == CriterionVerdict.MET
    assert cr.evidence_sentences == [0, 2]


def test_criterion_result_without_evidence():
    """Evidence sentences are optional."""
    mr = ModelResponse(text="test", input_tokens=0, output_tokens=0, latency_ms=0, estimated_cost=0)
    cr = CriterionResult(
        verdict=CriterionVerdict.NOT_MET,
        reasoning="Patient does not meet criterion",
        model_response=mr,
    )
    assert cr.evidence_sentences == []


def test_phase0_sample():
    a = CriterionAnnotation(
        annotation_id=1,
        patient_id="P1",
        note="patient",
        trial_id="NCT1",
        trial_title="t",
        criterion_type="inclusion",
        criterion_text="c",
        expert_label=CriterionVerdict.MET,
        expert_label_raw="included",
        expert_sentences=[],
        gpt4_label=CriterionVerdict.MET,
        gpt4_label_raw="included",
        gpt4_explanation="ok",
        explanation_correctness="Correct",
    )
    sample = Phase0Sample(pairs=[a])
    assert len(sample.pairs) == 1


def test_run_result():
    rr = RunResult(run_id="test-001", model_name="medgemma", results=[], metrics={})
    assert rr.run_id == "test-001"
