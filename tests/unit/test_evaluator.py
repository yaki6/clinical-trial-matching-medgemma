"""Tests for validate evaluator.

The evaluator is the REUSABLE CORE — it works with raw text inputs,
not benchmark-specific data structures.
"""

import asyncio
from unittest.mock import AsyncMock

from trialmatch.models.schema import CriterionVerdict, ModelResponse
from trialmatch.validate.evaluator import (
    build_criterion_prompt,
    evaluate_criterion,
    parse_criterion_verdict,
)

# --- Prompt building tests (reusable interface) ---


def test_build_prompt_contains_patient_note():
    prompt = build_criterion_prompt(
        patient_note="45-year-old male with NSCLC",
        criterion_text="Age >= 18",
        criterion_type="inclusion",
    )
    assert "45-year-old male with NSCLC" in prompt


def test_build_prompt_contains_criterion():
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Histologically confirmed NSCLC",
        criterion_type="inclusion",
    )
    assert "Histologically confirmed NSCLC" in prompt


def test_build_prompt_contains_criterion_type():
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="exclusion",
    )
    assert "exclusion" in prompt.lower()


def test_build_prompt_asks_for_met_not_met_unknown():
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "MET" in prompt
    assert "NOT_MET" in prompt
    assert "UNKNOWN" in prompt


def test_build_prompt_asks_for_evidence_sentences():
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "evidence_sentences" in prompt


# --- Verdict parsing tests ---


def test_parse_verdict_met():
    v, r, e = parse_criterion_verdict(
        '{"verdict": "MET", "reasoning": "meets criterion", "evidence_sentences": "0, 2"}'
    )
    assert v == CriterionVerdict.MET
    assert r == "meets criterion"
    assert e == [0, 2]


def test_parse_verdict_not_met():
    v, r, e = parse_criterion_verdict(
        '{"verdict": "NOT_MET", "reasoning": "does not meet", "evidence_sentences": ""}'
    )
    assert v == CriterionVerdict.NOT_MET
    assert e == []


def test_parse_verdict_unknown():
    v, r, e = parse_criterion_verdict('{"verdict": "UNKNOWN", "reasoning": "insufficient info"}')
    assert v == CriterionVerdict.UNKNOWN


def test_parse_verdict_markdown_wrapped():
    raw = '```json\n{"verdict": "MET", "reasoning": "ok"}\n```'
    v, r, e = parse_criterion_verdict(raw)
    assert v == CriterionVerdict.MET


def test_parse_verdict_fallback_met():
    v, r, e = parse_criterion_verdict("The patient MEETS this criterion clearly.")
    assert v == CriterionVerdict.MET


def test_parse_verdict_fallback_not_met():
    v, r, e = parse_criterion_verdict("The patient does NOT_MET this criterion.")
    assert v == CriterionVerdict.NOT_MET


def test_parse_verdict_fallback_unknown():
    v, r, e = parse_criterion_verdict("I cannot determine eligibility from the note.")
    assert v == CriterionVerdict.UNKNOWN


# --- End-to-end evaluation (reusable interface) ---


def test_evaluate_criterion_returns_result():
    """evaluate_criterion takes raw text — no benchmark data structures."""
    mock_adapter = AsyncMock()
    mock_adapter.generate.return_value = ModelResponse(
        text='{"verdict": "NOT_MET", "reasoning": "age exclusion", "evidence_sentences": "0"}',
        input_tokens=200,
        output_tokens=30,
        latency_ms=500.0,
        estimated_cost=0.01,
    )

    result = asyncio.run(
        evaluate_criterion(
            patient_note="30-year-old female with breast cancer",
            criterion_text="Age >= 40 years",
            criterion_type="inclusion",
            adapter=mock_adapter,
        )
    )
    assert result.verdict == CriterionVerdict.NOT_MET
    assert result.evidence_sentences == [0]


def test_evaluate_criterion_no_benchmark_coupling():
    """Verify evaluate_criterion doesn't import any benchmark/data modules."""
    import inspect

    from trialmatch.validate import evaluator

    source = inspect.getsource(evaluator)
    assert "from trialmatch.data" not in source
    assert "from trialmatch.cli" not in source
    assert "CriterionAnnotation" not in source
    assert "Phase0Sample" not in source
