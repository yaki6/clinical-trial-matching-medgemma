"""Tests for validate evaluator.

The evaluator is the REUSABLE CORE — it works with raw text inputs,
not benchmark-specific data structures.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from trialmatch.models.schema import CriterionVerdict, ModelResponse
from trialmatch.validate.evaluator import (
    EXCLUSION_INSTRUCTIONS,
    INCLUSION_INSTRUCTIONS,
    build_criterion_prompt,
    clean_model_response,
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


def test_build_prompt_inclusion_has_inclusion_instructions():
    """Inclusion prompt must contain inclusion-specific instructions."""
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Age >= 18",
        criterion_type="inclusion",
    )
    assert INCLUSION_INSTRUCTIONS in prompt
    assert EXCLUSION_INSTRUCTIONS not in prompt


def test_build_prompt_exclusion_has_exclusion_instructions():
    """Exclusion prompt must contain exclusion-specific instructions."""
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Active respiratory distress",
        criterion_type="exclusion",
    )
    assert EXCLUSION_INSTRUCTIONS in prompt
    assert INCLUSION_INSTRUCTIONS not in prompt


def test_build_prompt_inclusion_exclusion_instructions_differ():
    """Inclusion and exclusion prompts must be different."""
    inclusion_prompt = build_criterion_prompt(
        patient_note="note",
        criterion_text="criterion",
        criterion_type="inclusion",
    )
    exclusion_prompt = build_criterion_prompt(
        patient_note="note",
        criterion_text="criterion",
        criterion_type="exclusion",
    )
    assert inclusion_prompt != exclusion_prompt


# --- Response cleaning tests ---


def test_clean_model_response_passthrough():
    """Clean JSON passes through unchanged."""
    raw = '{"verdict": "MET", "reasoning": "ok", "evidence_sentences": [0]}'
    assert clean_model_response(raw) == raw


def test_clean_model_response_strips_start_of_turn():
    """MedGemma prompt echo is stripped."""
    raw = "<start_of_turn>user\nsome prompt<end_of_turn>\n<start_of_turn>model\n{\"verdict\": \"MET\"}"
    cleaned = clean_model_response(raw)
    assert "<start_of_turn>" not in cleaned
    assert '{"verdict": "MET"}' in cleaned


def test_clean_model_response_strips_unused_tokens():
    """MedGemma thinking tokens are stripped."""
    raw = "<unused94>thought\nsome thinking here<unused95>\n{\"verdict\": \"NOT_MET\"}"
    cleaned = clean_model_response(raw)
    assert "<unused94>" not in cleaned
    assert "<unused95>" not in cleaned
    assert '{"verdict": "NOT_MET"}' in cleaned


def test_clean_model_response_strips_end_of_turn():
    raw = '{"verdict": "MET"}<end_of_turn>'
    cleaned = clean_model_response(raw)
    assert "<end_of_turn>" not in cleaned


def test_clean_model_response_strips_bare_thought_prefix():
    """Bare 'thought' prefix without unused token wrappers is stripped (pair #16 bug)."""
    raw = 'thought The user wants me to evaluate this...\n{"verdict": "NOT_MET", "reasoning": "ok"}'
    cleaned = clean_model_response(raw)
    assert not cleaned.startswith("thought")
    assert '{"verdict": "NOT_MET"' in cleaned


def test_clean_model_response_strips_bare_thought_case_insensitive():
    """Bare 'Thought' (capital T) prefix is also stripped."""
    raw = 'Thought some internal monologue\n{"verdict": "MET"}'
    cleaned = clean_model_response(raw)
    assert not cleaned.lower().startswith("thought")


# --- Verdict parsing tests ---


def test_parse_verdict_met():
    v, r, e = parse_criterion_verdict(
        '{"verdict": "MET", "reasoning": "meets criterion", "evidence_sentences": "0, 2"}'
    )
    assert v == CriterionVerdict.MET
    assert r == "meets criterion"
    assert e == [0, 2]


def test_parse_verdict_met_array_evidence():
    """evidence_sentences as JSON array of ints."""
    v, r, e = parse_criterion_verdict(
        '{"verdict": "MET", "reasoning": "ok", "evidence_sentences": [0, 2]}'
    )
    assert v == CriterionVerdict.MET
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


def test_parse_verdict_medgemma_prompt_echo():
    """MedGemma response with full prompt echo is parsed correctly."""
    raw = (
        "<start_of_turn>user\nYou are a clinical expert...<end_of_turn>\n"
        "<start_of_turn>model\n"
        '```json\n{"verdict": "NOT_MET", "reasoning": "patient is 25", "evidence_sentences": [0]}\n```'
    )
    v, r, e = parse_criterion_verdict(raw)
    assert v == CriterionVerdict.NOT_MET
    assert e == [0]


def test_parse_verdict_fallback_met():
    v, r, e = parse_criterion_verdict("The patient MEETS this criterion clearly.")
    assert v == CriterionVerdict.MET


def test_parse_verdict_fallback_not_met():
    v, r, e = parse_criterion_verdict("The patient does NOT_MET this criterion.")
    assert v == CriterionVerdict.NOT_MET


def test_parse_verdict_fallback_unknown():
    v, r, e = parse_criterion_verdict("I cannot determine eligibility from the note.")
    assert v == CriterionVerdict.UNKNOWN


def test_parse_verdict_no_false_positive_committed():
    """Words like COMMITTED should not trigger MET verdict."""
    v, r, e = parse_criterion_verdict("The patient is committed to treatment but data is unclear.")
    assert v == CriterionVerdict.UNKNOWN


def test_parse_verdict_no_false_positive_submitted():
    """Words like SUBMITTED should not trigger MET verdict."""
    v, r, e = parse_criterion_verdict("The lab results were submitted but inconclusive.")
    assert v == CriterionVerdict.UNKNOWN


# --- End-to-end evaluation (reusable interface) ---


def test_evaluate_criterion_returns_result():
    """evaluate_criterion takes raw text — no benchmark data structures."""
    mock_adapter = AsyncMock()
    mock_adapter.name = "test-model"
    mock_adapter.generate.return_value = ModelResponse(
        text='{"verdict": "NOT_MET", "reasoning": "age exclusion", "evidence_sentences": [0]}',
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


def test_evaluate_criterion_timeout():
    """Timeout returns UNKNOWN without crashing."""
    import asyncio as _asyncio

    mock_adapter = AsyncMock()
    mock_adapter.name = "slow-model"
    mock_adapter.generate.side_effect = _asyncio.TimeoutError()

    result = asyncio.run(
        evaluate_criterion(
            patient_note="Patient note",
            criterion_text="Some criterion",
            criterion_type="inclusion",
            adapter=mock_adapter,
            timeout_seconds=0.001,
        )
    )
    assert result.verdict == CriterionVerdict.UNKNOWN
    assert "timeout" in result.reasoning.lower()


def test_evaluate_criterion_no_benchmark_coupling():
    """Verify evaluate_criterion doesn't import any benchmark/data modules."""
    import inspect

    from trialmatch.validate import evaluator

    source = inspect.getsource(evaluator)
    assert "from trialmatch.data" not in source
    assert "from trialmatch.cli" not in source
    assert "CriterionAnnotation" not in source
    assert "Phase0Sample" not in source
