"""Tests for validate evaluator.

The evaluator is the REUSABLE CORE — it works with raw text inputs,
not benchmark-specific data structures.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from trialmatch.models.schema import CriterionVerdict, ModelResponse
from trialmatch.validate.evaluator import (
    ELIGIBLE_LABEL_TO_VERDICT,
    EXCLUSION_INSTRUCTIONS,
    INCLUSION_INSTRUCTIONS,
    NATIVE_LABEL_TO_VERDICT,
    build_criterion_prompt,
    build_labeling_prompt,
    build_reasoning_prompt,
    clean_model_response,
    evaluate_criterion,
    evaluate_criterion_two_stage,
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


def test_build_prompt_asks_for_native_labels_inclusion():
    """Inclusion prompt uses simplified eligibility labels: eligible / not eligible."""
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "eligible" in prompt
    assert "not eligible" in prompt
    assert "unknown" in prompt


def test_build_prompt_asks_for_native_labels_exclusion():
    """Exclusion prompt uses simplified eligibility labels: eligible / not eligible."""
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="exclusion",
    )
    assert "eligible" in prompt
    assert "not eligible" in prompt
    assert "unknown" in prompt


def test_build_prompt_uses_label_key_not_verdict():
    """Prompt asks model to output 'label' key, not 'verdict'."""
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert '"label"' in prompt


def test_build_prompt_asks_for_evidence_sentences():
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "evidence_sentences" in prompt


def test_build_prompt_contains_cwa_instruction():
    """Prompt contains Closed World Assumption instruction."""
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "does not mention" in prompt
    assert "assume" in prompt


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


# --- Verdict parsing tests (native label format) ---


def test_parse_native_label_included():
    """JSON with 'label': 'included' → MET."""
    v, r, e = parse_criterion_verdict(
        '{"label": "included", "reasoning": "meets criterion", "evidence_sentences": [0, 2]}'
    )
    assert v == CriterionVerdict.MET
    assert r == "meets criterion"
    assert e == [0, 2]


def test_parse_native_label_not_included():
    """JSON with 'label': 'not included' → NOT_MET."""
    v, r, e = parse_criterion_verdict(
        '{"label": "not included", "reasoning": "does not meet", "evidence_sentences": []}'
    )
    assert v == CriterionVerdict.NOT_MET
    assert e == []


def test_parse_native_label_excluded():
    """JSON with 'label': 'excluded' → NOT_MET."""
    v, r, e = parse_criterion_verdict(
        '{"label": "excluded", "reasoning": "patient has condition", "evidence_sentences": [1]}'
    )
    assert v == CriterionVerdict.NOT_MET
    assert e == [1]


def test_parse_native_label_not_excluded():
    """JSON with 'label': 'not excluded' → MET."""
    v, r, e = parse_criterion_verdict(
        '{"label": "not excluded", "reasoning": "no evidence", "evidence_sentences": []}'
    )
    assert v == CriterionVerdict.MET


def test_parse_native_label_not_enough_info():
    """JSON with 'label': 'not enough information' → UNKNOWN."""
    v, r, e = parse_criterion_verdict(
        '{"label": "not enough information", "reasoning": "insufficient info"}'
    )
    assert v == CriterionVerdict.UNKNOWN


def test_parse_native_label_case_insensitive():
    """Native labels are case-insensitive."""
    v, r, e = parse_criterion_verdict(
        '{"label": "Excluded", "reasoning": "ok"}'
    )
    assert v == CriterionVerdict.NOT_MET


def test_native_label_to_verdict_matches_hf_loader():
    """NATIVE_LABEL_TO_VERDICT must mirror hf_loader.LABEL_MAP exactly."""
    from trialmatch.data.hf_loader import LABEL_MAP

    for label, verdict in LABEL_MAP.items():
        assert NATIVE_LABEL_TO_VERDICT[label] == verdict, (
            f"Mismatch for '{label}': evaluator={NATIVE_LABEL_TO_VERDICT[label]}, "
            f"hf_loader={verdict}"
        )


# --- Legacy verdict format (backwards compat) ---


def test_parse_verdict_legacy_met():
    """Legacy JSON with 'verdict' key still works."""
    v, r, e = parse_criterion_verdict(
        '{"verdict": "MET", "reasoning": "meets criterion", "evidence_sentences": "0, 2"}'
    )
    assert v == CriterionVerdict.MET
    assert r == "meets criterion"
    assert e == [0, 2]


def test_parse_verdict_legacy_not_met():
    v, r, e = parse_criterion_verdict(
        '{"verdict": "NOT_MET", "reasoning": "does not meet", "evidence_sentences": ""}'
    )
    assert v == CriterionVerdict.NOT_MET
    assert e == []


def test_parse_verdict_legacy_unknown():
    v, r, e = parse_criterion_verdict('{"verdict": "UNKNOWN", "reasoning": "insufficient info"}')
    assert v == CriterionVerdict.UNKNOWN


def test_parse_verdict_label_takes_precedence_over_verdict():
    """When both 'label' and 'verdict' are present, 'label' wins."""
    v, r, e = parse_criterion_verdict(
        '{"label": "excluded", "verdict": "MET", "reasoning": "label should win"}'
    )
    assert v == CriterionVerdict.NOT_MET


def test_parse_verdict_markdown_wrapped():
    raw = '```json\n{"label": "eligible", "reasoning": "ok"}\n```'
    v, r, e = parse_criterion_verdict(raw)
    assert v == CriterionVerdict.MET


def test_parse_verdict_medgemma_prompt_echo():
    """MedGemma response with full prompt echo is parsed correctly."""
    raw = (
        "<start_of_turn>user\nYou are a clinical expert...<end_of_turn>\n"
        "<start_of_turn>model\n"
        '```json\n{"label": "eligible", "reasoning": "patient is 25",'
        ' "evidence_sentences": [0]}\n```'
    )
    v, r, e = parse_criterion_verdict(raw)
    assert v == CriterionVerdict.MET
    assert e == [0]


# --- Keyword fallback tests ---


def test_parse_verdict_fallback_native_included():
    v, r, e = parse_criterion_verdict("The patient is clearly included in this criterion.")
    assert v == CriterionVerdict.MET


def test_parse_verdict_fallback_native_excluded():
    v, r, e = parse_criterion_verdict("The patient should be excluded from this trial.")
    assert v == CriterionVerdict.NOT_MET


def test_parse_verdict_fallback_native_not_excluded():
    v, r, e = parse_criterion_verdict("The patient is not excluded by this criterion.")
    assert v == CriterionVerdict.MET


def test_parse_verdict_fallback_native_not_included():
    v, r, e = parse_criterion_verdict("The patient is not included based on available data.")
    assert v == CriterionVerdict.NOT_MET


def test_parse_verdict_fallback_legacy_met():
    v, r, e = parse_criterion_verdict("The patient MEETS this criterion clearly.")
    assert v == CriterionVerdict.MET


def test_parse_verdict_fallback_legacy_not_met():
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
        text='{"label": "not eligible", "reasoning": "age exclusion", "evidence_sentences": [0]}',
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


# --- Simplified eligibility label tests ---


def test_parse_eligible_label():
    """JSON with 'label': 'eligible' → MET."""
    v, r, e = parse_criterion_verdict(
        '{"label": "eligible", "reasoning": "patient qualifies", "evidence_sentences": [1]}'
    )
    assert v == CriterionVerdict.MET
    assert r == "patient qualifies"
    assert e == [1]


def test_parse_not_eligible_label():
    """JSON with 'label': 'not eligible' → NOT_MET."""
    v, r, e = parse_criterion_verdict(
        '{"label": "not eligible", "reasoning": "does not qualify", "evidence_sentences": [0]}'
    )
    assert v == CriterionVerdict.NOT_MET
    assert r == "does not qualify"
    assert e == [0]


def test_parse_unknown_label():
    """JSON with 'label': 'unknown' → UNKNOWN."""
    v, r, e = parse_criterion_verdict(
        '{"label": "unknown", "reasoning": "insufficient data", "evidence_sentences": []}'
    )
    assert v == CriterionVerdict.UNKNOWN
    assert r == "insufficient data"


def test_parse_eligible_label_case_insensitive():
    """Eligibility labels are case-insensitive."""
    v, _, _ = parse_criterion_verdict('{"label": "Eligible", "reasoning": "ok"}')
    assert v == CriterionVerdict.MET
    v2, _, _ = parse_criterion_verdict('{"label": "Not Eligible", "reasoning": "ok"}')
    assert v2 == CriterionVerdict.NOT_MET
    v3, _, _ = parse_criterion_verdict('{"label": "UNKNOWN", "reasoning": "ok"}')
    assert v3 == CriterionVerdict.UNKNOWN


def test_parse_legacy_native_labels_still_work():
    """Old TrialGPT-native labels still parse correctly (backward compat)."""
    v1, _, _ = parse_criterion_verdict('{"label": "included", "reasoning": "ok"}')
    assert v1 == CriterionVerdict.MET
    v2, _, _ = parse_criterion_verdict('{"label": "not included", "reasoning": "ok"}')
    assert v2 == CriterionVerdict.NOT_MET
    v3, _, _ = parse_criterion_verdict('{"label": "excluded", "reasoning": "ok"}')
    assert v3 == CriterionVerdict.NOT_MET
    v4, _, _ = parse_criterion_verdict('{"label": "not excluded", "reasoning": "ok"}')
    assert v4 == CriterionVerdict.MET
    v5, _, _ = parse_criterion_verdict('{"label": "not enough information", "reasoning": "ok"}')
    assert v5 == CriterionVerdict.UNKNOWN


def test_parse_verdict_fallback_eligible_keyword():
    """Keyword fallback: 'eligible' in text → MET."""
    v, _, _ = parse_criterion_verdict("The patient is eligible for this criterion.")
    assert v == CriterionVerdict.MET


def test_parse_verdict_fallback_not_eligible_keyword():
    """Keyword fallback: 'not eligible' in text → NOT_MET."""
    v, _, _ = parse_criterion_verdict("The patient is not eligible for this criterion.")
    assert v == CriterionVerdict.NOT_MET


def test_eligible_label_to_verdict_complete():
    """ELIGIBLE_LABEL_TO_VERDICT covers all 3 verdict classes."""
    verdicts = set(ELIGIBLE_LABEL_TO_VERDICT.values())
    assert CriterionVerdict.MET in verdicts
    assert CriterionVerdict.NOT_MET in verdicts
    assert CriterionVerdict.UNKNOWN in verdicts
    assert len(ELIGIBLE_LABEL_TO_VERDICT) == 3


def test_build_prompt_exclusion_explains_eligibility_mapping():
    """Exclusion prompt explains that having the condition means NOT eligible."""
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Active hepatitis B",
        criterion_type="exclusion",
    )
    assert "HAS" in prompt
    assert "NOT eligible" in prompt or "not eligible" in prompt.lower()


def test_build_prompt_contains_consistency_instruction():
    """Prompt contains the label-reasoning consistency instruction."""
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "CRITICAL" in prompt
    assert "consistent" in prompt.lower()


def test_build_prompt_exclusion_cwa_strengthened():
    """Exclusion prompt contains strengthened CWA instruction about unmentioned conditions."""
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Active hepatitis B",
        criterion_type="exclusion",
    )
    assert "does not mention the excluded condition" in prompt


# --- Two-stage prompt tests ---


def test_build_reasoning_prompt_contains_patient_note():
    """Stage 1 prompt includes patient note."""
    prompt = build_reasoning_prompt(
        patient_note="45-year-old with NSCLC",
        criterion_text="Age >= 18",
        criterion_type="inclusion",
    )
    assert "45-year-old with NSCLC" in prompt


def test_build_reasoning_prompt_contains_criterion():
    """Stage 1 prompt includes criterion text."""
    prompt = build_reasoning_prompt(
        patient_note="Patient note",
        criterion_text="Histologically confirmed NSCLC",
        criterion_type="inclusion",
    )
    assert "Histologically confirmed NSCLC" in prompt


def test_build_reasoning_prompt_asks_factual_question():
    """Stage 1 asks factual YES/NO/INSUFFICIENT DATA, not eligibility labels."""
    prompt = build_reasoning_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="exclusion",
    )
    assert "YES" in prompt
    assert "NO" in prompt
    assert "INSUFFICIENT DATA" in prompt
    # Should NOT ask for eligible/not eligible JSON
    assert '"label"' not in prompt


def test_build_reasoning_prompt_requests_plain_text():
    """Stage 1 requests plain text output, not JSON."""
    prompt = build_reasoning_prompt(
        patient_note="Note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "plain text" in prompt.lower()
    assert "no JSON" in prompt


def test_build_reasoning_prompt_contains_cwa():
    """Stage 1 includes closed world assumption instruction."""
    prompt = build_reasoning_prompt(
        patient_note="Note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "does not mention" in prompt
    assert "assume" in prompt


def test_build_reasoning_prompt_contains_cwa_exception():
    """Stage 1 CWA has exceptions for procedural/safety requirements."""
    prompt = build_reasoning_prompt(
        patient_note="Note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "EXCEPTION" in prompt
    assert "Procedural" in prompt or "procedural" in prompt.lower()
    assert "INSUFFICIENT DATA" in prompt
    assert "contraception" in prompt.lower()


def test_build_reasoning_prompt_contains_severity_subquestion():
    """Stage 1 asks severity/specificity sub-question (Q4) after general condition."""
    prompt = build_reasoning_prompt(
        patient_note="Note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "GENERAL condition" in prompt
    assert "SPECIFIC requirements" in prompt
    assert "MATCHES" in prompt
    assert "DOES NOT MATCH" in prompt
    # Severity examples
    assert "mild" in prompt.lower()
    assert "severe" in prompt.lower()


def test_build_labeling_prompt_contains_reasoning():
    """Stage 2 prompt includes Stage 1 reasoning text."""
    prompt = build_labeling_prompt(
        stage1_reasoning="The patient has dementia based on sentences 1, 2.",
        criterion_text="Diagnosis of Dementia",
        criterion_type="exclusion",
    )
    assert "The patient has dementia" in prompt


def test_build_labeling_prompt_contains_criterion_type_instructions():
    """Stage 2 includes criterion type instructions (inclusion or exclusion)."""
    excl_prompt = build_labeling_prompt(
        stage1_reasoning="Analysis text",
        criterion_text="Active hepatitis",
        criterion_type="exclusion",
    )
    assert EXCLUSION_INSTRUCTIONS in excl_prompt

    incl_prompt = build_labeling_prompt(
        stage1_reasoning="Analysis text",
        criterion_text="Age >= 18",
        criterion_type="inclusion",
    )
    assert INCLUSION_INSTRUCTIONS in incl_prompt


def test_build_labeling_prompt_requests_json():
    """Stage 2 requests JSON output with label key."""
    prompt = build_labeling_prompt(
        stage1_reasoning="Analysis",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert '"label"' in prompt
    assert "JSON" in prompt


def test_build_labeling_prompt_contains_mapping_rules():
    """Stage 2 prompt has explicit label mapping rules for inclusion and exclusion."""
    prompt = build_labeling_prompt(
        stage1_reasoning="Analysis",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "LABEL MAPPING RULES" in prompt
    assert "INCLUSION criteria" in prompt
    assert "EXCLUSION criteria" in prompt
    assert "MATCHES" in prompt
    assert "DOES NOT MATCH" in prompt


def test_build_labeling_prompt_contains_contradiction_check():
    """Stage 2 prompt has contradiction detection instruction."""
    prompt = build_labeling_prompt(
        stage1_reasoning="Analysis",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "CONTRADICTION CHECK" in prompt
    assert "REASONING CONTENT" in prompt


def test_build_labeling_prompt_contains_abstention_guardrail():
    """Stage 2 prompt prevents over-abstention to unknown on clear negatives."""
    prompt = build_labeling_prompt(
        stage1_reasoning="Analysis",
        criterion_text="Criterion",
        criterion_type="exclusion",
    )
    assert "Do NOT downgrade" in prompt
    assert "HIGH confidence" in prompt


# --- Two-stage evaluate function tests ---


def test_evaluate_criterion_two_stage_returns_result():
    """Two-stage evaluation produces correct verdict from both stages."""
    mock_reasoning = AsyncMock()
    mock_reasoning.name = "medgemma-27b"
    mock_reasoning.generate.return_value = ModelResponse(
        text=(
            "The patient is 25 years old. The criterion requires "
            "age >= 18. YES, the patient satisfies this."
        ),
        input_tokens=200,
        output_tokens=50,
        latency_ms=500.0,
        estimated_cost=0.005,
    )

    mock_labeling = AsyncMock()
    mock_labeling.name = "gemini-3-pro"
    mock_labeling.generate.return_value = ModelResponse(
        text=(
            '{"label": "eligible", "reasoning": "patient is 25, '
            'meets age requirement", "evidence_sentences": [0]}'
        ),
        input_tokens=100,
        output_tokens=30,
        latency_ms=200.0,
        estimated_cost=0.001,
    )

    result = asyncio.run(
        evaluate_criterion_two_stage(
            patient_note="25-year-old male",
            criterion_text="Age >= 18",
            criterion_type="inclusion",
            reasoning_adapter=mock_reasoning,
            labeling_adapter=mock_labeling,
        )
    )

    assert result.verdict == CriterionVerdict.MET
    assert result.evidence_sentences == [0]
    assert result.stage1_reasoning is not None
    assert "25 years old" in result.stage1_reasoning
    assert result.stage1_response is not None
    assert result.stage1_response.estimated_cost == 0.005


def test_evaluate_criterion_two_stage_exclusion_inversion_fix():
    """Two-stage fixes exclusion inversion.

    MedGemma identifies condition, Gemini labels correctly.
    """
    # MedGemma correctly identifies the patient has dementia
    mock_reasoning = AsyncMock()
    mock_reasoning.name = "medgemma-27b"
    mock_reasoning.generate.return_value = ModelResponse(
        text=(
            "The patient has progressive memory loss and severe "
            "cognitive deficits. YES, the patient has dementia."
        ),
        input_tokens=200,
        output_tokens=50,
        latency_ms=500.0,
        estimated_cost=0.005,
    )

    # Gemini maps: exclusion + patient HAS condition → not eligible
    mock_labeling = AsyncMock()
    mock_labeling.name = "gemini-3-pro"
    mock_labeling.generate.return_value = ModelResponse(
        text=(
            '{"label": "not eligible", '
            '"reasoning": "patient has dementia, excluded", '
            '"evidence_sentences": [1, 2]}'
        ),
        input_tokens=100,
        output_tokens=30,
        latency_ms=200.0,
        estimated_cost=0.001,
    )

    result = asyncio.run(
        evaluate_criterion_two_stage(
            patient_note="62-year-old with progressive memory loss and cognitive deficits",
            criterion_text="Diagnosis of Dementia",
            criterion_type="exclusion",
            reasoning_adapter=mock_reasoning,
            labeling_adapter=mock_labeling,
        )
    )

    assert result.verdict == CriterionVerdict.NOT_MET


def test_evaluate_criterion_two_stage_reasoning_timeout():
    """Stage 1 timeout returns UNKNOWN with no stage1_reasoning."""
    mock_reasoning = AsyncMock()
    mock_reasoning.name = "slow-model"
    mock_reasoning.generate.side_effect = TimeoutError()

    mock_labeling = AsyncMock()
    mock_labeling.name = "gemini"

    result = asyncio.run(
        evaluate_criterion_two_stage(
            patient_note="Note",
            criterion_text="Criterion",
            criterion_type="inclusion",
            reasoning_adapter=mock_reasoning,
            labeling_adapter=mock_labeling,
            timeout_seconds=0.001,
        )
    )

    assert result.verdict == CriterionVerdict.UNKNOWN
    assert "Stage 1 timeout" in result.reasoning
    # Labeling adapter should NOT have been called
    mock_labeling.generate.assert_not_called()


def test_evaluate_criterion_two_stage_labeling_timeout():
    """Stage 2 timeout returns UNKNOWN but preserves stage1_reasoning."""
    mock_reasoning = AsyncMock()
    mock_reasoning.name = "medgemma"
    mock_reasoning.generate.return_value = ModelResponse(
        text="The patient meets the criterion. YES.",
        input_tokens=100,
        output_tokens=20,
        latency_ms=300.0,
        estimated_cost=0.002,
    )

    mock_labeling = AsyncMock()
    mock_labeling.name = "gemini"
    mock_labeling.generate.side_effect = TimeoutError()

    result = asyncio.run(
        evaluate_criterion_two_stage(
            patient_note="Note",
            criterion_text="Criterion",
            criterion_type="inclusion",
            reasoning_adapter=mock_reasoning,
            labeling_adapter=mock_labeling,
            timeout_seconds=0.001,
        )
    )

    assert result.verdict == CriterionVerdict.UNKNOWN
    assert "Stage 2 timeout" in result.reasoning
    assert result.stage1_reasoning is not None
    assert result.stage1_response is not None


def test_criterion_result_backward_compatible():
    """CriterionResult without two-stage fields works as before."""
    from trialmatch.models.schema import CriterionResult

    result = CriterionResult(
        verdict=CriterionVerdict.MET,
        reasoning="ok",
        model_response=ModelResponse(
            text="ok",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100.0,
            estimated_cost=0.001,
        ),
    )
    assert result.stage1_reasoning is None
    assert result.stage1_response is None
