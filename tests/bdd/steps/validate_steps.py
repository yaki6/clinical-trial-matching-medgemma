"""BDD step definitions for criterion evaluation."""

import inspect

import pytest
from pytest_bdd import given, parsers, scenario, then, when

from trialmatch.models.schema import CriterionVerdict
from trialmatch.validate.evaluator import build_criterion_prompt, parse_criterion_verdict


@scenario(
    "../../../features/validate/criterion_evaluation.feature",
    "Prompt contains patient note and criterion",
)
def test_prompt_contents():
    pass


@scenario(
    "../../../features/validate/criterion_evaluation.feature",
    "Parse MET verdict from JSON",
)
def test_parse_met():
    pass


@scenario(
    "../../../features/validate/criterion_evaluation.feature",
    "Parse NOT_MET verdict from JSON",
)
def test_parse_not_met():
    pass


@scenario(
    "../../../features/validate/criterion_evaluation.feature",
    "Evaluator is reusable (no benchmark coupling)",
)
def test_no_coupling():
    pass


# --- Shared context ---


@pytest.fixture()
def context():
    return {}


# --- Prompt building steps ---


@given(parsers.parse('a patient note "{note}"'), target_fixture="patient_note")
def given_patient_note(note):
    return note


@given(parsers.parse('an inclusion criterion "{criterion}"'), target_fixture="criterion_text")
def given_criterion(criterion):
    return criterion


@when("I build the criterion evaluation prompt", target_fixture="prompt")
def build_prompt(patient_note, criterion_text):
    return build_criterion_prompt(
        patient_note=patient_note,
        criterion_text=criterion_text,
        criterion_type="inclusion",
    )


@then("the prompt contains the patient note")
def prompt_has_note(prompt, patient_note):
    assert patient_note in prompt


@then("the prompt contains the criterion text")
def prompt_has_criterion(prompt, criterion_text):
    assert criterion_text in prompt


@then("the prompt asks for eligible, not eligible, or unknown")
def prompt_has_verdicts(prompt):
    prompt_lower = prompt.lower()
    assert '"label": "<eligible|not eligible|unknown>"' in prompt_lower


# --- Verdict parsing steps ---


@given(parsers.parse("the model returns '{raw_text}'"), target_fixture="model_output")
def given_model_output(raw_text):
    return raw_text


@when("I parse the criterion verdict", target_fixture="parsed")
def parse_verdict(model_output):
    v, r, e = parse_criterion_verdict(model_output)
    return {"verdict": v, "reasoning": r, "evidence": e}


@then("the verdict is MET")
def verdict_is_met(parsed):
    assert parsed["verdict"] == CriterionVerdict.MET


@then("the verdict is NOT_MET")
def verdict_is_not_met(parsed):
    assert parsed["verdict"] == CriterionVerdict.NOT_MET


@then(parsers.parse('the reasoning contains "{text}"'))
def reasoning_contains(parsed, text):
    assert text in parsed["reasoning"]


@then(parsers.parse("evidence sentences include {idx:d}"))
def evidence_includes(parsed, idx):
    assert idx in parsed["evidence"]


@then("evidence sentences are empty")
def evidence_empty(parsed):
    assert parsed["evidence"] == []


# --- Coupling check steps ---


@given("the evaluate_criterion function", target_fixture="evaluator_source")
def given_evaluator():
    from trialmatch.validate import evaluator

    return inspect.getsource(evaluator)


@then("it does not import from trialmatch.data")
def no_data_import(evaluator_source):
    assert "from trialmatch.data" not in evaluator_source


@then("it does not import from trialmatch.cli")
def no_cli_import(evaluator_source):
    assert "from trialmatch.cli" not in evaluator_source
