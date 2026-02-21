"""Tests for profile_adapter: nsclc_trial_profiles.json -> PRESCREEN input."""

from __future__ import annotations

import pytest

from trialmatch.ingest.profile_adapter import adapt_profile_for_prescreen, load_profiles
from trialmatch.prescreen.agent import _format_key_facts

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_profile(key_facts: list | None = None, text: str = "Patient note") -> dict:
    """Build a minimal profile dict for testing."""
    p: dict = {"profile_text": text}
    if key_facts is not None:
        p["key_facts"] = key_facts
    return p


# ---------------------------------------------------------------------------
# adapt_profile_for_prescreen
# ---------------------------------------------------------------------------


def test_flat_string_value():
    """String value passes through unchanged."""
    profile = _make_profile(
        key_facts=[
            {
                "field": "primary_diagnosis",
                "value": "Lung adenocarcinoma",
                "evidence_span": None,
                "required": True,
                "notes": None,
            }
        ]
    )
    _note, facts = adapt_profile_for_prescreen(profile)
    assert facts["primary_diagnosis"] == "Lung adenocarcinoma"


def test_nested_dict_flattened():
    """Nested dict (demographics) is flattened to 'key: val; key: val' string."""
    profile = _make_profile(
        key_facts=[
            {
                "field": "demographics",
                "value": {"age": "43", "sex": "female"},
                "evidence_span": None,
                "required": True,
                "notes": None,
            }
        ]
    )
    _note, facts = adapt_profile_for_prescreen(profile)
    assert facts["demographics"] == "age: 43; sex: female"


def test_nested_dict_with_list():
    """Nested dict with a list sub-value joins items with commas."""
    profile = _make_profile(
        key_facts=[
            {
                "field": "imaging_findings",
                "value": {
                    "description": "CT shows mass",
                    "findings": ["mass in left lobe", "pleural effusion"],
                },
                "evidence_span": None,
                "required": True,
                "notes": None,
            }
        ]
    )
    _note, facts = adapt_profile_for_prescreen(profile)
    result = facts["imaging_findings"]
    assert "description: CT shows mass" in result
    assert "findings: mass in left lobe, pleural effusion" in result


def test_nested_dict_skips_none_values():
    """None values inside nested dicts are omitted."""
    profile = _make_profile(
        key_facts=[
            {
                "field": "histopathology",
                "value": {"cell_type": "signet-ring", "grade": None},
                "evidence_span": None,
                "required": False,
                "notes": None,
            }
        ]
    )
    _note, facts = adapt_profile_for_prescreen(profile)
    assert "grade" not in facts["histopathology"]
    assert "cell_type: signet-ring" in facts["histopathology"]


def test_list_value_passthrough():
    """List value passes through as a list (not stringified)."""
    profile = _make_profile(
        key_facts=[
            {
                "field": "key_findings",
                "value": ["finding A", "finding B"],
                "evidence_span": None,
                "required": True,
                "notes": None,
            }
        ]
    )
    _note, facts = adapt_profile_for_prescreen(profile)
    assert facts["key_findings"] == ["finding A", "finding B"]


def test_empty_key_facts():
    """Empty key_facts list returns empty dict."""
    profile = _make_profile(key_facts=[])
    _note, facts = adapt_profile_for_prescreen(profile)
    assert facts == {}


def test_missing_key_facts():
    """Profile without 'key_facts' key returns empty dict."""
    profile = {"profile_text": "Some note"}
    _note, facts = adapt_profile_for_prescreen(profile)
    assert facts == {}


def test_patient_note_returned():
    """Patient note text is returned as-is."""
    profile = _make_profile(key_facts=[], text="EHR clinical note text")
    note, _facts = adapt_profile_for_prescreen(profile)
    assert note == "EHR clinical note text"


# ---------------------------------------------------------------------------
# load_profiles
# ---------------------------------------------------------------------------


def test_load_profiles_count():
    """load_profiles returns all 37 profiles from nsclc_trial_profiles.json."""
    profiles = load_profiles()
    assert len(profiles) == 37


def test_load_profiles_structure():
    """Each profile has required keys."""
    profiles = load_profiles()
    first = profiles[0]
    assert "topic_id" in first
    assert "profile_text" in first
    assert "key_facts" in first


# ---------------------------------------------------------------------------
# Roundtrip: adapter output -> _format_key_facts
# ---------------------------------------------------------------------------


def test_roundtrip_with_format_key_facts():
    """Adapted output works with agent's _format_key_facts() without errors."""
    profile = _make_profile(
        key_facts=[
            {
                "field": "primary_diagnosis",
                "value": "Lung adenocarcinoma",
                "evidence_span": None,
                "required": True,
                "notes": None,
            },
            {
                "field": "demographics",
                "value": {"age": "43", "sex": "female"},
                "evidence_span": None,
                "required": True,
                "notes": None,
            },
            {
                "field": "key_findings",
                "value": ["dyspnea", "wheezing"],
                "evidence_span": None,
                "required": True,
                "notes": None,
            },
        ]
    )
    _note, facts = adapt_profile_for_prescreen(profile)
    result = _format_key_facts(facts)

    assert "primary_diagnosis: Lung adenocarcinoma" in result
    assert "demographics: age: 43; sex: female" in result
    assert "key_findings: dyspnea, wheezing" in result


def test_roundtrip_empty_facts():
    """Empty adapted output produces the expected fallback string."""
    profile = _make_profile(key_facts=[])
    _note, facts = adapt_profile_for_prescreen(profile)
    result = _format_key_facts(facts)
    assert "No structured key facts" in result


@pytest.mark.parametrize(
    "value_type",
    ["string", "bool", "int"],
    ids=["string_val", "bool_val", "int_val"],
)
def test_roundtrip_scalar_types(value_type: str):
    """Scalar types (bool, int, string) pass through and format cleanly."""
    values = {"string": "never smoker", "bool": True, "int": 3}
    profile = _make_profile(
        key_facts=[
            {
                "field": "test_field",
                "value": values[value_type],
                "evidence_span": None,
                "required": False,
                "notes": None,
            }
        ]
    )
    _note, facts = adapt_profile_for_prescreen(profile)
    result = _format_key_facts(facts)
    assert "test_field" in result
