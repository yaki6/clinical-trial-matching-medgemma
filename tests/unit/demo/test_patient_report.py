"""Unit tests for demo/components/patient_report.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PATIENT_REPORT_PATH = REPO_ROOT / "demo" / "components" / "patient_report.py"


def _load_patient_report_module():
    spec = importlib.util.spec_from_file_location("demo_patient_report", PATIENT_REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_patient_report_filters_to_eligible_and_uncertain_only():
    patient_report = _load_patient_report_module()
    trials = [
        {"nct_id": "NCT100", "title": "A", "verdict": "ELIGIBLE", "criteria": []},
        {"nct_id": "NCT200", "title": "B", "verdict": "UNCERTAIN", "criteria": []},
        {"nct_id": "NCT300", "title": "C", "verdict": "EXCLUDED", "criteria": []},
        {"nct_id": "NCT400", "title": "D", "verdict": "NOT_RELEVANT", "criteria": []},
    ]

    report_data = patient_report.build_patient_report_data(trials)

    assert report_data["included_trial_count"] == 2
    assert [t["nct_id"] for t in report_data["eligible_trials"]] == ["NCT100"]
    assert [t["nct_id"] for t in report_data["uncertain_trials"]] == ["NCT200"]


def test_render_markdown_contains_two_sections_with_counts():
    patient_report = _load_patient_report_module()
    trials = [
        {"nct_id": "NCT100", "title": "Eligible trial", "verdict": "ELIGIBLE", "criteria": []},
        {"nct_id": "NCT200", "title": "Uncertain trial", "verdict": "UNCERTAIN", "criteria": []},
    ]

    report_data = patient_report.build_patient_report_data(trials)
    markdown = patient_report.render_patient_report_markdown(report_data)

    assert "### Likely Matches (ELIGIBLE): 1" in markdown
    assert "### Needs More Information (UNCERTAIN): 1" in markdown


def test_derive_uncertain_gap_checklist_maps_keyword_rules():
    patient_report = _load_patient_report_module()
    criteria = [
        {"text": "ECOG 0-1 required", "verdict": "UNKNOWN"},
        {"text": "EGFR mutation must be documented", "verdict": "UNKNOWN"},
        {"text": "Requires RECIST measurable lesion by CT", "verdict": "UNKNOWN"},
    ]

    checklist = patient_report.derive_uncertain_gap_checklist(criteria)

    assert any("ECOG/Karnofsky score" in item for item in checklist)
    assert any("genomic profiling" in item for item in checklist)
    assert any("measurable disease" in item for item in checklist)


def test_derive_uncertain_gap_checklist_deduplicates_items():
    patient_report = _load_patient_report_module()
    criteria = [
        {"text": "ECOG 0-1", "verdict": "UNKNOWN"},
        {"text": "performance status must be <= 1", "verdict": "UNKNOWN"},
    ]

    checklist = patient_report.derive_uncertain_gap_checklist(criteria)

    assert len(checklist) == 1
    assert "ECOG/Karnofsky score" in checklist[0]


def test_derive_uncertain_gap_checklist_caps_to_five_items():
    patient_report = _load_patient_report_module()
    criteria = [
        {"text": "ECOG 0-1", "verdict": "UNKNOWN"},
        {"text": "EGFR mutation required", "verdict": "UNKNOWN"},
        {"text": "Stage IV metastatic disease", "verdict": "UNKNOWN"},
        {"text": "RECIST measurable lesion", "verdict": "UNKNOWN"},
        {"text": "Creatinine and bilirubin must be normal", "verdict": "UNKNOWN"},
        {"text": "Prior treatment line required", "verdict": "UNKNOWN"},
        {"text": "Brain metastases status", "verdict": "UNKNOWN"},
    ]

    checklist = patient_report.derive_uncertain_gap_checklist(criteria)
    assert len(checklist) == 5


def test_build_patient_report_replaces_raw_error_reasoning():
    patient_report = _load_patient_report_module()
    trials = [
        {
            "nct_id": "NCT200",
            "title": "Uncertain trial",
            "verdict": "UNCERTAIN",
            "criteria": [
                {
                    "text": "ECOG 0-1 required",
                    "verdict": "UNKNOWN",
                    "reasoning": (
                        "Error: Client error '400 Bad Request' for url "
                        "'https://example.com/v1/predict'"
                    ),
                }
            ],
        }
    ]

    report_data = patient_report.build_patient_report_data(trials)
    uncertain = report_data["uncertain_trials"][0]
    combined_summary = " ".join(uncertain["reasoning_summary"])

    assert "Evaluation unavailable for this criterion in this run." in combined_summary
    assert "Bad Request" not in combined_summary

    markdown = patient_report.render_patient_report_markdown(report_data)
    assert "Bad Request" not in markdown


def test_build_patient_report_handles_missing_criteria():
    patient_report = _load_patient_report_module()
    trials = [
        {"nct_id": "NCT500", "title": "Sparse trial", "verdict": "UNCERTAIN"},
    ]

    report_data = patient_report.build_patient_report_data(trials)

    assert report_data["included_trial_count"] == 1
    assert report_data["uncertain_count"] == 1
    assert report_data["uncertain_trials"][0]["gap_checklist"] == []


def test_render_markdown_has_stable_structure_and_links():
    patient_report = _load_patient_report_module()
    trials = [
        {
            "nct_id": "NCT0001",
            "title": "Stable Structure Trial",
            "phase": ["Phase 2"],
            "status": "RECRUITING",
            "verdict": "ELIGIBLE",
            "criteria": [{"text": "Age >= 18", "verdict": "MET", "reasoning": "Age is documented."}],
        }
    ]

    report_data = patient_report.build_patient_report_data(trials)
    markdown = patient_report.render_patient_report_markdown(report_data)

    assert "## Doctor Discussion Report" in markdown
    assert "[NCT0001](https://clinicaltrials.gov/study/NCT0001)" in markdown
    assert "- Why this may fit:" in markdown
