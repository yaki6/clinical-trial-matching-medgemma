"""Tests for evaluation metrics."""

from trialmatch.evaluation.metrics import (
    TrialVerdict,
    aggregate_to_trial_verdict,
    compute_evidence_overlap,
    compute_metrics,
)
from trialmatch.models.schema import CriterionVerdict


def test_compute_metrics_perfect():
    predicted = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    actual = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    m = compute_metrics(predicted, actual)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["cohens_kappa"] == 1.0


def test_compute_metrics_all_wrong():
    predicted = [CriterionVerdict.UNKNOWN, CriterionVerdict.UNKNOWN, CriterionVerdict.UNKNOWN]
    actual = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    m = compute_metrics(predicted, actual)
    assert m["accuracy"] < 0.5


def test_compute_metrics_confusion_matrix():
    predicted = [CriterionVerdict.MET, CriterionVerdict.NOT_MET]
    actual = [CriterionVerdict.MET, CriterionVerdict.MET]
    m = compute_metrics(predicted, actual)
    assert "confusion_matrix" in m
    assert isinstance(m["confusion_matrix"], list)


def test_compute_metrics_per_class_f1():
    predicted = [CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.NOT_MET]
    actual = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.NOT_MET]
    m = compute_metrics(predicted, actual)
    assert "f1_per_class" in m
    assert "MET" in m["f1_per_class"]
    assert "NOT_MET" in m["f1_per_class"]


def test_compute_metrics_met_not_met_f1():
    """Core metric: F1 on just MET and NOT_MET classes."""
    predicted = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    actual = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    m = compute_metrics(predicted, actual)
    assert "f1_met_not_met" in m
    assert m["f1_met_not_met"] == 1.0


def test_evidence_overlap_identical():
    assert compute_evidence_overlap([0, 1, 2], [0, 1, 2]) == 1.0


def test_evidence_overlap_disjoint():
    assert compute_evidence_overlap([0, 1], [2, 3]) == 0.0


def test_evidence_overlap_partial():
    overlap = compute_evidence_overlap([0, 1, 2], [1, 2, 3])
    assert 0.0 < overlap < 1.0


def test_evidence_overlap_empty():
    assert compute_evidence_overlap([], []) == 1.0


def test_evidence_overlap_one_empty():
    assert compute_evidence_overlap([0, 1], []) == 0.0


# --- Trial-level aggregation tests ---


def test_aggregate_all_met_inclusions_all_not_met_exclusions_is_eligible():
    """All inclusions MET + all exclusions NOT_MET → ELIGIBLE."""
    results = [
        (CriterionVerdict.MET, "inclusion"),
        (CriterionVerdict.MET, "inclusion"),
        (CriterionVerdict.NOT_MET, "exclusion"),
    ]
    assert aggregate_to_trial_verdict(results) == TrialVerdict.ELIGIBLE


def test_aggregate_any_exclusion_met_is_excluded():
    """All inclusions MET but any exclusion MET → EXCLUDED."""
    results = [
        (CriterionVerdict.MET, "inclusion"),
        (CriterionVerdict.MET, "exclusion"),
    ]
    assert aggregate_to_trial_verdict(results) == TrialVerdict.EXCLUDED


def test_aggregate_any_inclusion_not_met_is_not_relevant():
    """Any inclusion NOT_MET → NOT_RELEVANT (definitive ineligibility)."""
    results = [
        (CriterionVerdict.NOT_MET, "inclusion"),
        (CriterionVerdict.MET, "inclusion"),
        (CriterionVerdict.NOT_MET, "exclusion"),
    ]
    assert aggregate_to_trial_verdict(results) == TrialVerdict.NOT_RELEVANT


def test_aggregate_unknown_inclusion_is_uncertain():
    """UNKNOWN inclusion → UNCERTAIN (cannot confirm eligibility)."""
    results = [
        (CriterionVerdict.MET, "inclusion"),
        (CriterionVerdict.UNKNOWN, "inclusion"),
        (CriterionVerdict.NOT_MET, "exclusion"),
    ]
    assert aggregate_to_trial_verdict(results) == TrialVerdict.UNCERTAIN


def test_aggregate_unknown_exclusion_with_met_inclusions_is_uncertain():
    """UNKNOWN exclusion (with all inclusions MET) → UNCERTAIN (cannot rule out exclusion)."""
    results = [
        (CriterionVerdict.MET, "inclusion"),
        (CriterionVerdict.UNKNOWN, "exclusion"),
    ]
    assert aggregate_to_trial_verdict(results) == TrialVerdict.UNCERTAIN


def test_aggregate_empty_is_uncertain():
    """Empty criterion list → UNCERTAIN."""
    assert aggregate_to_trial_verdict([]) == TrialVerdict.UNCERTAIN


def test_aggregate_not_relevant_takes_priority_over_excluded():
    """NOT_MET inclusion takes priority over MET exclusion."""
    results = [
        (CriterionVerdict.NOT_MET, "inclusion"),
        (CriterionVerdict.MET, "exclusion"),
    ]
    assert aggregate_to_trial_verdict(results) == TrialVerdict.NOT_RELEVANT
