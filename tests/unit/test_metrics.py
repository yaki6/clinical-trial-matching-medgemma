"""Tests for evaluation metrics."""

from trialmatch.evaluation.metrics import compute_evidence_overlap, compute_metrics
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
