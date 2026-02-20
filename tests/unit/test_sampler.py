"""Tests for stratified sampler."""

from pathlib import Path

from trialmatch.data.hf_loader import load_annotations_from_file
from trialmatch.data.sampler import filter_by_keywords, stratified_sample

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_stratified_sample_counts():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    sample = stratified_sample(annotations, n_pairs=3, seed=42)
    assert len(sample.pairs) == 3


def test_stratified_sample_deterministic():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    s1 = stratified_sample(annotations, n_pairs=3, seed=42)
    s2 = stratified_sample(annotations, n_pairs=3, seed=42)
    ids1 = [a.annotation_id for a in s1.pairs]
    ids2 = [a.annotation_id for a in s2.pairs]
    assert ids1 == ids2


def test_stratified_sample_covers_labels():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    sample = stratified_sample(annotations, n_pairs=3, seed=42)
    labels = {a.expert_label for a in sample.pairs}
    assert len(labels) >= 2


def test_stratified_sample_respects_n():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    sample = stratified_sample(annotations, n_pairs=2, seed=42)
    assert len(sample.pairs) == 2


# --- filter_by_keywords tests ---


def test_filter_by_keywords_matches_case_insensitively():
    """Keywords match regardless of case."""
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    # Fixture notes contain "non-small cell lung cancer" — match with uppercase
    result = filter_by_keywords(annotations, ["NSCLC", "NON-SMALL CELL"])
    # At least some annotations should match (fixture has NSCLC patient)
    assert len(result) > 0
    # All returned annotations must contain at least one keyword (case-insensitive)
    import re
    for a in result:
        matches = any(
            re.search(re.escape(kw), a.note, re.IGNORECASE) for kw in ["NSCLC", "NON-SMALL CELL"]
        )
        assert matches


def test_filter_by_keywords_returns_empty_when_no_matches():
    """Returns empty list when no annotations match keywords."""
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    result = filter_by_keywords(annotations, ["XYZZY_NONEXISTENT_CONDITION_12345"])
    assert result == []


def test_filter_by_keywords_returns_all_when_no_keywords():
    """Returns all annotations when keywords list is empty."""
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    result = filter_by_keywords(annotations, [])
    assert len(result) == len(annotations)
