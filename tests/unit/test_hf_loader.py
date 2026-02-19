"""Tests for HuggingFace TrialGPT data loader."""

from pathlib import Path

from trialmatch.data.hf_loader import (
    load_annotations_from_file,
    map_label,
    parse_sentence_indices,
)
from trialmatch.models.schema import CriterionAnnotation, CriterionVerdict

FIXTURES = Path(__file__).parent.parent / "fixtures"


# --- Label mapping tests ---


def test_map_label_included():
    assert map_label("included") == CriterionVerdict.MET


def test_map_label_not_excluded():
    assert map_label("not excluded") == CriterionVerdict.MET


def test_map_label_excluded():
    assert map_label("excluded") == CriterionVerdict.NOT_MET


def test_map_label_not_included():
    assert map_label("not included") == CriterionVerdict.NOT_MET


def test_map_label_not_enough_info():
    assert map_label("not enough information") == CriterionVerdict.UNKNOWN


def test_map_label_not_applicable():
    assert map_label("not applicable") == CriterionVerdict.UNKNOWN


def test_map_label_unknown_fallback():
    assert map_label("something_weird") == CriterionVerdict.UNKNOWN


# --- Sentence parsing tests ---


def test_parse_sentence_indices_valid():
    assert parse_sentence_indices("0, 1, 3") == [0, 1, 3]


def test_parse_sentence_indices_single():
    assert parse_sentence_indices("0") == [0]


def test_parse_sentence_indices_empty():
    assert parse_sentence_indices("") == []


def test_parse_sentence_indices_none():
    assert parse_sentence_indices(None) == []


# --- Loading from fixture file ---


def test_load_annotations_from_file():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    assert len(annotations) == 4
    assert all(isinstance(a, CriterionAnnotation) for a in annotations)


def test_load_annotations_label_mapping():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    assert annotations[0].expert_label == CriterionVerdict.MET
    assert annotations[1].expert_label == CriterionVerdict.MET
    assert annotations[2].expert_label == CriterionVerdict.NOT_MET
    assert annotations[3].expert_label == CriterionVerdict.UNKNOWN


def test_load_annotations_preserves_raw_labels():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    assert annotations[0].expert_label_raw == "included"
    assert annotations[2].expert_label_raw == "excluded"


def test_load_annotations_parses_sentences():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    assert annotations[0].expert_sentences == [0]
    assert annotations[1].expert_sentences == []
