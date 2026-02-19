"""Tests for stratified sampler."""

from pathlib import Path

from trialmatch.data.hf_loader import load_annotations_from_file
from trialmatch.data.sampler import stratified_sample

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
