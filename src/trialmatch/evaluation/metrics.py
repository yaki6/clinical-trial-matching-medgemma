"""Evaluation metrics for criterion-level benchmark."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from trialmatch.models.schema import CriterionVerdict

LABELS = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
LABEL_NAMES = [v.value for v in LABELS]


def compute_metrics(
    predicted: list[CriterionVerdict],
    actual: list[CriterionVerdict],
) -> dict[str, Any]:
    """Compute all benchmark evaluation metrics."""
    pred_str = [v.value for v in predicted]
    actual_str = [v.value for v in actual]

    acc = accuracy_score(actual_str, pred_str)
    f1_mac = f1_score(actual_str, pred_str, labels=LABEL_NAMES, average="macro", zero_division=0)
    kappa = cohen_kappa_score(actual_str, pred_str, labels=LABEL_NAMES)
    cm = confusion_matrix(actual_str, pred_str, labels=LABEL_NAMES)
    f1_per = f1_score(actual_str, pred_str, labels=LABEL_NAMES, average=None, zero_division=0)

    # Core metric: F1 on MET + NOT_MET only (excluding UNKNOWN)
    met_nm_labels = [CriterionVerdict.MET.value, CriterionVerdict.NOT_MET.value]
    f1_met_nm = f1_score(
        actual_str, pred_str, labels=met_nm_labels, average="macro", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_met_not_met": float(f1_met_nm),
        "cohens_kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
        "f1_per_class": {name: float(f1) for name, f1 in zip(LABEL_NAMES, f1_per)},
    }


def compute_evidence_overlap(
    predicted_sentences: list[int],
    expert_sentences: list[int],
) -> float:
    """Compute Jaccard similarity between predicted and expert evidence sentences."""
    pred_set = set(predicted_sentences)
    expert_set = set(expert_sentences)

    if not pred_set and not expert_set:
        return 1.0

    if not pred_set or not expert_set:
        return 0.0

    intersection = pred_set & expert_set
    union = pred_set | expert_set
    return len(intersection) / len(union)
