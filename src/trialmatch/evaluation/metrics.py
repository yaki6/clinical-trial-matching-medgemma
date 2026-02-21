"""Evaluation metrics for criterion-level benchmark."""

from __future__ import annotations

import enum
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from trialmatch.models.schema import CriterionVerdict


class TrialVerdict(enum.StrEnum):
    ELIGIBLE = "ELIGIBLE"          # inclusion MET + all exclusions NOT_MET → qrel 2
    EXCLUDED = "EXCLUDED"          # inclusion MET + any exclusion MET → qrel 1
    NOT_RELEVANT = "NOT_RELEVANT"  # any inclusion NOT_MET → qrel 0
    UNCERTAIN = "UNCERTAIN"        # UNKNOWN in critical path → qrel 0 (flagged)


def aggregate_to_trial_verdict(
    criterion_results: list[tuple[CriterionVerdict, str]],
) -> TrialVerdict:
    """Aggregate criterion-level verdicts to trial-level per sot-annotation-requirements.

    Args:
        criterion_results: List of (verdict, criterion_type) tuples for one (patient, trial).
            criterion_type is "inclusion" or "exclusion".

    Priority: NOT_RELEVANT > EXCLUDED > UNCERTAIN > ELIGIBLE.
    """
    inclusion_verdicts = [v for v, t in criterion_results if t == "inclusion"]
    exclusion_verdicts = [v for v, t in criterion_results if t == "exclusion"]

    # Definitive ineligibility: any inclusion fails
    if CriterionVerdict.NOT_MET in inclusion_verdicts:
        return TrialVerdict.NOT_RELEVANT

    # Definitive exclusion: any exclusion criterion met
    if CriterionVerdict.MET in exclusion_verdicts:
        return TrialVerdict.EXCLUDED

    # Cannot confirm eligibility: unknown inclusions
    if CriterionVerdict.UNKNOWN in inclusion_verdicts:
        return TrialVerdict.UNCERTAIN

    # Cannot rule out exclusion: unknown exclusions
    if CriterionVerdict.UNKNOWN in exclusion_verdicts:
        return TrialVerdict.UNCERTAIN

    # All inclusions MET + all exclusions NOT_MET
    if inclusion_verdicts and all(v == CriterionVerdict.MET for v in inclusion_verdicts):
        return TrialVerdict.ELIGIBLE

    return TrialVerdict.UNCERTAIN

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
        "f1_per_class": {name: float(f1) for name, f1 in zip(LABEL_NAMES, f1_per, strict=True)},
    }


def compute_stratified_metrics(
    predicted: list[CriterionVerdict],
    actual: list[CriterionVerdict],
    criterion_types: list[str],
) -> dict[str, Any]:
    """Compute metrics stratified by criterion type (inclusion vs exclusion).

    Returns overall metrics plus per-type breakdown and calibration stats.
    """
    overall = compute_metrics(predicted, actual)

    # Per-type metrics
    by_type: dict[str, Any] = {}
    for ctype in ("inclusion", "exclusion"):
        mask = [t == ctype for t in criterion_types]
        pred_sub = [p for p, m in zip(predicted, mask, strict=True) if m]
        actual_sub = [a for a, m in zip(actual, mask, strict=True) if m]
        if pred_sub:
            by_type[ctype] = compute_metrics(pred_sub, actual_sub)
        else:
            by_type[ctype] = {"accuracy": None, "note": "no pairs of this type"}

    overall["by_criterion_type"] = by_type

    # Calibration: MET bias and UNKNOWN rate difference
    n = len(predicted) if predicted else 1
    pred_met_rate = sum(1 for p in predicted if p == CriterionVerdict.MET) / n
    actual_met_rate = sum(1 for a in actual if a == CriterionVerdict.MET) / n
    pred_unk_rate = sum(1 for p in predicted if p == CriterionVerdict.UNKNOWN) / n
    actual_unk_rate = sum(1 for a in actual if a == CriterionVerdict.UNKNOWN) / n

    overall["calibration"] = {
        "met_bias": float(pred_met_rate - actual_met_rate),
        "unknown_rate_diff": float(pred_unk_rate - actual_unk_rate),
    }

    return overall


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
