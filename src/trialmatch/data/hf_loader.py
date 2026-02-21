"""Load TrialGPT criterion annotations from HuggingFace.

Primary data source for Phase 0 and Tier A benchmarks (ADR-006).
Maps 6-class HF labels to 3-class CriterionVerdict (MET/NOT_MET/UNKNOWN).

Usage:
    # From HuggingFace (requires internet)
    annotations = load_annotations()

    # From local fixture file (for testing)
    annotations = load_annotations_from_file(path)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

from trialmatch.models.schema import CriterionAnnotation, CriterionVerdict

logger = structlog.get_logger()

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
DATASET_SPLIT = "train"

# 6-class -> 3-class label mapping (ADR-006)
LABEL_MAP: dict[str, CriterionVerdict] = {
    "included": CriterionVerdict.MET,
    "not excluded": CriterionVerdict.MET,
    "excluded": CriterionVerdict.NOT_MET,
    "not included": CriterionVerdict.NOT_MET,
    "not enough information": CriterionVerdict.UNKNOWN,
    "not applicable": CriterionVerdict.UNKNOWN,
}


def map_label(raw_label: str) -> CriterionVerdict:
    """Map a 6-class HF label to 3-class CriterionVerdict."""
    return LABEL_MAP.get(raw_label.strip().lower(), CriterionVerdict.UNKNOWN)


def parse_sentence_indices(raw: str | None) -> list[int]:
    """Parse comma-separated sentence indices from HF dataset field."""
    if not raw or not str(raw).strip():
        return []
    try:
        return [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        return []


def _row_to_annotation(row: dict) -> CriterionAnnotation:
    """Convert a single HF dataset row to CriterionAnnotation."""
    return CriterionAnnotation(
        annotation_id=row["annotation_id"],
        patient_id=str(row["patient_id"]),
        note=row["note"],
        trial_id=str(row["trial_id"]),
        trial_title=row["trial_title"],
        criterion_type=row["criterion_type"],
        criterion_text=row["criterion_text"],
        expert_label=map_label(row["expert_eligibility"]),
        expert_label_raw=row["expert_eligibility"],
        expert_sentences=parse_sentence_indices(row.get("expert_sentences")),
        gpt4_label=map_label(row["gpt4_eligibility"]),
        gpt4_label_raw=row["gpt4_eligibility"],
        gpt4_explanation=row.get("gpt4_explanation", ""),
        explanation_correctness=row.get("explanation_correctness", ""),
    )


def load_annotations() -> list[CriterionAnnotation]:
    """Load all annotations from HuggingFace dataset.

    Requires: `pip install datasets`
    Downloads ~5 MB on first call, cached thereafter.
    Rows with null criterion_text are skipped (data quality filter).
    """
    from datasets import load_dataset

    logger.info("loading_hf_dataset", dataset_id=DATASET_ID, split=DATASET_SPLIT)
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    annotations = []
    skipped = 0
    for row in ds:
        if not row.get("criterion_text"):
            skipped += 1
            continue
        annotations.append(_row_to_annotation(row))
    if skipped:
        logger.warning("skipped_null_criterion_text", count=skipped)
    logger.info("loaded_annotations", count=len(annotations))
    return annotations


def load_annotations_from_file(path: Path) -> list[CriterionAnnotation]:
    """Load annotations from a local JSON file (for testing / offline use)."""
    with open(path) as f:
        rows = json.load(f)
    annotations = []
    skipped = 0
    for row in rows:
        if not row.get("criterion_text"):
            skipped += 1
            continue
        annotations.append(_row_to_annotation(row))
    if skipped:
        logger.warning("skipped_null_criterion_text", count=skipped, source=str(path))
    return annotations
