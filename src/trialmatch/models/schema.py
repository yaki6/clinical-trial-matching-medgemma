"""Domain models for trialmatch benchmark.

These models serve both the Phase 0 benchmark (TrialGPT HF data)
and the future e2e clinical trial matching pipeline.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class CriterionVerdict(enum.StrEnum):
    """Criterion-level eligibility verdict.

    Used by both the benchmark evaluator and the e2e pipeline.
    """

    MET = "MET"
    NOT_MET = "NOT_MET"
    UNKNOWN = "UNKNOWN"


class CriterionAnnotation(BaseModel):
    """A single criterion annotation from the TrialGPT HF dataset.

    Each row represents one (patient, criterion) pair with expert
    and GPT-4 labels.
    """

    annotation_id: int
    patient_id: str
    note: str
    trial_id: str
    trial_title: str
    criterion_type: str
    criterion_text: str
    expert_label: CriterionVerdict
    expert_label_raw: str
    expert_sentences: list[int]
    gpt4_label: CriterionVerdict
    gpt4_label_raw: str
    gpt4_explanation: str
    explanation_correctness: str


class ModelResponse(BaseModel):
    """Raw model API response metadata."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost: float
    token_count_estimated: bool = False


class CriterionResult(BaseModel):
    """Result of evaluating a single criterion against a patient.

    This is the output of evaluate_criterion() — used by both
    the benchmark and the e2e pipeline.
    """

    verdict: CriterionVerdict
    reasoning: str
    evidence_sentences: list[int] = Field(default_factory=list)
    model_response: ModelResponse


class Phase0Sample(BaseModel):
    """Stratified sample of criterion annotations for Phase 0."""

    pairs: list[CriterionAnnotation]


class RunResult(BaseModel):
    """Complete results for one model's benchmark run."""

    run_id: str
    model_name: str
    results: list[CriterionResult]
    metrics: dict[str, Any]
