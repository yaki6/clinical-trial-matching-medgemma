"""Criterion-level eligibility evaluator.

REUSABLE CORE: This module evaluates whether a patient meets a single
eligibility criterion. It takes raw text inputs and returns a structured
verdict. No dependency on benchmark data, HF dataset, or TrialGPT format.

Used by:
- Phase 0 benchmark (via cli/phase0.py)
- Future e2e clinical trial searching pipeline
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from trialmatch.models.schema import CriterionResult, CriterionVerdict, ModelResponse

if TYPE_CHECKING:
    from trialmatch.models.base import ModelAdapter

PROMPT_TEMPLATE = """You are a clinical trial eligibility assessment expert.

Given a patient's clinical note and a single eligibility criterion from a clinical trial,
determine whether the patient meets this criterion.

Criterion Type: {criterion_type}  (inclusion or exclusion)

Criterion:
{criterion_text}

Patient Note:
{patient_note}

Respond in JSON format:
{{
  "verdict": "MET" | "NOT_MET" | "UNKNOWN",
  "reasoning": "Step-by-step explanation citing specific evidence from the patient note",
  "evidence_sentences": "Comma-separated indices of sentences that support your verdict"
}}

Definitions:
- MET: The patient clearly satisfies this criterion based on the available information
- NOT_MET: The patient clearly does not satisfy this criterion
- UNKNOWN: There is not enough information in the patient note to determine this"""


def build_criterion_prompt(
    patient_note: str,
    criterion_text: str,
    criterion_type: str,
) -> str:
    """Build the evaluation prompt from raw text inputs.

    This is the reusable prompt builder — takes strings, not domain objects.
    """
    return PROMPT_TEMPLATE.format(
        patient_note=patient_note,
        criterion_text=criterion_text,
        criterion_type=criterion_type,
    )


def parse_criterion_verdict(raw_text: str) -> tuple[CriterionVerdict, str, list[int]]:
    """Parse model output into (verdict, reasoning, evidence_sentences).

    Tries JSON first, then markdown-wrapped JSON, then keyword extraction.
    """
    # Try direct JSON parse
    try:
        data = json.loads(raw_text)
        return (
            CriterionVerdict(data["verdict"]),
            data.get("reasoning", ""),
            _parse_evidence(data.get("evidence_sentences", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Try markdown-wrapped JSON
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return (
                CriterionVerdict(data["verdict"]),
                data.get("reasoning", ""),
                _parse_evidence(data.get("evidence_sentences", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fallback: keyword extraction using word boundaries to avoid false positives
    # (e.g., "COMMITTED" contains "MET" but is not a verdict)
    if re.search(r"\bNOT_MET\b", raw_text, re.IGNORECASE):
        return CriterionVerdict.NOT_MET, raw_text, []
    if re.search(r"\bMET\b|\bMEETS\b", raw_text, re.IGNORECASE):
        return CriterionVerdict.MET, raw_text, []

    return CriterionVerdict.UNKNOWN, raw_text, []


def _parse_evidence(raw: str | int | list | None) -> list[int]:
    """Parse evidence sentence indices from various formats."""
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if not raw or not str(raw).strip():
        return []
    try:
        return [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        return []


async def evaluate_criterion(
    patient_note: str,
    criterion_text: str,
    criterion_type: str,
    adapter: ModelAdapter,
    max_tokens: int = 2048,
) -> CriterionResult:
    """Evaluate a single criterion against a patient note.

    This is the REUSABLE ENTRY POINT. Takes raw text, returns structured result.
    No dependency on benchmark data structures.

    Args:
        patient_note: Full patient clinical note text.
        criterion_text: Single eligibility criterion text.
        criterion_type: "inclusion" or "exclusion".
        adapter: Any ModelAdapter implementation.
        max_tokens: Max tokens for model response.

    Returns:
        CriterionResult with verdict, reasoning, evidence, and model metadata.
    """
    prompt = build_criterion_prompt(
        patient_note=patient_note,
        criterion_text=criterion_text,
        criterion_type=criterion_type,
    )
    response: ModelResponse = await adapter.generate(prompt, max_tokens=max_tokens)
    verdict, reasoning, evidence = parse_criterion_verdict(response.text)

    return CriterionResult(
        verdict=verdict,
        reasoning=reasoning,
        evidence_sentences=evidence,
        model_response=response,
    )
