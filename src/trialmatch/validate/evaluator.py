"""Criterion-level eligibility evaluator.

REUSABLE CORE: This module evaluates whether a patient meets a single
eligibility criterion. It takes raw text inputs and returns a structured
verdict. No dependency on benchmark data, HF dataset, or TrialGPT format.

Used by:
- Phase 0 benchmark (via cli/phase0.py)
- Future e2e clinical trial searching pipeline
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING

import structlog

from trialmatch.models.schema import CriterionResult, CriterionVerdict, ModelResponse

if TYPE_CHECKING:
    from trialmatch.models.base import ModelAdapter

logger = structlog.get_logger()

INCLUSION_INSTRUCTIONS = (
    'For this INCLUSION criterion, respond with one of: "included", "not included", '
    'or "not enough information".\n'
    "- included: the patient meets this inclusion requirement based on the note.\n"
    "- not included: the patient does NOT meet this inclusion requirement.\n"
    '- not enough information: cannot determine from the note.'
)

EXCLUSION_INSTRUCTIONS = (
    'For this EXCLUSION criterion, respond with one of: "excluded", "not excluded", '
    'or "not enough information".\n'
    "- excluded: the patient HAS this characteristic and would be excluded from the trial.\n"
    "- not excluded: the patient does NOT have this characteristic.\n"
    '- not enough information: cannot determine from the note.'
)

# Maps TrialGPT-native labels to internal CriterionVerdict.
# Mirrors hf_loader.LABEL_MAP so model output and ground-truth use the same semantics.
NATIVE_LABEL_TO_VERDICT: dict[str, CriterionVerdict] = {
    "included": CriterionVerdict.MET,
    "not excluded": CriterionVerdict.MET,
    "excluded": CriterionVerdict.NOT_MET,
    "not included": CriterionVerdict.NOT_MET,
    "not enough information": CriterionVerdict.UNKNOWN,
    "not applicable": CriterionVerdict.UNKNOWN,
}

PROMPT_TEMPLATE = """You are a clinical trial eligibility assessment expert.

Given a patient's clinical note and a single eligibility criterion, determine the
patient's eligibility status for this criterion.

Important: if the patient note does not mention a medically important fact, you can
assume that the fact is not true for the patient (e.g., if the note does not mention
any allergies, assume the patient has no known allergies).

Criterion Type: {criterion_type}
{criterion_type_instructions}

Criterion:
{criterion_text}

Patient Note:
{patient_note}

Think step by step:
1. What does this criterion specifically require?
2. What does the patient note state about this? Consider both explicit statements and reasonable inferences from the absence of information.
3. Based on the evidence, determine the label.

Respond ONLY with valid JSON (no prose before or after):
{{
  "label": "<your label>",
  "reasoning": "Step-by-step explanation citing specific sentence indices",
  "evidence_sentences": [0, 1, 2]
}}

evidence_sentences: JSON array of integer sentence indices from the patient note (e.g. [0, 2])."""


def build_criterion_prompt(
    patient_note: str,
    criterion_text: str,
    criterion_type: str,
) -> str:
    """Build the evaluation prompt from raw text inputs.

    This is the reusable prompt builder — takes strings, not domain objects.
    Selects inclusion or exclusion instructions based on criterion_type.
    """
    instructions = (
        INCLUSION_INSTRUCTIONS if criterion_type == "inclusion" else EXCLUSION_INSTRUCTIONS
    )
    return PROMPT_TEMPLATE.format(
        patient_note=patient_note,
        criterion_text=criterion_text,
        criterion_type=criterion_type,
        criterion_type_instructions=instructions,
    )


def clean_model_response(raw_text: str) -> str:
    """Strip model-specific artifacts before JSON parsing.

    Handles MedGemma prompt echo, thinking tokens, and chat turn markers.
    Applied universally — clean responses (Gemini) pass through unchanged.
    """
    # 1. Strip everything before the last <start_of_turn>model marker
    if "<start_of_turn>model" in raw_text:
        raw_text = raw_text.rsplit("<start_of_turn>model", 1)[1]

    # 2. Strip <unused\d+> thinking token blocks (MedGemma internal monologue)
    #    These appear as <unused94>thought ... <unused95> before the actual JSON
    raw_text = re.sub(r"<unused\d+>.*?<unused\d+>", "", raw_text, flags=re.DOTALL)
    # Also strip any remaining isolated <unused\d+> tokens
    raw_text = re.sub(r"<unused\d+>", "", raw_text)
    # Strip bare "thought" prefix (MedGemma without unused token wrapper)
    raw_text = re.sub(r"^\s*thought\s+", "", raw_text, flags=re.IGNORECASE | re.MULTILINE)

    # 3. Strip chat turn end markers
    raw_text = raw_text.replace("<end_of_turn>", "")

    return raw_text.strip()


def _extract_verdict_from_json(data: dict) -> tuple[CriterionVerdict, str, list[int]]:
    """Extract verdict from parsed JSON dict, trying 'label' then 'verdict' keys."""
    reasoning = data.get("reasoning", "")
    evidence = _parse_evidence(data.get("evidence_sentences", []))

    # Prefer "label" key (TrialGPT-native format)
    if "label" in data:
        label = data["label"].strip().lower()
        if label in NATIVE_LABEL_TO_VERDICT:
            return NATIVE_LABEL_TO_VERDICT[label], reasoning, evidence

    # Fallback: "verdict" key (legacy MET/NOT_MET/UNKNOWN format)
    if "verdict" in data:
        return CriterionVerdict(data["verdict"]), reasoning, evidence

    raise KeyError("No 'label' or 'verdict' key found")


def parse_criterion_verdict(raw_text: str) -> tuple[CriterionVerdict, str, list[int]]:
    """Parse model output into (verdict, reasoning, evidence_sentences).

    Tries JSON first (with 'label' then 'verdict' key), then markdown-wrapped
    JSON, then keyword extraction for native labels, then MET/NOT_MET keywords.
    """
    cleaned = clean_model_response(raw_text)

    # Try direct JSON parse
    try:
        data = json.loads(cleaned)
        return _extract_verdict_from_json(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Try markdown-wrapped JSON
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return _extract_verdict_from_json(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Try extracting just the JSON object (find outermost braces)
    brace_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            return _extract_verdict_from_json(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fallback: keyword extraction for native TrialGPT labels
    cleaned_lower = cleaned.lower()
    if "not included" in cleaned_lower:
        return CriterionVerdict.NOT_MET, cleaned, []
    if "not excluded" in cleaned_lower:
        return CriterionVerdict.MET, cleaned, []
    if "excluded" in cleaned_lower:
        return CriterionVerdict.NOT_MET, cleaned, []
    if "included" in cleaned_lower:
        return CriterionVerdict.MET, cleaned, []
    if "not enough information" in cleaned_lower:
        return CriterionVerdict.UNKNOWN, cleaned, []

    # Last resort: legacy MET/NOT_MET keyword extraction
    if re.search(r"\bNOT_MET\b", cleaned, re.IGNORECASE):
        return CriterionVerdict.NOT_MET, cleaned, []
    if re.search(r"\bMET\b|\bMEETS\b", cleaned, re.IGNORECASE):
        return CriterionVerdict.MET, cleaned, []

    return CriterionVerdict.UNKNOWN, cleaned, []


def _parse_evidence(raw: str | int | list | None) -> list[int]:
    """Parse evidence sentence indices from various formats."""
    if isinstance(raw, list):
        try:
            return [int(x) for x in raw if str(x).strip().isdigit() or isinstance(x, int)]
        except (ValueError, TypeError):
            return []
    if not raw or not str(raw).strip():
        return []
    try:
        return [int(x.strip()) for x in str(raw).split(",") if x.strip().isdigit()]
    except ValueError:
        return []


async def evaluate_criterion(
    patient_note: str,
    criterion_text: str,
    criterion_type: str,
    adapter: ModelAdapter,
    max_tokens: int = 1024,
    timeout_seconds: float = 300.0,
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
        timeout_seconds: Per-call timeout. Returns UNKNOWN on timeout.
    """
    prompt = build_criterion_prompt(
        patient_note=patient_note,
        criterion_text=criterion_text,
        criterion_type=criterion_type,
    )

    call_start = time.perf_counter()
    try:
        response: ModelResponse = await asyncio.wait_for(
            adapter.generate(prompt, max_tokens=max_tokens),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - call_start) * 1000
        logger.warning(
            "evaluate_criterion_timeout",
            adapter=adapter.name,
            timeout_seconds=timeout_seconds,
            elapsed_ms=elapsed_ms,
        )
        return CriterionResult(
            verdict=CriterionVerdict.UNKNOWN,
            reasoning=f"Timeout: model did not respond within {timeout_seconds}s",
            evidence_sentences=[],
            model_response=ModelResponse(
                text="TIMEOUT",
                input_tokens=len(prompt) // 4,
                output_tokens=0,
                latency_ms=elapsed_ms,
                estimated_cost=0.0,
                token_count_estimated=True,
            ),
        )

    verdict, reasoning, evidence = parse_criterion_verdict(response.text)

    return CriterionResult(
        verdict=verdict,
        reasoning=reasoning,
        evidence_sentences=evidence,
        model_response=response,
    )
