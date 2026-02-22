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
    'For this INCLUSION criterion, determine if the patient is eligible.\n'
    '- "eligible": the patient satisfies this inclusion requirement based on the note.\n'
    '- "not eligible": the patient does NOT satisfy this inclusion requirement.\n'
    '- "unknown": cannot determine from the note.'
)

EXCLUSION_INSTRUCTIONS = (
    'For this EXCLUSION criterion, determine if the patient is eligible.\n'
    'A patient who HAS the excluded condition is NOT eligible.\n'
    'A patient who does NOT have the excluded condition IS eligible.\n'
    'If the note does not mention the excluded condition, '
    'the patient does NOT have it → patient IS eligible.\n'
    '- "eligible": the patient does NOT have this exclusion condition and can participate.\n'
    '- "not eligible": the patient HAS this exclusion condition and cannot participate.\n'
    '- "unknown": cannot determine from the note.'
)

# Maps simplified eligibility labels to CriterionVerdict.
# These are the primary labels used in prompts (eligible / not eligible / unknown).
ELIGIBLE_LABEL_TO_VERDICT: dict[str, CriterionVerdict] = {
    "eligible": CriterionVerdict.MET,
    "not eligible": CriterionVerdict.NOT_MET,
    "unknown": CriterionVerdict.UNKNOWN,
}

# Maps TrialGPT-native labels to internal CriterionVerdict (backward-compatible fallback).
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
3. Based on the evidence, is the patient eligible or not eligible for this criterion?

CRITICAL: Your JSON "label" field MUST be consistent with your reasoning.
If your reasoning concludes the patient is not eligible, your label MUST be "not eligible".

Respond ONLY with valid JSON (no prose before or after):
{{
  "label": "<eligible|not eligible|unknown>",
  "reasoning": "Step-by-step explanation citing specific sentence indices",
  "evidence_sentences": [0, 1, 2]
}}

evidence_sentences: JSON array of integer sentence indices from the patient note (e.g. [0, 2])."""


# --- Two-stage prompts ---

STAGE1_REASONING_PROMPT = """You are a medical expert analyzing a patient's clinical note.

Closed World Assumption (CWA):
CWA APPLIES to: medical conditions, diagnoses, symptoms, allergies, and
disease history. If the note does not mention a medical condition, assume
the patient does not have it (e.g., no mention of diabetes = no diabetes).

CWA DOES NOT APPLY to:
- Actions, behaviors, or compliance (contraception use, willingness to
  follow up, substance abstinence, diet adherence)
- Test results, lab values, or diagnostic findings not in this note
- Ongoing treatments, concomitant medications, or procedures
- Prior surgeries or procedures (silence ≠ never happened)
- Lifestyle or social history (smoking, alcohol, occupation)
For any of these, if not documented, answer INSUFFICIENT DATA.

Criterion Type: {criterion_type}
Criterion: {criterion_text}

Patient Note:
{patient_note}

Analyze this criterion against the patient note:

1. What does this criterion specifically require?
   Note any severity, staging, timing, or specificity requirements.
   If negated (e.g., "No history of X"), identify the UNDERLYING
   CONDITION being checked (e.g., "history of X"). Q3 should be
   about the underlying condition, not the negation.

2. What does the patient note explicitly state about this?
   Cite specific sentences by index.

3. Does the patient have the GENERAL condition described?
   - If the criterion requires a formal DIAGNOSIS (e.g., "Diagnosis of
     dementia"): only answer YES if the note documents a diagnosis, not
     merely symptoms suggestive of the condition.
   - If the criterion describes symptoms or clinical presentation:
     symptoms suffice.
   - Consider whether the patient's symptoms could have a more likely
     alternative explanation before concluding they match the criterion.
   Answer: YES / NO / INSUFFICIENT DATA

3b. Does the criterion specify a particular SEVERITY, GRADE, or STAGE
    (e.g., mild, moderate, severe, early, advanced, Rutherford stage 2)?
    If so, does the patient's documented condition match that specific
    level?
    Answer: SEVERITY MATCHES / SEVERITY DOES NOT MATCH / NO SEVERITY SPECIFIED

4. If YES to #3: Does the criterion have additional SPECIFIC qualifiers
   beyond the general condition?
   - If yes, does the patient match them?
     (subtype, timing/recency, threshold, modality)
     Answer: MATCHES / DOES NOT MATCH / INSUFFICIENT DATA
   - If the criterion has no specific qualifiers: Answer: N/A
   If NO to #3, skip this question.

5. How confident are you? HIGH / MEDIUM / LOW

You MUST use the exact answer keywords specified for each question.

Respond in plain text (no JSON). Focus on clinical accuracy."""


STAGE2_LABELING_PROMPT = """You are a clinical trial eligibility label assignment system.

A medical AI analyzed a patient's clinical note against an eligibility criterion.
Your task: assign the correct eligibility label based on the medical analysis.

Criterion Type: {criterion_type}
{criterion_type_instructions}

Criterion: {criterion_text}

Medical Analysis:
{stage1_reasoning}

LABEL MAPPING RULES:

For INCLUSION criteria:
- Analysis says YES + MATCHES specifics (or N/A) → "eligible"
- Analysis says YES + DOES NOT MATCH specifics → "not eligible"
- Analysis says YES + SEVERITY DOES NOT MATCH → "not eligible"
- Analysis says YES + INSUFFICIENT DATA on specifics → "unknown"
- Analysis says NO → "not eligible"
- Analysis says INSUFFICIENT DATA → "unknown"

For EXCLUSION criteria:
- Analysis says YES (patient HAS the excluded condition) + MATCHES → "not eligible"
- Analysis says YES + DOES NOT MATCH specifics → "eligible"
- Analysis says YES + SEVERITY DOES NOT MATCH → "eligible"
- Analysis says NO (patient does NOT have it) → "eligible"
- Analysis says INSUFFICIENT DATA → "unknown"

IMPORTANT RULES:
1. When the analysis gives a clear NO with HIGH confidence, output
   "not eligible" (inclusion) or "eligible" (exclusion). Do NOT
   downgrade to "unknown" unless you find a specific error.
2. SEVERITY CHECK: Pay close attention to severity qualifiers in the
   criterion (mild, moderate, severe, acute, chronic, early, advanced).
   If the analysis mentions a DIFFERENT severity than the criterion
   requires (e.g., criterion says "mild" but patient has "severe"),
   the criterion is NOT met — even if the general condition is present.
3. CONTRADICTION CHECK: If the analysis conclusion (YES/NO) contradicts
   its own reasoning text (e.g., says "NO" but the reasoning describes
   the patient clearly having the condition), rely on the REASONING
   CONTENT to determine the correct label. If genuinely ambiguous,
   output "unknown".

Respond ONLY with valid JSON:
{{
  "label": "<eligible|not eligible|unknown>",
  "reasoning": "Brief explanation of how you mapped the clinical finding to eligibility",
  "evidence_sentences": [0, 1, 2]
}}"""


def build_reasoning_prompt(
    patient_note: str,
    criterion_text: str,
    criterion_type: str,
) -> str:
    """Build Stage 1 reasoning prompt (factual clinical analysis, no eligibility label)."""
    return STAGE1_REASONING_PROMPT.format(
        patient_note=patient_note,
        criterion_text=criterion_text,
        criterion_type=criterion_type,
    )


def build_labeling_prompt(
    stage1_reasoning: str,
    criterion_text: str,
    criterion_type: str,
) -> str:
    """Build Stage 2 labeling prompt (Gemini assigns eligibility label from reasoning)."""
    instructions = (
        INCLUSION_INSTRUCTIONS if criterion_type == "inclusion" else EXCLUSION_INSTRUCTIONS
    )
    return STAGE2_LABELING_PROMPT.format(
        stage1_reasoning=stage1_reasoning,
        criterion_text=criterion_text,
        criterion_type=criterion_type,
        criterion_type_instructions=instructions,
    )


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

    # Prefer "label" key — try simplified eligibility labels first, then native fallback
    if "label" in data:
        label = data["label"].strip().lower()
        if label in ELIGIBLE_LABEL_TO_VERDICT:
            return ELIGIBLE_LABEL_TO_VERDICT[label], reasoning, evidence
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

    # Fallback: keyword extraction for eligibility labels (primary)
    cleaned_lower = cleaned.lower()
    if "not eligible" in cleaned_lower:
        return CriterionVerdict.NOT_MET, cleaned, []
    if "eligible" in cleaned_lower:
        return CriterionVerdict.MET, cleaned, []
    if "unknown" in cleaned_lower:
        return CriterionVerdict.UNKNOWN, cleaned, []

    # Fallback: keyword extraction for native TrialGPT labels (backward compat)
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
    max_tokens: int = 512,
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


async def evaluate_criterion_two_stage(
    patient_note: str,
    criterion_text: str,
    criterion_type: str,
    reasoning_adapter: ModelAdapter,
    labeling_adapter: ModelAdapter,
    max_tokens_reasoning: int = 2048,
    max_tokens_labeling: int = 256,
    timeout_seconds: float = 300.0,
) -> CriterionResult:
    """Two-stage evaluation: MedGemma reasons, Gemini labels.

    Stage 1: MedGemma performs factual clinical analysis (plain text, no JSON).
    Stage 2: Gemini reads the analysis and assigns an eligibility label (JSON).

    This separates medical reasoning (MedGemma's strength) from label assignment
    (Gemini's strength), fixing exclusion inversion and instruction-following errors.
    """
    # Stage 1: MedGemma medical reasoning
    reasoning_prompt = build_reasoning_prompt(patient_note, criterion_text, criterion_type)
    call_start = time.perf_counter()

    try:
        reasoning_response: ModelResponse = await asyncio.wait_for(
            reasoning_adapter.generate(reasoning_prompt, max_tokens=max_tokens_reasoning),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - call_start) * 1000
        logger.warning(
            "two_stage_reasoning_timeout",
            adapter=reasoning_adapter.name,
            timeout_seconds=timeout_seconds,
        )
        return CriterionResult(
            verdict=CriterionVerdict.UNKNOWN,
            reasoning=f"Stage 1 timeout: reasoning model did not respond within {timeout_seconds}s",
            evidence_sentences=[],
            model_response=ModelResponse(
                text="TIMEOUT",
                input_tokens=len(reasoning_prompt) // 4,
                output_tokens=0,
                latency_ms=elapsed_ms,
                estimated_cost=0.0,
                token_count_estimated=True,
            ),
        )

    stage1_text = clean_model_response(reasoning_response.text)

    # Stage 2: Gemini label assignment
    labeling_prompt = build_labeling_prompt(
        stage1_reasoning=stage1_text,
        criterion_text=criterion_text,
        criterion_type=criterion_type,
    )

    try:
        labeling_response: ModelResponse = await asyncio.wait_for(
            labeling_adapter.generate(labeling_prompt, max_tokens=max_tokens_labeling),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - call_start) * 1000
        logger.warning(
            "two_stage_labeling_timeout",
            adapter=labeling_adapter.name,
            timeout_seconds=timeout_seconds,
        )
        return CriterionResult(
            verdict=CriterionVerdict.UNKNOWN,
            reasoning=f"Stage 2 timeout: labeling model did not respond within {timeout_seconds}s",
            evidence_sentences=[],
            model_response=ModelResponse(
                text="TIMEOUT",
                input_tokens=len(labeling_prompt) // 4,
                output_tokens=0,
                latency_ms=elapsed_ms,
                estimated_cost=0.0,
                token_count_estimated=True,
            ),
            stage1_reasoning=stage1_text,
            stage1_response=reasoning_response,
        )

    verdict, reasoning, evidence = parse_criterion_verdict(labeling_response.text)

    return CriterionResult(
        verdict=verdict,
        reasoning=reasoning,
        evidence_sentences=evidence,
        model_response=labeling_response,
        stage1_reasoning=stage1_text,
        stage1_response=reasoning_response,
    )
