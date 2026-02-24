"""Gemini agentic PRESCREEN loop.

Orchestrates Gemini 3 Pro with two tools:
  - search_trials: CT.gov API v2 search (with demographic filters)
  - get_trial_details: full eligibility criteria fetch

Gemini reasons autonomously about which tools to call, stopping when it
has exhausted major query angles or reached max_tool_calls. The architecture
is disease-agnostic — Gemini adapts its search strategy to any patient profile.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING, Any

import structlog
from google.genai import types as genai_types

from trialmatch.prescreen.ctgov_client import CTGovClient
from trialmatch.prescreen.schema import PresearchResult, ToolCallRecord, TrialCandidate
from trialmatch.prescreen.tools import PRESCREEN_TOOLS, ToolExecutor

if TYPE_CHECKING:
    from trialmatch.models.gemini import GeminiAdapter

logger = structlog.get_logger()

MAX_TOOL_CALLS_DEFAULT = 20

# Imported from gemini.py pricing constants (kept in sync)
_COST_PER_1M_INPUT = 1.25
_COST_PER_1M_OUTPUT = 10.00

TraceCallback = Any

PRESCREEN_SYSTEM_PROMPT = """\
You are a clinical trial search specialist. Your job: find ALL potentially \
relevant trials for a patient on ClinicalTrials.gov — maximize recall.

## Search Execution Plan

You will receive a Search Checklist with condition terms from MedGemma \
(a medical AI). You MUST search EACH term systematically:

For EACH condition term in the checklist:
1. Call search_trials with that term as the condition parameter
2. Do NOT use age or sex filters — many trials lack structured age data \
   and the filter silently excludes them, hurting recall

After exhausting all condition terms:
3. Run 2-3 eligibility_keywords-only searches (no condition) for biomarkers \
   or treatment status from the checklist
4. Call get_trial_details for the 5-10 most promising trials across all searches

## Why This Matters

CT.gov indexes trials under diverse condition terms. A patient's relevant \
trials may be indexed under the specific diagnosis, a broader disease family, \
a clinical presentation, an anatomical category, or a broad umbrella term. \
A single search term misses trials indexed under different vocabulary. \
MedGemma provides diverse terms — search ALL of them.

## Tool Use Rules
- Call search_trials ONCE per condition term — one term per call
- Use page_size=50 for all searches to maximize coverage
- Do NOT combine multiple condition terms in one search call
- Do NOT stop after finding a few trials — exhaust ALL checklist terms
- Call get_trial_details for AT MOST 3 trials — prioritize MORE searches over details
- Every search call you skip to fetch details is ~50 trials you'll never see

## Final Response
Summarize: condition terms searched, total unique trials found, top \
candidates with fit reasoning, and key eligibility unknowns.\
"""

PRESCREEN_USER_TEMPLATE = """\
## Patient Demographics
- Age: {age}
- Sex: {sex} (API value: {sex_api})

## Patient Profile

{patient_note}

## Extracted Key Facts

{key_facts_text}

{clinical_guidance_section}

{search_checklist}

Search ClinicalTrials.gov for ALL potentially relevant trials for this patient.
Do NOT use age or sex filters — they exclude trials with missing structured data.\
"""

# ---------------------------------------------------------------------------
# MedGemma clinical reasoning (pre-search guidance)
# ---------------------------------------------------------------------------

CLINICAL_REASONING_PROMPT = """\
You are a clinical expert guiding clinical trial search on ClinicalTrials.gov.

Given this patient, generate search guidance in EXACTLY this format:

CONDITION TERMS (list 12-15, most specific to most broad):
1. [exact diagnosis subtype]
2. [standard disease name]
3. [broader disease family or related condition]
4. [key clinical presentation or complication — e.g., "Malignant Pleural Effusion"]
5. [another clinical presentation if relevant]
6. [related procedure or treatment that trials target — e.g., "Pleurodesis"]
7. [anatomical/organ-based category]
8. [MeSH heading for the condition — e.g., "Lung Neoplasms" for lung cancer]
9. [key complication with qualifier — e.g., "Malignant" + the complication]
10. [related disease that shares treatment — e.g., "Pleural Diseases"]
11. [broadest umbrella for basket trials — e.g., "Solid Tumor", "Neoplasms"]
12. [another broad umbrella — e.g., "Metastatic Cancer", "Advanced Cancer"]
13-15. [any additional MeSH terms, procedures, or synonyms — more is better]

ELIGIBILITY KEYWORDS (3-5 terms commonly found in trial eligibility criteria):
- [biomarker or mutation relevant to this patient]
- [treatment status keyword]
- [histology or clinical phenotype keyword]

TREATMENT LINE: [treatment-naive / first-line / second-line / later-line]

CLINICAL REASONING:
- Primary diagnosis with key supporting evidence
- Molecular/genomic considerations (likely driver mutations based on histology, \
demographics, smoking status)
- Key comorbidities that may affect trial eligibility
- Any distinctive features that should guide or constrain the search

Patient Profile:
{patient_note}

Key Facts:
{key_facts_text}

IMPORTANT: Each condition term becomes a SEPARATE CT.gov search query. \
Trials are indexed under many different vocabulary terms. A search for a \
clinical presentation may find trials that a search for the primary \
diagnosis misses entirely. Be comprehensive — list every term that could \
surface relevant trials.\
"""

FALLBACK_SEARCH_CHECKLIST = """\
## Search Checklist
MedGemma guidance was unavailable. Search broadly using the patient profile:
1. Search the primary diagnosis from the key facts
2. Search broader disease family terms
3. Search key clinical presentations or complications
4. Search broad umbrella categories (e.g., the organ system, "solid tumor")
5. Run 2-3 eligibility keyword searches for biomarkers or treatment status
"""


def _parse_clinical_guidance(guidance_text: str) -> dict[str, Any]:
    """Parse MedGemma's structured output into components.

    Extracts condition terms, eligibility keywords, and treatment line
    from the structured format. Returns empty lists on parse failure
    (Gemini still gets the raw text as unstructured guidance).
    """
    result: dict[str, Any] = {
        "condition_terms": [],
        "eligibility_keywords": [],
        "treatment_line": "",
        "raw_text": guidance_text,
    }
    if not guidance_text:
        return result

    # Extract numbered items after CONDITION header
    cond_match = re.search(
        r"CONDITION\s+TERMS?\s*[:(].*?\)?\s*:?\s*\n(.*?)(?=\nELIGIBILITY|\nTREATMENT|\nCLINICAL|\Z)",
        guidance_text,
        re.DOTALL | re.IGNORECASE,
    )
    if cond_match:
        block = cond_match.group(1)
        # Match numbered items: "1. term" or "1) term" or "- term"
        terms = re.findall(r"(?:^\d+[\.\)]\s*|^-\s+)(.+)", block, re.MULTILINE)
        result["condition_terms"] = [t.strip().strip('"').strip("'") for t in terms if t.strip()]

    # Extract dash-prefixed items after ELIGIBILITY header
    elig_match = re.search(
        r"ELIGIBILITY\s+KEYWORDS?\s*[:(].*?\)?\s*:?\s*\n(.*?)(?=\nTREATMENT|\nCLINICAL|\Z)",
        guidance_text,
        re.DOTALL | re.IGNORECASE,
    )
    if elig_match:
        block = elig_match.group(1)
        keywords = re.findall(r"(?:^-\s+|^\d+[\.\)]\s*)(.+)", block, re.MULTILINE)
        result["eligibility_keywords"] = [k.strip().strip('"').strip("'") for k in keywords if k.strip()]

    # Extract treatment line
    tl_match = re.search(r"TREATMENT\s+LINE\s*:\s*(.+)", guidance_text, re.IGNORECASE)
    if tl_match:
        result["treatment_line"] = tl_match.group(1).strip()

    return result


def _build_search_checklist(parsed: dict[str, Any]) -> str:
    """Generate an explicit search checklist from parsed MedGemma guidance.

    Terms are dynamically populated from MedGemma output — no hardcoded diseases.
    """
    lines = ["## Search Checklist (search EACH term):"]
    for term in parsed.get("condition_terms", []):
        lines.append(f'- [ ] Condition: "{term}"')

    keywords = parsed.get("eligibility_keywords", [])
    if keywords:
        lines.append("\n## Eligibility Keyword Searches:")
        for kw in keywords:
            lines.append(f'- [ ] Keywords: "{kw}"')

    return "\n".join(lines)


async def _get_gemini_fallback_guidance(
    gemini_adapter: Any,
    patient_note: str,
    key_facts_text: str,
    topic_id: str = "",
    trace_callback: TraceCallback | None = None,
) -> str:
    """Use Gemini as fallback when MedGemma is unavailable.

    Calls Gemini with the same CLINICAL_REASONING_PROMPT to generate
    condition terms and search guidance.
    """
    prompt = CLINICAL_REASONING_PROMPT.format(
        patient_note=patient_note.strip(),
        key_facts_text=key_facts_text,
    )
    _emit_trace(
        trace_callback,
        "gemini_fallback_guidance_prompt",
        {"topic_id": topic_id, "prompt_len": len(prompt)},
    )
    try:
        start = time.perf_counter()
        response = await asyncio.to_thread(
            gemini_adapter._client.models.generate_content,
            model=gemini_adapter._model,
            contents=prompt,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = response.text or ""
        logger.info(
            "gemini_fallback_guidance_complete",
            topic_id=topic_id,
            latency_ms=f"{latency_ms:.0f}",
            guidance_len=len(text),
        )
        _emit_trace(
            trace_callback,
            "gemini_fallback_guidance_response",
            {"topic_id": topic_id, "latency_ms": latency_ms, "guidance": text[:500]},
        )
        return text.strip()
    except Exception as exc:
        logger.warning(
            "gemini_fallback_guidance_failed",
            topic_id=topic_id,
            error=str(exc)[:200],
        )
        return ""


async def run_prescreen_agent(
    patient_note: str,
    key_facts: dict[str, Any],
    ingest_source: str,
    gemini_adapter: GeminiAdapter,
    medgemma_adapter: Any | None = None,
    allow_gemini_fallback: bool = False,
    max_tool_calls: int = MAX_TOOL_CALLS_DEFAULT,
    topic_id: str = "",
    on_tool_call: Any | None = None,
    on_agent_text: Any | None = None,
    trace_callback: TraceCallback | None = None,
) -> PresearchResult:
    """Run the Gemini agentic PRESCREEN loop for one patient.

    Args:
        patient_note: Raw patient note text (from INGEST output or gold SoT).
        key_facts: Structured key facts dict from INGEST (may be empty for baseline runs).
        ingest_source: "gold" | "model_medgemma" | "model_gemini" — for cache isolation.
        gemini_adapter: Configured GeminiAdapter (provides raw genai.Client access).
        medgemma_adapter: If provided, used for clinical reasoning pre-search step.
        allow_gemini_fallback: If True, uses Gemini for guidance when MedGemma unavailable.
            If False and no medgemma_adapter provided, raises ValueError.
        max_tool_calls: Hard cap on total tool calls (safety budget guard).
        topic_id: Patient/topic identifier for tracing.
        on_tool_call: Optional sync callback(ToolCallRecord) — invoked after each tool execution.
        on_agent_text: Optional sync callback(str) — invoked when agent produces reasoning text.
        trace_callback: Optional callback(event_name, payload) for local trace recording.

    Returns:
        PresearchResult with deduplicated TrialCandidate list + full agent trace.
    """
    run_start = time.perf_counter()

    ctgov = CTGovClient()
    try:
        executor = ToolExecutor(ctgov=ctgov)

        # Extract demographics from key_facts for API filter injection
        age = key_facts.get("age", "unknown")
        sex_raw = str(key_facts.get("sex", key_facts.get("gender", "unknown"))).strip()
        sex_api = _normalize_sex(sex_raw)

        # MedGemma clinical reasoning pre-search step (required unless fallback allowed)
        clinical_guidance = ""
        parsed_guidance: dict[str, Any] = {}
        medgemma_calls = 0
        medgemma_cost = 0.0

        if medgemma_adapter is not None:
            clinical_guidance, medgemma_calls, medgemma_cost = await _get_clinical_guidance(
                medgemma_adapter=medgemma_adapter,
                patient_note=patient_note,
                key_facts_text=_format_key_facts(key_facts),
                topic_id=topic_id,
                on_agent_text=on_agent_text,
                require_success=not allow_gemini_fallback,
                trace_callback=trace_callback,
            )
            if clinical_guidance:
                parsed_guidance = _parse_clinical_guidance(clinical_guidance)
        elif allow_gemini_fallback:
            logger.warning("prescreen_no_medgemma_using_gemini_fallback", topic_id=topic_id)
            clinical_guidance = await _get_gemini_fallback_guidance(
                gemini_adapter, patient_note, _format_key_facts(key_facts),
                topic_id=topic_id, trace_callback=trace_callback,
            )
            if clinical_guidance:
                parsed_guidance = _parse_clinical_guidance(clinical_guidance)
        else:
            raise ValueError(
                "MedGemma adapter required for PRESCREEN. "
                "Pass medgemma_adapter or set allow_gemini_fallback=True."
            )

        _emit_trace(
            trace_callback,
            "medgemma_guidance_parsed",
            {
                "topic_id": topic_id,
                "condition_terms": parsed_guidance.get("condition_terms", []),
                "eligibility_keywords": parsed_guidance.get("eligibility_keywords", []),
                "parse_success": bool(parsed_guidance.get("condition_terms")),
            },
        )

        # Build user message sections
        clinical_guidance_section = ""
        if clinical_guidance:
            clinical_guidance_section = (
                "## MedGemma Clinical Reasoning\n\n" + clinical_guidance
            )

        search_checklist = (
            _build_search_checklist(parsed_guidance)
            if parsed_guidance.get("condition_terms")
            else FALLBACK_SEARCH_CHECKLIST
        )

        user_message = PRESCREEN_USER_TEMPLATE.format(
            patient_note=patient_note.strip(),
            key_facts_text=_format_key_facts(key_facts),
            age=age,
            sex=sex_raw,
            sex_api=sex_api,
            clinical_guidance_section=clinical_guidance_section,
            search_checklist=search_checklist,
        )
        _emit_trace(
            trace_callback,
            "prescreen_prompt",
            {
                "topic_id": topic_id,
                "ingest_source": ingest_source,
                "model": gemini_adapter.name,
                "system_instruction": PRESCREEN_SYSTEM_PROMPT,
                "user_message": user_message,
                "max_tool_calls": max_tool_calls,
            },
        )

        contents: list[genai_types.Content] = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)],
            )
        ]

        config = genai_types.GenerateContentConfig(
            system_instruction=PRESCREEN_SYSTEM_PROMPT,
            tools=[PRESCREEN_TOOLS],
            tool_config=genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(mode="AUTO")
            ),
            max_output_tokens=4096,
        )

        tool_call_records: list[ToolCallRecord] = []
        call_index = 0
        candidates_by_nct: dict[str, dict[str, Any]] = {}
        found_by_query: dict[str, list[str]] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        final_text = ""

        # +5 extra iterations to give Gemini room to generate its final summary
        for _iteration in range(max_tool_calls + 5):
            response = await _generate_with_retry(
                client=gemini_adapter._client,  # noqa: SLF001 — intentional internal access
                model=gemini_adapter._model,  # noqa: SLF001
                contents=contents,
                config=config,
            )

            # Accumulate token counts
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_input_tokens += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                total_output_tokens += (
                    getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                )

            if not response.candidates:
                logger.warning("prescreen_empty_candidates", topic_id=topic_id)
                break

            model_content = response.candidates[0].content
            contents.append(model_content)

            parts = model_content.parts or []
            turn_texts: list[str] = []

            # Collect any text the model produced (append across turns)
            for part in parts:
                if getattr(part, "text", None):
                    turn_texts.append(part.text)
                    if final_text:
                        final_text += "\n" + part.text
                    else:
                        final_text = part.text
                    if on_agent_text:
                        on_agent_text(part.text)

            # Extract function calls from this turn
            function_calls = [
                part.function_call
                for part in parts
                if getattr(part, "function_call", None)
            ]
            _emit_trace(
                trace_callback,
                "gemini_turn",
                {
                    "topic_id": topic_id,
                    "turn_index": _iteration,
                    "texts": turn_texts,
                    "function_calls": [
                        {
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                        }
                        for fc in function_calls
                    ],
                    "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0)
                    if getattr(response, "usage_metadata", None)
                    else 0,
                    "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0)
                    if getattr(response, "usage_metadata", None)
                    else 0,
                },
            )

            if not function_calls:
                # Gemini has stopped calling tools — done
                break

            # Budget guard: close all pending function_calls with error responses so
            # the conversation remains structurally valid (function_call must always be
            # answered by a function_response before any subsequent user turn).
            if call_index >= max_tool_calls:
                logger.warning(
                    "prescreen_tool_budget_exceeded",
                    topic_id=topic_id,
                    max_tool_calls=max_tool_calls,
                )
                budget_msg = (
                    f"Tool budget of {max_tool_calls} calls exhausted. "
                    "Stop searching and summarise findings so far."
                )
                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                function_response=genai_types.FunctionResponse(
                                    name=fc.name,
                                    response={"error": budget_msg},
                                )
                            )
                            for fc in function_calls
                        ],
                    )
                )
                continue

            # Execute all function calls in this turn, then return results together
            function_response_parts: list[genai_types.Part] = []

            for fc in function_calls:
                fc_name: str = fc.name
                fc_args: dict[str, Any] = dict(fc.args) if fc.args else {}

                call_start = time.perf_counter()
                error_msg: str | None = None
                result: Any = {}
                summary = ""

                try:
                    result, summary = await executor.execute(fc_name, fc_args)

                    # Collect candidates from search results
                    if fc_name == "search_trials" and isinstance(result, dict):
                        query_label = _describe_query(fc_args)
                        for trial in result.get("trials", []):
                            nct = trial.get("nct_id", "")
                            if nct:
                                # Merge: don't overwrite richer data with sparser data
                                if nct not in candidates_by_nct:
                                    candidates_by_nct[nct] = dict(trial)
                                found_by_query.setdefault(nct, []).append(query_label)

                    # Enrich existing candidate with full eligibility data
                    elif fc_name == "get_trial_details" and isinstance(result, dict):
                        nct = result.get("nct_id", "")
                        if nct:
                            candidates_by_nct.setdefault(nct, {}).update(result)

                except Exception as exc:
                    error_msg = str(exc)
                    logger.error(
                        "prescreen_tool_error",
                        tool=fc_name,
                        topic_id=topic_id,
                        error=error_msg,
                    )
                    result = {"error": error_msg}
                    summary = f"Error in {fc_name}: {error_msg[:120]}"

                latency_ms = (time.perf_counter() - call_start) * 1000

                record = ToolCallRecord(
                    call_index=call_index,
                    tool_name=fc_name,
                    args=fc_args,
                    result_summary=summary,
                    result_count=(
                        result.get("count", 0)
                        if isinstance(result, dict) and "count" in result
                        else 0
                    ),
                    latency_ms=latency_ms,
                    error=error_msg,
                )
                tool_call_records.append(record)
                call_index += 1
                _emit_trace(
                    trace_callback,
                    "tool_call_result",
                    {
                        "topic_id": topic_id,
                        "call_index": record.call_index,
                        "tool_name": fc_name,
                        "args": fc_args,
                        "result": result,
                        "result_summary": summary,
                        "latency_ms": latency_ms,
                        "error": error_msg,
                    },
                )

                if on_tool_call:
                    on_tool_call(record)

                function_response_parts.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=fc_name,
                            response={"result": result},
                        )
                    )
                )

            # Feed all results back to Gemini in one user Content
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=function_response_parts,
                )
            )

        # Build final candidate list
        candidates = _build_candidates(candidates_by_nct, found_by_query)

        # Collect condition terms actually searched by Gemini
        condition_terms_searched = [
            tc.args.get("condition", "")
            for tc in tool_call_records
            if tc.tool_name == "search_trials" and tc.args.get("condition")
        ]

        gemini_cost = (
            total_input_tokens * _COST_PER_1M_INPUT + total_output_tokens * _COST_PER_1M_OUTPUT
        ) / 1_000_000

        total_latency_ms = (time.perf_counter() - run_start) * 1000

        total_medgemma_calls = executor.medgemma_calls + medgemma_calls
        total_medgemma_cost = executor.medgemma_cost + medgemma_cost

        logger.info(
            "prescreen_complete",
            topic_id=topic_id,
            unique_trials=len(candidates),
            tool_calls=call_index,
            latency_ms=f"{total_latency_ms:.0f}",
            gemini_cost=f"${gemini_cost:.4f}",
            medgemma_calls=total_medgemma_calls,
        )
        _emit_trace(
            trace_callback,
            "prescreen_complete",
            {
                "topic_id": topic_id,
                "candidate_count": len(candidates),
                "candidate_nct_ids": [c.nct_id for c in candidates],
                "tool_calls": call_index,
                "gemini_input_tokens": total_input_tokens,
                "gemini_output_tokens": total_output_tokens,
                "gemini_estimated_cost": gemini_cost,
                "medgemma_calls": total_medgemma_calls,
                "medgemma_estimated_cost": total_medgemma_cost,
                "latency_ms": total_latency_ms,
                "agent_reasoning": final_text,
            },
        )

        return PresearchResult(
            topic_id=topic_id,
            ingest_source=ingest_source,
            candidates=candidates,
            agent_reasoning=final_text,
            tool_call_trace=tool_call_records,
            total_api_calls=call_index,
            total_unique_nct_ids=len(candidates_by_nct),
            gemini_input_tokens=total_input_tokens,
            gemini_output_tokens=total_output_tokens,
            gemini_estimated_cost=gemini_cost,
            medgemma_calls=total_medgemma_calls,
            medgemma_estimated_cost=total_medgemma_cost,
            latency_ms=total_latency_ms,
            medgemma_guidance_raw=clinical_guidance,
            medgemma_condition_terms=parsed_guidance.get("condition_terms", []),
            medgemma_eligibility_keywords=parsed_guidance.get("eligibility_keywords", []),
            condition_terms_searched=condition_terms_searched,
        )

    finally:
        await ctgov.aclose()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _get_clinical_guidance(
    medgemma_adapter: Any,
    patient_note: str,
    key_facts_text: str,
    topic_id: str = "",
    on_agent_text: Any | None = None,
    require_success: bool = False,
    trace_callback: TraceCallback | None = None,
) -> tuple[str, int, float]:
    """Call MedGemma for clinical reasoning to guide search strategy.

    Returns (guidance_text, call_count, estimated_cost).
    On failure, returns empty guidance unless require_success is True.
    """
    prompt = CLINICAL_REASONING_PROMPT.format(
        patient_note=patient_note.strip(),
        key_facts_text=key_facts_text,
    )
    _emit_trace(
        trace_callback,
        "medgemma_guidance_prompt",
        {
            "topic_id": topic_id,
            "prompt": prompt,
        },
    )

    try:
        start = time.perf_counter()
        response = await medgemma_adapter.generate(prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        # Clean MedGemma output (strip chat markers, thinking tokens)
        from trialmatch.validate.evaluator import clean_model_response

        guidance = clean_model_response(response.text).strip()

        logger.info(
            "medgemma_clinical_reasoning_complete",
            topic_id=topic_id,
            latency_ms=f"{latency_ms:.0f}",
            guidance_len=len(guidance),
        )
        _emit_trace(
            trace_callback,
            "medgemma_guidance_response",
            {
                "topic_id": topic_id,
                "model": getattr(medgemma_adapter, "name", ""),
                "latency_ms": latency_ms,
                "estimated_cost": response.estimated_cost,
                "raw_response": response.text,
                "guidance": guidance,
            },
        )

        if on_agent_text:
            on_agent_text(f"[MedGemma clinical reasoning] {guidance[:200]}...")

        return guidance, 1, response.estimated_cost

    except Exception as exc:
        err = str(exc)[:200]
        logger.warning(
            "medgemma_clinical_reasoning_failed",
            topic_id=topic_id,
            error=err,
        )
        _emit_trace(
            trace_callback,
            "medgemma_guidance_failed",
            {
                "topic_id": topic_id,
                "error": err,
                "required": require_success,
            },
        )
        if require_success:
            logger.error(
                "medgemma_clinical_reasoning_required_failed",
                topic_id=topic_id,
                error=err,
            )
            _emit_trace(
                trace_callback,
                "medgemma_guidance_required_failed",
                {
                    "topic_id": topic_id,
                    "error": err,
                },
            )
            msg = f"MedGemma clinical guidance required but failed: {err}"
            raise RuntimeError(msg) from exc
        return "", 0, 0.0


def _emit_trace(
    trace_callback: TraceCallback | None,
    event_name: str,
    payload: dict[str, Any],
) -> None:
    """Emit trace events without affecting runtime behavior on callback errors."""
    if trace_callback is None:
        return
    try:
        trace_callback(event_name, payload)
    except Exception:
        logger.warning("prescreen_trace_callback_failed", event=event_name, exc_info=True)


async def _generate_with_retry(
    client: Any,
    model: str,
    contents: list,
    config: Any,
    max_retries: int = 3,
) -> Any:
    """Wrap genai generate_content with transient-error retry."""
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            err_str = str(exc)
            is_transient = any(
                marker in err_str
                for marker in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand")
            )
            if is_transient and attempt < max_retries - 1:
                wait = 2.0**attempt
                logger.warning(
                    "prescreen_gemini_transient",
                    attempt=attempt + 1,
                    wait=wait,
                    error=err_str[:100],
                )
                await asyncio.sleep(wait)
                continue
            raise

    msg = f"Gemini PRESCREEN API unavailable after {max_retries} retries"
    raise RuntimeError(msg)  # pragma: no cover


def _format_key_facts(key_facts: dict[str, Any]) -> str:
    """Format key_facts dict into a readable bullet list."""
    if not key_facts:
        return "(No structured key facts — search from free text only)"
    lines = []
    for key, val in key_facts.items():
        if isinstance(val, list):
            lines.append(f"- {key}: {', '.join(str(v) for v in val)}")
        elif val is not None:
            lines.append(f"- {key}: {val}")
    return "\n".join(lines) if lines else "(Empty key facts dict)"


def _normalize_sex(raw: str) -> str:
    """Map free-text sex/gender to CT.gov API enum: MALE, FEMALE, or ALL."""
    lower = raw.lower().strip()
    if lower in ("female", "f", "woman"):
        return "FEMALE"
    if lower in ("male", "m", "man"):
        return "MALE"
    return "ALL"


def _describe_query(args: dict[str, Any]) -> str:
    """Produce a short human-readable label for a search_trials call args."""
    parts = []
    if cond := args.get("condition"):
        parts.append(f"cond='{cond[:50]}'")
    if intr := args.get("intervention"):
        parts.append(f"intr='{intr[:30]}'")
    if kw := args.get("eligibility_keywords"):
        parts.append(f"kw='{kw[:50]}'")
    return ", ".join(parts) or "broad search"


MAX_CANDIDATES = 200  # High for recall — downstream VALIDATE filters


def _score_candidate(data: dict[str, Any], query_count: int) -> float:
    """Heuristic relevance score for ranking. Higher = better.

    Scoring signals:
    - query_count: found by multiple searches → more relevant (3 pts each)
    - Phase II/III: more clinically advanced (2 pts)
    - RECRUITING status: can actually enroll (1 pt)
    - Has eligibility_criteria: we fetched details → agent thought it was relevant (2 pts)
    """
    score = query_count * 3.0

    phases = data.get("phase") or []
    if any(p in ("PHASE2", "PHASE3") for p in phases):
        score += 2.0
    elif any(p in ("PHASE1",) for p in phases):
        score += 1.0

    if data.get("status") == "RECRUITING":
        score += 1.0

    if data.get("eligibility_criteria"):
        score += 2.0

    return score


def _build_candidates(
    candidates_by_nct: dict[str, dict[str, Any]],
    found_by_query: dict[str, list[str]],
) -> list[TrialCandidate]:
    """Build TrialCandidate list from accumulated search data.

    Scores candidates by heuristic relevance and caps at MAX_CANDIDATES.
    """
    scored: list[tuple[float, str, dict[str, Any]]] = []

    for nct_id, data in candidates_by_nct.items():
        query_count = len(found_by_query.get(nct_id, []))
        score = _score_candidate(data, query_count)
        scored.append((score, nct_id, data))

    # Sort by score desc, then nct_id for determinism
    scored.sort(key=lambda t: (-t[0], t[1]))

    # Prune to top N
    scored = scored[:MAX_CANDIDATES]

    result: list[TrialCandidate] = []
    for _score, nct_id, data in scored:
        try:
            candidate = TrialCandidate(
                nct_id=nct_id,
                title=data.get("title") or data.get("brief_title", ""),
                brief_title=data.get("brief_title", ""),
                status=data.get("status", "UNKNOWN"),
                phase=data.get("phase") or [],
                conditions=data.get("conditions") or [],
                interventions=data.get("interventions") or [],
                sponsor=data.get("sponsor", ""),
                enrollment=data.get("enrollment"),
                start_date=data.get("start_date"),
                primary_completion_date=data.get("primary_completion_date"),
                locations_count=data.get("locations_count"),
                study_type=data.get("study_type", ""),
                found_by_queries=found_by_query.get(nct_id, []),
            )
            result.append(candidate)
        except Exception:
            logger.warning("prescreen_candidate_build_failed", nct_id=nct_id, exc_info=True)

    logger.info(
        "prescreen_candidates_ranked",
        total_raw=len(candidates_by_nct),
        after_pruning=len(result),
        max_cap=MAX_CANDIDATES,
    )
    return result
