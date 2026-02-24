"""Adaptive Gemini agentic PRESCREEN loop.

Orchestrates Gemini 3 Pro with three tools:
  - search_trials: CT.gov API v2 search
  - get_trial_details: full eligibility criteria fetch
  - consult_medical_expert: MedGemma 27B on-demand clinical expertise

Gemini reasons autonomously about which tools to call, dynamically deciding
search terms and strategy based on intermediate results. MedGemma is available
as an on-demand expert for medical terminology questions.
"""

from __future__ import annotations

import asyncio
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

MAX_TOOL_CALLS_DEFAULT = 25

# Imported from gemini.py pricing constants (kept in sync)
_COST_PER_1M_INPUT = 1.25
_COST_PER_1M_OUTPUT = 10.00

TraceCallback = Any

PRESCREEN_SYSTEM_PROMPT = """\
You are a clinical trial search agent. Your goal: find ALL potentially \
relevant trials on ClinicalTrials.gov for a given patient. \
Maximize recall — missing an eligible trial is worse than including \
an irrelevant one.

You have {max_tool_calls} tool calls. Use them wisely.

## Tools

### search_trials
Search ClinicalTrials.gov. Key parameters:
- **condition**: Disease or condition (e.g., "mesothelioma", "pleural effusion"). \
  This is the primary search lever. CT.gov indexes trials under many different \
  vocabulary terms — a trial for "pleural effusion" will NOT appear in a \
  "mesothelioma" search. Use diverse condition terms across multiple calls.
- **intervention**: Drug or therapy name (e.g., "pembrolizumab", "cisplatin").
- **eligibility_keywords**: Free-text searched inside eligibility criteria \
  (e.g., "EGFR L858R", "treatment naive", "prior platinum"). Best for \
  biomarkers and clinical phenotype terms that appear in inclusion/exclusion text.
- **status**: Recruitment status filter. Default: ["RECRUITING"]. \
  Set to ["RECRUITING", "NOT_YET_RECRUITING", "COMPLETED", "ACTIVE_NOT_RECRUITING"] \
  for historical or comprehensive searches.
- **phase**, **location**, **study_type**: Optional narrowing filters.
- **page_size**: Results per call (default 50, max 100).
- **age/sex filters**: Available but NOT recommended — many trials lack \
  structured age data and the filter silently excludes them, hurting recall.

One condition term per call. Diverse terms across calls > repeated similar terms.

### get_trial_details
Fetch full eligibility criteria for a specific trial by NCT ID. \
Use when you need the inclusion/exclusion text to assess patient fit. \
Each call costs one tool use — prefer more searches over more detail fetches.

### consult_medical_expert
Ask MedGemma, a specialized medical AI, a clinical question. Use when you \
need domain expertise you're uncertain about — e.g., what related conditions \
share treatment approaches, what molecular features are clinically relevant, \
or what terminology CT.gov uses for a specific disease. \
Each call takes ~10 seconds. Typical usage: 1-3 calls per patient.\
"""

PRESCREEN_USER_TEMPLATE = """\
## Patient Demographics
- Age: {age}
- Sex: {sex} (API value: {sex_api})

## Patient Profile

{patient_note}

## Extracted Key Facts

{key_facts_text}

Find ALL potentially relevant clinical trials for this patient.\
"""


async def run_prescreen_agent(
    patient_note: str,
    key_facts: dict[str, Any],
    ingest_source: str,
    gemini_adapter: GeminiAdapter,
    medgemma_adapter: Any | None = None,
    max_tool_calls: int = 25,
    topic_id: str = "",
    on_tool_call: Any | None = None,
    on_agent_text: Any | None = None,
    trace_callback: TraceCallback | None = None,
) -> PresearchResult:
    """Run the adaptive Gemini agentic PRESCREEN loop for one patient.

    Args:
        patient_note: Raw patient note text (from INGEST output or gold SoT).
        key_facts: Structured key facts dict from INGEST (may be empty).
        ingest_source: "gold" | "model_medgemma" | "model_gemini" — for cache isolation.
        gemini_adapter: Configured GeminiAdapter (provides raw genai.Client access).
        medgemma_adapter: If provided, exposed as consult_medical_expert tool.
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
        executor = ToolExecutor(ctgov=ctgov, medgemma=medgemma_adapter)

        # Extract demographics from key_facts
        age = key_facts.get("age", "unknown")
        sex_raw = str(key_facts.get("sex", key_facts.get("gender", "unknown"))).strip()
        sex_api = _normalize_sex(sex_raw)

        system_prompt = PRESCREEN_SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls)

        user_message = PRESCREEN_USER_TEMPLATE.format(
            patient_note=patient_note.strip(),
            key_facts_text=_format_key_facts(key_facts),
            age=age,
            sex=sex_raw,
            sex_api=sex_api,
        )
        _emit_trace(
            trace_callback,
            "prescreen_prompt",
            {
                "topic_id": topic_id,
                "ingest_source": ingest_source,
                "model": gemini_adapter.name,
                "system_instruction": system_prompt,
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
            system_instruction=system_prompt,
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
                client=gemini_adapter._client,  # noqa: SLF001
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

            for part in parts:
                if getattr(part, "text", None):
                    turn_texts.append(part.text)
                    if final_text:
                        final_text += "\n" + part.text
                    else:
                        final_text = part.text
                    if on_agent_text:
                        on_agent_text(part.text)

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
                break

            # Budget guard
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

            # Execute all function calls in this turn
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

                    if fc_name == "search_trials" and isinstance(result, dict):
                        query_label = _describe_query(fc_args)
                        for trial in result.get("trials", []):
                            nct = trial.get("nct_id", "")
                            if nct:
                                if nct not in candidates_by_nct:
                                    candidates_by_nct[nct] = dict(trial)
                                found_by_query.setdefault(nct, []).append(query_label)

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

            contents.append(
                genai_types.Content(
                    role="user",
                    parts=function_response_parts,
                )
            )

        # Build final candidate list
        candidates = _build_candidates(candidates_by_nct, found_by_query)

        condition_terms_searched = [
            tc.args.get("condition", "")
            for tc in tool_call_records
            if tc.tool_name == "search_trials" and tc.args.get("condition")
        ]

        gemini_cost = (
            total_input_tokens * _COST_PER_1M_INPUT + total_output_tokens * _COST_PER_1M_OUTPUT
        ) / 1_000_000

        total_latency_ms = (time.perf_counter() - run_start) * 1000

        logger.info(
            "prescreen_complete",
            topic_id=topic_id,
            unique_trials=len(candidates),
            tool_calls=call_index,
            latency_ms=f"{total_latency_ms:.0f}",
            gemini_cost=f"${gemini_cost:.4f}",
            medgemma_calls=executor.medgemma_calls,
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
                "medgemma_calls": executor.medgemma_calls,
                "medgemma_estimated_cost": executor.medgemma_cost,
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
            medgemma_calls=executor.medgemma_calls,
            medgemma_estimated_cost=executor.medgemma_cost,
            latency_ms=total_latency_ms,
            medgemma_guidance_raw="",
            medgemma_condition_terms=[],
            medgemma_eligibility_keywords=[],
            condition_terms_searched=condition_terms_searched,
        )

    finally:
        await ctgov.aclose()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
