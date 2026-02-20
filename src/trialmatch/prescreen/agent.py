"""Gemini agentic PRESCREEN loop.

Orchestrates Gemini 3 Pro with three tools:
  - search_trials: CT.gov API v2 search
  - get_trial_details: full eligibility criteria fetch
  - normalize_medical_terms: MedGemma medical term normalization

Gemini reasons autonomously about which tools to call, stopping when it
has exhausted major query angles or reached max_tool_calls. The architecture
is disease-agnostic — Gemini adapts its search strategy to any patient profile.
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
    from trialmatch.models.base import ModelAdapter
    from trialmatch.models.gemini import GeminiAdapter

logger = structlog.get_logger()

MAX_TOOL_CALLS_DEFAULT = 12

# Imported from gemini.py pricing constants (kept in sync)
_COST_PER_1M_INPUT = 1.25
_COST_PER_1M_OUTPUT = 10.00

PRESCREEN_SYSTEM_PROMPT = """\
You are a clinical trial search specialist helping match patients to eligible clinical trials.

Given a patient's medical notes and extracted key clinical facts, search ClinicalTrials.gov
comprehensively to find all recruiting trials the patient may be eligible for.

## Search Strategy

Follow this layered approach to maximize recall:

1. **Normalize ambiguous terms first** — Before searching, call normalize_medical_terms for:
   - Biomarkers / mutations (e.g., "EGFR L858R") → get HUGO notation + CT.gov search variants
   - Drug names (e.g., "Tagrisso") → get INN + brand name variants
   - Diagnoses → get MeSH terms and CT.gov-indexed synonyms
   - Clinical phenotypes (e.g., "non-smoker") → get exact phrasing used in eligibility text

2. **Search in layers** — Use multiple complementary queries:
   - **Condition layer**: Primary diagnosis + major subtypes
   - **Phenotype layer**: Clinical characteristics (smoking status, ECOG, treatment history)
     — use eligibility_keywords for phrases like "never smoker", "treatment naive", "prior platinum"
   - **Biomarker layer**: Specific mutations/alterations in eligibility_keywords
   - **Intervention layer**: Prior therapies to find post-progression trials
   - **Broad fallback**: Condition alone if specific searches miss trials

3. **Investigate promising trials** — Use get_trial_details for trials where:
   - The title suggests relevance but you need to verify inclusion/exclusion criteria
   - A trial is borderline — eligibility text is needed to assess fit

4. **Stop when confident** — Stop after exhausting major query angles.
   Additional searches are redundant when they return the same NCT IDs.

## Key Principles

- Phenotype-based queries often outperform biomarker queries: trials may say
  "actionable genomic alteration" instead of a specific mutation name
- Run 3–6 searches with different parameter combinations to maximize recall
- For post-progression patients, search by prior therapy name in eligibility_keywords
- Deduplication happens automatically — call search_trials multiple times without worry

## Final Response

After completing your searches, provide:
- Summary of what you searched and why
- Top candidate trials with brief reasoning for fit
- Any important caveats or disqualifying factors noticed
"""

PRESCREEN_USER_TEMPLATE = """\
## Patient Profile

{patient_note}

## Extracted Key Facts

{key_facts_text}

Please search ClinicalTrials.gov to find all recruiting clinical trials that this patient \
may be eligible for. Use multiple complementary search strategies to maximize recall.\
"""


async def run_prescreen_agent(
    patient_note: str,
    key_facts: dict[str, Any],
    ingest_source: str,
    gemini_adapter: GeminiAdapter,
    medgemma_adapter: ModelAdapter,
    max_tool_calls: int = MAX_TOOL_CALLS_DEFAULT,
    topic_id: str = "",
) -> PresearchResult:
    """Run the Gemini agentic PRESCREEN loop for one patient.

    Args:
        patient_note: Raw patient note text (from INGEST output or gold SoT).
        key_facts: Structured key facts dict from INGEST (may be empty for baseline runs).
        ingest_source: "gold" | "model_medgemma" | "model_gemini" — for cache isolation.
        gemini_adapter: Configured GeminiAdapter (provides raw genai.Client access).
        medgemma_adapter: MedGemma adapter used by the normalize_medical_terms tool.
        max_tool_calls: Hard cap on total tool calls (safety budget guard).
        topic_id: Patient/topic identifier for tracing.

    Returns:
        PresearchResult with deduplicated TrialCandidate list + full agent trace.
    """
    run_start = time.perf_counter()

    ctgov = CTGovClient()
    try:
        executor = ToolExecutor(ctgov=ctgov, medgemma=medgemma_adapter)

        user_message = PRESCREEN_USER_TEMPLATE.format(
            patient_note=patient_note.strip(),
            key_facts_text=_format_key_facts(key_facts),
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

            # Collect any text the model produced (updated on each turn)
            for part in model_content.parts:
                if getattr(part, "text", None):
                    final_text = part.text

            # Extract function calls from this turn
            function_calls = [
                part.function_call
                for part in model_content.parts
                if getattr(part, "function_call", None)
            ]

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

                tool_call_records.append(
                    ToolCallRecord(
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
                )
                call_index += 1

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
        )

    finally:
        await ctgov.aclose()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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


def _build_candidates(
    candidates_by_nct: dict[str, dict[str, Any]],
    found_by_query: dict[str, list[str]],
) -> list[TrialCandidate]:
    """Build TrialCandidate list from accumulated search data.

    Candidates found by more queries are ranked first (higher recall signal).
    """
    result: list[TrialCandidate] = []

    for nct_id, data in candidates_by_nct.items():
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

    # Primary sort: trials found by more queries first (relevance signal)
    # Secondary sort: stable by nct_id for determinism
    result.sort(key=lambda c: (-len(c.found_by_queries), c.nct_id))
    return result
