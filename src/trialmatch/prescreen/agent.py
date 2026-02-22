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

MAX_TOOL_CALLS_DEFAULT = 8

# Imported from gemini.py pricing constants (kept in sync)
_COST_PER_1M_INPUT = 1.25
_COST_PER_1M_OUTPUT = 10.00

PRESCREEN_SYSTEM_PROMPT = """\
You are a clinical trial search specialist. Your job: find the TOP 5-15 most \
relevant recruiting trials for a patient — quality over quantity.

## Search Strategy — BROAD FIRST, then NARROW

You MUST follow this layered approach. Do NOT combine many filters on a single search.

### Search 1: BROAD condition search (NO eligibility_keywords)
- Use the standard disease term for the condition field \
  (e.g., "non-small cell lung cancer", NOT "NSCLC adenocarcinoma")
- Pass age and sex filters to exclude ineligible trials
- Do NOT add eligibility_keywords — this first search establishes the baseline

### Search 2: Histology/subtype search
- Use a more specific condition term (e.g., "lung adenocarcinoma")
- Still use age/sex filters, but NO eligibility_keywords yet

### Search 3-4: Targeted eligibility keyword searches
- NOW add eligibility_keywords for specific clinical features:
  - Biomarkers/mutations: "EGFR", "ALK", "treatment naive", "first line"
  - Clinical phenotype: relevant to patient profile
- Use broad condition (e.g., "NSCLC" or "non-small cell lung cancer")
- Age/sex filters optional here (they may over-restrict combined with keywords)

### Search 5 (if needed): Intervention-based
- Search by prior therapy or drug class if patient has treatment history

## Demographic Filters

The patient's age and sex are provided. Use them on broad searches (Search 1-2) \
to exclude ineligible trials. Format: "43 Years" for age. \
On narrow keyword searches (Search 3+), age/sex filters are optional — \
the eligibility_keywords already restrict results.

## Key Rules

- Use standard CT.gov terms for condition: "non-small cell lung cancer" (not "NSCLC \
  adenocarcinoma"), "breast cancer" (not "breast carcinoma"), etc.
- Keep page_size at 10
- Call get_trial_details for the 3-5 most promising trials only
- Stop after 4-6 searches or when you have 5+ relevant trials

## Final Response

Summarize: search strategy, top 5 candidates with fit reasoning, key unknowns.
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

Search ClinicalTrials.gov for the TOP 5-15 recruiting trials this patient may be eligible for.

Remember: Start with a BROAD condition-only search (Search 1), then narrow. \
Use age="{age} Years" and sex="{sex_api}" on broad searches. \
Do NOT combine eligibility_keywords with age/sex on the same search.\
"""

# ---------------------------------------------------------------------------
# MedGemma clinical reasoning (pre-search guidance)
# ---------------------------------------------------------------------------

CLINICAL_REASONING_PROMPT = """\
You are a clinical oncology expert analyzing a patient profile to guide \
clinical trial search on ClinicalTrials.gov.

Given the patient profile below, provide a concise clinical reasoning summary:

1. **Standard CT.gov condition terms**: What are the 2-3 best condition search \
   terms for this patient? Use standard terms indexed by ClinicalTrials.gov \
   (e.g., "non-small cell lung cancer", "lung adenocarcinoma").

2. **Most likely molecular drivers**: Based on the patient's demographics, \
   histology, and smoking status, what are the most probable oncogenic driver \
   mutations? Rank by probability. For example, a young female never-smoker \
   with lung adenocarcinoma has ~50-60% probability of EGFR activating mutation.

3. **Priority eligibility keywords**: What 3-5 eligibility_keywords should be \
   searched on CT.gov? These should be terms commonly found in trial eligibility \
   criteria text (e.g., "EGFR", "treatment naive", "first line").

4. **Treatment line**: Is this patient treatment-naive, first-line, or later-line? \
   This determines which trials are appropriate.

5. **Clinical phenotype hints**: Any distinctive features (rare histology, \
   specific comorbidities) that should guide or constrain the search.

Patient Profile:
{patient_note}

Key Facts:
{key_facts_text}

Respond in plain text, concise bullet points. Focus on actionable search guidance."""


async def run_prescreen_agent(
    patient_note: str,
    key_facts: dict[str, Any],
    ingest_source: str,
    gemini_adapter: GeminiAdapter,
    medgemma_adapter: Any | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS_DEFAULT,
    topic_id: str = "",
    on_tool_call: Any | None = None,
    on_agent_text: Any | None = None,
) -> PresearchResult:
    """Run the Gemini agentic PRESCREEN loop for one patient.

    Args:
        patient_note: Raw patient note text (from INGEST output or gold SoT).
        key_facts: Structured key facts dict from INGEST (may be empty for baseline runs).
        ingest_source: "gold" | "model_medgemma" | "model_gemini" — for cache isolation.
        gemini_adapter: Configured GeminiAdapter (provides raw genai.Client access).
        medgemma_adapter: If provided, used for clinical reasoning pre-search step.
        max_tool_calls: Hard cap on total tool calls (safety budget guard).
        topic_id: Patient/topic identifier for tracing.
        on_tool_call: Optional sync callback(ToolCallRecord) — invoked after each tool execution.
        on_agent_text: Optional sync callback(str) — invoked when agent produces reasoning text.

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

        # MedGemma clinical reasoning pre-search step (optional)
        clinical_guidance = ""
        medgemma_calls = 0
        medgemma_cost = 0.0
        if medgemma_adapter is not None:
            clinical_guidance, medgemma_calls, medgemma_cost = await _get_clinical_guidance(
                medgemma_adapter=medgemma_adapter,
                patient_note=patient_note,
                key_facts_text=_format_key_facts(key_facts),
                topic_id=topic_id,
                on_agent_text=on_agent_text,
            )

        clinical_guidance_section = ""
        if clinical_guidance:
            clinical_guidance_section = (
                "## MedGemma Clinical Reasoning (use this to guide your search strategy)\n\n"
                + clinical_guidance
            )

        user_message = PRESCREEN_USER_TEMPLATE.format(
            patient_note=patient_note.strip(),
            key_facts_text=_format_key_facts(key_facts),
            age=age,
            sex=sex_raw,
            sex_api=sex_api,
            clinical_guidance_section=clinical_guidance_section,
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

            # Collect any text the model produced (updated on each turn)
            for part in parts:
                if getattr(part, "text", None):
                    final_text = part.text
                    if on_agent_text:
                        on_agent_text(part.text)

            # Extract function calls from this turn
            function_calls = [
                part.function_call
                for part in parts
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
) -> tuple[str, int, float]:
    """Call MedGemma for clinical reasoning to guide search strategy.

    Returns (guidance_text, call_count, estimated_cost).
    On failure, returns empty guidance (search proceeds without it).
    """
    prompt = CLINICAL_REASONING_PROMPT.format(
        patient_note=patient_note.strip(),
        key_facts_text=key_facts_text,
    )

    try:
        start = time.perf_counter()
        response = await medgemma_adapter.generate(prompt, max_tokens=512)
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

        if on_agent_text:
            on_agent_text(f"[MedGemma clinical reasoning] {guidance[:200]}...")

        return guidance, 1, response.estimated_cost

    except Exception as exc:
        logger.warning(
            "medgemma_clinical_reasoning_failed",
            topic_id=topic_id,
            error=str(exc)[:200],
        )
        return "", 0, 0.0


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


MAX_CANDIDATES = 20  # Hard cap — downstream VALIDATE is expensive per trial


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
