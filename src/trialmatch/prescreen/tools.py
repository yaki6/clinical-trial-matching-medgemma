"""Tool definitions and dispatch for the PRESCREEN agent.

This module owns:
  1. The Gemini function-calling schemas (what the model sees).
  2. The async tool executor (what actually runs when the model calls a tool).

The three tools exposed to Gemini are:
  - search_trials            → CT.gov API v2 /studies search
  - get_trial_details        → CT.gov API v2 /studies/{nct_id}
  - consult_medical_expert   → MedGemma 27B on-demand clinical expertise

Design note: tool schemas are purposefully kept narrow.  Gemini should
not be handed every CT.gov query parameter — just the ones that produce
reliable results (validated against the live API).
"""

from __future__ import annotations

import re
import time
from typing import Any

import structlog
from google.genai import types as genai_types

from trialmatch.prescreen.ctgov_client import CTGovClient, parse_search_results, parse_study_summary

logger = structlog.get_logger()

MAX_ELIGIBILITY_CHARS = 2000  # ~500 tokens — captures first ~20 criteria lines


def _truncate_eligibility(text: str) -> str:
    """Truncate eligibility criteria to prevent context window bloat."""
    if len(text) <= MAX_ELIGIBILITY_CHARS:
        return text
    truncated = text[:MAX_ELIGIBILITY_CHARS]
    last_newline = truncated.rfind('\n')
    if last_newline > MAX_ELIGIBILITY_CHARS * 0.7:
        truncated = truncated[:last_newline]
    return truncated + "\n[... truncated for brevity]"


# ---------------------------------------------------------------------------
# 1. Tool schemas (Gemini function declarations)
# ---------------------------------------------------------------------------

_SEARCH_TRIALS_DECL = genai_types.FunctionDeclaration(
    name="search_trials",
    description=(
        "Search ClinicalTrials.gov for recruiting clinical trials. "
        "Use this to find trials matching a patient's condition, biomarkers, interventions, "
        "or clinical phenotype characteristics (e.g. 'never smoker', 'treatment naive'). "
        "You can call this multiple times with different parameters to broaden or narrow results. "
        "IMPORTANT: A single query often misses trials that describe the same condition in "
        "different words. Run 2-4 complementary searches with varying specificity."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "condition": genai_types.Schema(
                type=genai_types.Type.STRING,
                description=(
                    "Disease or condition to search for in the trial's condition field. "
                    "Examples: 'non-small cell lung cancer', 'NSCLC', 'breast cancer', "
                    "'type 2 diabetes'. You may also combine clinical characteristics here, "
                    "e.g. 'NSCLC never smoker' or 'breast cancer HER2 positive'."
                ),
            ),
            "intervention": genai_types.Schema(
                type=genai_types.Type.STRING,
                description=(
                    "Drug, device, or intervention to filter by. "
                    "Useful to find trials involving a prior therapy (post-progression trials) "
                    "or a specific drug class. Example: 'osimertinib', 'pembrolizumab', "
                    "'CAR-T therapy'."
                ),
            ),
            "eligibility_keywords": genai_types.Schema(
                type=genai_types.Type.STRING,
                description=(
                    "Keywords searched inside the eligibility criteria text. "
                    "Best for specific biomarkers or clinical phenotype phrases that trials "
                    "write explicitly in their inclusion/exclusion criteria. "
                    "Examples: 'EGFR L858R', 'KRAS G12C', 'never smoker', 'ALK rearrangement', "
                    "'MET exon 14', 'treatment naive', 'prior platinum chemotherapy'. "
                    "Avoid vague phrases — use exact medical terms as a clinician would write."
                ),
            ),
            "status": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
                description=(
                    "Filter by recruitment status. Default: ['RECRUITING']. "
                    "Options: 'RECRUITING', 'NOT_YET_RECRUITING', 'ACTIVE_NOT_RECRUITING', "
                    "'COMPLETED'."
                ),
            ),
            "phase": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
                description=(
                    "Filter by trial phase. Leave empty to include all phases (recommended). "
                    "Options: 'PHASE1', 'PHASE2', 'PHASE3', 'PHASE4', 'EARLY_PHASE1'. "
                    "Note: many oncology trials are 'PHASE1' or combined 'PHASE1'+'PHASE2' — "
                    "restricting to PHASE2/3 only misses important options for heavily "
                    "pretreated patients."
                ),
            ),
            "location": genai_types.Schema(
                type=genai_types.Type.STRING,
                description=(
                    "Geographic location to filter trial sites. "
                    "City, state, or country. Examples: 'Boston', 'California', 'Germany'."
                ),
            ),
            "study_type": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"],
                description=(
                    "Filter by study type. Omit to search all study types. "
                    "Use INTERVENTIONAL for clinical trials, "
                    "OBSERVATIONAL for observational studies."
                ),
            ),
            "page_size": genai_types.Schema(
                type=genai_types.Type.INTEGER,
                description="Number of results to return. Default 50. Max 100.",
            ),
        },
    ),
)

_GET_DETAILS_DECL = genai_types.FunctionDeclaration(
    name="get_trial_details",
    description=(
        "Fetch complete eligibility criteria and protocol details for a specific trial "
        "by its NCT ID. Use this when a trial from search_trials looks promising and "
        "you need the full inclusion/exclusion criteria text to assess patient fit. "
        "Do not call this for every search result — only for trials worth closer examination."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=["nct_id"],
        properties={
            "nct_id": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="NCT identifier, e.g. 'NCT05456256'.",
            ),
        },
    ),
)

_CONSULT_EXPERT_DECL = genai_types.FunctionDeclaration(
    name="consult_medical_expert",
    description=(
        "Ask a specialized medical AI (MedGemma 27B) a clinical question. "
        "Use this when you need domain expertise that you're uncertain about:\n"
        "- What condition terms or synonyms to search for a diagnosis\n"
        "- Whether a specific biomarker is relevant to the patient's condition\n"
        "- What comorbidities or clinical presentations are associated with the diagnosis\n"
        "- Whether a trial's intervention is appropriate for the patient's treatment line\n"
        "- Medical terminology clarification for eligibility criteria interpretation\n\n"
        "COST: Each call takes ~8-15 seconds. Use judiciously — 2-3 calls per patient "
        "is typical. Do NOT call this for every search result or minor question."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=["question"],
        properties={
            "question": genai_types.Schema(
                type=genai_types.Type.STRING,
                description=(
                    "The clinical question to ask. Be specific and include patient context. "
                    "Good: 'What condition terms should I search on CT.gov for a 61-year-old "
                    "male with epithelioid mesothelioma? Include related presentations and "
                    "broader categories.' "
                    "Bad: 'What is mesothelioma?'"
                ),
            ),
        },
    ),
)

PRESCREEN_TOOLS = genai_types.Tool(
    function_declarations=[
        _SEARCH_TRIALS_DECL,
        _GET_DETAILS_DECL,
        _CONSULT_EXPERT_DECL,
    ]
)

# ---------------------------------------------------------------------------
# 2. Tool executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Dispatches Gemini tool calls to CT.gov API and MedGemma expert.

    Three tools: search_trials, get_trial_details, consult_medical_expert.
    Keeps running totals for cost/latency tracking.
    """

    def __init__(self, ctgov: CTGovClient, medgemma: Any = None,
                 default_status: list[str] | None = None):
        self._ctgov = ctgov
        self._medgemma = medgemma
        self._default_status = default_status  # None = use ["RECRUITING"]
        self.medgemma_calls: int = 0
        self.medgemma_cost: float = 0.0
        self._search_cache: dict[tuple, tuple[dict, str]] = {}

    async def execute(self, tool_name: str, args: dict[str, Any]) -> tuple[Any, str]:
        """Execute a tool call. Returns (raw_result, summary_string).

        The raw_result is what Gemini gets back.
        The summary_string is stored in ToolCallRecord for human review.
        """
        if tool_name == "search_trials":
            return await self._search_trials(**args)
        if tool_name == "get_trial_details":
            return await self._get_trial_details(**args)
        if tool_name == "consult_medical_expert":
            return await self._consult_medical_expert(**args)
        raise ValueError(f"Unknown tool: {tool_name}")

    async def _search_trials(
        self,
        condition: str | None = None,
        intervention: str | None = None,
        eligibility_keywords: str | None = None,
        status: list[str] | None = None,
        phase: list[str] | None = None,
        location: str | None = None,
        min_age: str | None = None,
        max_age: str | None = None,
        sex: str | None = None,
        study_type: str | None = None,
        page_size: int = 50,
        **_ignored: Any,
    ) -> tuple[dict, str]:
        # Dedup: normalize and check cache
        cache_key = (
            (condition or "").strip().lower(),
            (intervention or "").strip().lower(),
            (eligibility_keywords or "").strip().lower(),
        )
        if cache_key in self._search_cache:
            cached_result, cached_summary = self._search_cache[cache_key]
            logger.info("search_dedup_cache_hit", condition=condition)
            return (
                {**cached_result, "note": "DUPLICATE SEARCH — you already searched this term. Use a DIFFERENT condition term."},
                f"(cached) {cached_summary}",
            )

        start = time.perf_counter()
        raw = await self._ctgov.search(
            condition=condition,
            intervention=intervention,
            eligibility_keywords=eligibility_keywords,
            status=status or self._default_status or ["RECRUITING"],
            phase=phase or None,
            location=location,
            sex=sex,
            min_age=min_age,
            max_age=max_age,
            study_type=study_type or None,
            page_size=min(int(page_size), 100),
        )
        latency = (time.perf_counter() - start) * 1000

        studies = parse_search_results(raw)
        summaries = [parse_study_summary(s) for s in studies]

        # Compact representation for Gemini — just what it needs to decide next steps
        compact = [
            {
                "nct_id": s["nct_id"],
                "brief_title": s["brief_title"],
                "phase": s["phase"],
                "conditions": s["conditions"][:3],
                "interventions": s["interventions"][:4],
                "status": s["status"],
                "sponsor": s["sponsor"],
                "enrollment": s["enrollment"],
            }
            for s in summaries
        ]

        total = raw.get("totalCount") or len(studies)
        summary = (
            f"Found {len(studies)} trials (API total: {total}), latency {latency:.0f}ms. "
            f"NCT IDs: {[s['nct_id'] for s in summaries]}"
        )
        logger.info("ctgov_search_complete", count=len(studies), query_condition=condition)

        result_dict = {"count": len(compact), "total_available": total, "trials": compact}
        self._search_cache[cache_key] = (result_dict, summary)
        return result_dict, summary

    async def _get_trial_details(self, nct_id: str, **_ignored: Any) -> tuple[dict, str]:
        start = time.perf_counter()
        raw = await self._ctgov.get_details(nct_id)
        latency = (time.perf_counter() - start) * 1000

        proto = raw.get("protocolSection", {})
        eligibility = proto.get("eligibilityModule", {})
        id_mod = proto.get("identificationModule", {})

        result = {
            "nct_id": nct_id,
            "title": id_mod.get("briefTitle", ""),
            "eligibility_criteria": _truncate_eligibility(
                eligibility.get("eligibilityCriteria", "")
            ),
            "minimum_age": eligibility.get("minimumAge", ""),
            "maximum_age": eligibility.get("maximumAge", ""),
            "sex": eligibility.get("sex", ""),
            "healthy_volunteers": eligibility.get("healthyVolunteers", ""),
        }
        brief = id_mod.get("briefTitle", "")
        summary = f"Fetched details for {nct_id} ({brief}), latency {latency:.0f}ms"
        logger.info("ctgov_details_fetched", nct_id=nct_id)
        return result, summary

    async def _consult_medical_expert(self, question: str, **_ignored: Any) -> tuple[dict, str]:
        """Call MedGemma 27B for on-demand clinical expertise."""
        if self._medgemma is None:
            return (
                {"error": "Medical expert not available. "
                 "Proceed with your own clinical knowledge."},
                "MedGemma unavailable — skipped",
            )

        prompt = (
            "You are a clinical expert advising a clinical trial search agent.\n"
            "The agent will search ClinicalTrials.gov using the condition terms you provide.\n\n"
            "IMPORTANT: You MUST end your response with a structured list in this exact format:\n"
            "SEARCH_TERMS:\n"
            "1. <term>\n"
            "2. <term>\n"
            "...\n\n"
            "Include ALL of these categories:\n"
            "- Exact diagnosis and synonyms\n"
            "- Histological/molecular subtypes\n"
            "- Broader parent categories (e.g., organ-level, system-level)\n"
            "- Related clinical presentations and symptoms the patient has\n"
            "- Related procedures the patient has undergone\n"
            "- Common comorbidity-associated trial categories\n\n"
            "Aim for 15-20 terms. More is better — the agent will search each one.\n\n"
            f"Question: {question}"
        )
        start = time.perf_counter()
        response = await self._medgemma.generate(prompt, max_tokens=1024)
        latency_ms = (time.perf_counter() - start) * 1000

        self.medgemma_calls += 1
        self.medgemma_cost += response.estimated_cost

        # Parse structured terms from response
        extracted_terms = _extract_search_terms(response.text)

        summary = f"MedGemma ({latency_ms:.0f}ms): {len(extracted_terms)} terms extracted"
        logger.info(
            "medgemma_expert_consulted",
            question=question[:100],
            latency_ms=f"{latency_ms:.0f}",
            terms_extracted=len(extracted_terms),
        )
        return {
            "answer": response.text,
            "search_terms": extracted_terms,
            "instruction": (
                "You MUST call search_trials once for EACH term in 'search_terms' above. "
                "Do not skip any. Do not repeat any. Check them off as you go."
            ),
        }, summary


def _extract_search_terms(text: str) -> list[str]:
    """Extract numbered search terms from MedGemma's response.

    Looks for a SEARCH_TERMS: section with numbered items.
    Falls back to extracting any numbered list items if section header not found.
    """
    terms: list[str] = []
    # Try to find SEARCH_TERMS section first — capture all lines after the header
    match = re.search(r"SEARCH_TERMS:\s*\n((?:[ \t]*(?:\d+[\.\)]|[-*]).+\n?)+)", text, re.IGNORECASE)
    if match:
        block = match.group(1)
    else:
        # Fallback: use the whole text
        block = text

    # Extract numbered items: "1. Term Name" or "- Term Name" or "* Term Name"
    for line in block.split("\n"):
        m = re.match(r"^\s*(?:\d+[\.\)]\s*\**|[-*]\s*\**)(.*?)(?:\**\s*[-:(].*)?$", line)
        if m:
            term = m.group(1).strip().strip("*").strip()
            if term and len(term) > 2 and len(term) < 100:
                terms.append(term)

    return terms
