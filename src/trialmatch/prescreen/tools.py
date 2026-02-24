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
            "min_age": genai_types.Schema(
                type=genai_types.Type.STRING,
                description=(
                    "Patient's age as lower bound. Format: 'X Years'. "
                    "Finds trials accepting patients this age or older. "
                    "Example: '65 Years'."
                ),
            ),
            "max_age": genai_types.Schema(
                type=genai_types.Type.STRING,
                description=(
                    "Patient's age as upper bound. Format: 'X Years'. "
                    "Finds trials accepting patients this age or younger. "
                    "Example: '75 Years'."
                ),
            ),
            "sex": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["MALE", "FEMALE", "ALL"],
                description="Patient's sex for eligibility filtering. Default ALL.",
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

    def __init__(self, ctgov: CTGovClient, medgemma: Any = None):
        self._ctgov = ctgov
        self._medgemma = medgemma
        self.medgemma_calls: int = 0
        self.medgemma_cost: float = 0.0

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
        start = time.perf_counter()
        raw = await self._ctgov.search(
            condition=condition,
            intervention=intervention,
            eligibility_keywords=eligibility_keywords,
            status=status or ["RECRUITING"],
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
        return {"count": len(compact), "total_available": total, "trials": compact}, summary

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
            "Answer concisely and actionably. Focus on specific medical terms, "
            "synonyms, and CT.gov condition vocabulary.\n\n"
            f"Question: {question}"
        )
        start = time.perf_counter()
        response = await self._medgemma.generate(prompt, max_tokens=1024)
        latency_ms = (time.perf_counter() - start) * 1000

        self.medgemma_calls += 1
        self.medgemma_cost += response.estimated_cost

        summary = f"MedGemma ({latency_ms:.0f}ms): {response.text[:150]}..."
        logger.info(
            "medgemma_expert_consulted",
            question=question[:100],
            latency_ms=f"{latency_ms:.0f}",
        )
        return {"answer": response.text}, summary
