"""Tool definitions and dispatch for the PRESCREEN agent.

This module owns:
  1. The Gemini function-calling schemas (what the model sees).
  2. The async tool executor (what actually runs when the model calls a tool).

The two tools exposed to Gemini are:
  - search_trials       → CT.gov API v2 /studies search
  - get_trial_details   → CT.gov API v2 /studies/{nct_id}

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
                    "Filter by study type. Default: INTERVENTIONAL (clinical trials "
                    "patients can enroll in). Use OBSERVATIONAL only if specifically needed."
                ),
            ),
            "page_size": genai_types.Schema(
                type=genai_types.Type.INTEGER,
                description="Number of results to return. Default 20. Max 50.",
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

# --- Commented out: normalize_medical_terms adds ~25s/call via MedGemma with
# --- near-zero value (echoes input unchanged). May revisit with better prompts.
# _NORMALIZE_TERMS_DECL = genai_types.FunctionDeclaration(
#     name="normalize_medical_terms",
#     description=(
#         "Ask MedGemma (a specialized medical language model) to produce "
#         "clinically correct search variants for a medical term. "
#         ...
#     ),
#     parameters=genai_types.Schema(...),
# )

PRESCREEN_TOOLS = genai_types.Tool(
    function_declarations=[
        _SEARCH_TRIALS_DECL,
        _GET_DETAILS_DECL,
    ]
)

# ---------------------------------------------------------------------------
# 2. Tool executor
# ---------------------------------------------------------------------------

# --- Commented out: MedGemma normalize prompts (see _NORMALIZE_TERMS_DECL above)
# MEDGEMMA_NORMALIZE_SYSTEM = (
#     "You are a medical terminology expert specializing in clinical trial search.\n"
#     ...
# )
# MEDGEMMA_NORMALIZE_USER = """Term: {raw_term} ..."""


class ToolExecutor:
    """Dispatches Gemini tool calls to CT.gov API.

    Keeps running totals for cost/latency tracking.
    """

    def __init__(self, ctgov: CTGovClient, medgemma: Any = None):
        self._ctgov = ctgov
        # medgemma kept as optional param for backward compat; not used currently
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
        # normalize_medical_terms commented out — see bottom of file
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
        page_size: int = 20,
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
            study_type=study_type or "INTERVENTIONAL",
            page_size=min(int(page_size), 50),
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
            "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
            "minimum_age": eligibility.get("minimumAge", ""),
            "maximum_age": eligibility.get("maximumAge", ""),
            "sex": eligibility.get("sex", ""),
            "healthy_volunteers": eligibility.get("healthyVolunteers", ""),
        }
        brief = id_mod.get("briefTitle", "")
        summary = f"Fetched details for {nct_id} ({brief}), latency {latency:.0f}ms"
        logger.info("ctgov_details_fetched", nct_id=nct_id)
        return result, summary

    # --- Commented out: normalize_medical_terms adds ~25s/call with near-zero value.
    # --- May revisit with better MedGemma prompts for search term generation.
    # async def _normalize_medical_terms(
    #     self, raw_term: str, term_type: str, patient_context: str = "", **_ignored: Any,
    # ) -> tuple[dict, str]:
    #     prompt = MEDGEMMA_NORMALIZE_USER.format(...)
    #     response = await self._medgemma.generate(full_prompt, max_tokens=512)
    #     self.medgemma_calls += 1
    #     self.medgemma_cost += response.estimated_cost
    #     ... (see git history for full implementation)
    #     return result, summary
