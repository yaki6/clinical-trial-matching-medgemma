"""Tool definitions and dispatch for the PRESCREEN agent.

This module owns:
  1. The Gemini function-calling schemas (what the model sees).
  2. The async tool executor (what actually runs when the model calls a tool).

The three tools exposed to Gemini are:
  - search_trials       → CT.gov API v2 /studies search
  - get_trial_details   → CT.gov API v2 /studies/{nct_id}
  - normalize_medical_terms → MedGemma: correct nomenclature + CT.gov search variants

Design note: tool schemas are purposefully kept narrow.  Gemini should
not be handed every CT.gov query parameter — just the ones that produce
reliable results (validated against the live API).
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog
from google.genai import types as genai_types

from trialmatch.prescreen.ctgov_client import CTGovClient, parse_search_results, parse_study_summary

if TYPE_CHECKING:
    from trialmatch.models.base import ModelAdapter

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

_NORMALIZE_TERMS_DECL = genai_types.FunctionDeclaration(
    name="normalize_medical_terms",
    description=(
        "Ask MedGemma (a specialized medical language model) to produce "
        "clinically correct search variants for a medical term. "
        "Use this BEFORE searching when the patient profile contains:\n"
        "  - A biomarker or mutation (e.g. 'EGFR L858R') — get HUGO/HGVS notation "
        "    and synonym variants ranked from most-specific to broadest.\n"
        "  - A drug name (e.g. 'Tagrisso') — get INN + brand name variants.\n"
        "  - A diagnosis (e.g. 'lung cancer') — get MeSH-preferred terms and common "
        "    CT.gov-indexed synonyms.\n"
        "  - A clinical phenotype (e.g. 'non-smoker') — get the exact phrasing used "
        "    in eligibility criteria text.\n"
        "Returns a JSON list of search_variants ordered from most specific to broadest, "
        "and disambiguation notes (e.g. EGFR gene vs eGFR renal function)."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=["raw_term", "term_type"],
        properties={
            "raw_term": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="The term as extracted from the patient profile, e.g. 'EGFR L858R'.",
            ),
            "term_type": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["biomarker", "condition", "drug", "phenotype"],
                description="Category of the term — determines normalization rules applied.",
            ),
            "patient_context": genai_types.Schema(
                type=genai_types.Type.STRING,
                description=(
                    "Brief patient context to help MedGemma disambiguate. "
                    "E.g. 'NSCLC patient with EGFR mutation, never smoker, post-TKI'."
                ),
            ),
        },
    ),
)

PRESCREEN_TOOLS = genai_types.Tool(
    function_declarations=[
        _SEARCH_TRIALS_DECL,
        _GET_DETAILS_DECL,
        _NORMALIZE_TERMS_DECL,
    ]
)

# ---------------------------------------------------------------------------
# 2. Tool executor
# ---------------------------------------------------------------------------

MEDGEMMA_NORMALIZE_SYSTEM = (
    "You are a medical terminology expert specializing in clinical trial search.\n"
    "Given a medical term and its type, return a JSON object with:\n"
    '- "normalized": the canonical form (HUGO gene name, INN drug name, MeSH disease term, etc.)\n'
    '- "search_variants": list of strings, most-to-least specific, for CT.gov eligibility search\n'
    '- "disambiguation": important notes (e.g. EGFR gene vs eGFR renal, drug brand vs generic)\n'
    '- "avoid": terms that look similar but mean something different (false-positive risk)\n'
    "\nReturn only valid JSON, no other text."
)

MEDGEMMA_NORMALIZE_USER = """Term: {raw_term}
Type: {term_type}
Patient context: {patient_context}

Return JSON with normalized, search_variants (list, most→least specific), disambiguation, avoid."""


class ToolExecutor:
    """Dispatches Gemini tool calls to CT.gov API or MedGemma.

    Keeps running totals for cost/latency tracking.
    """

    def __init__(self, ctgov: CTGovClient, medgemma: ModelAdapter):
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
        if tool_name == "normalize_medical_terms":
            return await self._normalize_medical_terms(**args)
        raise ValueError(f"Unknown tool: {tool_name}")

    async def _search_trials(
        self,
        condition: str | None = None,
        intervention: str | None = None,
        eligibility_keywords: str | None = None,
        status: list[str] | None = None,
        phase: list[str] | None = None,
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

    async def _normalize_medical_terms(
        self,
        raw_term: str,
        term_type: str,
        patient_context: str = "",
        **_ignored: Any,
    ) -> tuple[dict, str]:
        prompt = MEDGEMMA_NORMALIZE_USER.format(
            raw_term=raw_term,
            term_type=term_type,
            patient_context=patient_context or "not specified",
        )
        full_prompt = f"{MEDGEMMA_NORMALIZE_SYSTEM}\n\n{prompt}"

        start = time.perf_counter()
        response = await self._medgemma.generate(full_prompt, max_tokens=512)
        latency = (time.perf_counter() - start) * 1000

        self.medgemma_calls += 1
        self.medgemma_cost += response.estimated_cost

        # Try to parse MedGemma's JSON response
        try:
            # Strip markdown fences if present
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            # Return raw text if JSON parsing fails
            result = {
                "normalized": raw_term,
                "search_variants": [raw_term],
                "disambiguation": "",
                "avoid": [],
                "raw_response": response.text,
            }

        summary = (
            f"MedGemma normalized '{raw_term}' ({term_type}): "
            f"variants={result.get('search_variants', [])}, latency {latency:.0f}ms"
        )
        logger.info(
            "medgemma_normalize_complete", term=raw_term, variants=result.get("search_variants")
        )
        return result, summary
