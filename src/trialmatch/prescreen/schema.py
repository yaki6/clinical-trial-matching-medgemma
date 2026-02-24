"""Domain models for the PRESCREEN component.

PRESCREEN converts a patient profile into a set of trial candidates
by running an agentic search against the ClinicalTrials.gov API v2.

These models are disease-agnostic — they work for any patient profile,
not just NSCLC.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    """Trace record for a single tool invocation by the PRESCREEN agent."""

    call_index: int
    tool_name: str  # "search_trials" | "get_trial_details" | "normalize_medical_terms"
    args: dict[str, Any]
    result_summary: str  # Short human-readable summary of the result
    result_count: int = 0  # Number of trials returned (0 for non-search tools)
    latency_ms: float = 0.0
    error: str | None = None  # Set if the tool call failed


class TrialCandidate(BaseModel):
    """A single clinical trial returned by the CT.gov API.

    Disease-agnostic — populated directly from API response fields.
    """

    nct_id: str
    title: str
    brief_title: str = ""
    status: str  # "RECRUITING" | "NOT_YET_RECRUITING" | ...
    phase: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    sponsor: str = ""
    enrollment: int | None = None
    start_date: str | None = None
    primary_completion_date: str | None = None
    locations_count: int | None = None
    study_type: str = ""
    found_by_queries: list[str] = Field(default_factory=list)
    # Which agent queries discovered this trial — for ranking/attribution


class PresearchResult(BaseModel):
    """Complete output of the PRESCREEN agent for one patient.

    Contains the ranked trial candidate list plus full agent trace for
    reproducibility and cost accounting.
    """

    topic_id: str
    ingest_source: str  # "gold" | "model_medgemma" | "model_gemini"
    candidates: list[TrialCandidate]  # Deduplicated, ranked by found_by_queries count
    agent_reasoning: str  # Final model text summarising the search strategy used
    tool_call_trace: list[ToolCallRecord]
    total_api_calls: int
    total_unique_nct_ids: int
    gemini_input_tokens: int = 0
    gemini_output_tokens: int = 0
    gemini_estimated_cost: float = 0.0
    medgemma_calls: int = 0
    medgemma_estimated_cost: float = 0.0
    latency_ms: float = 0.0
    medgemma_guidance_raw: str = ""
    medgemma_condition_terms: list[str] = Field(default_factory=list)
    medgemma_eligibility_keywords: list[str] = Field(default_factory=list)
    condition_terms_searched: list[str] = Field(default_factory=list)
