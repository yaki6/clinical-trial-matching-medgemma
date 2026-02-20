"""Async ClinicalTrials.gov API v2 client.

Wraps the public REST API at https://clinicaltrials.gov/api/v2/
with rate limiting (40 req/min), retry on transient errors, and
pagination support.

This client is intentionally thin — it returns raw dicts from the
API. Schema parsing happens in tools.py.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import structlog

logger = structlog.get_logger()

CTGOV_BASE = "https://clinicaltrials.gov/api/v2"
RATE_LIMIT_RPM = 40
_MIN_INTERVAL = 60.0 / RATE_LIMIT_RPM  # seconds between requests


class CTGovClient:
    """Async HTTP client for ClinicalTrials.gov API v2.

    Instantiate once per agent run and reuse — it tracks rate-limit
    state across all calls.
    """

    def __init__(self, timeout_seconds: float = 30.0, max_retries: int = 3):
        self._http = httpx.AsyncClient(
            base_url=CTGOV_BASE,
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
        )
        self._last_call_time: float = 0.0
        self._max_retries = max_retries

    async def _get(self, path: str, params: dict) -> dict:
        """Rate-limited GET with retry on 429 / 5xx."""
        # Enforce rate limit
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < _MIN_INTERVAL:
            await asyncio.sleep(_MIN_INTERVAL - elapsed)
        self._last_call_time = time.monotonic()

        for attempt in range(self._max_retries):
            try:
                resp = await self._http.get(path, params=params)
                if resp.status_code == 429:
                    wait = 2.0**attempt
                    logger.warning("ctgov_rate_limited", wait=wait, attempt=attempt)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1.5**attempt)
                    continue
                msg = f"CT.gov API unavailable after {self._max_retries} retries: {exc}"
                raise RuntimeError(msg) from exc

        raise RuntimeError("CT.gov API: max retries exceeded")

    async def search(
        self,
        *,
        condition: str | None = None,
        intervention: str | None = None,
        eligibility_keywords: str | None = None,
        status: list[str] | None = None,
        phase: list[str] | None = None,
        advanced_query: str | None = None,
        page_size: int = 20,
        page_token: str | None = None,
    ) -> dict:
        """Search for studies. Returns raw API response dict.

        At least one of condition, intervention, eligibility_keywords, or
        advanced_query must be provided.
        """
        params: dict = {"pageSize": min(page_size, 100), "format": "json"}

        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        if eligibility_keywords:
            params["query.term"] = eligibility_keywords
        if advanced_query:
            # Essie syntax — supports AREA[FieldName] expressions
            existing_term = params.get("query.term", "")
            if existing_term:
                params["query.term"] = f"{existing_term} AND ({advanced_query})"
            else:
                params["query.term"] = advanced_query
        if status:
            params["filter.overallStatus"] = ",".join(status)
        if phase:
            params["filter.phase"] = ",".join(phase)
        if page_token:
            params["pageToken"] = page_token

        logger.debug("ctgov_search", params={k: v for k, v in params.items() if k != "format"})
        return await self._get("/studies", params)

    async def get_details(self, nct_id: str) -> dict:
        """Fetch full study record for a single NCT ID."""
        logger.debug("ctgov_get_details", nct_id=nct_id)
        return await self._get(f"/studies/{nct_id}", {"format": "json"})

    async def aclose(self) -> None:
        await self._http.aclose()


def parse_search_results(raw: dict) -> list[dict]:
    """Extract the list of study summaries from a search API response."""
    return raw.get("studies", [])


def parse_study_summary(study: dict) -> dict:
    """Flatten a study search result into a simple dict."""
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    desc_mod = proto.get("descriptionModule", {})
    cond_mod = proto.get("conditionsModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    design_mod = proto.get("designModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

    interventions = [i.get("name", "") for i in arms_mod.get("interventions", [])]

    phases = design_mod.get("phases", [])

    loc_count = None
    # location count lives in contactsLocationsModule
    contacts_mod = proto.get("contactsLocationsModule", {})
    if "locations" in contacts_mod:
        loc_count = len(contacts_mod["locations"])

    return {
        "nct_id": id_mod.get("nctId", ""),
        "title": id_mod.get("officialTitle", id_mod.get("briefTitle", "")),
        "brief_title": id_mod.get("briefTitle", ""),
        "status": status_mod.get("overallStatus", ""),
        "phase": phases,
        "conditions": cond_mod.get("conditions", []),
        "interventions": interventions,
        "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
        "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
        "start_date": status_mod.get("startDateStruct", {}).get("date"),
        "primary_completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date"),
        "locations_count": loc_count,
        "study_type": design_mod.get("studyType", ""),
        "brief_summary": desc_mod.get("briefSummary", ""),
    }
