"""Unit tests for the PRESCREEN module.

Tests cover:
  - CTGovClient: parameter mapping, response parsing
  - ToolExecutor: search_trials, get_trial_details, normalize_medical_terms dispatch
  - agent helpers: _format_key_facts, _describe_query, _build_candidates
  - run_prescreen_agent: agentic loop with mocked Gemini + CT.gov

Async functions are called via asyncio.run() per project convention.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trialmatch.prescreen.agent import (
    _build_candidates,
    _describe_query,
    _format_key_facts,
    run_prescreen_agent,
)
from trialmatch.prescreen.ctgov_client import CTGovClient, parse_search_results, parse_study_summary
from trialmatch.prescreen.schema import PresearchResult, TrialCandidate
from trialmatch.prescreen.tools import ToolExecutor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_search_response() -> dict:
    """Minimal CT.gov search API response with two studies."""
    return {
        "totalCount": 2,
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT11111111",
                        "briefTitle": "Trial Alpha",
                        "officialTitle": "A Phase 2 Study of Drug Alpha",
                    },
                    "statusModule": {"overallStatus": "RECRUITING"},
                    "conditionsModule": {"conditions": ["Lung Cancer", "NSCLC"]},
                    "armsInterventionsModule": {
                        "interventions": [{"name": "Drug Alpha"}, {"name": "Placebo"}]
                    },
                    "designModule": {
                        "phases": ["PHASE2"],
                        "studyType": "INTERVENTIONAL",
                        "enrollmentInfo": {"count": 120},
                    },
                    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor Corp"}},
                    "contactsLocationsModule": {"locations": [{}, {}, {}]},
                    "descriptionModule": {"briefSummary": "Phase 2 study."},
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT22222222",
                        "briefTitle": "Trial Beta",
                    },
                    "statusModule": {"overallStatus": "RECRUITING"},
                    "conditionsModule": {"conditions": ["Breast Cancer"]},
                    "armsInterventionsModule": {"interventions": []},
                    "designModule": {
                        "phases": ["PHASE3"],
                        "studyType": "INTERVENTIONAL",
                        "enrollmentInfo": {"count": 300},
                    },
                    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "BigPharma"}},
                    "contactsLocationsModule": {},
                    "descriptionModule": {},
                }
            },
        ],
    }


@pytest.fixture
def sample_details_response() -> dict:
    """Minimal CT.gov get-details API response."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT11111111",
                "briefTitle": "Trial Alpha",
            },
            "eligibilityModule": {
                "eligibilityCriteria": (
                    "Inclusion Criteria:\n- NSCLC\n- Never smoker\n\n"
                    "Exclusion Criteria:\n- Prior chemotherapy"
                ),
                "minimumAge": "18 Years",
                "maximumAge": "N/A",
                "sex": "ALL",
                "healthyVolunteers": "No",
            },
        }
    }


# ---------------------------------------------------------------------------
# CTGovClient parser tests (sync)
# ---------------------------------------------------------------------------


class TestCTGovClientParsers:
    def test_parse_search_results_extracts_studies(self, sample_search_response):
        studies = parse_search_results(sample_search_response)
        assert len(studies) == 2

    def test_parse_search_results_empty(self):
        assert parse_search_results({}) == []

    def test_parse_study_summary_fields(self, sample_search_response):
        study = sample_search_response["studies"][0]
        summary = parse_study_summary(study)

        assert summary["nct_id"] == "NCT11111111"
        assert summary["brief_title"] == "Trial Alpha"
        assert summary["status"] == "RECRUITING"
        assert "PHASE2" in summary["phase"]
        assert "Lung Cancer" in summary["conditions"]
        assert "Drug Alpha" in summary["interventions"]
        assert summary["sponsor"] == "Sponsor Corp"
        assert summary["enrollment"] == 120
        assert summary["locations_count"] == 3
        assert summary["study_type"] == "INTERVENTIONAL"

    def test_parse_study_summary_missing_fields(self, sample_search_response):
        """Missing optional fields should not raise — return empty/None defaults."""
        study = sample_search_response["studies"][1]
        summary = parse_study_summary(study)

        assert summary["nct_id"] == "NCT22222222"
        assert summary["locations_count"] is None
        assert summary["brief_summary"] == ""

    def test_parse_study_summary_interventions_list(self, sample_search_response):
        study = sample_search_response["studies"][0]
        summary = parse_study_summary(study)
        assert isinstance(summary["interventions"], list)
        assert len(summary["interventions"]) == 2


# ---------------------------------------------------------------------------
# CTGovClient HTTP tests (async via asyncio.run)
# ---------------------------------------------------------------------------


class TestCTGovClientHttp:
    def test_search_builds_correct_params(self, sample_search_response):
        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_search_response
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                result = await client.search(
                    condition="lung cancer",
                    intervention="pembrolizumab",
                    status=["RECRUITING"],
                    phase=["PHASE2", "PHASE3"],
                    page_size=10,
                )

            called_params = mock_get.call_args[1]["params"]
            assert called_params["query.cond"] == "lung cancer"
            assert called_params["query.intr"] == "pembrolizumab"
            assert called_params["filter.overallStatus"] == "RECRUITING"
            assert called_params["aggFilters"] == "phase:2,phase:3"
            assert called_params["pageSize"] == 10
            assert result == sample_search_response
            await client.aclose()

        asyncio.run(_run())

    def test_search_eligibility_keywords_uses_query_term(self):
        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"studies": []}
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                await client.search(eligibility_keywords="never smoker")

            called_params = mock_get.call_args[1]["params"]
            assert called_params["query.term"] == "never smoker"
            await client.aclose()

        asyncio.run(_run())

    def test_get_details_calls_correct_path(self, sample_details_response):
        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_details_response
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                result = await client.get_details("NCT11111111")

            called_path = mock_get.call_args[0][0]
            assert "NCT11111111" in called_path
            assert result == sample_details_response
            await client.aclose()

        asyncio.run(_run())

    def test_page_size_capped_at_100(self):
        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"studies": []}
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                await client.search(condition="cancer", page_size=200)

            called_params = mock_get.call_args[1]["params"]
            assert called_params["pageSize"] <= 100
            await client.aclose()

        asyncio.run(_run())

    def test_429_retry_updates_last_call_time(self):
        """After a 429 backoff sleep, _last_call_time must be reset so the
        subsequent _get call does not double-wait the full _MIN_INTERVAL again."""

        async def _run():
            client = CTGovClient(max_retries=2)

            resp_429 = MagicMock()
            resp_429.status_code = 429
            resp_429.raise_for_status = MagicMock()

            resp_200 = MagicMock()
            resp_200.status_code = 200
            resp_200.json.return_value = {"studies": []}
            resp_200.raise_for_status = MagicMock()

            call_count = 0

            async def mock_get(path, *, params):
                nonlocal call_count
                call_count += 1
                return resp_429 if call_count == 1 else resp_200

            with (
                patch.object(client._http, "get", side_effect=mock_get),
                patch("trialmatch.prescreen.ctgov_client.asyncio.sleep", new=AsyncMock()),
            ):
                await client.search(condition="cancer")
                # After 429 + sleep, _last_call_time should have been updated
                # (i.e. it should NOT still be the value set before the retry loop)
                assert client._last_call_time > 0

            await client.aclose()

        asyncio.run(_run())

    def test_phase_filter_uses_aggfilters_not_filter_phase(self):
        """Phase filtering must use aggFilters param (not the invalid filter.phase)."""

        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"studies": []}
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                await client.search(
                    condition="cancer", phase=["EARLY_PHASE1", "PHASE1", "PHASE3"]
                )

            called_params = mock_get.call_args[1]["params"]
            assert "filter.phase" not in called_params
            assert called_params["aggFilters"] == "phase:early1,phase:1,phase:3"
            await client.aclose()

        asyncio.run(_run())

    def test_location_maps_to_query_locn(self):
        """location param should map to query.locn in the API call."""

        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"studies": []}
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                await client.search(condition="cancer", location="Boston")

            called_params = mock_get.call_args[1]["params"]
            assert called_params["query.locn"] == "Boston"
            await client.aclose()

        asyncio.run(_run())

    def test_sex_filter_uses_aggfilters(self):
        """sex='FEMALE' should produce aggFilters containing 'sex:f'."""

        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"studies": []}
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                await client.search(condition="cancer", sex="FEMALE")

            called_params = mock_get.call_args[1]["params"]
            assert called_params["aggFilters"] == "sex:f"
            await client.aclose()

        asyncio.run(_run())

    def test_phase_and_sex_compose_in_aggfilters(self):
        """Phase and sex values should be comma-joined in a single aggFilters param."""

        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"studies": []}
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                await client.search(condition="cancer", phase=["PHASE2"], sex="MALE")

            called_params = mock_get.call_args[1]["params"]
            assert called_params["aggFilters"] == "phase:2,sex:m"
            await client.aclose()

        asyncio.run(_run())

    def test_age_filter_builds_advanced_query(self):
        """min_age and max_age should produce Essie AREA expressions in query.term."""

        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"studies": []}
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                await client.search(
                    condition="cancer", min_age="18 Years", max_age="75 Years"
                )

            called_params = mock_get.call_args[1]["params"]
            term = called_params["query.term"]
            assert "AREA[MinimumAge]RANGE[MIN, 18 Years]" in term
            assert "AREA[MaximumAge]RANGE[75 Years, MAX]" in term
            await client.aclose()

        asyncio.run(_run())

    def test_age_and_eligibility_keywords_compose(self):
        """Age Essie expressions should AND-compose with eligibility_keywords."""

        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"studies": []}
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                await client.search(
                    eligibility_keywords="never smoker", min_age="65 Years"
                )

            called_params = mock_get.call_args[1]["params"]
            term = called_params["query.term"]
            assert "never smoker" in term
            assert "AREA[MinimumAge]RANGE[MIN, 65 Years]" in term
            assert " AND " in term
            await client.aclose()

        asyncio.run(_run())

    def test_sex_all_omitted_from_aggfilters(self):
        """sex='ALL' should not add anything to aggFilters."""

        async def _run():
            client = CTGovClient()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"studies": []}
            mock_resp.raise_for_status = MagicMock()

            with patch.object(
                client._http, "get", new=AsyncMock(return_value=mock_resp)
            ) as mock_get:
                await client.search(condition="cancer", sex="ALL")

            called_params = mock_get.call_args[1]["params"]
            assert "aggFilters" not in called_params
            await client.aclose()

        asyncio.run(_run())

    def test_400_raises_value_error(self):
        """A 400 Bad Request should raise ValueError immediately (no retry)."""

        async def _run():
            client = CTGovClient(max_retries=3)
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_resp.text = "filter.phase is unknown parameter"
            mock_resp.raise_for_status = MagicMock()

            with (
                patch.object(client._http, "get", new=AsyncMock(return_value=mock_resp)),
                pytest.raises(ValueError, match="400 Bad Request"),
            ):
                await client.search(condition="cancer")
            await client.aclose()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# ToolExecutor tests (async via asyncio.run)
# ---------------------------------------------------------------------------


def _make_executor(ctgov_search_response=None):
    ctgov = AsyncMock(spec=CTGovClient)
    ctgov.search = AsyncMock(return_value=ctgov_search_response or {"studies": [], "totalCount": 0})
    ctgov.get_details = AsyncMock(
        return_value={
            "protocolSection": {
                "identificationModule": {"nctId": "NCT11111111", "briefTitle": "T"},
                "eligibilityModule": {"eligibilityCriteria": "Inclusion: NSCLC"},
            }
        }
    )

    medgemma = MagicMock()
    medgemma.generate = AsyncMock(
        return_value=MagicMock(
            text=json.dumps(
                {
                    "normalized": "EGFR",
                    "search_variants": ["EGFR L858R", "EGFR exon 21"],
                    "disambiguation": "EGFR gene, not eGFR renal function",
                    "avoid": ["eGFR"],
                }
            ),
            estimated_cost=0.0001,
        )
    )
    return ToolExecutor(ctgov=ctgov, medgemma=medgemma)


class TestToolExecutorDispatch:
    def test_search_trials_returns_compact_list(self, sample_search_response):
        executor = _make_executor(ctgov_search_response=sample_search_response)
        result, summary = asyncio.run(
            executor.execute("search_trials", {"condition": "lung cancer"})
        )

        assert result["count"] == 2
        assert len(result["trials"]) == 2
        assert result["trials"][0]["nct_id"] == "NCT11111111"
        assert "NCT11111111" in summary

    def test_search_trials_compact_fields(self, sample_search_response):
        executor = _make_executor(ctgov_search_response=sample_search_response)
        result, _ = asyncio.run(executor.execute("search_trials", {"condition": "cancer"}))

        trial = result["trials"][0]
        assert set(trial.keys()) >= {
            "nct_id",
            "brief_title",
            "phase",
            "conditions",
            "interventions",
            "status",
        }
        assert "eligibility_criteria" not in trial

    def test_get_trial_details_returns_eligibility(self):
        executor = _make_executor()
        result, summary = asyncio.run(
            executor.execute("get_trial_details", {"nct_id": "NCT11111111"})
        )

        assert result["nct_id"] == "NCT11111111"
        assert "eligibility_criteria" in result
        assert "NSCLC" in result["eligibility_criteria"]
        assert "NCT11111111" in summary

    @pytest.mark.skip(reason="normalize_medical_terms commented out — may revisit with better MedGemma prompts")
    def test_normalize_medical_terms_parses_json(self):
        executor = _make_executor()
        result, summary = asyncio.run(
            executor.execute(
                "normalize_medical_terms",
                {"raw_term": "EGFR L858R", "term_type": "biomarker", "patient_context": "NSCLC"},
            )
        )

        assert result["normalized"] == "EGFR"
        assert "EGFR L858R" in result["search_variants"]
        assert executor.medgemma_calls == 1
        assert "EGFR L858R" in summary

    @pytest.mark.skip(reason="normalize_medical_terms commented out — may revisit with better MedGemma prompts")
    def test_normalize_falls_back_on_invalid_json(self):
        ctgov = AsyncMock(spec=CTGovClient)
        medgemma = MagicMock()
        medgemma.generate = AsyncMock(
            return_value=MagicMock(text="This is not JSON at all", estimated_cost=0.0)
        )
        executor = ToolExecutor(ctgov=ctgov, medgemma=medgemma)
        result, _ = asyncio.run(
            executor.execute(
                "normalize_medical_terms",
                {"raw_term": "mystery term", "term_type": "condition"},
            )
        )

        assert result["normalized"] == "mystery term"
        assert result["search_variants"] == ["mystery term"]
        assert "raw_response" in result

    def test_unknown_tool_raises(self):
        executor = _make_executor()
        with pytest.raises(ValueError, match="Unknown tool"):
            asyncio.run(executor.execute("nonexistent_tool", {}))


# ---------------------------------------------------------------------------
# Agent helper function tests (sync)
# ---------------------------------------------------------------------------


class TestAgentHelpers:
    def test_format_key_facts_bullet_list(self):
        facts = {"diagnosis": "NSCLC", "mutations": ["EGFR L858R", "TP53"], "age": 62}
        text = _format_key_facts(facts)
        assert "diagnosis: NSCLC" in text
        assert "mutations: EGFR L858R, TP53" in text
        assert "age: 62" in text

    def test_format_key_facts_empty(self):
        text = _format_key_facts({})
        assert "No structured key facts" in text

    def test_format_key_facts_none_values_skipped(self):
        facts = {"diagnosis": "NSCLC", "stage": None}
        text = _format_key_facts(facts)
        assert "diagnosis: NSCLC" in text
        assert "stage" not in text

    def test_describe_query_all_params(self):
        label = _describe_query(
            {
                "condition": "lung cancer",
                "intervention": "pembrolizumab",
                "eligibility_keywords": "never smoker",
            }
        )
        assert "lung cancer" in label
        assert "pembrolizumab" in label
        assert "never smoker" in label

    def test_describe_query_condition_only(self):
        label = _describe_query({"condition": "breast cancer"})
        assert "breast cancer" in label
        assert "intr=" not in label

    def test_describe_query_empty(self):
        assert _describe_query({}) == "broad search"

    def test_build_candidates_deduplication(self):
        candidates = {
            "NCT00001": {"nct_id": "NCT00001", "brief_title": "Trial A", "status": "RECRUITING"},
            "NCT00002": {"nct_id": "NCT00002", "brief_title": "Trial B", "status": "RECRUITING"},
        }
        found_by = {
            "NCT00001": ["q1", "q2", "q3"],
            "NCT00002": ["q1"],
        }
        result = _build_candidates(candidates, found_by)

        assert len(result) == 2
        assert isinstance(result[0], TrialCandidate)
        assert result[0].nct_id == "NCT00001"
        assert len(result[0].found_by_queries) == 3

    def test_build_candidates_stable_sort_by_nct(self):
        """Ties broken by NCT ID for determinism."""
        candidates = {
            "NCT00002": {"nct_id": "NCT00002", "brief_title": "B", "status": "RECRUITING"},
            "NCT00001": {"nct_id": "NCT00001", "brief_title": "A", "status": "RECRUITING"},
        }
        found_by = {"NCT00001": ["q1"], "NCT00002": ["q1"]}
        result = _build_candidates(candidates, found_by)

        nct_ids = [c.nct_id for c in result]
        assert nct_ids == sorted(nct_ids)

    def test_build_candidates_missing_found_by(self):
        candidates = {"NCT99999": {"nct_id": "NCT99999", "status": "RECRUITING"}}
        result = _build_candidates(candidates, {})
        assert result[0].found_by_queries == []

    def test_build_candidates_handles_bad_data_gracefully(self):
        candidates = {"NCT00001": {"nct_id": "NCT00001", "status": "RECRUITING"}}
        result = _build_candidates(candidates, {})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# run_prescreen_agent integration-style tests (mocked Gemini, async via asyncio.run)
# ---------------------------------------------------------------------------


def _make_gemini_adapter(turns: list[dict]):
    """Build a mock GeminiAdapter that cycles through predefined response turns."""
    adapter = MagicMock()
    adapter._model = "gemini-3-pro-preview"
    adapter._client = MagicMock()

    responses = []
    for turn in turns:
        fc_parts = []
        for fc in turn.get("function_calls", []):
            mock_fc = MagicMock()
            mock_fc.name = fc["name"]
            mock_fc.args = fc["args"]
            part = MagicMock()
            part.function_call = mock_fc
            part.text = None
            fc_parts.append(part)

        text_parts = []
        if turn.get("text"):
            text_part = MagicMock()
            text_part.function_call = None
            text_part.text = turn["text"]
            text_parts.append(text_part)

        content = MagicMock()
        content.parts = fc_parts + text_parts

        candidate = MagicMock()
        candidate.content = content

        usage = MagicMock()
        usage.prompt_token_count = turn.get("input_tokens", 100)
        usage.candidates_token_count = turn.get("output_tokens", 50)

        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata = usage
        responses.append(response)

    call_iter = iter(responses)

    def side_effect(*args, **kwargs):
        return next(call_iter)

    adapter._client.models.generate_content.side_effect = side_effect
    return adapter


class TestRunPresearchAgent:
    def test_agent_no_tool_calls(self):
        """Gemini immediately returns text with no tool calls → empty candidates."""
        adapter = _make_gemini_adapter(
            [{"text": "No relevant trials found for this patient.", "function_calls": []}]
        )

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.aclose = AsyncMock()
        mock_medgemma = MagicMock()

        with patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov):
            result = asyncio.run(
                run_prescreen_agent(
                    patient_note="62yo female, advanced NSCLC",
                    key_facts={"diagnosis": "NSCLC"},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=mock_medgemma,
                    topic_id="test-001",
                )
            )

        assert isinstance(result, PresearchResult)
        assert result.topic_id == "test-001"
        assert result.ingest_source == "gold"
        assert result.candidates == []
        assert result.total_api_calls == 0
        assert "No relevant trials" in result.agent_reasoning

    def test_trace_callback_records_prompt_and_tool_result(self, sample_search_response):
        """Trace callback captures prescreen prompt, model turns, and raw tool outputs."""
        adapter = _make_gemini_adapter(
            [
                {"function_calls": [{"name": "search_trials", "args": {"condition": "NSCLC"}}]},
                {"text": "Done.", "function_calls": []},
            ]
        )

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.search = AsyncMock(return_value=sample_search_response)
        mock_ctgov.aclose = AsyncMock()
        events: list[tuple[str, dict]] = []

        with patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov):
            asyncio.run(
                run_prescreen_agent(
                    patient_note="NSCLC patient",
                    key_facts={"diagnosis": "NSCLC"},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=None,
                    topic_id="trace-test-001",
                    trace_callback=lambda event, payload: events.append((event, payload)),
                )
            )

        event_names = [name for name, _payload in events]
        assert "prescreen_prompt" in event_names
        assert "gemini_turn" in event_names
        assert "tool_call_result" in event_names

        tool_payload = next(payload for name, payload in events if name == "tool_call_result")
        assert tool_payload["tool_name"] == "search_trials"
        assert tool_payload["result"]["count"] == 2

    def test_guidance_required_fails_when_medgemma_fails(self):
        """When guidance is required, MedGemma failure should abort PRESCREEN."""
        adapter = _make_gemini_adapter([{"text": "unused", "function_calls": []}])

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.aclose = AsyncMock()
        mock_medgemma = MagicMock()
        mock_medgemma.generate = AsyncMock(side_effect=RuntimeError("guidance endpoint down"))

        with (
            patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov),
            pytest.raises(
                RuntimeError,
                match="MedGemma clinical guidance required but failed",
            ),
        ):
            asyncio.run(
                run_prescreen_agent(
                    patient_note="NSCLC patient",
                    key_facts={"diagnosis": "NSCLC"},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=mock_medgemma,
                    require_clinical_guidance=True,
                    topic_id="test-guidance-required",
                )
            )

        mock_ctgov.aclose.assert_awaited_once()

    def test_guidance_optional_continues_when_medgemma_fails(self):
        """When guidance is optional, MedGemma failure should not abort PRESCREEN."""
        adapter = _make_gemini_adapter(
            [{"text": "Completed with optional guidance fallback.", "function_calls": []}]
        )

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.aclose = AsyncMock()
        mock_medgemma = MagicMock()
        mock_medgemma.generate = AsyncMock(side_effect=RuntimeError("guidance endpoint down"))

        with patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov):
            result = asyncio.run(
                run_prescreen_agent(
                    patient_note="NSCLC patient",
                    key_facts={"diagnosis": "NSCLC"},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=mock_medgemma,
                    require_clinical_guidance=False,
                    topic_id="test-guidance-optional",
                )
            )

        assert isinstance(result, PresearchResult)
        assert result.total_api_calls == 0
        assert "optional guidance fallback" in result.agent_reasoning
        mock_ctgov.aclose.assert_awaited_once()

    def test_agent_single_search_tool_call(self, sample_search_response):
        """Gemini calls search_trials once, then returns summary."""
        adapter = _make_gemini_adapter(
            [
                {"function_calls": [{"name": "search_trials", "args": {"condition": "NSCLC"}}]},
                {"text": "Found 2 trials: NCT11111111 and NCT22222222.", "function_calls": []},
            ]
        )

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.search = AsyncMock(return_value=sample_search_response)
        mock_ctgov.aclose = AsyncMock()
        mock_medgemma = MagicMock()

        with patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov):
            result = asyncio.run(
                run_prescreen_agent(
                    patient_note="NSCLC patient",
                    key_facts={},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=mock_medgemma,
                    topic_id="test-002",
                )
            )

        assert result.total_unique_nct_ids == 2
        assert result.total_api_calls == 1
        assert len(result.candidates) == 2
        assert len(result.tool_call_trace) == 1
        assert result.tool_call_trace[0].tool_name == "search_trials"
        assert result.tool_call_trace[0].error is None
        assert "Found 2 trials" in result.agent_reasoning

    def test_agent_deduplicates_across_searches(self, sample_search_response):
        """Same NCT ID from two searches → one candidate with two found_by_queries."""
        adapter = _make_gemini_adapter(
            [
                {"function_calls": [{"name": "search_trials", "args": {"condition": "NSCLC"}}]},
                {
                    "function_calls": [
                        {"name": "search_trials", "args": {"eligibility_keywords": "never smoker"}}
                    ]
                },
                {"text": "Search complete.", "function_calls": []},
            ]
        )

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.search = AsyncMock(return_value=sample_search_response)
        mock_ctgov.aclose = AsyncMock()
        mock_medgemma = MagicMock()

        with patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov):
            result = asyncio.run(
                run_prescreen_agent(
                    patient_note="NSCLC never smoker",
                    key_facts={},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=mock_medgemma,
                )
            )

        nct_candidate = next(c for c in result.candidates if c.nct_id == "NCT11111111")
        assert len(nct_candidate.found_by_queries) == 2

    def test_agent_cost_tracking(self):
        """Token counts and cost should be aggregated across all Gemini turns."""
        adapter = _make_gemini_adapter(
            [{"text": "Done", "function_calls": [], "input_tokens": 500, "output_tokens": 100}]
        )

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.aclose = AsyncMock()
        mock_medgemma = MagicMock()

        with patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov):
            result = asyncio.run(
                run_prescreen_agent(
                    patient_note="Patient",
                    key_facts={},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=mock_medgemma,
                )
            )

        assert result.gemini_input_tokens == 500
        assert result.gemini_output_tokens == 100
        assert result.gemini_estimated_cost > 0
        assert result.latency_ms > 0

    def test_agent_respects_max_tool_calls(self, sample_search_response):
        """Agent stops executing tools when max_tool_calls budget is reached."""
        many_turns = [
            {"function_calls": [{"name": "search_trials", "args": {"condition": "cancer"}}]}
            for _ in range(8)
        ]
        many_turns.append({"text": "Stopped.", "function_calls": []})
        adapter = _make_gemini_adapter(many_turns)

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.search = AsyncMock(return_value=sample_search_response)
        mock_ctgov.aclose = AsyncMock()
        mock_medgemma = MagicMock()

        with patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov):
            result = asyncio.run(
                run_prescreen_agent(
                    patient_note="Patient",
                    key_facts={},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=mock_medgemma,
                    max_tool_calls=3,
                )
            )

        assert result.total_api_calls <= 3

    def test_budget_guard_sends_function_response_not_text(self, sample_search_response):
        """Budget guard must close pending function_calls with function_response parts,
        not a plain text user message — otherwise the conversation structure is invalid."""
        from google.genai import types as genai_types

        many_turns = [
            {"function_calls": [{"name": "search_trials", "args": {"condition": "cancer"}}]}
            for _ in range(5)
        ]
        many_turns.append({"text": "Budget reached, stopping.", "function_calls": []})
        adapter = _make_gemini_adapter(many_turns)

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.search = AsyncMock(return_value=sample_search_response)
        mock_ctgov.aclose = AsyncMock()
        mock_medgemma = MagicMock()

        # Capture a shallow copy of the contents list on each generate_content call.
        # Shallow copy is essential: the agent mutates the same list in-place by appending
        # model responses after each call, so a reference would be stale by assertion time.
        captured_contents = []
        original_side_effect = adapter._client.models.generate_content.side_effect

        def capturing_side_effect(*args, **kwargs):
            contents = kwargs.get("contents", args[1] if len(args) > 1 else [])
            captured_contents.append(list(contents))  # shallow copy to freeze the list
            return original_side_effect(*args, **kwargs)

        adapter._client.models.generate_content.side_effect = capturing_side_effect

        with patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov):
            result = asyncio.run(
                run_prescreen_agent(
                    patient_note="Patient",
                    key_facts={},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=mock_medgemma,
                    max_tool_calls=2,
                )
            )

        assert result.total_api_calls <= 2

        # The budget guard Content is injected right before the next generate_content call.
        # In the last such call, the final element of captured_contents is our injected
        # function_response Content (a real genai_types.Content, not a MagicMock).
        # Verify it has function_response parts and NOT a plain text part.
        assert len(captured_contents) >= 3, "Expected at least 3 generate_content calls"

        # Find a Content with function_response parts — that's our budget guard.
        # Skip the initial prompt Content (which is also a real genai_types.Content but has text).
        budget_guard_found = False
        for call_contents in captured_contents:
            for item in call_contents:
                if not isinstance(item, genai_types.Content):
                    continue
                # Check if ANY part has function_response set
                has_fr = any(
                    getattr(p, "function_response", None) is not None for p in item.parts
                )
                if has_fr:
                    budget_guard_found = True
                    for part in item.parts:
                        assert part.function_response is not None, (
                            "Budget guard part must have function_response set"
                        )
                        assert not part.text, (
                            "Budget guard must not inject plain text — use function_response"
                        )
                    break
            if budget_guard_found:
                break

        assert budget_guard_found, "Budget guard Content with function_response was never found"

    def test_agent_tool_error_recorded_not_raised(self):
        """A CT.gov API error should be recorded in trace but not crash the agent."""
        adapter = _make_gemini_adapter(
            [
                {"function_calls": [{"name": "search_trials", "args": {"condition": "cancer"}}]},
                {"text": "Completed despite error.", "function_calls": []},
            ]
        )

        mock_ctgov = MagicMock(spec=CTGovClient)
        mock_ctgov.search = AsyncMock(side_effect=RuntimeError("CT.gov 503 error"))
        mock_ctgov.aclose = AsyncMock()
        mock_medgemma = MagicMock()

        with patch("trialmatch.prescreen.agent.CTGovClient", return_value=mock_ctgov):
            result = asyncio.run(
                run_prescreen_agent(
                    patient_note="Patient",
                    key_facts={},
                    ingest_source="gold",
                    gemini_adapter=adapter,
                    medgemma_adapter=mock_medgemma,
                )
            )

        assert len(result.tool_call_trace) == 1
        assert result.tool_call_trace[0].error is not None
        assert "503" in result.tool_call_trace[0].error
        assert result.candidates == []
