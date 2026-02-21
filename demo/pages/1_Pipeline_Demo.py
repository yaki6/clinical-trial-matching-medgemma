"""Pipeline Demo page -- main interactive demo for TrialMatch.

Wires the real PRESCREEN agent and VALIDATE evaluator into a Streamlit UI
with agent trace visualization and per-criterion eligibility results.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Add demo root to path for component imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache_manager import (
    list_cached_patients,
    load_validate_results,
    save_ingest_result,
    save_prescreen_result,
    save_validate_results,
)
from components.patient_card import render_patient_card
from components.pipeline_viewer import (
    render_ingest_step,
    render_prescreen_placeholder,
    render_validate_placeholder,
)

# ---------------------------------------------------------------------------
# Profile loading -- try adapter first, fall back to raw JSON
# ---------------------------------------------------------------------------
PROFILES_PATH = Path(__file__).resolve().parents[2] / "nsclc_trial_profiles.json"
CACHED_RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "cached_runs"

try:
    from trialmatch.ingest.profile_adapter import adapt_profile_for_prescreen, load_profiles
except ImportError:

    def load_profiles(path: str | Path | None = None) -> list[dict]:
        """Fallback loader when profile_adapter is not yet available."""
        p = Path(path) if path else PROFILES_PATH
        with open(p) as f:
            data = json.load(f)
        return data.get("profiles", [])

    def adapt_profile_for_prescreen(profile: dict) -> tuple[str, dict]:
        """Fallback adapter: flatten key_facts list-of-objects to dict."""
        patient_note = profile.get("profile_text", "")
        key_facts_list = profile.get("key_facts", [])
        result = {}
        for kf in key_facts_list:
            field = kf.get("field", "unknown")
            result[field] = kf.get("value")
        return patient_note, result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from Streamlit's sync context."""
    return asyncio.run(coro)


def _init_adapters():
    """Initialize model adapters from env vars. Returns (gemini, medgemma) or (None, None)."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    errors = []
    if not api_key:
        errors.append(
            "GOOGLE_API_KEY not set. Required for PRESCREEN agent (Gemini orchestration)."
        )
    if not hf_token:
        errors.append("HF_TOKEN not set. Required for MedGemma normalization.")

    if errors:
        for e in errors:
            st.error(e)
        return None, None

    from trialmatch.models.gemini import GeminiAdapter
    from trialmatch.models.medgemma import MedGemmaAdapter

    gemini = GeminiAdapter(api_key=api_key)
    medgemma = MedGemmaAdapter(hf_token=hf_token)
    return gemini, medgemma


def _get_validate_adapter(model_choice: str):
    """Create a model adapter for VALIDATE based on sidebar selection."""
    if model_choice == "Gemini 3 Pro":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            st.error("GOOGLE_API_KEY not set.")
            return None
        from trialmatch.models.gemini import GeminiAdapter

        return GeminiAdapter(api_key=api_key)
    elif model_choice == "MedGemma 27B":
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            st.error("HF_TOKEN not set.")
            return None
        from trialmatch.models.medgemma import MedGemmaAdapter

        endpoint_27b = os.environ.get(
            "MEDGEMMA_27B_ENDPOINT_URL",
            "https://wu5nclwms3ctrwd1.us-east-1.aws.endpoints.huggingface.cloud",
        )
        return MedGemmaAdapter(
            hf_token=hf_token,
            endpoint_url=endpoint_27b,
            model_name="medgemma-27b-text-it",
        )
    else:  # MedGemma 4B
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            st.error("HF_TOKEN not set.")
            return None
        from trialmatch.models.medgemma import MedGemmaAdapter

        return MedGemmaAdapter(hf_token=hf_token)


def parse_eligibility_criteria(criteria_text: str) -> list[dict]:
    """Split eligibility criteria text into individual criteria with types."""
    criteria = []
    current_type = "inclusion"

    for line in criteria_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "inclusion" in line.lower() and "criteria" in line.lower():
            current_type = "inclusion"
            continue
        if "exclusion" in line.lower() and "criteria" in line.lower():
            current_type = "exclusion"
            continue
        # Strip bullet markers
        if line.startswith(("*", "-", "~")):
            line = line.lstrip("*-~ ").strip()
        if len(line) > 10:
            criteria.append({"text": line, "type": current_type})

    return criteria


def _compute_trial_verdict(results: list[tuple[dict, object]]) -> str:
    """Compute trial-level eligibility verdict from criterion results.

    ELIGIBLE: all inclusion MET, no exclusion NOT_MET
    EXCLUDED: any exclusion NOT_MET or critical inclusion NOT_MET
    UNCERTAIN: any UNKNOWN on critical criteria
    """
    has_unknown = False
    for criterion, cr in results:
        verdict_val = cr.verdict.value
        ctype = criterion["type"]

        if ctype == "exclusion" and verdict_val == "NOT_MET":
            # NOT_MET on exclusion means patient does NOT have the exclusion trait = good
            pass
        elif ctype == "exclusion" and verdict_val == "MET":
            # MET on exclusion means patient HAS the exclusion trait = excluded
            return "EXCLUDED"
        elif ctype == "inclusion" and verdict_val == "NOT_MET":
            return "EXCLUDED"
        elif verdict_val == "UNKNOWN":
            has_unknown = True

    if has_unknown:
        return "UNCERTAIN"
    return "ELIGIBLE"


VERDICT_BADGES = {
    "ELIGIBLE": ":green[ELIGIBLE]",
    "EXCLUDED": ":red[EXCLUDED]",
    "UNCERTAIN": ":orange[UNCERTAIN]",
}

CRITERION_ICONS = {
    "MET": "🟢",
    "NOT_MET": "🔴",
    "UNKNOWN": "🟡",
}


# ---------------------------------------------------------------------------
# Load profiles once per session
# ---------------------------------------------------------------------------
@st.cache_data
def _load_profiles() -> list[dict]:
    return load_profiles(str(PROFILES_PATH))


profiles = _load_profiles()

# Build a mapping of topic_id -> profile for the selector
profile_map = {p["topic_id"]: p for p in profiles}
topic_ids = list(profile_map.keys())

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("TrialMatch Pipeline")

selected_topic = st.sidebar.selectbox(
    "Select Patient",
    topic_ids,
    format_func=lambda tid: f"{tid}",
)

if selected_topic:
    preview_text = profile_map[selected_topic].get("profile_text", "")[:200]
    st.sidebar.caption(f"**{selected_topic}**: {preview_text}...")

pipeline_mode = st.sidebar.radio("Pipeline Mode", ["live", "cached"])

validate_model = st.sidebar.selectbox(
    "VALIDATE Model",
    ["Gemini 3 Pro", "MedGemma 27B", "MedGemma 4B"],
    help="Model used for criterion-level eligibility evaluation",
)

max_trials = st.sidebar.slider("Max trials to evaluate", 1, 10, 3)
max_criteria = st.sidebar.slider("Max criteria per trial", 5, 30, 10)

# Show cached status indicator
cached_patients = list_cached_patients()
if cached_patients:
    label = ", ".join(cached_patients[:5])
    suffix = "..." if len(cached_patients) > 5 else ""
    st.sidebar.caption(f"Cached: {len(cached_patients)} patients ({label}{suffix})")

run_button = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Pipeline Demo")

if not selected_topic:
    st.info("Select a patient from the sidebar to begin.")
    st.stop()

profile = profile_map[selected_topic]

# -- Patient card --
render_patient_card(profile)

st.divider()

# -- INGEST step --
patient_note, key_facts = adapt_profile_for_prescreen(profile)
render_ingest_step(key_facts)

# ---------------------------------------------------------------------------
# PRESCREEN step
# ---------------------------------------------------------------------------
if run_button and pipeline_mode == "live":
    gemini_adapter, medgemma_adapter = _init_adapters()

    if gemini_adapter is None or medgemma_adapter is None:
        st.error("Cannot run pipeline: missing API credentials. Set env vars and reload.")
        st.stop()

    from trialmatch.prescreen.agent import run_prescreen_agent

    with st.status("PRESCREEN: Searching ClinicalTrials.gov...", expanded=True) as status:
        st.write("Running agentic search with Gemini 3 Pro orchestration...")

        try:
            prescreen_result = _run_async(
                run_prescreen_agent(
                    patient_note=patient_note,
                    key_facts=key_facts,
                    ingest_source="gold",
                    gemini_adapter=gemini_adapter,
                    medgemma_adapter=medgemma_adapter,
                    topic_id=selected_topic,
                )
            )
        except Exception as exc:
            st.error(f"PRESCREEN failed: {exc}")
            status.update(label="PRESCREEN failed", state="error")
            st.stop()

        # -- Agent trace visualization (P0 for Special Award) --
        st.write(f"**{len(prescreen_result.tool_call_trace)} tool calls made**")
        for tc in prescreen_result.tool_call_trace:
            icon = "🔧" if not tc.error else "⚠️"
            with st.expander(f"{icon} {tc.tool_name} (call #{tc.call_index})", expanded=False):
                st.json(tc.args)
                st.caption(tc.result_summary)
                if tc.error:
                    st.error(f"Error: {tc.error}")
                st.caption(f"Latency: {tc.latency_ms:.0f}ms")

        # -- Agent reasoning summary --
        if prescreen_result.agent_reasoning:
            with st.expander("Agent Reasoning Summary", expanded=False):
                st.markdown(prescreen_result.agent_reasoning)

        # -- Candidate trials table --
        n_candidates = len(prescreen_result.candidates)
        st.write(f"**Found {n_candidates} candidate trials**")

        if prescreen_result.candidates:
            trial_data = []
            for c in prescreen_result.candidates:
                trial_data.append(
                    {
                        "NCT ID": c.nct_id,
                        "Title": c.brief_title or c.title,
                        "Phase": ", ".join(c.phase) if c.phase else "N/A",
                        "Status": c.status,
                        "Conditions": ", ".join(c.conditions[:3]),
                        "Relevance": len(c.found_by_queries),
                    }
                )
            st.dataframe(trial_data, use_container_width=True)

        # -- Cost summary --
        st.caption(
            f"Cost: Gemini ${prescreen_result.gemini_estimated_cost:.4f} "
            f"+ MedGemma ${prescreen_result.medgemma_estimated_cost:.4f} "
            f"| Latency: {prescreen_result.latency_ms / 1000:.1f}s"
        )

        status.update(label=f"PRESCREEN complete: {n_candidates} trials found", state="complete")

    # Store for VALIDATE
    st.session_state["prescreen_result"] = prescreen_result

    # Auto-save to cache for replay
    save_prescreen_result(selected_topic, prescreen_result)
    save_ingest_result(selected_topic, patient_note, key_facts)
    st.caption(f"Results cached for replay: {selected_topic}")

elif run_button and pipeline_mode == "cached":
    # Load cached PRESCREEN result if available
    cached_path = CACHED_RUNS_DIR / selected_topic / "prescreen_result.json"
    if cached_path.exists():
        with open(cached_path) as f:
            cached_data = json.load(f)

        from trialmatch.prescreen.schema import PresearchResult

        prescreen_result = PresearchResult(**cached_data)
        st.session_state["prescreen_result"] = prescreen_result

        with st.expander("Step 2: PRESCREEN -- Trial Search (cached)", expanded=True):
            st.caption(f"Loaded cached result: {len(prescreen_result.candidates)} trials")

            # Show tool trace from cache
            st.write(f"**{len(prescreen_result.tool_call_trace)} tool calls recorded**")
            for tc in prescreen_result.tool_call_trace:
                icon = "🔧" if not tc.error else "⚠️"
                with st.expander(f"{icon} {tc.tool_name} (call #{tc.call_index})", expanded=False):
                    st.json(tc.args)
                    st.caption(tc.result_summary)
                    if tc.error:
                        st.error(f"Error: {tc.error}")
                    st.caption(f"Latency: {tc.latency_ms:.0f}ms")

            if prescreen_result.candidates:
                trial_data = []
                for c in prescreen_result.candidates:
                    trial_data.append(
                        {
                            "NCT ID": c.nct_id,
                            "Title": c.brief_title or c.title,
                            "Phase": ", ".join(c.phase) if c.phase else "N/A",
                            "Status": c.status,
                            "Conditions": ", ".join(c.conditions[:3]),
                            "Relevance": len(c.found_by_queries),
                        }
                    )
                st.dataframe(trial_data, use_container_width=True)
    else:
        with st.expander("Step 2: PRESCREEN -- Trial Search (cached)", expanded=True):
            st.warning(
                f"No cached run found for {selected_topic}. "
                f"Expected at: {cached_path}\n\n"
                "Run in LIVE mode first to generate a cached result, "
                "or add cached data to the demo/data/cached_runs/ directory."
            )
else:
    render_prescreen_placeholder()

# ---------------------------------------------------------------------------
# VALIDATE step
# ---------------------------------------------------------------------------
prescreen_result = st.session_state.get("prescreen_result")

if run_button and pipeline_mode == "cached" and prescreen_result and prescreen_result.candidates:
    # -- Cached VALIDATE path --
    cached_validate = load_validate_results(selected_topic)
    if cached_validate:
        with st.expander("Step 3: VALIDATE -- Eligibility Check (cached)", expanded=True):
            for nct_id, data in cached_validate.items():
                st.write(f"**{nct_id}**")
                for c in data["criteria"]:
                    icon = CRITERION_ICONS.get(c["verdict"], "---")
                    st.markdown(
                        f"{icon} **{c['verdict']}** [{c['type'].upper()}] -- {c['text'][:100]}"
                    )
                verdict = data["verdict"]
                st.markdown(f"**Trial verdict: {VERDICT_BADGES.get(verdict, verdict)}**")
                st.divider()

        st.session_state["trial_verdicts"] = {
            nct: d["verdict"] for nct, d in cached_validate.items()
        }
        # Store empty validate_results (detailed per-criterion objects not available from cache)
        st.session_state["validate_results"] = {}
    else:
        with st.expander("Step 3: VALIDATE -- Eligibility Check (cached)", expanded=True):
            st.warning(
                f"No cached VALIDATE results for {selected_topic}. "
                "Run in LIVE mode first to generate cached results."
            )

elif run_button and pipeline_mode == "live" and prescreen_result and prescreen_result.candidates:
    from trialmatch.validate.evaluator import evaluate_criterion

    validate_adapter = _get_validate_adapter(validate_model)

    if validate_adapter is None:
        st.error("Cannot run VALIDATE: missing API credentials.")
        st.stop()

    # Collect eligibility criteria from candidates that had get_trial_details called.
    # The agent merges eligibility_criteria into candidates_by_nct, but TrialCandidate
    # Pydantic model doesn't persist it. We need to check the raw tool call results
    # for get_trial_details calls to find criteria text.
    #
    # Strategy: look through tool_call_trace for get_trial_details calls and match
    # NCT IDs. If no criteria available, we note it.
    trial_criteria_map: dict[str, str] = {}
    for tc in prescreen_result.tool_call_trace:
        if tc.tool_name == "get_trial_details" and not tc.error:
            # The args contain nct_id
            nct_id = tc.args.get("nct_id", "")
            # The result_summary has the NCT ID but not the full criteria text.
            # We don't have the raw result in the trace. For live runs, the data
            # was merged into candidates_by_nct (but lost in Pydantic serialization).
            # This is a limitation -- for now we note it.
            if nct_id:
                trial_criteria_map[nct_id] = ""  # placeholder

    # For the demo, we'll try to fetch criteria via CT.gov API for top candidates
    # that don't have cached criteria
    top_candidates = prescreen_result.candidates[:max_trials]

    validate_results: dict[str, list[tuple[dict, object]]] = {}
    trial_verdicts: dict[str, str] = {}

    with st.status("VALIDATE: Evaluating eligibility criteria...", expanded=True) as status:
        from trialmatch.prescreen.ctgov_client import CTGovClient

        ctgov = CTGovClient()
        try:
            for trial in top_candidates:
                st.write(f"**Evaluating: {trial.nct_id}** -- {trial.brief_title or trial.title}")

                # Fetch eligibility criteria if not already available
                criteria_text = trial_criteria_map.get(trial.nct_id, "")
                if not criteria_text:
                    try:
                        raw_details = _run_async(ctgov.get_details(trial.nct_id))
                        proto = raw_details.get("protocolSection", {})
                        eligibility = proto.get("eligibilityModule", {})
                        criteria_text = eligibility.get("eligibilityCriteria", "")
                    except Exception as exc:
                        st.warning(f"Could not fetch criteria for {trial.nct_id}: {exc}")
                        criteria_text = ""

                if not criteria_text:
                    st.warning(f"No eligibility criteria available for {trial.nct_id}")
                    continue

                criteria = parse_eligibility_criteria(criteria_text)
                if not criteria:
                    st.warning(f"Could not parse criteria for {trial.nct_id}")
                    continue

                st.caption(
                    f"Evaluating {min(len(criteria), max_criteria)} of {len(criteria)} criteria..."
                )

                results = []
                for criterion in criteria[:max_criteria]:
                    try:
                        cr = _run_async(
                            evaluate_criterion(
                                patient_note=patient_note,
                                criterion_text=criterion["text"],
                                criterion_type=criterion["type"],
                                adapter=validate_adapter,
                            )
                        )
                        results.append((criterion, cr))
                    except Exception as exc:
                        st.warning(f"Criterion eval failed: {exc}")
                        continue

                if results:
                    validate_results[trial.nct_id] = results

                    # Display per-criterion results
                    for criterion, cr in results:
                        icon = CRITERION_ICONS.get(cr.verdict.value, "⚪")
                        ctype_label = criterion["type"].upper()
                        text_preview = criterion["text"][:100]
                        st.markdown(
                            f"{icon} **{cr.verdict.value}** [{ctype_label}] -- {text_preview}"
                        )
                        with st.expander("Reasoning", expanded=False):
                            st.write(cr.reasoning)

                    # Compute trial-level verdict
                    verdict = _compute_trial_verdict(results)
                    trial_verdicts[trial.nct_id] = verdict
                    st.markdown(f"**Trial verdict: {VERDICT_BADGES.get(verdict, verdict)}**")

                st.divider()

            status.update(label="VALIDATE complete", state="complete")
        finally:
            _run_async(ctgov.aclose())

    st.session_state["validate_results"] = validate_results
    st.session_state["trial_verdicts"] = trial_verdicts

    # Auto-save validate results to cache for replay
    validate_cache: dict[str, dict] = {}
    for nct_id, results in validate_results.items():
        validate_cache[nct_id] = {
            "verdict": trial_verdicts.get(nct_id, "UNCERTAIN"),
            "criteria": [
                {
                    "text": criterion["text"],
                    "type": criterion["type"],
                    "verdict": cr.verdict.value,
                    "reasoning": cr.reasoning,
                    "evidence_sentences": cr.evidence_sentences,
                }
                for criterion, cr in results
            ],
        }
    save_validate_results(selected_topic, validate_cache)
    st.caption("VALIDATE results cached for replay")

elif run_button and prescreen_result and not prescreen_result.candidates:
    with st.expander("Step 3: VALIDATE -- Eligibility Check", expanded=True):
        st.warning("No candidate trials found by PRESCREEN. Nothing to validate.")
elif not run_button:
    render_validate_placeholder()

# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Results")

trial_verdicts = st.session_state.get("trial_verdicts", {})
validate_results = st.session_state.get("validate_results", {})
prescreen_for_results = st.session_state.get("prescreen_result")

cached_validate_data = load_validate_results(selected_topic) if pipeline_mode == "cached" else None

if trial_verdicts and prescreen_for_results:
    # Build ranked results table
    results_data = []
    for trial in prescreen_for_results.candidates:
        if trial.nct_id in trial_verdicts:
            verdict = trial_verdicts[trial.nct_id]

            # Handle both live (list-of-tuples) and cached (dict) formats
            if validate_results.get(trial.nct_id):
                criteria_results = validate_results[trial.nct_id]
                met_count = sum(1 for _, cr in criteria_results if cr.verdict.value == "MET")
                total_count = len(criteria_results)
            elif cached_validate_data and trial.nct_id in cached_validate_data:
                cached_criteria = cached_validate_data[trial.nct_id].get("criteria", [])
                met_count = sum(1 for c in cached_criteria if c["verdict"] == "MET")
                total_count = len(cached_criteria)
            else:
                met_count = 0
                total_count = 0

            results_data.append(
                {
                    "NCT ID": trial.nct_id,
                    "Title": trial.brief_title or trial.title,
                    "Verdict": verdict,
                    "Criteria Met": f"{met_count}/{total_count}",
                    "Link": f"https://clinicaltrials.gov/study/{trial.nct_id}",
                }
            )

    if results_data:
        # Sort: ELIGIBLE first, then UNCERTAIN, then EXCLUDED
        verdict_order = {"ELIGIBLE": 0, "UNCERTAIN": 1, "EXCLUDED": 2}
        results_data.sort(key=lambda r: verdict_order.get(r["Verdict"], 3))

        for r in results_data:
            badge = VERDICT_BADGES.get(r["Verdict"], r["Verdict"])
            st.markdown(
                f"**{badge}** | [{r['NCT ID']}]({r['Link']}) | "
                f"{r['Title'][:80]} | Criteria: {r['Criteria Met']}"
            )

        st.divider()
        st.caption(
            f"Evaluated {len(results_data)} trials with {validate_model}. "
            f"Pipeline mode: {pipeline_mode}."
        )
elif run_button:
    st.info("No results to display. Check pipeline output above for details.")
else:
    st.caption("Run the pipeline to see matching results.")
