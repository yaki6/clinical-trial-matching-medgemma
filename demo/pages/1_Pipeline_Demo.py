"""Pipeline Demo page -- main interactive demo for TrialMatch.

Wires the real PRESCREEN agent and VALIDATE evaluator into a Streamlit UI
with agent trace visualization and per-criterion eligibility results.

Supports two modes:
- **Patient mode** (default): simplified, patient-friendly UX with disclaimers
- **Dev mode**: full pipeline traces, token counts, cost breakdowns, all controls
  Activate via ?dev=1 query param or the Settings toggle in the sidebar.
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
from components.patient_card import extract_friendly_label, render_patient_card
from components.pipeline_viewer import (
    render_ingest_step,
    render_prescreen_placeholder,
    render_validate_placeholder,
)
from components.results_summary import render_results_summary
from components.trial_card import render_trial_card

# ---------------------------------------------------------------------------
# Dev mode feature flag
# ---------------------------------------------------------------------------


def _is_dev_mode() -> bool:
    """Check if dev mode is active via query param or session state."""
    if st.query_params.get("dev") == "1":
        st.session_state["dev_mode"] = True
    return st.session_state.get("dev_mode", False)


DEV_MODE = _is_dev_mode()

# ---------------------------------------------------------------------------
# Medical disclaimer text
# ---------------------------------------------------------------------------
_MEDICAL_DISCLAIMER = (
    "This is an AI-assisted screening tool for informational purposes only. "
    "It does not constitute medical advice. Always consult with your healthcare "
    "provider and the trial's research team before making decisions about "
    "clinical trial participation."
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


def _friendly_patient_label(profile: dict) -> str:
    """Build patient-friendly label from profile data.

    Delegates to the shared extract_friendly_label() in patient_card.py.
    """
    tid = profile.get("topic_id", "Unknown")
    suffix = extract_friendly_label(profile)  # e.g. " (43F, Lung Adenocarcinoma)"
    return f"{tid}{suffix}" if suffix else tid


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


def _get_validate_adapters(mode: str):
    """Create model adapter(s) for VALIDATE based on sidebar selection.

    Returns (reasoning_adapter, labeling_adapter) for two-stage mode,
    or (adapter, None) for single-stage mode.
    """
    if mode == "Two-Stage (MedGemma \u2192 Gemini)":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        hf_token = os.environ.get("HF_TOKEN", "")
        errors = []
        if not api_key:
            errors.append("GOOGLE_API_KEY not set (needed for Gemini labeling).")
        if not hf_token:
            errors.append("HF_TOKEN not set (needed for MedGemma reasoning).")
        if errors:
            for e in errors:
                st.error(e)
            return None, None

        from trialmatch.models.gemini import GeminiAdapter
        from trialmatch.models.medgemma import MedGemmaAdapter

        medgemma = MedGemmaAdapter(hf_token=hf_token)
        gemini = GeminiAdapter(api_key=api_key)
        return medgemma, gemini  # (reasoning, labeling)

    elif mode == "Gemini 3 Pro (single)":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            st.error("GOOGLE_API_KEY not set.")
            return None, None
        from trialmatch.models.gemini import GeminiAdapter

        return GeminiAdapter(api_key=api_key), None

    else:  # MedGemma 4B (single)
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            st.error("HF_TOKEN not set.")
            return None, None
        from trialmatch.models.medgemma import MedGemmaAdapter

        return MedGemmaAdapter(hf_token=hf_token), None


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
    "MET": "\U0001f7e2",
    "NOT_MET": "\U0001f534",
    "UNKNOWN": "\U0001f7e1",
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
# Handle force-live from patient mode (no cache available)
# ---------------------------------------------------------------------------
if st.session_state.get("force_live"):
    pipeline_mode = "live"
    st.session_state["force_live"] = False
    run_button = True  # auto-trigger the pipeline
else:
    pipeline_mode = None
    run_button = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("TrialMatch Pipeline")

if DEV_MODE:
    # ----- Dev mode sidebar: ALL controls -----
    selected_topic = st.sidebar.selectbox(
        "Select Patient",
        topic_ids,
        format_func=lambda tid: f"{tid}",
    )

    if selected_topic:
        preview_text = profile_map[selected_topic].get("profile_text", "")[:200]
        st.sidebar.caption(f"**{selected_topic}**: {preview_text}...")

    if pipeline_mode is None:
        pipeline_mode = st.sidebar.radio("Pipeline Mode", ["live", "cached"])

    validate_mode = st.sidebar.selectbox(
        "VALIDATE Mode",
        [
            "Two-Stage (MedGemma \u2192 Gemini)",
            "Gemini 3 Pro (single)",
            "MedGemma 4B (single)",
        ],
        help=(
            "Two-stage uses MedGemma for medical reasoning "
            "+ Gemini for label assignment (80% vs 75%)"
        ),
    )

    max_trials = st.sidebar.slider("Max trials to evaluate", 1, 10, 3)
    max_criteria = st.sidebar.slider("Max criteria per trial", 5, 30, 10)

    # Show cached status indicator
    cached_patients = list_cached_patients()
    if cached_patients:
        label = ", ".join(cached_patients[:5])
        suffix = "..." if len(cached_patients) > 5 else ""
        st.sidebar.caption(f"Cached: {len(cached_patients)} patients ({label}{suffix})")

    if not run_button:
        run_button = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)

else:
    # ----- Patient mode sidebar: simplified -----
    selected_topic = st.sidebar.selectbox(
        "Select Patient",
        topic_ids,
        format_func=lambda tid: _friendly_patient_label(profile_map[tid]),
    )

    if selected_topic:
        preview_text = profile_map[selected_topic].get("profile_text", "")[:200]
        st.sidebar.caption(f"**{selected_topic}**: {preview_text}...")

    # Set patient-mode defaults (hidden from sidebar)
    if pipeline_mode is None:
        pipeline_mode = "cached"
    validate_mode = "Two-Stage (MedGemma \u2192 Gemini)"
    max_trials = 3
    max_criteria = 10

    if not run_button:
        run_button = st.sidebar.button(
            "Search for Trials", type="primary", use_container_width=True
        )

# Settings expander at bottom of sidebar (always visible)
with st.sidebar.expander("Settings", expanded=False, icon=":material/settings:"):
    dev_toggle = st.checkbox(
        "Developer mode",
        value=DEV_MODE,
        help="Show full pipeline traces, token counts, and costs",
    )
    if dev_toggle != DEV_MODE:
        st.session_state["dev_mode"] = dev_toggle
        st.rerun()

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if DEV_MODE:
    st.title("Pipeline Demo")
else:
    st.title("Find Clinical Trials")
    st.caption(_MEDICAL_DISCLAIMER)

if not selected_topic:
    st.info("Select a patient from the sidebar to begin.")
    st.stop()

profile = profile_map[selected_topic]

# -- Patient card --
render_patient_card(profile, dev_mode=DEV_MODE)

st.divider()

# -- INGEST step --
patient_note, key_facts = adapt_profile_for_prescreen(profile)
render_ingest_step(key_facts, dev_mode=DEV_MODE)

# ---------------------------------------------------------------------------
# PRESCREEN step
# ---------------------------------------------------------------------------
if run_button and pipeline_mode == "live":
    gemini_adapter, medgemma_adapter = _init_adapters()

    if gemini_adapter is None:
        st.error("Cannot run pipeline: missing GOOGLE_API_KEY. Set env vars and reload.")
        st.stop()

    from trialmatch.prescreen.agent import run_prescreen_agent

    if DEV_MODE:
        # ---- Dev mode: full agent trace streaming ----
        # Container for live-streaming agent tool calls
        prescreen_status = st.status("PRESCREEN: Searching ClinicalTrials.gov...", expanded=True)
        prescreen_log = prescreen_status.container()
        prescreen_log.write("Running agentic search with Gemini Flash orchestration...")

        # Live-streaming callbacks -- called from within the agent loop
        _tool_counter = {"n": 0}

        def _on_tool_call(tc):
            """Stream each tool call to Streamlit as it completes."""
            _tool_counter["n"] += 1
            icon = "\U0001f527" if not tc.error else "\u26a0\ufe0f"
            label = (
                f"{icon} **{tc.tool_name}** (call #{tc.call_index}) \u2014 {tc.latency_ms:.0f}ms"
            )
            if tc.result_count:
                label += f" \u2014 {tc.result_count} results"
            prescreen_log.write(label)
            if tc.result_summary:
                prescreen_log.caption(tc.result_summary[:200])
            if tc.error:
                prescreen_log.error(f"Error: {tc.error}")

        def _on_agent_text(text):
            """Stream agent reasoning text as it arrives."""
            if text and len(text.strip()) > 20:
                prescreen_log.info(f"Agent: {text[:300]}...")

        with prescreen_status:
            try:
                prescreen_result = _run_async(
                    run_prescreen_agent(
                        patient_note=patient_note,
                        key_facts=key_facts,
                        ingest_source="gold",
                        gemini_adapter=gemini_adapter,
                        topic_id=selected_topic,
                        on_tool_call=_on_tool_call,
                        on_agent_text=_on_agent_text,
                    )
                )
            except Exception as exc:
                st.error(f"PRESCREEN failed: {exc}")
                prescreen_status.update(label="PRESCREEN failed", state="error")
                st.stop()

            # -- Final agent trace (collapsed details) --
            prescreen_log.write(f"**{len(prescreen_result.tool_call_trace)} tool calls completed**")
            for tc in prescreen_result.tool_call_trace:
                icon = "\U0001f527" if not tc.error else "\u26a0\ufe0f"
                with prescreen_log.expander(
                    f"{icon} {tc.tool_name} (call #{tc.call_index})", expanded=False
                ):
                    st.json(tc.args)
                    st.caption(tc.result_summary)
                    if tc.error:
                        st.error(f"Error: {tc.error}")
                    st.caption(f"Latency: {tc.latency_ms:.0f}ms")

            # -- Agent reasoning summary --
            if prescreen_result.agent_reasoning:
                with prescreen_log.expander("Agent Reasoning Summary", expanded=False):
                    st.markdown(prescreen_result.agent_reasoning)

            # -- Candidate trials table --
            n_candidates = len(prescreen_result.candidates)
            prescreen_log.write(f"**Found {n_candidates} candidate trials**")

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
                prescreen_log.dataframe(trial_data, use_container_width=True)

            # -- Cost summary --
            prescreen_log.caption(
                f"Cost: Gemini ${prescreen_result.gemini_estimated_cost:.4f} "
                f"+ MedGemma ${prescreen_result.medgemma_estimated_cost:.4f} "
                f"| Latency: {prescreen_result.latency_ms / 1000:.1f}s"
            )

            prescreen_status.update(
                label=f"PRESCREEN complete: {n_candidates} trials found", state="complete"
            )

    else:
        # ---- Patient mode: simplified live search ----
        with st.spinner("Searching for matching clinical trials..."):
            try:
                prescreen_result = _run_async(
                    run_prescreen_agent(
                        patient_note=patient_note,
                        key_facts=key_facts,
                        ingest_source="gold",
                        gemini_adapter=gemini_adapter,
                        topic_id=selected_topic,
                    )
                )
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                st.stop()

        n_candidates = len(prescreen_result.candidates)
        st.success(f"Found {n_candidates} clinical trials that may be relevant to your condition.")

    # Store for VALIDATE
    st.session_state["prescreen_result"] = prescreen_result

    # Auto-save to cache for replay
    save_prescreen_result(selected_topic, prescreen_result)
    save_ingest_result(selected_topic, patient_note, key_facts)
    if DEV_MODE:
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

        if DEV_MODE:
            # ---- Dev mode: full cached prescreen display ----
            with st.expander("Step 2: PRESCREEN -- Trial Search (cached)", expanded=True):
                st.caption(f"Loaded cached result: {len(prescreen_result.candidates)} trials")

                # Show tool trace from cache
                st.write(f"**{len(prescreen_result.tool_call_trace)} tool calls recorded**")
                for tc in prescreen_result.tool_call_trace:
                    icon = "\U0001f527" if not tc.error else "\u26a0\ufe0f"
                    exp_label = f"{icon} {tc.tool_name} (call #{tc.call_index})"
                    with st.expander(exp_label, expanded=False):
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
            # ---- Patient mode: simple cached result summary ----
            n_candidates = len(prescreen_result.candidates)
            st.success(
                f"Found {n_candidates} clinical trials that may be relevant to your condition."
            )

    else:
        if DEV_MODE:
            with st.expander("Step 2: PRESCREEN -- Trial Search (cached)", expanded=True):
                st.warning(
                    f"No cached run found for {selected_topic}. "
                    f"Expected at: {cached_path}\n\n"
                    "Run in LIVE mode first to generate a cached result, "
                    "or add cached data to the demo/data/cached_runs/ directory."
                )
        else:
            # ---- Patient mode: no cache, offer live run ----
            st.warning("Pre-computed results are not yet available for this patient.")
            if st.button("Run live search (may take 1-2 minutes)"):
                st.session_state["force_live"] = True
                st.rerun()
else:
    render_prescreen_placeholder(dev_mode=DEV_MODE)

# ---------------------------------------------------------------------------
# VALIDATE step
# ---------------------------------------------------------------------------
prescreen_result = st.session_state.get("prescreen_result")

if run_button and pipeline_mode == "cached" and prescreen_result and prescreen_result.candidates:
    # -- Cached VALIDATE path --
    cached_validate = load_validate_results(selected_topic)
    if cached_validate:
        if DEV_MODE:
            # ---- Dev mode: full cached validate display ----
            with st.expander("Step 3: VALIDATE -- Eligibility Check (cached)", expanded=True):
                cached_mode = None
                for nct_id, data in cached_validate.items():
                    cached_mode = data.get("mode", "single")
                    st.write(f"**{nct_id}**")
                    for c in data["criteria"]:
                        icon = CRITERION_ICONS.get(c["verdict"], "---")
                        st.markdown(
                            f"{icon} **{c['verdict']}** [{c['type'].upper()}] -- {c['text'][:100]}"
                        )
                        if c.get("stage1_reasoning"):
                            with st.expander("Stage 1: MedGemma Medical Reasoning", expanded=False):
                                st.write(c["stage1_reasoning"])
                            with st.expander("Stage 2: Gemini Label Assignment", expanded=False):
                                st.write(c.get("reasoning", ""))
                        elif c.get("reasoning"):
                            with st.expander("Reasoning", expanded=False):
                                st.write(c["reasoning"])
                    verdict = data["verdict"]
                    st.markdown(f"**Trial verdict: {VERDICT_BADGES.get(verdict, verdict)}**")
                    st.divider()
                if cached_mode:
                    mode_display = (
                        "Two-Stage (MedGemma \u2192 Gemini)"
                        if cached_mode == "two_stage"
                        else "Single-Stage"
                    )
                    st.caption(f"Evaluation mode: {mode_display}")
        else:
            # ---- Patient mode: render trial cards from cache ----
            # (trial cards are rendered in the Results panel below)
            pass

        st.session_state["trial_verdicts"] = {
            nct: d["verdict"] for nct, d in cached_validate.items()
        }
        # Store cached validate data for patient-mode rendering
        st.session_state["validate_results"] = {}
        st.session_state["cached_validate_data"] = cached_validate
    else:
        if DEV_MODE:
            with st.expander("Step 3: VALIDATE -- Eligibility Check (cached)", expanded=True):
                st.warning(
                    f"No cached VALIDATE results for {selected_topic}. "
                    "Run in LIVE mode first to generate cached results."
                )
        else:
            st.info("Eligibility evaluation results are not yet available for this patient.")

elif run_button and pipeline_mode == "live" and prescreen_result and prescreen_result.candidates:
    from trialmatch.validate.evaluator import evaluate_criterion, evaluate_criterion_two_stage

    is_two_stage = validate_mode == "Two-Stage (MedGemma \u2192 Gemini)"
    reasoning_adapter, labeling_adapter = _get_validate_adapters(validate_mode)

    if reasoning_adapter is None:
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

    # Use a single persistent event loop for VALIDATE to avoid httpx
    # "Event loop is closed" errors. asyncio.run() creates+destroys a loop
    # per call, but httpx.AsyncClient binds its connection pool to the first
    # loop. Subsequent asyncio.run() calls create new loops where the pool
    # is already closed.
    loop = asyncio.new_event_loop()

    if DEV_MODE:
        # ---- Dev mode: full VALIDATE display with metrics ----
        with st.status("VALIDATE: Evaluating eligibility criteria...", expanded=True) as status:
            from trialmatch.prescreen.ctgov_client import CTGovClient

            ctgov = CTGovClient()
            try:
                for trial in top_candidates:
                    trial_title = trial.brief_title or trial.title
                    st.write(f"**Evaluating: {trial.nct_id}** -- {trial_title}")

                    # Fetch eligibility criteria if not already available
                    criteria_text = trial_criteria_map.get(trial.nct_id, "")
                    if not criteria_text:
                        try:
                            raw_details = loop.run_until_complete(ctgov.get_details(trial.nct_id))
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

                    n_eval = min(len(criteria), max_criteria)
                    st.caption(f"Evaluating {n_eval} of {len(criteria)} criteria...")

                    results = []
                    for criterion in criteria[:max_criteria]:
                        try:
                            if is_two_stage:
                                cr = loop.run_until_complete(
                                    evaluate_criterion_two_stage(
                                        patient_note=patient_note,
                                        criterion_text=criterion["text"],
                                        criterion_type=criterion["type"],
                                        reasoning_adapter=reasoning_adapter,
                                        labeling_adapter=labeling_adapter,
                                    )
                                )
                            else:
                                cr = loop.run_until_complete(
                                    evaluate_criterion(
                                        patient_note=patient_note,
                                        criterion_text=criterion["text"],
                                        criterion_type=criterion["type"],
                                        adapter=reasoning_adapter,
                                    )
                                )
                            results.append((criterion, cr))
                        except Exception as exc:
                            st.warning(f"Criterion eval failed: {exc}")
                            continue

                    if results:
                        validate_results[trial.nct_id] = results

                        # Display per-criterion results with inline metrics
                        for criterion, cr in results:
                            icon = CRITERION_ICONS.get(cr.verdict.value, "\u26aa")
                            ctype_label = criterion["type"].upper()
                            text_preview = criterion["text"][:100]
                            st.markdown(
                                f"{icon} **{cr.verdict.value}** [{ctype_label}] -- {text_preview}"
                            )

                            if is_two_stage and cr.stage1_response:
                                # Two-stage: show both stages' metrics
                                s1 = cr.stage1_response
                                s2 = cr.model_response
                                total_latency = (s1.latency_ms + s2.latency_ms) / 1000
                                total_cost = s1.estimated_cost + s2.estimated_cost
                                s1_lat = s1.latency_ms / 1000
                                s2_lat = s2.latency_ms / 1000
                                st.caption(
                                    f"Stage 1: MedGemma "
                                    f"{s1.input_tokens}\u2192"
                                    f"{s1.output_tokens} tok, "
                                    f"{s1_lat:.1f}s | "
                                    f"Stage 2: Gemini "
                                    f"{s2.input_tokens}\u2192"
                                    f"{s2.output_tokens} tok, "
                                    f"{s2_lat:.1f}s | "
                                    f"Total: {total_latency:.1f}s, "
                                    f"${total_cost:.4f}"
                                )
                                with st.expander(
                                    "Stage 1: MedGemma Medical Reasoning",
                                    expanded=False,
                                ):
                                    st.write(cr.stage1_reasoning)
                                with st.expander(
                                    "Stage 2: Gemini Label Assignment",
                                    expanded=False,
                                ):
                                    st.write(cr.reasoning)
                            else:
                                # Single-stage: show single model metrics
                                mr = cr.model_response
                                latency_s = mr.latency_ms / 1000
                                model_label = (
                                    "MedGemma 4B" if "MedGemma" in validate_mode else "Gemini 3 Pro"
                                )
                                st.caption(
                                    f"{model_label} | "
                                    f"{mr.input_tokens}\u2192{mr.output_tokens} tokens | "
                                    f"{latency_s:.1f}s | "
                                    f"${mr.estimated_cost:.4f}"
                                )
                                with st.expander("Reasoning", expanded=False):
                                    st.write(cr.reasoning)

                        # Per-trial summary: totals across all criteria
                        if is_two_stage:
                            total_latency_ms = sum(
                                cr.model_response.latency_ms
                                + (cr.stage1_response.latency_ms if cr.stage1_response else 0)
                                for _, cr in results
                            )
                            total_cost = sum(
                                cr.model_response.estimated_cost
                                + (cr.stage1_response.estimated_cost if cr.stage1_response else 0)
                                for _, cr in results
                            )
                        else:
                            total_latency_ms = sum(
                                cr.model_response.latency_ms for _, cr in results
                            )
                            total_cost = sum(cr.model_response.estimated_cost for _, cr in results)
                        verdict = _compute_trial_verdict(results)
                        trial_verdicts[trial.nct_id] = verdict
                        verdict_badge = VERDICT_BADGES.get(verdict, verdict)
                        mode_label = "Two-Stage" if is_two_stage else validate_mode
                        st.markdown(
                            f"**Trial {trial.nct_id}:** {len(results)} criteria | "
                            f"{mode_label} | "
                            f"{total_latency_ms / 1000:.1f}s total | "
                            f"${total_cost:.4f} | "
                            f"Verdict: {verdict_badge}"
                        )

                    st.divider()

                status.update(label="VALIDATE complete", state="complete")
            finally:
                loop.run_until_complete(ctgov.aclose())
                loop.close()

    else:
        # ---- Patient mode: simplified VALIDATE display ----
        with st.spinner("Checking your eligibility for each trial..."):
            from trialmatch.prescreen.ctgov_client import CTGovClient

            ctgov = CTGovClient()
            try:
                for trial in top_candidates:
                    # Fetch eligibility criteria if not already available
                    criteria_text = trial_criteria_map.get(trial.nct_id, "")
                    if not criteria_text:
                        try:
                            raw_details = loop.run_until_complete(ctgov.get_details(trial.nct_id))
                            proto = raw_details.get("protocolSection", {})
                            eligibility = proto.get("eligibilityModule", {})
                            criteria_text = eligibility.get("eligibilityCriteria", "")
                        except Exception:
                            criteria_text = ""

                    if not criteria_text:
                        continue

                    criteria = parse_eligibility_criteria(criteria_text)
                    if not criteria:
                        continue

                    results = []
                    for criterion in criteria[:max_criteria]:
                        try:
                            if is_two_stage:
                                cr = loop.run_until_complete(
                                    evaluate_criterion_two_stage(
                                        patient_note=patient_note,
                                        criterion_text=criterion["text"],
                                        criterion_type=criterion["type"],
                                        reasoning_adapter=reasoning_adapter,
                                        labeling_adapter=labeling_adapter,
                                    )
                                )
                            else:
                                cr = loop.run_until_complete(
                                    evaluate_criterion(
                                        patient_note=patient_note,
                                        criterion_text=criterion["text"],
                                        criterion_type=criterion["type"],
                                        adapter=reasoning_adapter,
                                    )
                                )
                            results.append((criterion, cr))
                        except Exception:
                            continue

                    if results:
                        validate_results[trial.nct_id] = results
                        verdict = _compute_trial_verdict(results)
                        trial_verdicts[trial.nct_id] = verdict

            finally:
                loop.run_until_complete(ctgov.aclose())
                loop.close()

        # Render trial cards immediately after evaluation
        for trial in top_candidates:
            if trial.nct_id in validate_results:
                trial_results = validate_results[trial.nct_id]
                criteria_for_card = []
                for criterion, cr in trial_results:
                    entry = {
                        "text": criterion["text"],
                        "type": criterion["type"],
                        "verdict": cr.verdict.value,
                        "reasoning": cr.reasoning,
                    }
                    if cr.stage1_reasoning:
                        entry["stage1_reasoning"] = cr.stage1_reasoning
                    criteria_for_card.append(entry)

                render_trial_card(
                    nct_id=trial.nct_id,
                    title=trial.brief_title or trial.title,
                    phase=", ".join(trial.phase) if trial.phase else "N/A",
                    status=trial.status,
                    verdict=trial_verdicts[trial.nct_id],
                    criteria=criteria_for_card,
                )

    st.session_state["validate_results"] = validate_results
    st.session_state["trial_verdicts"] = trial_verdicts

    # Auto-save validate results to cache for replay
    validate_cache: dict[str, dict] = {}
    for nct_id, results in validate_results.items():
        criteria_cache = []
        for criterion, cr in results:
            entry = {
                "text": criterion["text"],
                "type": criterion["type"],
                "verdict": cr.verdict.value,
                "reasoning": cr.reasoning,
                "evidence_sentences": cr.evidence_sentences,
            }
            if cr.stage1_reasoning:
                entry["stage1_reasoning"] = cr.stage1_reasoning
            criteria_cache.append(entry)
        validate_cache[nct_id] = {
            "verdict": trial_verdicts.get(nct_id, "UNCERTAIN"),
            "mode": "two_stage" if is_two_stage else "single",
            "criteria": criteria_cache,
        }
    save_validate_results(selected_topic, validate_cache)
    if DEV_MODE:
        st.caption("VALIDATE results cached for replay")

elif run_button and prescreen_result and not prescreen_result.candidates:
    if DEV_MODE:
        with st.expander("Step 3: VALIDATE -- Eligibility Check", expanded=True):
            st.warning("No candidate trials found by PRESCREEN. Nothing to validate.")
    else:
        st.info("No matching clinical trials were found for your condition.")
elif not run_button:
    render_validate_placeholder(dev_mode=DEV_MODE)

# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------
st.divider()

if DEV_MODE:
    st.subheader("Results")
else:
    st.subheader("Your Results")

trial_verdicts = st.session_state.get("trial_verdicts", {})
validate_results = st.session_state.get("validate_results", {})
prescreen_for_results = st.session_state.get("prescreen_result")

cached_validate_data = st.session_state.get("cached_validate_data") or (
    load_validate_results(selected_topic) if pipeline_mode == "cached" else None
)

if trial_verdicts and prescreen_for_results:
    if DEV_MODE:
        # ---- Dev mode: existing results table ----
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
                f"Evaluated {len(results_data)} trials with {validate_mode}. "
                f"Pipeline mode: {pipeline_mode}."
            )

    else:
        # ---- Patient mode: results summary + trial cards ----
        eligible_count = sum(1 for v in trial_verdicts.values() if v == "ELIGIBLE")
        uncertain_count = sum(1 for v in trial_verdicts.values() if v == "UNCERTAIN")
        excluded_count = sum(1 for v in trial_verdicts.values() if v == "EXCLUDED")

        render_results_summary(
            total_trials=len(trial_verdicts),
            eligible_count=eligible_count,
            uncertain_count=uncertain_count,
            excluded_count=excluded_count,
        )

        # Sort trials: ELIGIBLE first, then UNCERTAIN, then EXCLUDED
        verdict_order = {"ELIGIBLE": 0, "UNCERTAIN": 1, "EXCLUDED": 2}
        sorted_trials = sorted(
            [t for t in prescreen_for_results.candidates if t.nct_id in trial_verdicts],
            key=lambda t: verdict_order.get(trial_verdicts.get(t.nct_id, ""), 3),
        )

        for trial in sorted_trials:
            nct_id = trial.nct_id
            verdict = trial_verdicts[nct_id]

            # Build criteria list for the trial card
            criteria_for_card = []
            if validate_results.get(nct_id):
                # Live results (list of tuples)
                for criterion, cr in validate_results[nct_id]:
                    entry = {
                        "text": criterion["text"],
                        "type": criterion["type"],
                        "verdict": cr.verdict.value,
                        "reasoning": cr.reasoning,
                    }
                    if cr.stage1_reasoning:
                        entry["stage1_reasoning"] = cr.stage1_reasoning
                    criteria_for_card.append(entry)
            elif cached_validate_data and nct_id in cached_validate_data:
                # Cached results (dict)
                criteria_for_card = cached_validate_data[nct_id].get("criteria", [])

            render_trial_card(
                nct_id=nct_id,
                title=trial.brief_title or trial.title,
                phase=", ".join(trial.phase) if trial.phase else "N/A",
                status=trial.status,
                verdict=verdict,
                criteria=criteria_for_card,
            )

        # Bottom disclaimer
        st.divider()
        st.caption(_MEDICAL_DISCLAIMER)

elif run_button:
    if DEV_MODE:
        st.info("No results to display. Check pipeline output above for details.")
    else:
        st.info("No matching results found. Please try a different patient or run a live search.")
else:
    if DEV_MODE:
        st.caption("Run the pipeline to see matching results.")
    else:
        st.caption("Click 'Search for Trials' above to find matching clinical trials.")
