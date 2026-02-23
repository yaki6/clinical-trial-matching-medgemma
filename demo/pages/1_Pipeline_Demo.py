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

from trialmatch.evaluation.metrics import aggregate_to_trial_verdict
from trialmatch.live_runtime import (
    HealthCheckResult,
    create_prescreen_adapters,
    create_validate_adapters,
    failed_preflight_checks,
    run_live_preflight,
)
from trialmatch.tracing.live_trace import LiveTraceRecorder, create_live_trace_recorder

from cache_manager import (
    list_cached_patients,
    load_validate_results,
    save_ingest_result,
    save_cached_manifest,
    save_prescreen_result,
    save_validate_results,
    validate_cached_run,
)
from components.patient_card import extract_friendly_label, render_medical_image, render_patient_card
from components.pipeline_viewer import (
    render_image_findings,
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
IMAGE_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "sot" / "ingest" / "medgemma_image_cache"
)

try:
    from trialmatch.ingest.profile_adapter import (
        adapt_harness_patient,
        adapt_profile_for_prescreen,
        get_image_path,
        load_demo_harness,
        load_profiles,
    )
except ImportError:

    def load_profiles(path: str | Path | None = None) -> list[dict]:
        """Fallback loader when profile_adapter is not yet available."""
        p = Path(path) if path else PROFILES_PATH
        with open(p) as f:
            data = json.load(f)
        return data.get("profiles", [])

    def load_demo_harness(path=None) -> list[dict]:
        """Fallback: load demo harness JSON."""
        p = Path(path) if path else (
            Path(__file__).resolve().parents[2] / "data" / "sot" / "ingest" / "nsclc_demo_harness.json"
        )
        with open(p) as f:
            data = json.load(f)
        return data.get("patients", [])

    def adapt_profile_for_prescreen(profile: dict) -> tuple[str, dict]:
        """Fallback adapter: flatten key_facts list-of-objects to dict."""
        patient_note = profile.get("profile_text", "")
        key_facts_list = profile.get("key_facts", [])
        result = {}
        for kf in key_facts_list:
            field = kf.get("field", "unknown")
            result[field] = kf.get("value")
        return patient_note, result

    def adapt_harness_patient(profile: dict, image_findings=None) -> tuple[str, dict]:
        """Fallback: adapt harness patient with optional image findings."""
        patient_note, key_facts = adapt_profile_for_prescreen(profile)
        if image_findings:
            key_facts["medgemma_imaging"] = image_findings
        return patient_note, key_facts

    def get_image_path(patient: dict, base_dir=None):
        """Fallback: resolve image path."""
        if not patient.get("image"):
            return None
        base = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
        return base / patient["image"]["file_path"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from Streamlit's sync context."""
    return asyncio.run(coro)


def _trace_event(
    recorder: LiveTraceRecorder | None,
    event_name: str,
    payload: dict | None = None,
) -> None:
    """Emit local trace event when recorder is available."""
    if recorder is None:
        return
    recorder.record(event_name, payload or {})


def _friendly_patient_label(profile: dict) -> str:
    """Build patient-friendly label from profile data.

    Delegates to the shared extract_friendly_label() in patient_card.py.
    """
    tid = profile.get("topic_id", "Unknown")
    suffix = extract_friendly_label(profile)  # e.g. " (43F, Lung Adenocarcinoma)"
    return f"{tid}{suffix}" if suffix else tid


def _load_image_cache(topic_id: str) -> dict | None:
    """Load cached MedGemma image extraction results for a patient."""
    cache_file = IMAGE_CACHE_DIR / f"{topic_id}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def _render_preflight_results(results: list[HealthCheckResult], dev_mode: bool) -> None:
    """Render live preflight status for dev and patient modes."""
    failed_checks = [r for r in results if not r.ok]

    if dev_mode:
        with st.expander("Live Preflight Checks", expanded=True):
            for r in results:
                icon = "✅" if r.ok else "⚠️"
                st.write(f"{icon} **{r.name}** — {r.latency_ms:.0f}ms")
                st.caption(r.detail)

    if failed_checks:
        names = ", ".join(r.name for r in failed_checks)
        st.error(f"Live preflight failed: {names}. Resolve checks before running live.")


def _filter_validate_to_prescreen(
    cached_validate: dict[str, dict] | None,
    prescreen_result,
) -> dict[str, dict]:
    """Keep only validate entries that correspond to PRESCREEN candidates."""
    if not cached_validate:
        return {}
    valid_ids = {c.nct_id for c in prescreen_result.candidates}
    return {nct: payload for nct, payload in cached_validate.items() if nct in valid_ids}


def _init_adapters():
    """Initialize model adapters from env vars. Returns (gemini, medgemma) or (None, None)."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not api_key:
        st.error(
            "GOOGLE_API_KEY not set. Required for PRESCREEN agent (Gemini orchestration)."
        )
        return None, None

    try:
        return create_prescreen_adapters(api_key=api_key, hf_token=hf_token)
    except ValueError as exc:
        st.error(str(exc))
        return None, None


def _get_validate_adapters(mode: str):
    """Create model adapter(s) for VALIDATE based on sidebar selection.

    Returns (reasoning_adapter, labeling_adapter) for two-stage mode,
    or (adapter, None) for single-stage mode.
    """
    try:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        hf_token = os.environ.get("HF_TOKEN", "")

        if mode == "Two-Stage (MedGemma \u2192 Gemini)":
            if not api_key:
                st.error("GOOGLE_API_KEY not set (needed for Gemini labeling).")
                return None, None
            return create_validate_adapters(mode, api_key=api_key, hf_token=hf_token)

        if mode == "Gemini 3 Pro (single)":
            if not api_key:
                st.error("GOOGLE_API_KEY not set.")
                return None, None
            return create_validate_adapters(mode, api_key=api_key)

        # MedGemma 4B (single)
        return create_validate_adapters(mode, hf_token=hf_token)
    except ValueError as exc:
        st.error(str(exc))
        return None, None


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
    """Aggregate criterion verdicts to trial-level using the benchmark logic.

    Delegates to aggregate_to_trial_verdict() so demo and benchmark stay in sync.
    """
    criterion_tuples = [(cr.verdict, criterion["type"]) for criterion, cr in results]
    return aggregate_to_trial_verdict(criterion_tuples).value


VERDICT_BADGES = {
    "ELIGIBLE": ":green[ELIGIBLE]",
    "EXCLUDED": ":red[EXCLUDED]",
    "NOT_RELEVANT": ":red[NOT RELEVANT]",
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
    """Load demo harness patients (5 curated NSCLC cases).

    Falls back to full nsclc_trial_profiles.json if harness is unavailable.
    """
    try:
        patients = load_demo_harness()
        if patients:
            return patients
    except Exception:
        pass
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
    cached_topic_ids = [tid for tid in topic_ids if tid in set(list_cached_patients())]
    topic_selector_ids = cached_topic_ids if cached_topic_ids else topic_ids

    selected_topic = st.sidebar.selectbox(
        "Select Patient",
        topic_selector_ids,
        format_func=lambda tid: _friendly_patient_label(profile_map[tid]),
    )

    if selected_topic:
        preview_text = profile_map[selected_topic].get("profile_text", "")[:200]
        st.sidebar.caption(f"**{selected_topic}**: {preview_text}...")
    if cached_topic_ids:
        st.sidebar.caption(
            "Demo-ready cached patients: " + ", ".join(cached_topic_ids)
        )

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

# -- Medical image for multimodal patients --
is_multimodal = profile.get("ingest_mode") == "multimodal"
image_cache = _load_image_cache(selected_topic) if is_multimodal else None

if is_multimodal:
    image_path = get_image_path(profile)
    if image_path and image_path.exists():
        render_medical_image(image_path, profile["image"], dev_mode=DEV_MODE)

st.divider()

# -- INGEST step --
# Don't merge image cache into display key_facts (shown separately below).
# PRESCREEN live runs merge image context via adapt_harness_patient(profile, image_cache).
patient_note, key_facts = adapt_harness_patient(profile)
render_ingest_step(key_facts, dev_mode=DEV_MODE)

# -- MedGemma image findings for multimodal patients --
if image_cache:
    render_image_findings(image_cache, dev_mode=DEV_MODE)

# Reset previous run outputs before a new run starts.
if run_button:
    st.session_state.pop("prescreen_result", None)
    st.session_state.pop("validate_results", None)
    st.session_state.pop("trial_verdicts", None)
    st.session_state.pop("cached_validate_data", None)

# Live run trace recorder (local JSONL artifact)
live_trace: LiveTraceRecorder | None = None
if run_button and pipeline_mode == "live":
    live_trace = create_live_trace_recorder(
        topic_id=selected_topic,
        pipeline_mode=pipeline_mode,
        validate_mode=validate_mode,
        dev_mode=DEV_MODE,
    )
    _trace_event(
        live_trace,
        "streamlit_live_context",
        {
            "selected_topic": selected_topic,
            "is_multimodal": is_multimodal,
            "max_trials": max_trials,
            "max_criteria": max_criteria,
        },
    )
    if DEV_MODE:
        st.caption(f"Local trace log: {live_trace.path}")

# ---------------------------------------------------------------------------
# PRESCREEN step
# ---------------------------------------------------------------------------
if run_button and pipeline_mode == "live":
    gemini_adapter, medgemma_adapter = _init_adapters()

    if gemini_adapter is None:
        _trace_event(
            live_trace,
            "live_run_blocked_missing_credentials",
            {
                "google_api_key_set": bool(os.environ.get("GOOGLE_API_KEY", "")),
                "hf_token_set": bool(os.environ.get("HF_TOKEN", "")),
            },
        )
        st.error("Cannot run pipeline: missing GOOGLE_API_KEY. Set env vars and reload.")
        st.stop()

    preflight_results = _run_async(
        run_live_preflight(
            gemini_adapter=gemini_adapter,
            medgemma_adapter=medgemma_adapter,
            include_ctgov=True,
        )
    )
    _trace_event(
        live_trace,
        "live_preflight_results",
        {
            "gemini_model": gemini_adapter.name,
            "medgemma_model": medgemma_adapter.name if medgemma_adapter else "",
            "checks": [
                {
                    "name": r.name,
                    "ok": r.ok,
                    "latency_ms": r.latency_ms,
                    "detail": r.detail,
                }
                for r in preflight_results
            ],
        },
    )
    _render_preflight_results(preflight_results, dev_mode=DEV_MODE)
    failed_checks = failed_preflight_checks(preflight_results)
    if failed_checks:
        failed_names = [r.name for r in failed_checks]
        _trace_event(
            live_trace,
            "live_run_blocked_preflight_failed",
            {
                "failed_checks": failed_names,
                "details": [
                    {
                        "name": r.name,
                        "latency_ms": r.latency_ms,
                        "detail": r.detail,
                    }
                    for r in failed_checks
                ],
            },
        )
        st.error(
            "Live mode is blocked because one or more preflight checks failed. "
            "Use cached mode for recording or fix endpoint/credentials first."
        )
        st.stop()

    from trialmatch.prescreen.agent import run_prescreen_agent

    if DEV_MODE:
        # ---- Dev mode: full agent trace streaming ----
        # Container for live-streaming agent tool calls
        prescreen_status = st.status("PRESCREEN: Searching ClinicalTrials.gov...", expanded=True)
        prescreen_log = prescreen_status.container()
        prescreen_log.write(
            f"Running agentic search with {gemini_adapter.name} orchestration..."
        )

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
                        medgemma_adapter=medgemma_adapter,
                        require_clinical_guidance=True,
                        topic_id=selected_topic,
                        on_tool_call=_on_tool_call,
                        on_agent_text=_on_agent_text,
                        trace_callback=live_trace.record if live_trace else None,
                    )
                )
            except Exception as exc:
                _trace_event(
                    live_trace,
                    "prescreen_failed",
                    {
                        "topic_id": selected_topic,
                        "error": str(exc)[:500],
                    },
                )
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
                        medgemma_adapter=medgemma_adapter,
                        require_clinical_guidance=True,
                        topic_id=selected_topic,
                        trace_callback=live_trace.record if live_trace else None,
                    )
                )
            except Exception as exc:
                _trace_event(
                    live_trace,
                    "prescreen_failed",
                    {
                        "topic_id": selected_topic,
                        "error": str(exc)[:500],
                    },
                )
                st.error(f"Search failed: {exc}")
                st.stop()

        n_candidates = len(prescreen_result.candidates)
        st.success(f"Found {n_candidates} clinical trials that may be relevant to your condition.")

    # Store for VALIDATE
    st.session_state["prescreen_result"] = prescreen_result
    _trace_event(
        live_trace,
        "prescreen_result_saved",
        {
            "topic_id": selected_topic,
            "candidate_count": len(prescreen_result.candidates),
            "tool_calls": prescreen_result.total_api_calls,
            "gemini_input_tokens": prescreen_result.gemini_input_tokens,
            "gemini_output_tokens": prescreen_result.gemini_output_tokens,
            "gemini_estimated_cost": prescreen_result.gemini_estimated_cost,
            "medgemma_estimated_cost": prescreen_result.medgemma_estimated_cost,
            "latency_ms": prescreen_result.latency_ms,
        },
    )

    # Auto-save to cache for replay
    prescreen_cache_path = save_prescreen_result(selected_topic, prescreen_result)
    ingest_cache_path = save_ingest_result(selected_topic, patient_note, key_facts)
    _trace_event(
        live_trace,
        "prescreen_cache_saved",
        {
            "prescreen_cache_path": str(prescreen_cache_path),
            "ingest_cache_path": str(ingest_cache_path),
        },
    )
    if DEV_MODE:
        st.caption(f"Results cached for replay: {selected_topic}")

elif run_button and pipeline_mode == "cached":
    cache_report = validate_cached_run(selected_topic)
    if not cache_report.valid:
        error_lines = "\n".join(f"- {e}" for e in cache_report.errors)
        st.error(
            "Cached demo data is invalid for this patient. "
            "Regenerate cache artifacts before recording.\n\n"
            f"{error_lines}"
        )
        st.stop()

    cached_path = CACHED_RUNS_DIR / selected_topic / "prescreen_result.json"
    with open(cached_path) as f:
        cached_data = json.load(f)

    from trialmatch.prescreen.schema import PresearchResult

    prescreen_result = PresearchResult(**cached_data)
    st.session_state["prescreen_result"] = prescreen_result

    if DEV_MODE:
        with st.expander("Step 2: PRESCREEN -- Trial Search (cached)", expanded=True):
            st.caption(
                "Loaded cached result: "
                f"{len(prescreen_result.candidates)} candidates"
            )
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
        n_candidates = len(prescreen_result.candidates)
        st.success(
            f"Loaded cached PRESCREEN output with {n_candidates} candidate trials."
        )
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
        filtered_validate = _filter_validate_to_prescreen(cached_validate, prescreen_result)
        skipped_count = len(cached_validate) - len(filtered_validate)
        cached_validate = filtered_validate

        if DEV_MODE:
            # ---- Dev mode: full cached validate display ----
            with st.expander("Step 3: VALIDATE -- Eligibility Check (cached)", expanded=True):
                cached_mode = None
                if skipped_count:
                    st.warning(
                        f"Ignored {skipped_count} cached validate trial(s) not present in "
                        "PRESCREEN candidates."
                    )
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
                    st.caption(
                        f"Evaluation mode: {mode_display} | "
                        f"PRESCREEN candidates: {len(prescreen_result.candidates)} | "
                        f"VALIDATE evaluated: {len(cached_validate)}"
                    )
        else:
            # ---- Patient mode: render trial cards from cache ----
            # (trial cards are rendered in the Results panel below)
            st.caption(
                f"PRESCREEN candidates: {len(prescreen_result.candidates)} | "
                f"VALIDATE evaluated: {len(cached_validate)}"
            )

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
        _trace_event(
            live_trace,
            "validate_blocked_missing_credentials",
            {
                "validate_mode": validate_mode,
                "google_api_key_set": bool(os.environ.get("GOOGLE_API_KEY", "")),
                "hf_token_set": bool(os.environ.get("HF_TOKEN", "")),
            },
        )
        st.error("Cannot run VALIDATE: missing API credentials.")
        st.stop()

    _trace_event(
        live_trace,
        "validate_started",
        {
            "validate_mode": validate_mode,
            "is_two_stage": is_two_stage,
            "reasoning_adapter": reasoning_adapter.name,
            "labeling_adapter": labeling_adapter.name if labeling_adapter else "",
            "candidate_pool_size": len(prescreen_result.candidates),
            "max_trials": max_trials,
            "max_criteria": max_criteria,
        },
    )

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
                            _trace_event(
                                live_trace,
                                "validate_ctgov_details_loaded",
                                {
                                    "trial_nct_id": trial.nct_id,
                                    "criteria_chars": len(criteria_text),
                                },
                            )
                        except Exception as exc:
                            _trace_event(
                                live_trace,
                                "validate_ctgov_details_failed",
                                {
                                    "trial_nct_id": trial.nct_id,
                                    "error": str(exc)[:500],
                                },
                            )
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
                    for idx, criterion in enumerate(criteria[:max_criteria]):
                        criterion_trace = None
                        if live_trace:
                            criterion_trace = (
                                lambda event_name, payload, *,
                                _trial_id=trial.nct_id,
                                _criterion=criterion,
                                _idx=idx: live_trace.record(
                                    event_name,
                                    {
                                        "trial_nct_id": _trial_id,
                                        "criterion_index": _idx,
                                        "criterion_type": _criterion["type"],
                                        "criterion_text": _criterion["text"],
                                        **payload,
                                    },
                                )
                            )
                        try:
                            if is_two_stage:
                                cr = loop.run_until_complete(
                                    evaluate_criterion_two_stage(
                                        patient_note=patient_note,
                                        criterion_text=criterion["text"],
                                        criterion_type=criterion["type"],
                                        reasoning_adapter=reasoning_adapter,
                                        labeling_adapter=labeling_adapter,
                                        trace_callback=criterion_trace,
                                    )
                                )
                            else:
                                cr = loop.run_until_complete(
                                    evaluate_criterion(
                                        patient_note=patient_note,
                                        criterion_text=criterion["text"],
                                        criterion_type=criterion["type"],
                                        adapter=reasoning_adapter,
                                        trace_callback=criterion_trace,
                                    )
                                )
                            results.append((criterion, cr))
                        except Exception as exc:
                            _trace_event(
                                live_trace,
                                "validate_criterion_failed",
                                {
                                    "trial_nct_id": trial.nct_id,
                                    "criterion_index": idx,
                                    "criterion_type": criterion["type"],
                                    "criterion_text": criterion["text"],
                                    "error": str(exc)[:500],
                                },
                            )
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
                        _trace_event(
                            live_trace,
                            "validate_trial_complete",
                            {
                                "trial_nct_id": trial.nct_id,
                                "criteria_evaluated": len(results),
                                "mode_label": mode_label,
                                "latency_ms": total_latency_ms,
                                "total_cost": total_cost,
                                "verdict": verdict,
                            },
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
                            _trace_event(
                                live_trace,
                                "validate_ctgov_details_loaded",
                                {
                                    "trial_nct_id": trial.nct_id,
                                    "criteria_chars": len(criteria_text),
                                },
                            )
                        except Exception:
                            _trace_event(
                                live_trace,
                                "validate_ctgov_details_failed",
                                {
                                    "trial_nct_id": trial.nct_id,
                                },
                            )
                            criteria_text = ""

                    if not criteria_text:
                        continue

                    criteria = parse_eligibility_criteria(criteria_text)
                    if not criteria:
                        continue

                    results = []
                    for idx, criterion in enumerate(criteria[:max_criteria]):
                        criterion_trace = None
                        if live_trace:
                            criterion_trace = (
                                lambda event_name, payload, *,
                                _trial_id=trial.nct_id,
                                _criterion=criterion,
                                _idx=idx: live_trace.record(
                                    event_name,
                                    {
                                        "trial_nct_id": _trial_id,
                                        "criterion_index": _idx,
                                        "criterion_type": _criterion["type"],
                                        "criterion_text": _criterion["text"],
                                        **payload,
                                    },
                                )
                            )
                        try:
                            if is_two_stage:
                                cr = loop.run_until_complete(
                                    evaluate_criterion_two_stage(
                                        patient_note=patient_note,
                                        criterion_text=criterion["text"],
                                        criterion_type=criterion["type"],
                                        reasoning_adapter=reasoning_adapter,
                                        labeling_adapter=labeling_adapter,
                                        trace_callback=criterion_trace,
                                    )
                                )
                            else:
                                cr = loop.run_until_complete(
                                    evaluate_criterion(
                                        patient_note=patient_note,
                                        criterion_text=criterion["text"],
                                        criterion_type=criterion["type"],
                                        adapter=reasoning_adapter,
                                        trace_callback=criterion_trace,
                                    )
                                )
                            results.append((criterion, cr))
                        except Exception as exc:
                            _trace_event(
                                live_trace,
                                "validate_criterion_failed",
                                {
                                    "trial_nct_id": trial.nct_id,
                                    "criterion_index": idx,
                                    "criterion_type": criterion["type"],
                                    "criterion_text": criterion["text"],
                                    "error": str(exc)[:500],
                                },
                            )
                            continue

                    if results:
                        validate_results[trial.nct_id] = results
                        verdict = _compute_trial_verdict(results)
                        trial_verdicts[trial.nct_id] = verdict
                        _trace_event(
                            live_trace,
                            "validate_trial_complete",
                            {
                                "trial_nct_id": trial.nct_id,
                                "criteria_evaluated": len(results),
                                "verdict": verdict,
                            },
                        )

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
    validate_cache_path = save_validate_results(selected_topic, validate_cache)
    manifest_path = save_cached_manifest(
        topic_id=selected_topic,
        prescreen_trial_ids=[c.nct_id for c in prescreen_result.candidates],
        validated_trial_ids=list(validate_cache.keys()),
        validate_mode="two_stage" if is_two_stage else "single",
    )
    _trace_event(
        live_trace,
        "validate_results_saved",
        {
            "topic_id": selected_topic,
            "trial_count": len(validate_cache),
            "cache_path": str(validate_cache_path),
            "manifest_path": str(manifest_path),
        },
    )
    if DEV_MODE:
        st.caption("VALIDATE results cached for replay")

elif run_button and prescreen_result and not prescreen_result.candidates:
    empty_validate_path = save_validate_results(selected_topic, {})
    manifest_path = save_cached_manifest(
        topic_id=selected_topic,
        prescreen_trial_ids=[],
        validated_trial_ids=[],
        validate_mode="not_run_no_candidates",
    )
    _trace_event(
        live_trace,
        "validate_results_saved_empty",
        {
            "topic_id": selected_topic,
            "cache_path": str(empty_validate_path),
            "manifest_path": str(manifest_path),
        },
    )
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
if cached_validate_data and prescreen_for_results:
    cached_validate_data = _filter_validate_to_prescreen(
        cached_validate_data,
        prescreen_for_results,
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
        prescreen_count = len(prescreen_for_results.candidates)
        evaluated_count = len(trial_verdicts)
        eligible_count = sum(1 for v in trial_verdicts.values() if v == "ELIGIBLE")
        uncertain_count = sum(1 for v in trial_verdicts.values() if v == "UNCERTAIN")
        excluded_count = sum(1 for v in trial_verdicts.values() if v == "EXCLUDED")

        st.caption(
            f"PRESCREEN candidates: {prescreen_count} | "
            f"VALIDATE evaluated: {evaluated_count}"
        )
        render_results_summary(
            total_trials=evaluated_count,
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
