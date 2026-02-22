# Streamlit Dev Mode & Patient UX Redesign

**Date**: 2026-02-22
**Status**: Implemented

---

## 1. Overview

The current Streamlit demo (`demo/pages/1_Pipeline_Demo.py`) exposes developer-oriented controls (model selector, token counts, tool call traces, raw reasoning chains) that confuse non-technical users. This plan introduces a **two-mode UI**:

| Mode | Audience | What they see |
|------|----------|---------------|
| **Patient mode** (default) | Patients, clinicians, demo reviewers | Clean results with plain-language explanations |
| **Dev mode** | Engineers, evaluators, Kaggle judges | Full pipeline traces, costs, tokens, raw model outputs |

---

## 2. Feature Flag: Dev Mode Toggle

### Implementation

**Mechanism**: `st.query_params` + sidebar toggle (sticky per session).

```
# Default (patient mode):
http://localhost:8501/Pipeline_Demo

# Dev mode via URL:
http://localhost:8501/Pipeline_Demo?dev=1

# Also toggleable via sidebar checkbox (hidden behind a small "gear" icon)
```

### Code pattern

```python
# At top of 1_Pipeline_Demo.py, after imports:
def _is_dev_mode() -> bool:
    """Check if dev mode is active via query param or session state."""
    if st.query_params.get("dev") == "1":
        st.session_state["dev_mode"] = True
    return st.session_state.get("dev_mode", False)

DEV_MODE = _is_dev_mode()
```

**Sidebar toggle** (only visible when already in dev mode, or via a subtle gear icon):

```python
# In sidebar, at bottom:
with st.sidebar.expander("Settings", expanded=False, icon=":material/settings:"):
    dev_toggle = st.checkbox("Developer mode", value=DEV_MODE,
                             help="Show full pipeline traces, token counts, and costs")
    if dev_toggle != DEV_MODE:
        st.session_state["dev_mode"] = dev_toggle
        st.rerun()
```

---

## 3. Patient Mode (Default) — What Changes

### 3.1 Sidebar — Simplified

**Current (7 controls):**
- Select Patient (selectbox)
- Pipeline Mode: live/cached (radio)
- VALIDATE Mode: 3-option dropdown
- Max trials (slider 1-10)
- Max criteria (slider 5-30)
- Cached patients list
- Run Pipeline button

**Patient mode (3 controls):**

| Control | Type | Default | Notes |
|---------|------|---------|-------|
| Select Patient | Selectbox | First patient | Keep — patients need to select their profile |
| Run Pipeline | Button (primary) | N/A | Keep — single action to start |
| Settings gear | Expander | Collapsed | Contains dev mode toggle only |

**Hidden in patient mode:**
- Pipeline Mode → hardcoded to `"cached"` (use pre-computed results for reliability; fall back to `"live"` only if no cache exists)
- VALIDATE Mode → hardcoded to `"Two-Stage (MedGemma → Gemini)"` (best accuracy: 80%)
- Max trials → hardcoded to `3`
- Max criteria → hardcoded to `10`
- Cached patients list → hidden

**Rationale for `cached` default**: Patients should see instant, reliable results. Live API calls are slow (~2 min), can fail, and cost money. The demo's value is in the results, not watching API calls stream.

### 3.2 INGEST Display — Patient-Friendly

**Current**: Expander titled "Step 1: INGEST -- Key Facts" with field/value table.

**Patient mode**:
- Rename to **"Your Clinical Profile"**
- Remove "Step 1: INGEST" jargon
- Keep key facts table but use friendlier field names:

| Internal field | Patient-facing label |
|---------------|---------------------|
| `primary_diagnosis` | Diagnosis |
| `demographics` | Age & Sex |
| `tobacco_use` | Smoking History |
| `histopathology` | Pathology Results |
| `imaging_findings` | Imaging Results |
| `missing_info` | Information Gaps |
| `symptoms` | Current Symptoms |

- Add a brief intro: *"Here's what we extracted from your medical record:"*
- Clinical Note expander: rename to **"View Full Medical Record"**

### 3.3 PRESCREEN Display — Patient-Friendly

**Current**: Shows tool call traces, agent reasoning, cost breakdown, streaming status.

**Patient mode**:
- Replace `st.status()` streaming with a simple spinner: *"Searching for matching clinical trials..."*
- **Hide entirely**: tool call traces, agent reasoning, cost/token metrics
- Show only the **result**: *"Found N clinical trials that may be relevant to your condition."*
- No `st.dataframe()` — instead show a brief list:
  - Trial title (human-readable)
  - Phase (e.g., "Phase 2" not raw list)
  - Status (e.g., "Recruiting" with green indicator)

### 3.4 VALIDATE Display — Patient-Friendly (Biggest Change)

**Current**: Per-criterion display with MET/NOT_MET/UNKNOWN icons, token counts, cost, expandable reasoning chains.

**Patient mode** — redesigned as a **trial card per trial**:

```
┌─────────────────────────────────────────────────────┐
│  Trial: A Study of Drug X for Advanced NSCLC        │
│  NCT12345678 · Phase 2 · Recruiting                 │
│                                                      │
│  ✅ You may be ELIGIBLE for this trial               │
│                                                      │
│  Criteria you meet:                                  │
│  ✓ Age requirement (18+)                             │
│  ✓ Diagnosis matches                                 │
│  ✓ No exclusionary conditions found                  │
│                                                      │
│  Needs further review:                               │
│  ? Lab values not available in your record           │
│                                                      │
│  ▸ View detailed eligibility breakdown               │
│  🔗 View on ClinicalTrials.gov                       │
└─────────────────────────────────────────────────────┘
```

**Key design decisions:**
1. **Verdict as plain English**: "You may be ELIGIBLE" / "You likely do NOT qualify" / "More information needed"
2. **Group by outcome** not by inclusion/exclusion: patients don't know this terminology
3. **Truncate criterion text** to plain-language summaries (first 60 chars + "...")
4. **Hide** token counts, latency, cost, stage1/stage2 reasoning
5. **"View detailed eligibility breakdown"** expander shows full criteria (still no tokens/cost)
6. **Disclaimer** at bottom: *"This is an AI-assisted screening tool. Always consult with your healthcare provider and the trial's research team before making decisions about clinical trial participation."*

### 3.5 Results Panel — Patient-Friendly

**Current**: Markdown lines with verdict badges, NCT links, criteria counts.

**Patient mode** — **summary card at the top** (before details):

```
┌─────────────────────────────────────────────────────┐
│  📋 Your Trial Matching Results                      │
│                                                      │
│  We found 3 clinical trials for your condition.      │
│                                                      │
│  ✅ 1 trial you may be eligible for                  │
│  ❓ 1 trial needs more information                   │
│  ❌ 1 trial you likely don't qualify for             │
│                                                      │
│  Scroll down for details on each trial.              │
└─────────────────────────────────────────────────────┘
```

- Sort: ELIGIBLE first (give patients hope)
- Each trial links to ClinicalTrials.gov
- Remove "Evaluated N trials with Two-Stage" caption → replace with timestamp

### 3.6 Remove/Relocate

| Element | Patient mode | Dev mode |
|---------|-------------|----------|
| "Ambiguities / Missing Info" expander | Rename to "Information gaps in your record" | Keep as-is |
| Pipeline step numbering ("Step 1/2/3") | Remove | Keep |
| Token counts | Hide | Show |
| Cost breakdown | Hide | Show |
| Latency metrics | Hide | Show |
| Tool call traces | Hide | Show in expanders |
| Agent reasoning | Hide | Show in expanders |
| Stage1/Stage2 reasoning | Hide | Show in expanders |
| Pipeline mode selector | Hide | Show |
| Validate mode selector | Hide | Show |
| Max trials/criteria sliders | Hide | Show |

---

## 4. Dev Mode — What's Shown

Everything currently shown, **plus** these additions:

### 4.1 Existing elements (keep all)
- Pipeline Mode radio (live/cached)
- VALIDATE Mode selector (3 options)
- Max trials / Max criteria sliders
- Cached patients indicator
- All tool call traces with args/results/latency
- Agent reasoning summary
- Per-criterion token counts, costs, latency
- Stage1 (MedGemma) + Stage2 (Gemini) reasoning expanders
- Cost summary per stage
- Candidate trials dataframe

### 4.2 New additions for dev mode
- **Run config summary**: At top of results, show which models were used, total tokens, total cost, total latency
- **Raw model response viewer**: New expander per criterion showing the raw JSON response from the model (before parsing)
- **Session state inspector**: Collapsible section at bottom showing `st.session_state` keys and sizes (for debugging)
- **Cache status**: Show whether result was served from cache or live, with cache file paths

---

## 5. Implementation Plan

### Files to modify

| File | Changes |
|------|---------|
| `demo/pages/1_Pipeline_Demo.py` | Add `_is_dev_mode()`, wrap all dev-only sections in `if DEV_MODE:`, add patient-friendly alternatives |
| `demo/components/patient_card.py` | Add `dev_mode` param; patient mode uses friendly field names |
| `demo/components/pipeline_viewer.py` | Add `dev_mode` param; patient mode uses simplified displays |
| `demo/components/trial_card.py` | **New file** — patient-friendly trial result card component |
| `demo/components/results_summary.py` | **New file** — summary card showing eligible/uncertain/excluded counts |
| `demo/app.py` | No changes needed (landing page is already clean) |
| `demo/pages/2_Benchmark.py` | Add dev mode guard — benchmark is dev-only; hide from patient mode sidebar |

### Phased approach

**Phase 1: Feature flag + sidebar cleanup** (~30 min)
- Add `_is_dev_mode()` to `1_Pipeline_Demo.py`
- Wrap sidebar controls in `if DEV_MODE:` blocks
- Set hardcoded defaults for patient mode

**Phase 2: Patient-friendly display components** (~45 min)
- Create `demo/components/trial_card.py` (patient-friendly trial result card)
- Create `demo/components/results_summary.py` (top-level summary)
- Modify `patient_card.py` for friendly field names in patient mode
- Modify `pipeline_viewer.py` for simplified INGEST display

**Phase 3: Wrap pipeline output in mode checks** (~45 min)
- PRESCREEN section: patient mode shows simple result, dev mode shows traces
- VALIDATE section: patient mode shows trial cards, dev mode shows full traces
- Results panel: patient mode shows summary + cards, dev mode shows current table
- Add disclaimer footer in patient mode

**Phase 4: Dev mode enhancements** (~15 min)
- Add run config summary
- Add cache status indicators
- Add session state inspector

---

## 6. Key Defaults for Patient Mode

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Pipeline mode | `cached` (fallback to `live`) | Instant reliable results; patients shouldn't wait ~2 min |
| VALIDATE mode | Two-Stage (MedGemma + Gemini) | Best accuracy (80%) |
| Max trials | 3 | Patients don't need 10 trials; 3 top matches is actionable |
| Max criteria | 10 | Sufficient for accurate verdict without overwhelming display |
| Benchmark page | Hidden | Not relevant to patients |

---

## 7. Questions for Alignment

1. **Patient selector format**: Currently shows raw `topic_id` (e.g., "mpx1016"). Should we show a friendlier label like "Patient A (43F, NSCLC)" in patient mode?

2. **Cached-only vs auto-fallback**: If no cache exists for a patient, should patient mode:
   - (a) Show an error: "Results not yet available for this patient"
   - (b) Silently fall back to live mode (slow but works)
   - (c) Show a warning and offer to run live

3. **Benchmark page visibility**: Should the Benchmark page be:
   - (a) Completely hidden in patient mode (remove from sidebar)
   - (b) Visible but marked as "Technical Details"
   - (c) Always visible

4. **Disclaimer placement**: Should the medical disclaimer appear:
   - (a) At the bottom of results only
   - (b) At the top of the page always
   - (c) Both top and bottom

---

## 8. Patient-Facing Copy

### Verdict translations

| Internal | Patient-facing |
|----------|---------------|
| ELIGIBLE | "You may be eligible for this trial" |
| EXCLUDED | "You likely do not qualify for this trial" |
| UNCERTAIN | "More information is needed to determine eligibility" |
| MET (inclusion) | "You meet this requirement" |
| NOT_MET (inclusion) | "You do not meet this requirement" |
| MET (exclusion) | "This exclusion applies to you" |
| NOT_MET (exclusion) | "This exclusion does not apply to you" |
| UNKNOWN | "Could not be determined from available records" |

### Section headers

| Internal | Patient-facing |
|----------|---------------|
| INGEST -- Key Facts | Your Clinical Profile |
| PRESCREEN -- Trial Search | Finding Matching Trials |
| VALIDATE -- Eligibility Check | Checking Your Eligibility |
| Results | Your Trial Matching Results |
