# Plan: Two-Patient Demo (mpx1016 + trec-20226) — INGEST + PRESCREEN + VALIDATE

## Context

The MedGemma Challenge demo needs a compelling end-to-end pipeline for Kaggle judges. Two patients: **mpx1016** (43F, lung adenocarcinoma, chest CT, multimodal) and **trec-20226** (61M, mesothelioma, text-only, with TREC ground truth).

### Architecture (post-adaptive-prescreen refactor)

The adaptive prescreen agent (`docs/plans/2026-02-24-adaptive-prescreen-agent.md`) is **already implemented**:
- `agent.py` has the new simplified system prompt with `{max_tool_calls}` budget
- `tools.py` has 3 tools: `search_trials`, `get_trial_details`, `consult_medical_expert`
- `run_prescreen_agent()` signature: `(patient_note, key_facts, ingest_source, gemini_adapter, medgemma_adapter=None, max_tool_calls=25, topic_id="", on_tool_call=None, on_agent_text=None, trace_callback=None)`
- **No** `require_clinical_guidance` parameter, **no** `_get_clinical_guidance()`, **no** checklist

### Critical Bug to Fix

`demo/pages/1_Pipeline_Demo.py:601` passes `require_clinical_guidance=True` to `run_prescreen_agent()` — **this parameter does not exist** and will cause TypeError at runtime. Must be removed.

### Problems to solve
1. **No fresh cached data** — current mpx1016 cache has only 1 trial (EXCLUDED). Need fresh run with >= 1 ELIGIBLE trial.
2. **Cached mode looks static** — judge sees "Loaded cached result" instead of compelling AI animation.
3. **Data display issues** — monospace rendering, collapsed images, truncated criterion text.

---

## Part A: Pipeline Runner Script

### New file: `scripts/run_demo_cache.py` (~300 lines)

Runs the full e2e pipeline live and writes demo cache. Supports both patients.

```
Phase 1 — INGEST:
  mpx1016: generate_with_image() on CT → merge into key_facts
  trec-20226: text-only, just adapt key_facts from structured_profile

Phase 2 — PRESCREEN (adaptive agent):
  Call run_prescreen_agent() — Gemini Pro orchestrates search_trials + consult_medical_expert
  mpx1016: default RECRUITING status
  trec-20226: all-status search (historical trials)

Phase 3 — VALIDATE (two-stage):
  Top 3 candidates: fetch criteria → evaluate_criterion_two_stage()
  Aggregate → trial-level verdicts

Phase 4 — Cache + manifest + (trec-20226 only: ground truth comparison)
```

**Key API calls** (no modification to existing code):

| Function | File | Notes |
|----------|------|-------|
| `generate_with_image()` | `src/trialmatch/models/vertex_medgemma.py` | mpx1016 only (4B imaging) |
| `create_imaging_adapter()` | `src/trialmatch/live_runtime.py` | Returns Vertex 4B adapter |
| `create_prescreen_adapters()` | `src/trialmatch/live_runtime.py` | Returns (GeminiAdapter, MedGemma27B) |
| `create_validate_adapters()` | `src/trialmatch/live_runtime.py` | Returns (reasoning, labeling) adapters |
| `adapt_harness_patient()` | `src/trialmatch/ingest/profile_adapter.py` | Flattens key_facts list→dict |
| `merge_image_findings()` | `src/trialmatch/ingest/profile_adapter.py` | Adds medgemma_imaging key |
| `load_demo_harness()` | `src/trialmatch/ingest/profile_adapter.py` | Loads nsclc_demo_harness.json |
| `get_image_path()` | `src/trialmatch/ingest/profile_adapter.py` | Resolves image path |
| `run_prescreen_agent()` | `src/trialmatch/prescreen/agent.py` | Adaptive agent (3 tools) |
| `evaluate_criterion_two_stage()` | `src/trialmatch/validate/evaluator.py` | MedGemma 27B → Gemini Pro |
| `aggregate_to_trial_verdict()` | `src/trialmatch/evaluation/metrics.py` | Criteria→trial verdict |
| `parse_eligibility_criteria()` | `demo/pages/1_Pipeline_Demo.py:253` | Splits criteria text |
| `save_ingest_result()` | `demo/cache_manager.py` | Saves (patient_note, key_facts) |
| `save_prescreen_result()` | `demo/cache_manager.py` | Saves PresearchResult via model_dump_json |
| `save_validate_results()` | `demo/cache_manager.py` | Saves dict keyed by nct_id |
| `save_cached_manifest()` | `demo/cache_manager.py` | Saves manifest with trial IDs |
| `validate_cached_run()` | `demo/cache_manager.py` | Cross-file consistency check |

**PRESCREEN call** (compatible with adaptive architecture):
```python
result = await run_prescreen_agent(
    patient_note=patient_note,
    key_facts=key_facts,
    ingest_source="gold",
    gemini_adapter=gemini_adapter,       # Gemini 3 Pro
    medgemma_adapter=medgemma_27b,       # consult_medical_expert tool
    max_tool_calls=25,
    topic_id=patient_id,
)
# NOTE: No require_clinical_guidance — that parameter was removed
```

**trec-20226 PRESCREEN config** — needs all-status search for historical trials. Since the adaptive agent's system prompt says `Default: ["RECRUITING"]`, the script must pass status override in key_facts or instruct via patient_note. Actually, the agent reads the system prompt which says `Set to ["RECRUITING", "NOT_YET_RECRUITING", "COMPLETED", "ACTIVE_NOT_RECRUITING"] for historical or comprehensive searches.` We add a note in the patient_note: "Search across all trial statuses including COMPLETED and TERMINATED, as this is a historical validation case."

### Execution sequence
```bash
uv run python scripts/deploy_vertex_4b.py deploy &
uv run python scripts/deploy_vertex_27b.py deploy &
# Wait for endpoints ready...
uv run python scripts/run_demo_cache.py --patient mpx1016 &
tail -f /tmp/demo_cache.log
uv run python scripts/run_demo_cache.py --patient trec-20226 &
tail -f /tmp/demo_cache_trec.log
# After completion:
uv run python scripts/deploy_vertex_4b.py undeploy &
uv run python scripts/deploy_vertex_27b.py undeploy &
```

---

## Part B: UI Enhancements for Demo Story

### B0. Fix `require_clinical_guidance` Bug

**File**: `demo/pages/1_Pipeline_Demo.py:601`

Remove the invalid parameter `require_clinical_guidance=True` from the `run_prescreen_agent()` call. This parameter does not exist in the current function signature (post-adaptive-prescreen refactor).

```python
# BEFORE (line 594-606):
prescreen_result = _run_async(
    run_prescreen_agent(
        ...
        require_clinical_guidance=True,  # ← REMOVE THIS LINE
        ...
    )
)

# AFTER:
prescreen_result = _run_async(
    run_prescreen_agent(
        patient_note=patient_note,
        key_facts=key_facts,
        ingest_source="gold",
        gemini_adapter=gemini_adapter,
        medgemma_adapter=medgemma_adapter,
        topic_id=selected_topic,
        on_tool_call=_on_tool_call,
        on_agent_text=_on_agent_text,
        trace_callback=live_trace.record if live_trace else None,
    )
)
```

### B1. INGEST Animation (cached mode)

**File**: `demo/pages/1_Pipeline_Demo.py` — insert after the `if run_button:` reset block (~line 467), before PRESCREEN section

```python
if run_button and pipeline_mode == "cached":
    with st.status("Step 1: INGEST — MedGemma 1.5 4B analyzing patient data...", expanded=True) as ingest_status:
        st.write("Reading clinical note...")
        time.sleep(0.8)
        st.write("Extracting structured key facts...")
        time.sleep(0.5)
        if is_multimodal and image_cache:
            st.write("MedGemma 1.5 4B analyzing CT image...")
            time.sleep(1.5)
            st.write("Merging image findings with clinical profile...")
            time.sleep(0.5)
        ingest_status.update(label="INGEST complete — Patient profile extracted", state="complete")
```

### B2. PRESCREEN Animation (cached mode, patient mode only)

**File**: `demo/pages/1_Pipeline_Demo.py` — modify the `else:` branch of the cached PRESCREEN block (lines 782-787)

Replace the flat "Loaded cached PRESCREEN output" with simulated search animation. The tool_call_trace now contains 3 tool types: `search_trials`, `get_trial_details`, `consult_medical_expert`. Show them with appropriate icons:

```python
else:  # patient mode
    with st.status("Step 2: PRESCREEN — Gemini Pro + MedGemma 27B searching trials...", expanded=True) as ps_status:
        st.write("Analyzing patient profile for search strategy...")
        time.sleep(1.0)
        for i, tc in enumerate(prescreen_result.tool_call_trace[:6]):
            if tc.tool_name == "consult_medical_expert":
                st.write(f"Consulting MedGemma medical expert...")
            elif tc.tool_name == "search_trials":
                cond = tc.args.get("condition", "")[:40]
                st.write(f"Searching ClinicalTrials.gov: {cond}...")
            elif tc.tool_name == "get_trial_details":
                nct = tc.args.get("nct_id", "")
                st.write(f"Reviewing trial details: {nct}...")
            time.sleep(0.6)
        n_candidates = len(prescreen_result.candidates)
        st.write(f"**Found {n_candidates} candidate trials**")
        time.sleep(0.5)
        ps_status.update(label=f"PRESCREEN complete — {n_candidates} trials found", state="complete")
    st.session_state["prescreen_result"] = prescreen_result
```

### B3. VALIDATE Animation (cached mode, patient mode only)

**File**: `demo/pages/1_Pipeline_Demo.py` — modify the cached VALIDATE patient-mode block (lines 843-848)

```python
else:  # patient mode
    with st.status("Step 3: VALIDATE — MedGemma 27B + Gemini Pro evaluating eligibility...", expanded=True) as val_status:
        for nct_id, data in cached_validate.items():
            st.write(f"**Evaluating: {nct_id}**")
            time.sleep(0.5)
            for c in data["criteria"][:5]:
                icon = {"MET": "🟢", "NOT_MET": "🔴", "UNKNOWN": "🟡"}.get(c["verdict"], "")
                st.write(f"  {icon} {c['verdict']} — {c['text'][:80]}...")
                time.sleep(0.3)
            verdict = data["verdict"]
            badge = VERDICT_BADGES.get(verdict, verdict)
            st.write(f"  Trial verdict: {badge}")
            time.sleep(0.3)
        val_status.update(label="VALIDATE complete — Eligibility determined", state="complete")
```

### B4. Model Attribution Labels

Update step headers to show multi-model orchestration. Note: PRESCREEN now uses **both** Gemini Pro (orchestrator) and MedGemma 27B (expert tool).

**`pipeline_viewer.py:render_ingest_step()`**:
- Dev mode: `"Step 1: INGEST — MedGemma 1.5 4B Key Fact Extraction"`
- Patient mode: `"Your Clinical Profile (AI-extracted)"`

**PRESCREEN status labels** (in `1_Pipeline_Demo.py`):
- `"Step 2: PRESCREEN — Gemini Pro + MedGemma 27B searching trials..."` (reflects adaptive agent with expert tool)

**VALIDATE status labels** (in `1_Pipeline_Demo.py`):
- `"Step 3: VALIDATE — MedGemma 27B + Gemini Pro evaluating eligibility..."`

### B5. Data Display Quality Fixes

**B5a. Clinical note → markdown**: `patient_card.py:75,87` — change `st.text()` to `st.markdown()` for `profile_text` (contains `## Clinical History` headers)

**B5b. Medical image expanded**: `patient_card.py:121` — change `expanded=False` to `expanded=True` in patient mode

**B5c. Reasoning → markdown**: `trial_card.py:137,140,143` — change `st.text()` to `st.markdown()` for `stage1_reasoning` and `reasoning`

**B5d. imaging_findings formatting**: `pipeline_viewer.py:render_ingest_step()` — detect dict values with "findings" key (list) and render as bulleted list instead of flat `"; ".join()`

**B5f. Criterion truncation**: `trial_card.py:37` — increase `_TRUNCATE_LEN` from 60 to 80

### B6. Pipeline Overview Header (patient mode only)

**File**: `demo/pages/1_Pipeline_Demo.py` — insert before INGEST step, after patient card

```python
if not DEV_MODE:
    with st.container(border=True):
        st.markdown("### How TrialMatch Works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Understand**")
            st.caption("MedGemma 1.5 4B reads your records & images")
        with col2:
            st.markdown("**2. Search**")
            st.caption("Gemini Pro + MedGemma 27B find matching trials")
        with col3:
            st.markdown("**3. Evaluate**")
            st.caption("MedGemma 27B + Gemini Pro check eligibility")
```

Note: Step 2 now says "Gemini Pro + MedGemma 27B" to reflect that the adaptive agent uses MedGemma as an on-demand expert tool during search.

---

## Part C: TREC-20226 Ground Truth Validation Patient

### C1. Add trec-20226 to Demo Harness

**File**: `data/harness schema/ingest/nsclc_demo_harness.json`

Add entry to `patients` array. Source data from `data/trec2022_ground_truth/patient.jsonl`.

```json
{
  "topic_id": "trec-20226",
  "source_dataset": "TREC 2022",
  "ingest_mode": "text",
  "ehr_text": "<full text from patient.jsonl>",
  "profile_text": "<formatted clinical summary from gpt4_summary + structured_profile>",
  "key_facts": [
    {"field": "primary_diagnosis", "value": "Malignant pleural mesothelioma (epithelioid type)", "evidence_span": "Biopsy shows proliferation of epithelioid-type cells with very long microvilli", "required": true, "notes": null},
    {"field": "demographics", "value": {"age": "61", "sex": "male"}, "evidence_span": "A 61-year-old man", "required": true, "notes": null},
    {"field": "key_findings", "value": ["Left-sided pleural effusion with nodular pleural thickening on CT", "Bloody pleural fluid on thoracentesis", "Biopsy: epithelioid-type cells with very long microvilli", "Decreased breath sounds at left lung base"], "evidence_span": null, "required": true, "notes": null},
    {"field": "comorbidities", "value": ["hypertension", "hypercholesterolemia", "peptic ulcer disease"], "evidence_span": "The patient's medical conditions include hypertension, hypercholesteremia and peptic ulcer disease", "required": true, "notes": null},
    {"field": "smoking_history", "value": "2 packs/day for 30 years (60 pack-years, current smoker)", "evidence_span": "He smokes 2 packs of cigarettes daily for the past 30 years", "required": false, "notes": null}
  ]
}
```

### C2. Pipeline Cache for trec-20226

Handled by `scripts/run_demo_cache.py --patient trec-20226`. Key differences from mpx1016:
- **INGEST**: Text-only (skip image step)
- **PRESCREEN**: Agent gets hint in patient_note to search all statuses for historical validation
- **Ground truth**: After VALIDATE, compare against `data/trec2022_ground_truth/qrels.tsv` (648 lines, format: `query-id\tcorpus-id\tscore` where score: 2=eligible, 1=excluded, 0=not_relevant)

### C3-C4. Ground Truth — Backend Only

Script computes and saves `ground_truth_comparison.json`. Not displayed in UI. For Kaggle writeup only.

---

## Part D: Files Changed Summary

| File | Action | What Changes |
|------|--------|-------------|
| `scripts/run_demo_cache.py` | **CREATE** | Pipeline runner + cache writer (both patients) |
| `data/harness schema/ingest/nsclc_demo_harness.json` | **MODIFY** | Add trec-20226 patient entry (C1) |
| `demo/pages/1_Pipeline_Demo.py` | **MODIFY** | Fix require_clinical_guidance bug (B0), animations (B1-B3), model labels (B4), pipeline overview (B6) |
| `demo/components/patient_card.py` | **MODIFY** | st.text→st.markdown (B5a), image expanded (B5b) |
| `demo/components/trial_card.py` | **MODIFY** | st.text→st.markdown (B5c), truncation 60→80 (B5f) |
| `demo/components/pipeline_viewer.py` | **MODIFY** | Model attribution labels (B4), imaging_findings formatting (B5d) |

---

## Execution Order

0. **Save this plan** to `docs/plans/2026-02-24-two-patient-demo.md`
1. **Fix bug B0** — remove `require_clinical_guidance=True` (blocks all live/cached runs)
2. **Add trec-20226 to demo harness** (C1)
3. **Create `scripts/run_demo_cache.py`** (Part A)
4. **UI changes** — B1-B6 modifications (can be done in parallel with endpoint deploy)
5. **Deploy endpoints** — both 4B and 27B (parallel, ~15-30 min)
6. **Run pipeline for mpx1016** → cache
7. **Run pipeline for trec-20226** → cache + ground truth
8. **Undeploy endpoints** immediately
9. **Test demo** — launch Streamlit, verify both patients

Steps 1-4 are code-only (no endpoints needed). Steps 5-8 need live endpoints.

---

## Fallback Strategies (ALL REQUIRE HUMAN APPROVAL)

1. **VALIDATE judges all trials EXCLUDED**: Script stops, reports. User decides.
2. **MedGemma 1.5 4B/27B endpoint fails**: Script stops, asks user.
3. **PRESCREEN finds too few candidates**: Script stops, asks user.
4. **trec-20226 ground truth recall is low**: Script stops, reports. For Kaggle writeup.

---

## Verification

### mpx1016 (multimodal):
1. `uv run streamlit run demo/app.py --server.port 8501`
2. Select mpx1016, cached mode → click "Search for Trials"
3. Verify: INGEST animation with CT image analysis step
4. Verify: PRESCREEN animation shows search_trials + consult_medical_expert tool calls
5. Verify: VALIDATE animation with criterion-by-criterion evaluation
6. Verify: >= 1 ELIGIBLE trial card
7. Verify: Pipeline overview header with 3-model narrative
8. Verify: Clinical note renders as markdown (not monospace)
9. Verify: CT image expanded by default in patient mode

### trec-20226 (text-only):
10. Select trec-20226, cached mode → click "Search for Trials"
11. Verify: INGEST animation without image step
12. Verify: PRESCREEN + VALIDATE animations work
13. Verify: `ground_truth_comparison.json` in cache

### Both patients:
14. Toggle dev/patient mode — both views work
15. Live mode: verify no TypeError from require_clinical_guidance
