# Design: MedGemma Impact Challenge Demo (v2)

**Date**: 2026-02-21
**Deadline**: 2026-02-24 (Kaggle submission close)
**Competition**: https://www.kaggle.com/competitions/med-gemma-impact-challenge

---

## Competition Requirements (Confirmed)

| Component | Requirement |
|-----------|-------------|
| **Video** | 3-minute max demo video |
| **Writeup** | 3-page max technical documentation (Kaggle Writeup format) |
| **Code** | Complete, reproducible source code |
| **Model** | Must use >= 1 HAI-DEF model (MedGemma). We use **4B + 27B** |
| **Prize pool** | $75,000 (main) + special awards |
| **Special awards** | Agent-based workflows, fine-tuned models, edge AI |
| **Platform** | Kaggle Writeups (can revise before deadline) |

### Judging Criteria (5 dimensions)

1. **Effective use of HAI-DEF models** — how models are applied, integrated, deployed
2. **Problem significance** — importance of the healthcare problem addressed
3. **Real-world impact** — potential for real deployment
4. **Technical feasibility** — working, reproducible solution
5. **Execution & communication quality** — code quality, presentation clarity

Key insight: judges evaluate "how models are applied in realistic healthcare contexts", not isolated model performance.

---

## What We Have

| Asset | Status | Details |
|-------|--------|---------|
| CLI benchmark pipeline | Ready | INGEST (gold) -> VALIDATE -> METRICS, 140 tests passing |
| PRESCREEN agent | Ready | Gemini agentic loop + CT.gov API + MedGemma normalize |
| 37 NSCLC patient profiles | Ready | `nsclc_trial_profiles.json` with key_facts pre-extracted |
| MedGemma 1.5 4B endpoint | Active | HF Inference, smoke tests passing |
| MedGemma 27B endpoint | Active | TGI on A100 80GB, smoke tests passing |
| Gemini 3 Pro | Active | API key validated |
| Phase 0 benchmark | NOT RUN | Config ready, needs live execution |

---

## Architecture (v2 — Streamlit)

```
Streamlit App (localhost:8501)
┌──────────────────────────────────────────────────────────┐
│  SIDEBAR                   │  MAIN PANEL                  │
│  ┌────────────────────┐    │  ┌────────────────────────┐  │
│  │ Patient Selector   │    │  │ Pipeline Viewer         │  │
│  │ (37 NSCLC cases)   │    │  │  ├─ INGEST (key facts) │  │
│  │                    │    │  │  ├─ PRESCREEN (CT.gov)  │  │
│  │ Run Pipeline [btn] │    │  │  └─ VALIDATE (criteria) │  │
│  │                    │    │  ├────────────────────────┤  │
│  │ ──────────────     │    │  │ Results Panel           │  │
│  │ Benchmark Tab [btn]│    │  │  ├─ Matched Trials      │  │
│  └────────────────────┘    │  │  ├─ Per-criterion table │  │
│                            │  │  └─ Trial links         │  │
│                            │  ├────────────────────────┤  │
│                            │  │ Benchmark Dashboard     │  │
│                            │  │  ├─ Accuracy/F1/Kappa   │  │
│                            │  │  ├─ Confusion matrices  │  │
│                            │  │  └─ Model comparison    │  │
│                            │  └────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
         │              │              │
     MedGemma 1.5 4B    MedGemma 27B    Gemini 3 Pro
      (HF TGI)      (HF TGI)     (AI Studio)
         │
     CT.gov API v2
```

No separate backend. Streamlit calls Python functions directly:
- `run_prescreen_agent()` for PRESCREEN
- `evaluate_criterion()` for VALIDATE
- Patient profiles loaded from `nsclc_trial_profiles.json`

### Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| UI | **Streamlit** | 10x faster than Next.js, built-in streaming, `st.status()` |
| Pipeline | Existing Python modules | Zero new backend code |
| Models | `huggingface_hub` + `google-genai` | Already implemented |
| QA + Recording | **Playwright CLI** | Doubles as QA harness and video recorder |

---

## Sample Patients (from `nsclc_trial_profiles.json`)

37 NSCLC patients with pre-extracted `key_facts`. For demo, select 2-3 representative cases:

| Case | Topic ID | Why Selected |
|------|----------|-------------|
| 1 | `mpx1016` | 43F, never-smoker, adenocarcinoma with signet-ring cells, imaging findings |
| 2 | `mpx1201` | 47M, heavy smoker, metastatic adenocarcinoma, multi-site disease |
| 3 | TBD | Pick one with biomarkers (EGFR/ALK) if available in dataset |

Each profile has: `profile_text` (EHR note), `key_facts` (structured), `ambiguities`.

---

## Demo Pages (Streamlit)

### Page 1: Pipeline Demo (main)

**Flow**: Select Patient -> Run Pipeline -> See Results

1. **Patient Selector** (sidebar): dropdown of 37 patients, shows profile summary on select
2. **INGEST step**: Display pre-extracted `key_facts` from JSON (demographics, diagnosis, imaging, symptoms, missing info). Uses `st.expander()` for detail.
3. **PRESCREEN step**: Run `run_prescreen_agent()` live with `st.status()` streaming. Shows:
   - Tool calls in real-time (search_trials, normalize_medical_terms, get_trial_details)
   - Candidate trials found (NCT IDs, titles, conditions)
   - Agent reasoning summary
4. **VALIDATE step**: For top-N candidate trials, run `evaluate_criterion()` per criterion with `st.status()`. Shows:
   - Per-criterion MET/NOT_MET/UNKNOWN verdict with reasoning
   - Evidence sentence highlights
   - Trial-level aggregation (ELIGIBLE/EXCLUDED/UNCERTAIN)
5. **Results Panel**: Final ranked trial list with ClinicalTrials.gov links

### Page 2: Benchmark Dashboard

Shows pre-computed Phase 0 results (loaded from `runs/<run_id>/`):

1. **Model Comparison Table**: accuracy, F1, Cohen's kappa for MedGemma 1.5 4B vs 27B vs Gemini vs GPT-4 baseline
2. **Confusion Matrices**: side-by-side heatmaps (3 models)
3. **Per-criterion Audit Table**: expandable, from `audit_table.md`
4. **Cost & Latency**: tokens, USD cost, avg latency per model
5. **Key Findings**: bullet points on MedGemma vs Gemini error patterns

---

## 3-Page Writeup Structure (Kaggle Writeup Format)

### Page 1: Problem & Approach

**Section 1.1: The Problem** (~300 words)
- Clinical trial matching is a critical bottleneck: <5% of cancer patients enroll in trials
- Manual eligibility screening takes 2+ hours per patient per trial
- Requires cross-referencing patient records against complex inclusion/exclusion criteria
- High stakes: wrong match = wasted time for patients and sites; missed match = lost treatment option

**Section 1.2: Our Approach** (~300 words)
- Three-stage pipeline: INGEST -> PRESCREEN -> VALIDATE
- MedGemma as domain-specialized medical reasoning engine
- Agent-based CT.gov search (qualifies for special award)
- Quantitative comparison vs general-purpose model (Gemini 3 Pro) and GPT-4 baseline

**Section 1.3: HAI-DEF Model Usage** (~200 words)
- MedGemma 1.5 4B: medical term normalization in PRESCREEN (search_variants for CT.gov)
- MedGemma 27B: criterion-level eligibility evaluation in VALIDATE
- Both: clinical reasoning with medical domain knowledge vs general-purpose Gemini 3 Pro

### Page 2: Architecture & Implementation

**Section 2.1: System Architecture** (~200 words + diagram)
- Pipeline diagram: INGEST -> PRESCREEN -> VALIDATE
- Component isolation: gold SoT for benchmark, model output for E2E
- Model adapter pattern: common interface across MedGemma (4B, 27B) and Gemini

**Section 2.2: PRESCREEN Agent** (~250 words + diagram)
- Gemini 3 Pro orchestrates multi-turn agentic loop
- Three tools: `normalize_medical_terms` (MedGemma), `search_trials` (CT.gov), `get_trial_details` (CT.gov)
- MedGemma produces CT.gov-optimized search variants (avoiding false positives)
- Budget guard, rate limiting, retry with backoff

**Section 2.3: VALIDATE Evaluator** (~200 words)
- Criterion-type aware prompting (inclusion vs exclusion)
- Structured JSON output with label, reasoning, evidence_sentences
- Label mapping: 6-class TrialGPT -> 3-class (MET/NOT_MET/UNKNOWN)

**Section 2.4: Reproducibility** (~150 words)
- Every run persists to `runs/<run_id>/` with config, results, metrics, cost
- Deterministic sampling (seed=42), version-pinned dependencies
- 140 unit tests, 4 BDD scenarios, 3 smoke tests

### Page 3: Results & Impact

**Section 3.1: Benchmark Results** (~300 words + table + confusion matrix)
- Phase 0: 20-pair criterion-level evaluation (3 models)
- Metrics table: accuracy, macro-F1, MET/NOT_MET F1, Cohen's kappa
- MedGemma vs Gemini error pattern analysis
- GPT-4 baseline comparison (pre-computed from HF dataset)

**Section 3.2: Demo Walkthrough** (~200 words)
- NSCLC patient -> PRESCREEN finds trials -> VALIDATE checks criteria -> ranked results
- Real-world scenario: oncologist uses tool to identify trials for a newly diagnosed patient

**Section 3.3: Real-World Impact** (~200 words)
- Reduce screening time from hours to minutes
- MedGemma's medical domain knowledge reduces false positives/negatives
- Agent workflow automates CT.gov search (currently manual)
- Path to deployment: integrate with EHR systems, clinical decision support

**Section 3.4: Limitations & Future Work** (~100 words)
- Current: text-only (4B multimodal planned)
- Dataset limited to TrialGPT annotations (1,024 pairs)
- No IRB-approved clinical validation yet
- Next: full 1,024-pair Tier A evaluation, INGEST model implementation

---

## Priority (v2 — revised)

### P0 — Must ship (submission-blocking)

1. **Run Phase 0 benchmark** (3 models, 20 pairs) — produces numbers for writeup
2. **Streamlit app**: patient selector + pipeline viewer + results panel
3. **Benchmark dashboard page** in Streamlit (from run artifacts)
4. **3-page Kaggle Writeup** (structure above)
5. **Playwright QA + demo recording** (3-min video)

### P1 — Strong differentiator

6. MedGemma 27B for VALIDATE (not just 4B) — show model size comparison
7. PRESCREEN agent live demo with real CT.gov queries
8. Error pattern analysis (MedGemma vs Gemini qualitative differences)

### P2 — If time permits

9. MedGemma 1.5 4B multimodal image input for imaging cases
10. ClinicalTrials.gov deep links for matched trials
11. Export report as PDF

---

## Milestone Plan (v2)

| Day | Date | Morning (focus) | Afternoon (focus) | Evening (polish) |
|-----|------|-----------------|-------------------|------------------|
| **1** | Feb 21 | **Run Phase 0 benchmark** (3-way). Fix any adapter issues. Collect metrics. | **Streamlit app scaffold**: patient selector (load nsclc_trial_profiles.json), INGEST display (key_facts from JSON), pipeline skeleton | Wire PRESCREEN agent into Streamlit with `st.status()` streaming |
| **2** | Feb 22 | **VALIDATE integration**: wire evaluate_criterion into Streamlit, results panel | **Benchmark dashboard page**: load run artifacts, display metrics table + confusion matrix charts | **Start writeup**: Page 1 (Problem & Approach) + Page 2 skeleton |
| **3** | Feb 23 | **Complete writeup**: Page 2 (Architecture) + Page 3 (Results). Insert benchmark numbers | **UI polish**: error handling, loading states, edge cases. Playwright QA pass | **Record demo**: QuickTime screen recording + script narration |
| **4** | Feb 24 | **Final QA**: Playwright full flow test. Fix any issues | **Writeup final edit**, code cleanup, README update | **Submit to Kaggle** before deadline |

### Critical Path

```
Day 1 AM: Phase 0 benchmark ──► Day 1 PM: Streamlit scaffold
                                      │
                    ┌─────────────────┘
                    ▼
Day 2 AM: VALIDATE in Streamlit ──► Day 2 PM: Benchmark dashboard
                                          │
                         ┌────────────────┘
                         ▼
Day 3 AM: Writeup (needs benchmark numbers) ──► Day 3 PM: QA + Video
                                                       │
                                      ┌────────────────┘
                                      ▼
Day 4: Final QA ──► Submit
```

Phase 0 benchmark results are the critical dependency — everything else (writeup, dashboard, demo narrative) depends on having real numbers.

---

## Streamlit File Structure

```
demo/
├── app.py                    # Main Streamlit app (multipage)
├── pages/
│   ├── 1_Pipeline_Demo.py    # Patient selector + pipeline runner
│   └── 2_Benchmark.py        # Pre-computed benchmark results
├── components/
│   ├── patient_card.py       # Patient profile display
│   ├── pipeline_viewer.py    # INGEST/PRESCREEN/VALIDATE steps
│   └── benchmark_charts.py   # Metrics visualizations
└── data/
    └── (symlink or copy of nsclc_trial_profiles.json)
```

---

## Playwright QA + Recording

Use `playwright-cli` skill for:

1. **QA harness** — automated flow: select patient -> run pipeline -> verify results render
2. **Demo recording** — capture the full pipeline execution as GIF/video for submission
3. **Regression** — re-run after UI changes to catch breakage

Playwright script covers:
- Navigate to Streamlit app
- Select patient from dropdown
- Click "Run Pipeline"
- Wait for each step to complete (INGEST -> PRESCREEN -> VALIDATE)
- Verify results panel shows trials
- Switch to Benchmark page
- Verify charts render

---

## Competition Story (refined)

**Narrative**: Clinical trial matching is the #1 bottleneck in oncology research — fewer than 5% of cancer patients find a matching trial. TrialMatch demonstrates how MedGemma's medical domain knowledge enables an AI-powered pipeline that:

1. **Understands** patient records (INGEST with key fact extraction)
2. **Searches** ClinicalTrials.gov autonomously (PRESCREEN agent — qualifies for **agent-based workflow award**)
3. **Evaluates** each eligibility criterion with clinical reasoning (VALIDATE)
4. **Measures** performance against expert annotations with rigorous benchmarking

Differentiators for judges:
- Uses **both MedGemma sizes** (4B for normalization, 27B for reasoning) — strong model utilization score
- **Quantitative evidence** (benchmark vs GPT-4 baseline) — not just a demo, but measured science
- **Agent architecture** — genuine agentic workflow, not scripted calls
- **37 real NSCLC patients** — realistic clinical data, not toy examples
- **140 tests** — production-quality engineering
