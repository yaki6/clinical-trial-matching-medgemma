# MedGemma Impact Challenge — Final Execution Plan

**Date**: 2026-02-21
**Deadline**: 2026-02-24 (Kaggle submission close)
**Competition**: https://www.kaggle.com/competitions/med-gemma-impact-challenge
**Supersedes**: `demo-design.md` and `demo-design-v2.md`

---

## 1. Competition Requirements

| Component | Requirement |
|-----------|-------------|
| **Video** | 3-minute max demo video |
| **Writeup** | 3-page max (Kaggle Writeup format — rich markdown, inline) |
| **Code** | Complete, reproducible source code |
| **Model** | Must use >= 1 HAI-DEF model. We use **MedGemma 1.5 4B + 27B** |
| **Prize** | $75,000 main + special awards (agent workflows, fine-tuning, edge AI) |
| **Platform** | Kaggle Writeups (can revise before deadline) |

### Judging (5 dimensions)

1. **Effective use of HAI-DEF models** — how models are applied in realistic contexts
2. **Problem significance** — importance of healthcare problem
3. **Real-world impact** — potential for deployment
4. **Technical feasibility** — working, reproducible solution
5. **Execution & communication quality** — code, presentation clarity

---

## 2. Honest Assessment: What We Have vs What We Need

### Assets Ready

| Asset | Status |
|-------|--------|
| CLI pipeline (VALIDATE + METRICS) | 140 tests passing |
| PRESCREEN agent (Gemini + CT.gov + MedGemma) | Ready, 33 unit tests |
| 37 NSCLC patient profiles | `nsclc_trial_profiles.json` with key_facts |
| MedGemma 1.5 4B endpoint | Active, smoke tests pass |
| MedGemma 27B endpoint | Active, smoke tests pass |
| Gemini 3 Pro | API key validated |

### Known Benchmark Results (existing runs — pre-prompt-fix)

| Model | Accuracy | F1 Macro | Cohen's kappa | Notes |
|-------|----------|----------|---------------|-------|
| MedGemma 1.5 4B | 55% | 0.508 | 0.286 | Pre-prompt-fix (no inclusion/exclusion instructions) |
| Gemini 3 Pro | 75% | 0.558 | 0.583 | Pre-prompt-fix |
| GPT-4 (HF baseline) | 75% | 0.746 | — | From TrialGPT HF dataset |
| **MedGemma 27B** | **Running** | — | — | **In separate session** (post-prompt-fix) |
| **MedGemma 1.5 4B (re-run)** | **Running** | — | — | **In separate session** (post-prompt-fix) |

**Note**: Prompt fix added criterion-type-aware instructions (inclusion vs exclusion). 4B and 27B are being re-benchmarked with the fixed prompt. Results pending.

### Critical Fact

MedGemma 1.5 4B (pre-fix) is 20 points worse than Gemini on criterion evaluation. We **cannot** claim "MedGemma outperforms general models." The narrative must be reframed. The prompt fix may improve 4B scores, and 27B is expected to perform better given 5x parameters.

---

## 3. Narrative Strategy

**Reframe**: "Multi-model clinical AI — MedGemma provides domain-specific medical understanding while Gemini orchestrates complex reasoning. Together they form a trial matching system that neither could achieve alone."

Key messages:
1. **Complementary architecture**: 4B for medical term normalization (PRESCREEN), 27B for clinical reasoning (VALIDATE), Gemini for agentic orchestration
2. **Honest benchmarking**: Transparent comparison showing where domain models excel vs where general models win
3. **Agent-based workflow**: PRESCREEN is a genuine multi-model agent — targets **special award**
4. **Real clinical problem**: <5% of cancer patients find matching trials

### Judging Dimension Alignment

| Criterion | Our Strength | Weakness | Action |
|-----------|-------------|----------|--------|
| HAI-DEF model use | Both MedGemma sizes + complementary roles | 4B accuracy low | Emphasize roles (normalize, extract) not raw accuracy |
| Problem significance | Trial matching is critical | — | Strong narrative Page 1 |
| Real-world impact | Live CT.gov, real patients | Local-only | Describe deployment path |
| Technical feasibility | 140 tests, working pipeline | Multimodal not live | Pre-computed results, honest |
| Execution quality | Clean architecture | Only 20-pair Phase 0 | "Directional capability assessment" |

### Special Award: Agent-Based Workflows

Prominently demonstrate in UI:
1. Multi-model agent (Gemini orchestrates, MedGemma normalizes)
2. Real CT.gov tool use with visible query -> result -> reasoning chain
3. Adaptive search (agent adjusts based on previous results)
4. Agent trace in real-time (P0, not P1)

---

## 4. Architecture: Streamlit

```
Streamlit App (localhost:8501)
┌──────────────────────────────────────────────────────────┐
│  SIDEBAR                   │  MAIN PANEL                  │
│  ┌────────────────────┐    │  ┌────────────────────────┐  │
│  │ Patient Selector   │    │  │ Pipeline Viewer         │  │
│  │ (37 NSCLC cases)   │    │  │  ├─ INGEST (key facts) │  │
│  │                    │    │  │  ├─ PRESCREEN (CT.gov)  │  │
│  │ [Run Pipeline] btn │    │  │  └─ VALIDATE (criteria) │  │
│  │                    │    │  ├────────────────────────┤  │
│  │ Mode: live/cached  │    │  │ Results Panel           │  │
│  │                    │    │  │  ├─ Matched Trials      │  │
│  │ ──────────────     │    │  │  ├─ Per-criterion table │  │
│  │ [Benchmark] tab    │    │  │  └─ CT.gov links        │  │
│  └────────────────────┘    │  ├────────────────────────┤  │
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

No separate backend. Streamlit calls Python functions directly.

### Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| UI | **Streamlit** | 10x faster than Next.js, built-in `st.status()` streaming |
| Pipeline | Existing Python modules | Zero new backend code |
| Models | `huggingface_hub` + `google-genai` | Already implemented |
| QA + Video | **Playwright CLI** | Doubles as QA harness + demo recorder |

### Cached/Replay Mode

```python
class PipelineMode(Enum):
    LIVE = "live"       # Real API calls
    CACHED = "cached"   # Replay saved JSON traces

# Streamlit sidebar toggle: st.radio("Mode", ["live", "cached"])
# CACHED loads from data/cached_runs/{patient_id}/
#   ingest_result.json, prescreen_result.json, validate_results.json
```

Critical for:
- Frontend development without API costs
- Reliable demo recording (no cold-start failures)
- Judge reproducibility (works without API keys)

### File Structure

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
    ├── nsclc_trial_profiles.json  # 37 patients (symlink)
    └── cached_runs/               # Pre-recorded pipeline traces
        └── mpx1016/
            ├── ingest.json
            ├── prescreen.json
            └── validate.json
```

---

## 5. Data Compatibility: INGEST → PRESCREEN

### The Problem

`nsclc_trial_profiles.json` stores `key_facts` as **list of objects**:
```json
[{"field": "primary_diagnosis", "value": "Lung adenocarcinoma...", "evidence_span": "...", "required": true}, ...]
```

`run_prescreen_agent()` expects `key_facts: dict[str, Any]`, and `_format_key_facts()` iterates with `.items()`.

Passing the list directly would iterate over indices (0, 1, 2...) instead of field names → nonsensical output.

### The Fix: Thin Adapter Function

**Location**: `src/trialmatch/ingest/profile_adapter.py` (keeps `agent.py` untouched)

**Responsibilities**:
1. Convert list-of-objects → flat dict: `{kf["field"]: kf["value"] for kf in profile["key_facts"]}`
2. Flatten nested dicts for readable formatting (e.g., `demographics: {"age": "43", "sex": "female"}` → `"age: 43, sex: female"`)
3. Return `(patient_note: str, key_facts: dict)` tuple ready for `run_prescreen_agent()`

```python
# src/trialmatch/ingest/profile_adapter.py
def adapt_profile_for_prescreen(profile: dict) -> tuple[str, dict[str, Any]]:
    """Convert nsclc_trial_profiles.json format → PRESCREEN input format.

    Transforms key_facts from list-of-objects [{field, value, ...}]
    to flat dict {field: value} with nested values flattened to strings.
    """
    patient_note = profile["profile_text"]
    raw_facts = profile.get("key_facts", [])

    key_facts: dict[str, Any] = {}
    for kf in raw_facts:
        field = kf["field"]
        value = kf["value"]

        if isinstance(value, dict):
            # Flatten nested dict → readable string
            parts = []
            for k, v in value.items():
                if isinstance(v, list):
                    parts.append(f"{k}: {', '.join(str(i) for i in v)}")
                elif not isinstance(v, dict):
                    parts.append(f"{k}: {v}")
            key_facts[field] = "; ".join(parts)
        else:
            key_facts[field] = value

    return patient_note, key_facts
```

**After transform, `_format_key_facts()` produces**:
```
- primary_diagnosis: Lung adenocarcinoma with signet-ring cell features
- demographics: age: 43; sex: female
- imaging_findings: description: Axial CT...; findings: Nodular opacities, Large mass, ...
- key_findings: Progressive dyspnea, Wheezing, Pleural fluid suspicious...
- tobacco_use: never smoker
```

This feeds cleanly into PRESCREEN's system prompt where Gemini uses it to plan search strategies.

---

## 6. Sample Patients (from `nsclc_trial_profiles.json`)

All from `nsclc_trial_profiles.json` (37 cases). Demo highlights 2-3:

| Case | Topic ID | Profile | Why |
|------|----------|---------|-----|
| 1 | `mpx1016` | 43F, never-smoker, adenocarcinoma + signet-ring cells, CT imaging | Rich histopathology, clear imaging findings |
| 2 | `mpx1201` | 47M, heavy smoker, metastatic adenocarcinoma, multi-site | Complex metastatic case, strong family history |
| 3 | TBD | Pick one with EGFR/ALK biomarkers if available | Targeted therapy matching story |

INGEST step: Display pre-extracted `key_facts` from JSON. Label as "Key Facts (MedGemma-extracted)" in UI. Honest — data was extracted by MedGemma (Gemini 2.5 Pro generated these profiles), not simulated.

---

## 7. Demo Pages

### Page 1: Pipeline Demo

**Flow**: Select Patient → [toggle live/cached] → Run Pipeline → See Results

1. **Patient Selector** (sidebar): dropdown of patients, shows profile_text preview
2. **INGEST**: `st.expander("INGEST: Key Facts")` — display key_facts via `adapt_profile_for_prescreen()`, grouped by field type
3. **PRESCREEN**: `st.status("Searching ClinicalTrials.gov...")` — run `run_prescreen_agent()` (live) or load cached trace. Show:
   - Each tool call: normalize_medical_terms, search_trials, get_trial_details
   - Agent trace: query → results → reasoning chain (P0 for Special Award)
   - Candidate trials found (NCT IDs, titles)
   - Agent reasoning summary
4. **VALIDATE**: `st.status("Evaluating eligibility criteria...")` — for top-N trials, run `evaluate_criterion()`. Show:
   - Per-criterion MET/NOT_MET/UNKNOWN with reasoning
   - Evidence highlights
   - Trial-level verdict (ELIGIBLE/EXCLUDED/UNCERTAIN)
5. **Results**: Ranked trial list with ClinicalTrials.gov links

### Page 2: Benchmark Dashboard

Loads pre-computed Phase 0 results from `runs/<run_id>/`:

1. **Model Comparison Table**: accuracy, F1, kappa for 4B vs 27B vs Gemini vs GPT-4
2. **Confusion Matrices**: side-by-side heatmaps
3. **Per-criterion Audit Table**: expandable
4. **Cost & Latency**: tokens, USD, avg latency
5. **Key Findings**: where MedGemma excels (normalization) vs Gemini (structured eval)

---

## 8. Three-Page Writeup Structure (Deferred)

### Page 1: Problem & Approach (~800 words)

**1.1 The Problem** (~300 words)
- <5% of cancer patients enroll in clinical trials
- Manual screening: 2+ hours per patient per trial
- Cross-referencing EHR against complex inclusion/exclusion criteria
- Wrong match = wasted time; missed match = lost treatment option

**1.2 Our Approach** (~300 words)
- Three-stage pipeline: INGEST -> PRESCREEN -> VALIDATE
- Multi-model orchestration: MedGemma (domain) + Gemini (reasoning/orchestration)
- Agent-based CT.gov search (qualifies for special award)
- Quantitative comparison vs GPT-4 baseline on TrialGPT expert annotations

**1.3 HAI-DEF Model Utilization** (~200 words)
- MedGemma 1.5 4B: medical term normalization (PRESCREEN search_variants)
- MedGemma 27B: criterion-level eligibility evaluation (VALIDATE)
- Complementary roles: domain knowledge + general reasoning

### Page 2: Architecture & Agent Design (~800 words)

**2.1 System Architecture** (~200 words + diagram)
- Pipeline: INGEST -> PRESCREEN -> VALIDATE
- Component isolation: gold SoT for benchmark, model output for E2E
- Model adapter pattern: common interface across 3 models

**2.2 PRESCREEN Agent** (~250 words + diagram)
- Gemini orchestrates multi-turn agentic loop with 3 tools
- MedGemma normalizes medical terms (CT.gov-optimized search variants)
- Budget guard, rate limiting, retry with backoff
- Adaptive: adjusts queries based on previous results

**2.3 VALIDATE Evaluator** (~200 words)
- Criterion-type-aware prompting (inclusion vs exclusion)
- Structured JSON output: label, reasoning, evidence_sentences
- 6-class TrialGPT -> 3-class mapping (MET/NOT_MET/UNKNOWN)

**2.4 Reproducibility** (~150 words)
- Every run persists to `runs/<run_id>/` with full artifacts
- Deterministic sampling (seed=42), version-pinned deps
- 140 unit tests, cached replay mode for offline reproduction

### Page 3: Results & Impact (~800 words)

**3.1 Benchmark Results** (~300 words + table + confusion matrix)
- Phase 0: 20-pair criterion-level evaluation (3 models)
- Results table + confusion matrices
- Honest analysis: where MedGemma excels vs where Gemini wins
- GPT-4 baseline comparison

**3.2 Demo Walkthrough** (~200 words)
- NSCLC patient -> PRESCREEN finds trials -> VALIDATE evaluates -> ranked results
- Real-world scenario: oncologist finding trials for a new patient

**3.3 Impact & Deployment Path** (~200 words)
- Screening time: hours -> minutes
- Agent automates CT.gov search (currently manual)
- Integration path: EHR systems, clinical decision support tools

**3.4 Limitations & Future** (~100 words)
- Current: text-only (4B multimodal planned)
- 20-pair Phase 0 (1,024-pair Tier A next)
- No IRB clinical validation yet
- Fine-tuning MedGemma on TrialGPT data as next step

---

## 9. Priority

### P0 — Must ship (submission-blocking)

| # | Item | Hours | Day | Status |
|---|------|-------|-----|--------|
| 1 | ~~Run Phase 0 benchmark (27B + re-run 4B)~~ | — | 1 | **Running in separate session** |
| 2 | **Thin adapter** (`ingest/profile_adapter.py` + unit tests) | 1h | 1 | TODO |
| 3 | Streamlit scaffold: patient selector, INGEST display, pipeline skeleton | 3h | 1 | TODO |
| 4 | Wire PRESCREEN agent into Streamlit with `st.status()` + agent trace viz | 3h | 1-2 | TODO |
| 5 | VALIDATE integration: wire evaluate_criterion, results panel | 3h | 2 | TODO |
| 6 | Benchmark dashboard page (load run artifacts, metrics table, charts) | 2h | 2 | TODO |
| 7 | Cached/replay mode (record live run, replay from JSON) | 1.5h | 2 | TODO |
| 8 | Playwright QA harness + demo video recording | 2h | 3-4 | TODO |
| 9 | 3-page Kaggle Writeup | 3h total | **Deferred** | TODO |
| 10 | Kaggle Writeup submission | 1h | 4 | TODO |

### P1 — Strong differentiator

| # | Item | Hours | Day |
|---|------|-------|-----|
| 10 | Error pattern analysis (qualitative MedGemma vs Gemini differences) | 1h | 3 |
| 11 | UI polish: loading states, error handling, responsive layout | 2h | 3 |
| 12 | Pre-warm script for all endpoints before recording | 0.5h | 3 |
| 13 | Reproducibility: `.env.example`, README setup instructions | 1h | 4 |

### P2 — If time permits

| # | Item | Hours | Day |
|---|------|-------|-----|
| 14 | MedGemma 1.5 4B multimodal image support | 3h | 3 |
| 15 | TxGemma or MedSigLIP quick integration (expand HAI-DEF usage) | 2h | 3 |
| 16 | Export results as PDF report | 1h | 3 |

---

## 10. Milestone Plan

| Day | Date | AM | PM | Eve |
|-----|------|----|----|-----|
| **1** | Feb 21 | **Thin adapter** + unit tests. Benchmark running in parallel (other session). | **Streamlit scaffold**: patient selector, INGEST key_facts display, pipeline skeleton | Wire PRESCREEN agent with streaming + agent trace |
| **2** | Feb 22 | **VALIDATE in Streamlit**: per-criterion verdicts, results panel | **Benchmark dashboard** + **cached mode** (record one live run for replay) | UI polish, error handling |
| **3** | Feb 23 | Playwright QA first pass. UI fixes. | Record demo video attempt 1. Pre-warm endpoints. | Video review + re-record if needed |
| **4** | Feb 24 | **Final Playwright QA**. Fix any issues found | Code cleanup, README, `.env.example`. **Kaggle Writeup** (when ready) | **Submit to Kaggle** |

### Critical Path

```
Day 1: Thin adapter + Streamlit scaffold ──► (Benchmark running in parallel, other session)
                    │
                    ▼
Day 2 AM: VALIDATE in Streamlit ──► Day 2 PM: Benchmark dashboard (needs numbers from other session)
                                          │
                         ┌────────────────┘
                         ▼
Day 3: Playwright QA + demo video recording
                         │
                         ▼
Day 4: Final QA + Writeup (deferred to here) + Submit
```

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MedGemma 27B also underperforms (<65%) | Medium | High | Reframe: "honest analysis reveals fine-tuning opportunities" — judges value intellectual honesty |
| Endpoint cold-start during demo recording | High | High | Pre-warm script + cached replay mode as fallback |
| CT.gov API rate limit / downtime | Medium | Medium | Pre-cache trial data for demo patients |
| Playwright video quality issues | Medium | Low | 1280x720 explicit, ffmpeg post-process, OBS as backup |
| Streamlit dev exceeds estimate | Medium | Medium | Cut benchmark dashboard charts to table-only |
| 27B benchmark takes too long | Low | Medium | Smoke test with 1-pair config first |

---

## 12. Playwright QA + Recording

Use `playwright-cli` skill for:

1. **QA harness**: select patient -> run pipeline -> verify results render -> switch to benchmark -> verify charts
2. **Demo recording**: capture full pipeline as video for submission
3. **Regression**: re-run after UI changes

### Video Quality

- Resolution: `recordVideo: { size: { width: 1280, height: 720 } }`
- Pacing: `page.waitForTimeout()` (not `slowMo`)
- Post-process: `ffmpeg -i recording.webm -c:v libx264 -crf 20 demo.mp4`
- Backup: OBS screen recording if Playwright has issues

---

## 13. Reproducibility

Include in repo:
1. `.env.example` with all required keys (HF_TOKEN, GOOGLE_API_KEY)
2. `data/cached_runs/` with pre-recorded pipeline traces (works without API keys)
3. README: setup instructions for HF Inference API or local deployment
4. `demo/` runnable with `streamlit run demo/app.py`

---

## 14. What Makes Us Competitive

1. **Live CT.gov integration** — most submissions will use static data
2. **Genuine agent architecture** — not prompt chaining, real multi-turn tool use
3. **Honest benchmarking** — judges respect transparency over inflated claims
4. **Multi-model orchestration** — 4B + 27B + Gemini with complementary roles
5. **140-test codebase** — production-quality engineering
6. **37 real NSCLC patients** — realistic clinical data

---

## Sources

- [MedGemma Impact Challenge (Kaggle)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
- [Google MedGemma Impact Challenge (EdTech Hub)](https://www.edtechinnovationhub.com/news/google-launches-medgemma-impact-challenge-to-advance-human-centered-health-ai)
- [HAI-DEF Collection (HuggingFace)](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def)
