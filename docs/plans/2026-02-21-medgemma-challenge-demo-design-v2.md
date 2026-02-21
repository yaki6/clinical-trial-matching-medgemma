# Design v2: MedGemma Impact Challenge Demo (Revised Assessment)

**Date**: 2026-02-21
**Deadline**: 2026-02-24 (Kaggle submission close)
**Competition**: https://www.kaggle.com/competitions/med-gemma-impact-challenge
**Supersedes**: `2026-02-21-medgemma-challenge-demo-design.md`

---

## Part 1: Flaws & Gaps in v1 Plan

### CRITICAL FLAWS (submission-threatening)

#### FLAW 1: Narrative Contradicts Benchmark Data

**v1 claim**: "MedGemma's medical domain knowledge provides measurable advantage over general-purpose models."

**Reality** (from `runs/` metrics):

| Model | Accuracy | F1 Macro | Cohen's κ |
|-------|----------|----------|-----------|
| MedGemma 4B | 55% | 0.508 | 0.286 |
| Gemini 3 Pro | 75% | 0.558 | 0.583 |
| GPT-4 (HF baseline) | 75% | 0.746 | — |

MedGemma 4B is **20 points worse** than Gemini on accuracy. The plan's story of "MedGemma superiority" would be immediately challenged by judges reviewing the benchmark table. This is the single biggest risk to the submission.

**Fix**: Reframe the narrative. Don't claim MedGemma beats general models. Instead:
1. Position MedGemma as the **domain-specific backbone in a multi-model orchestration system** — it handles medical term normalization (PRESCREEN), patient profile extraction (INGEST), while Gemini handles general reasoning/orchestration.
2. Emphasize that **MedGemma 27B has not yet been benchmarked** (see Flaw 2) and is expected to perform significantly better than 4B.
3. Frame the benchmark as **honest comparative analysis** showing where domain models excel (e.g., medical term normalization, image understanding) vs. where general models win (structured reasoning tasks).

#### FLAW 2: MedGemma 27B Never Benchmarked

All 7 existing Phase 0 runs are either 4B or Gemini. **Zero MedGemma 27B runs exist.** The 27B endpoint was deployed and smoke-tested but never put through the criterion evaluation benchmark. The 3-way comparison config exists (`phase0_three_way.yaml`) but was never executed.

**Impact**: Without 27B results, we can't:
- Tell the 4B vs 27B comparison story
- Know if 27B actually outperforms 4B (it should, given 5x parameters)
- Claim meaningful benchmark analysis in the technical document

**Fix**: Run 27B benchmark IMMEDIATELY as Day 1 priority. If 27B achieves ~70-80% accuracy (close to Gemini), the narrative becomes much stronger: "27B matches general-purpose models while 4B provides multimodal capability."

#### FLAW 3: INGEST Module is Empty

`src/trialmatch/ingest/__init__.py` contains nothing. The pipeline diagram shows INGEST as step 1, but there is literally no code to execute it.

For the demo, when a user selects a patient and clicks "Run Pipeline":
- Step 1 (INGEST) would have **nothing to execute**
- Either the demo skips INGEST (feels incomplete) or shows fake progress (dishonest)

**Fix**: Two options:
- **Option A (Recommended, 2-3h)**: Build a minimal INGEST module that calls MedGemma to extract key facts from patient notes. The adapter already exists. We just need a prompt template + JSON parser, similar to how `evaluator.py` works.
- **Option B (Fallback, 30min)**: Pre-compute INGEST outputs and store in sample JSON fixtures. The demo loads pre-computed results but shows them in real-time with simulated delay. Label clearly as "pre-extracted" in the UI.

#### FLAW 4: No Sample Patient Data Exists

`data/samples/` does not exist. No JSON fixtures exist in `data/`. The file `nsclc_trial_profiles.json` sits at the repo root (37 profiles) but:
- It's not in the expected location
- It's not formatted as demo fixtures
- It may not have the right structure for the demo API

**Fix**: Create `data/samples/` with 3 curated JSON fixtures on Day 1. Source from `nsclc_trial_profiles.json` or synthesize from TrialGPT HF cases.

#### FLAW 5: MedGemma 4B Multimodal Not Supported in Code

The `MedGemmaAdapter` uses `text_generation` API exclusively. Even though MedGemma 4B is a multimodal model, the current adapter **cannot send images**. Building multimodal support requires:
- Switching to `chat_completion` or `image_text_to_text` API
- Base64 image encoding
- Updated prompt template for image + text inputs
- Different response parsing

The plan lists "MedGemma 4B multimodal image processing" as P1 (Day 3), but this is a significant code change to a core module.

**Fix**: Demote multimodal to P2. For Patient 2 (EHR + image), use pre-extracted key facts. Show the medical image in the UI as context but note it was "pre-processed by MedGemma 4B." This is honest and avoids a risky code change.

### SIGNIFICANT GAPS (quality-impacting)

#### GAP 1: Submission Format Misunderstood

**v1 assumption**: Submit a zip with video + PDF + code.

**Reality**: Submissions are via **Kaggle Writeups** — a rich multimedia post format on Kaggle's platform. This is not a file upload. It requires:
- Creating a Kaggle Writeup post (markdown-like format)
- Embedding the demo video (likely YouTube/Vimeo link or direct upload)
- Writing the technical overview inline or as attached PDF
- Linking to source code (GitHub or Kaggle notebook)

**Fix**: Investigate Kaggle Writeup format early (Day 1). Create a draft Writeup shell on Day 1 or 2, not Day 4.

#### GAP 2: Special Awards Strategy Underexplored

Three special award categories exist:
1. **Agent-based workflows** — our PRESCREEN module qualifies
2. **Novel fine-tuned model adaptations** — we haven't fine-tuned anything
3. **Effective edge AI deployment** — not relevant

The v1 plan mentions targeting (1) but doesn't detail what judges look for. For agent-based workflows, judges likely want to see:
- Clear multi-step reasoning with tool use
- Visible agent decision-making (why it chose certain searches)
- Error recovery and adaptive behavior
- Real-time visualization of the agent's thought process

**Fix**: Make PRESCREEN agent visualization a P0, not P1. The agent trace (tool calls, queries, results) should be prominently displayed in the UI. This is our strongest differentiator for the special award.

#### GAP 3: PRESCREEN Demo Reliability Risk

The PRESCREEN agent depends on three live services simultaneously:
1. CT.gov API (40 req/min, can be slow/down)
2. Gemini 3 Pro (rate-limited, can 503)
3. MedGemma endpoint (scale-to-zero, 15min cold start)

If any one fails during demo recording, the entire pipeline breaks.

**Fix**:
- Pre-warm all endpoints before demo recording (`health_check()` calls)
- Build a **cached/replay mode**: Record a real PRESCREEN run, save the full trace, then replay it during demo if live mode fails. The UI shows the same real-time experience but from cached data.
- This is also useful for development — no need to make live API calls while building the frontend.

#### GAP 4: No Error Recovery Plan

What if:
- MedGemma 27B accuracy is also poor (< 60%)?
- Model endpoints go down during demo recording?
- CT.gov API changes or blocks us?
- Playwright recording has quality issues?

No Plan B exists.

**Fix**: Define fallbacks:
- **Poor 27B results**: Shift narrative to "honest benchmarking reveals insights for future fine-tuning" — judges value intellectual honesty
- **Endpoint down**: Use cached mode (see GAP 3 fix)
- **Playwright issues**: Have manual screen recording ready (OBS/macOS) as backup
- **CT.gov issues**: Pre-cache trial data for demo patients

#### GAP 5: Technical Document Has No Outline

The 3-page tech doc is listed for Day 4 with zero structure. For a competition, this document heavily influences judging. Leaving it to the last day is risky.

**Fix**: Draft outline on Day 1, fill in sections progressively:
- Page 1: Problem statement + why MedGemma
- Page 2: Architecture (INGEST → PRESCREEN → VALIDATE) + agent design
- Page 3: Results (benchmark table, confusion matrices, key findings)

#### GAP 6: Reproducibility Challenges

Competition requires "reproducible source code" but:
- MedGemma 4B/27B HF Inference Endpoints are private/paid
- MedGemma 27B was self-deployed on A100 80GB (~$5/hr)
- External reproducer needs HF Pro subscription or own deployment

**Fix**: Include in README:
1. Instructions for using HF Inference API (paid) OR local deployment with `transformers`
2. A `--mock` mode that uses cached model responses for reproducibility without API access
3. Clear `.env.example` with all required keys

#### GAP 7: HAI-DEF Model Utilization Could Be Stronger

The plan uses MedGemma 4B + 27B. But the HAI-DEF collection has 17+ models. Using more models (even briefly) strengthens the "effective use of HAI-DEF models" judging criterion.

**Consider adding**:
- **TxGemma** (therapeutics prediction) — for ranking matched trials by drug mechanism relevance. Even a simple call in VALIDATE to check drug interactions would add value.
- **MedSigLIP** — for image-text alignment scoring on Patient 2's medical image. Quick to integrate, strong multimodal story.

**Fix**: Evaluate whether a quick TxGemma or MedSigLIP integration is feasible in < 2 hours. If yes, add on Day 3.

#### GAP 8: Demo Video Strategy Needs Planning

Playwright video recording has known limitations:
- Hardcoded 1 Mbit/s bitrate (grainy)
- Default 800x800 resolution (blurry)
- `slowMo` applies to ALL actions (inflates recording time)

**Fix**:
- Use `recordVideo: { size: { width: 1280, height: 720 } }` explicitly
- Use `page.waitForTimeout()` for selective pacing, not `slowMo`
- Post-process with `ffmpeg -i recording.webm -c:v libx264 -crf 20 demo.mp4`
- Have OBS as backup recording method

### MINOR ISSUES

#### ISSUE 1: SSE Implementation Gotchas

Several known pitfalls the plan doesn't address:
- **POST-based SSE**: Native `EventSource` only supports GET. Need `@microsoft/fetch-event-source` for POST requests with JSON body.
- **CORS**: FastAPI CORSMiddleware must be added early for preflight requests.
- **Disconnect detection**: Must call `request.is_disconnected()` in the generator to stop wasting LLM API calls when client navigates away.
- **Mid-stream errors**: Cannot change HTTP status after streaming starts. Must yield error events as data.

#### ISSUE 2: Token Cost Estimation is Rough

`MedGemmaAdapter` estimates tokens at 4 chars/token with fixed pricing ($0.01/1M input, $0.03/1M output). This is very approximate. For a competition demo this is fine, but the technical document should acknowledge the estimation method.

#### ISSUE 3: Gemini Adapter Uses `asyncio.to_thread`

Both MedGemma and Gemini adapters wrap synchronous SDK calls with `asyncio.to_thread`. This works but creates thread pool pressure under concurrent calls. For a single-user demo this is fine, but should be noted.

---

## Part 2: Revised Design

### Narrative Reframe

**Before (v1)**: "MedGemma beats general models"
**After (v2)**: "Multi-model clinical AI: MedGemma provides domain-specific medical understanding while Gemini orchestrates complex reasoning — together they form a clinical trial matching system that neither could achieve alone"

Key messages:
1. **Complementary model architecture**: 4B for multimodal medical extraction, 27B for text-heavy clinical reasoning, Gemini for agentic orchestration + tool use
2. **Honest benchmarking**: Transparent comparison revealing where domain models shine (PRESCREEN normalization, INGEST medical extraction) vs. where general models win (structured criterion evaluation)
3. **Agent-based workflow**: PRESCREEN is a genuine multi-model agent with CT.gov tool use — not a gimmick
4. **Real clinical problem**: Trial matching impacts millions of patients annually

### Revised Architecture

```
                          Next.js Frontend (localhost:3000)
┌────────────────────────────────────────────────────────────────────┐
│  Patient Selector (3 pre-loaded cases)                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│  │ Patient 1 │  │ Patient 2 │  │ Patient 3 │                       │
│  │ Text EHR  │  │ EHR+Image │  │ Complex   │                       │
│  │  → 27B    │  │  → 4B     │  │  → Both   │                       │
│  └──────────┘  └──────────┘  └──────────┘                         │
│                                                                     │
│  Pipeline Viewer (SSE real-time)                                   │
│  ┌─ INGEST ─────────────────────────────────────────────────────┐  │
│  │  Status: ✅ | Key Facts: 12 extracted | Model: MedGemma 27B  │  │
│  │  [▼ Expand: extracted demographics, conditions, labs, meds]  │  │
│  ├─ PRESCREEN ──────────────────────────────────────────────────┤  │
│  │  Status: ⏳ | Queries: 4/6 | Trials Found: 8 unique          │  │
│  │  [▼ Agent Trace: query→results→reasoning for each tool call] │  │
│  ├─ VALIDATE ───────────────────────────────────────────────────┤  │
│  │  Status: 🔄 | Criteria: 3/12 evaluated | MET: 2, NOT_MET: 1 │  │
│  │  [▼ Per-criterion: verdict + reasoning + evidence sentences]  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Results Summary + Benchmark Dashboard (tabbed)                    │
└────────────────────────────────────────────────────────────────────┘
           │ SSE stream (POST)                    │ REST (GET)
           ▼                                      ▼
┌────────────────────────────────────────────────────────────────────┐
│  FastAPI Backend (localhost:8000)                                   │
│                                                                     │
│  POST /api/pipeline/run                                            │
│   ├─ Load patient from data/samples/{id}.json                      │
│   ├─ INGEST: MedGemma → extract key facts (or load pre-computed)  │
│   ├─ PRESCREEN: Gemini agent + CT.gov + MedGemma normalize        │
│   └─ VALIDATE: MedGemma 27B → criterion verdicts (live or cached) │
│                                                                     │
│  GET /api/benchmark/results — Phase 0 metrics (from runs/)        │
│  GET /api/health — ping all 3 model endpoints                     │
│  GET /api/patients — list sample patients with metadata            │
│                                                                     │
│  Mode: live (default) | cached (fallback for demo recording)       │
└────────────────────────────────────────────────────────────────────┘
        │              │              │
   MedGemma 4B    MedGemma 27B    Gemini 3 Pro
   (HF Inference)  (HF Inference)  (AI Studio)
```

### Revised Sample Patients

Source from existing `nsclc_trial_profiles.json` (37 profiles at repo root) + TrialGPT HF dataset:

| # | Name | Description | Primary Model | Why This Case |
|---|------|-------------|---------------|---------------|
| 1 | NSCLC EGFR+ | 62F, lung adenocarcinoma, EGFR L858R, never smoker | MedGemma 27B | Rich biomarker + phenotype data, many matching trials |
| 2 | Diabetic Retinopathy | 58M, T2DM + retinal image, fundoscopy findings | MedGemma 4B | Demonstrates multimodal (pre-extracted), straightforward text |
| 3 | Multi-condition Elderly | 74M, CKD + CHF + atrial fib, 8 medications | Both models | Complex polypharmacy, demonstrates multi-model routing |

**Key change**: Patient 2 should come from TrialGPT HF dataset or be synthesized, since NSCLC profiles don't include medical images. Alternatively, use a pre-computed multimodal result with a real medical image for display.

### Revised Pipeline with Cached Mode

```python
# Pipeline mode switch
class PipelineMode(Enum):
    LIVE = "live"          # Real API calls (default for development)
    CACHED = "cached"      # Replay saved trace (for demo recording)
    HYBRID = "hybrid"      # INGEST cached, PRESCREEN+VALIDATE live

# Each pipeline step can independently use cached or live mode
```

This is critical for:
- Frontend development without burning API credits
- Reliable demo recording
- Reproducibility for judges

### Revised Priority

#### P0 — Must Ship (submission-blocking)

| # | Item | Est. Hours | Day | New? |
|---|------|-----------|-----|------|
| 1 | **Run MedGemma 27B Phase 0 benchmark** | 0.5 (waiting) | 1 | NEW |
| 2 | FastAPI backend with SSE + cached mode | 3-4h | 1 | Enhanced |
| 3 | 3 sample patient JSON fixtures in `data/samples/` | 1h | 1 | Same |
| 4 | Minimal INGEST module (MedGemma key fact extraction) | 2-3h | 1 | NEW |
| 5 | Next.js frontend: Patient Selector + Pipeline Viewer + Results | 6-8h | 2 | Same |
| 6 | **PRESCREEN agent trace visualization** (upgraded from P1) | 2h | 2 | Upgraded |
| 7 | Kaggle Writeup draft shell (not Day 4) | 0.5h | 2 | NEW |
| 8 | 3-page technical document (start Day 2, finish Day 4) | 3h total | 2-4 | Moved earlier |
| 9 | Playwright demo recording + post-processing | 2h | 4 | Same |
| 10 | Final Kaggle Writeup submission | 1h | 4 | Same |

#### P1 — Strong Differentiator

| # | Item | Est. Hours | Day |
|---|------|-----------|-----|
| 11 | 3-model comparison view (side-by-side verdicts) | 2h | 3 |
| 12 | Benchmark dashboard (charts from Phase 0 results) | 2h | 3 |
| 13 | Endpoint pre-warm + cached replay mode for recording | 1h | 3 |
| 14 | UI polish: loading states, error handling, responsive | 2h | 3 |

#### P2 — If Time Permits

| # | Item | Est. Hours | Day |
|---|------|-----------|-----|
| 15 | MedGemma 4B multimodal image support in adapter | 3h | 3 |
| 16 | TxGemma integration for drug mechanism scoring | 2h | 3 |
| 17 | ClinicalTrials.gov links + export report | 1h | 3 |
| 18 | Playwright e2e test suite (beyond recording) | 1.5h | 3 |

### Revised Milestone Plan

| Day | Date | Focus | Deliverables | Risk Mitigation |
|-----|------|-------|-------------|-----------------|
| **1** | Feb 21 | Backend + Data + Benchmark | FastAPI backend (SSE + cached mode), 3 sample fixtures, INGEST module (minimal), **kick off 27B benchmark immediately** | If 27B results are poor, reframe narrative to "honest comparative analysis" |
| **2** | Feb 22 | Frontend Core + Writeup | Patient Selector, Pipeline Viewer with agent trace, Results Panel, wired to backend. Draft Kaggle Writeup shell. Start tech doc. | If frontend is slow, cut benchmark dashboard to Day 3 |
| **3** | Feb 23 | Polish + Differentiation | Model comparison view, benchmark dashboard, UI polish, cached mode testing, endpoint pre-warming scripts | If models are down, verify cached mode works end-to-end |
| **4** | Feb 24 | Ship | Finalize tech doc, record demo video (try live first, cached backup), finalize Kaggle Writeup, submit | Backup: OBS screen recording if Playwright has quality issues |

### Technical Document Outline

**Page 1: Problem & Approach**
- Clinical trial matching problem: 80% of trials fail to recruit on time
- Why MedGemma: domain-specific medical knowledge for extraction + normalization
- Multi-model architecture: MedGemma (domain) + Gemini (orchestration)
- Our approach: INGEST → PRESCREEN → VALIDATE pipeline

**Page 2: Architecture & Agent Design**
- System diagram (3-tier: Frontend → Backend → Models)
- PRESCREEN agent: Gemini orchestrator + CT.gov tools + MedGemma normalization
- VALIDATE evaluator: criterion-type-aware prompts, JSON-structured outputs
- Design principles: component isolation, run determinism, cost tracking

**Page 3: Results & Impact**
- Phase 0 benchmark table (3 models × 20 pairs × criterion-level metrics)
- Key findings: where MedGemma excels (extraction, normalization) vs. where it trails (structured reasoning)
- Confusion matrices for each model
- Real-world impact: how this system could be deployed in clinical settings
- Future work: fine-tuning on TrialGPT data, full Tier A evaluation

### SSE Implementation Spec

```python
# Backend: FastAPI + sse-starlette
from sse_starlette.sse import EventSourceResponse

@app.post("/api/pipeline/run")
async def run_pipeline(request: Request, payload: PipelineRequest):
    async def event_generator():
        try:
            # Step 1: INGEST
            yield {"event": "step", "data": json.dumps({
                "step": "ingest", "status": "running",
                "message": "Extracting key facts from patient note..."
            })}

            if await request.is_disconnected():  # CRITICAL: check disconnect
                return

            ingest_result = await run_ingest(payload.patient_id, mode=pipeline_mode)

            yield {"event": "step", "data": json.dumps({
                "step": "ingest", "status": "complete",
                "data": ingest_result
            })}

            # Step 2: PRESCREEN (stream agent tool calls)
            yield {"event": "step", "data": json.dumps({
                "step": "prescreen", "status": "running",
                "message": "Searching ClinicalTrials.gov..."
            })}

            async for tool_event in run_prescreen_stream(ingest_result, mode=pipeline_mode):
                if await request.is_disconnected():
                    return
                yield {"event": "tool_call", "data": json.dumps(tool_event)}

            # ... VALIDATE step similarly

        except Exception as e:
            yield {"event": "error", "data": json.dumps({
                "step": "pipeline", "message": str(e)
            })}

    return EventSourceResponse(
        event_generator(),
        ping=15,
        headers={"X-Accel-Buffering": "no"}  # Prevent proxy buffering
    )
```

```typescript
// Frontend: @microsoft/fetch-event-source (NOT native EventSource)
// Native EventSource doesn't support POST requests
import { fetchEventSource } from '@microsoft/fetch-event-source';

useEffect(() => {
  const ctrl = new AbortController();

  fetchEventSource('http://localhost:8000/api/pipeline/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patient_id: selectedPatient }),
    signal: ctrl.signal,
    onmessage(ev) {
      const data = JSON.parse(ev.data);
      dispatch({ type: ev.event, payload: data });
    },
    onerror(err) {
      // Don't retry on 4xx errors
      if (err.status >= 400 && err.status < 500) throw err;
    },
  });

  return () => ctrl.abort();  // CRITICAL: cleanup on unmount
}, [selectedPatient]);
```

---

## Part 3: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MedGemma 27B also underperforms (< 65%) | Medium | High | Reframe as "honest analysis revealing fine-tuning opportunities" |
| Model endpoint cold-start during demo | High | High | Pre-warm script + cached replay mode |
| CT.gov API rate limit during demo | Medium | Medium | Pre-cache trial data for demo patients |
| Playwright video quality insufficient | Medium | Low | OBS backup recording, ffmpeg post-processing |
| Frontend development exceeds estimate | Medium | High | Cut benchmark dashboard (P1) to save time |
| Kaggle Writeup format unknown specifics | Low | Medium | Research on Day 1, draft shell early |
| 27B benchmark takes too long (token limits) | Low | Medium | Use phase0_test.yaml (1 pair) first as smoke test |

---

## Part 4: Competition Strategy Summary

### Judging Criteria Alignment

| Criterion | Our Strength | Our Weakness | Action |
|-----------|-------------|-------------|--------|
| Effective HAI-DEF model use | 2 MedGemma sizes + Gemini | 4B accuracy is low | Emphasize complementary roles, not raw accuracy |
| Problem importance | Trial matching is critical healthcare problem | — | Strong narrative in tech doc page 1 |
| Real-world impact | Working pipeline, real CT.gov data | Local-only demo | Describe deployment path in tech doc |
| Technical feasibility | Full working codebase, 140 tests | Multimodal not live | Demonstrate with pre-computed results |
| Execution quality | Clean architecture, comprehensive benchmark | Only 20-pair Phase 0 | Present as "directional capability assessment" |

### Special Award Target: Agent-Based Workflows

To win this award, we need to prominently demonstrate:
1. **Multi-model agent**: Gemini orchestrates, MedGemma normalizes medical terms
2. **Real tool use**: CT.gov API searches with visible query → result → reasoning chain
3. **Adaptive search strategy**: Agent changes queries based on previous results
4. **Visible decision-making**: Agent trace shown in real-time in UI
5. **Error recovery**: Agent handles empty results, adjusts strategy

This is why PRESCREEN agent visualization is upgraded to P0.

### What Makes Us Competitive

1. **Only submission with live CT.gov integration** (most will use static data)
2. **Genuine agent architecture** (not just prompt chaining)
3. **Honest benchmarking** (judges respect transparency over inflated claims)
4. **Multi-model orchestration** (4B + 27B + Gemini working together)
5. **140-test codebase** (production-quality engineering)

---

## Sources

- [MedGemma Impact Challenge (Kaggle)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
- [MedGemma Impact Challenge Rules](https://www.kaggle.com/competitions/med-gemma-impact-challenge/rules)
- [Google MedGemma Impact Challenge coverage (EdTech Innovation Hub)](https://www.edtechinnovationhub.com/news/google-launches-medgemma-impact-challenge-to-advance-human-centered-health-ai)
- [HAI-DEF Collection (HuggingFace)](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def)
- [HAI-DEF Developer Site (Google)](https://developers.google.com/health-ai-developer-foundations)
- [Kaggle announcement (X/Twitter)](https://x.com/kaggle/status/2011188975155707905)
- [sse-starlette (GitHub)](https://github.com/sysid/sse-starlette)
- [@microsoft/fetch-event-source (npm)](https://www.npmjs.com/package/@microsoft/fetch-event-source)
- [Playwright Video Recording docs](https://playwright.dev/docs/videos)
- [MedGemma 1.5 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
