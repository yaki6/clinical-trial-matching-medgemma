# TrialMatch — Business Stakeholder Testing Guide

> **Date**: 2026-02-22
> **App URL**: http://localhost:8501
> **Duration**: ~15 minutes
> **No technical setup required** — just open the URL in Chrome/Firefox/Safari

---

## What You're Testing

TrialMatch is an AI-powered clinical trial matching tool. It takes a cancer patient's medical record and automatically finds relevant clinical trials, then evaluates whether the patient meets each trial's eligibility criteria.

**The pipeline has 3 stages:**

| Stage | What it does | Model used |
|-------|-------------|------------|
| **INGEST** | Extracts key medical facts from patient notes | MedGemma 4B |
| **PRESCREEN** | Searches ClinicalTrials.gov for matching trials | Gemini 3 Pro |
| **VALIDATE** | Evaluates patient eligibility per criterion | MedGemma 27B + Gemini 3 Pro |

---

## Test Scenario 1: Pipeline Demo (Cached Mode)

**Goal**: Verify the core user flow works with pre-computed results.

### Steps

1. Open **http://localhost:8501** in your browser
2. Click **"Pipeline Demo"** in the left sidebar

3. **Select a patient**:
   - In the sidebar, find the patient dropdown
   - Select **mpx1016** (a 43-year-old female with lung adenocarcinoma)
   - This patient has cached results for instant demo

4. **Review INGEST results** (Key Facts panel):
   - You should see extracted patient information:
     - Age, sex, diagnosis
     - Molecular markers (e.g., ALK fusion, RET fusion)
     - Treatment history, smoking status
   - **Check**: Do the key facts look reasonable for a lung cancer patient?
   - **Check**: Is the information clearly labeled and readable?

5. **Review PRESCREEN results** (Trial Candidates):
   - You should see a list of clinical trials found on ClinicalTrials.gov
   - Each trial shows: NCT ID, title, phase, recruitment status
   - **Check**: Are the trials relevant to lung cancer?
   - **Check**: Are trial titles and metadata displayed correctly?

6. **Review VALIDATE results** (Eligibility Cards):
   - Each trial should have a colored verdict badge:
     - **Green** = "You may be eligible for this trial"
     - **Red** = "You likely do not qualify for this trial"
     - **Orange** = "More information is needed"
   - Each card groups criteria into:
     - Criteria you meet (green checkmarks)
     - Criteria not met (red crosses)
     - Needs further review (orange question marks)
   - **Check**: Are the verdicts logical? (e.g., age >= 18 should be MET for a 43-year-old)
   - **Check**: Is the language patient-friendly and not overly technical?

7. **Expand detailed breakdown**:
   - Click **"View detailed eligibility breakdown"** on any trial card
   - You should see per-criterion reasoning with two stages:
     - Stage 1: MedGemma medical reasoning (clinical analysis)
     - Stage 2: Gemini label assignment (final verdict)
   - **Check**: Does the reasoning make clinical sense?
   - **Check**: Is there a link to ClinicalTrials.gov at the bottom of each card?

### Expected Results for mpx1016

| Trial | Expected Verdict | Why |
|-------|-----------------|-----|
| NCT06196424 (Family History Study) | EXCLUDED | Missing family cancer history data |
| Other NSCLC trials | Varies | Based on specific eligibility criteria |

---

## Test Scenario 2: Patient Mode vs Dev Mode

**Goal**: Verify both UX modes work correctly for different audiences.

### Patient Mode (Default)

1. Open **http://localhost:8501/Pipeline_Demo**
2. Confirm you see:
   - Medical disclaimer at the top
   - Patient-friendly language throughout
   - No technical jargon (no "tokens", "costs", "latency")
   - Simple loading states
3. **Check**: Would a real patient understand this interface?

### Dev Mode

1. Add `?dev=1` to the URL: **http://localhost:8501/Pipeline_Demo?dev=1**
   — OR scroll to the bottom of the sidebar and toggle **"Developer mode"**
2. Confirm you see additional controls:
   - Pipeline Mode toggle (Live / Cached)
   - VALIDATE Mode selector (Two-Stage / Gemini / MedGemma)
   - Max trials slider
   - Max criteria per trial slider
   - Cost breakdowns and token counts
   - Full agent trace details
3. **Check**: Does toggling between modes change the displayed information?

---

## Test Scenario 3: Benchmark Dashboard

**Goal**: Review model comparison metrics and quality evidence.

### Steps

1. Click **"Benchmark"** in the left sidebar
2. You should see:

   **a) Model Comparison Table**
   - Multiple models listed (MedGemma 27B, MedGemma 4B, Gemini 3 Pro, GPT-4 baseline)
   - Accuracy, F1 scores, Cohen's Kappa metrics
   - **Check**: Are numbers displayed correctly? (e.g., 70%, 75%, not 0.7, 0.75)

   **b) Confusion Matrices**
   - Color-coded heatmaps showing prediction accuracy per class
   - **Check**: Are labels readable (MET, NOT_MET, UNKNOWN)?

   **c) Per-Class F1 Bar Charts**
   - Grouped bar chart comparing models across classes
   - **Check**: Is the chart interactive (hover for values)?

   **d) Cost & Latency Table**
   - Per-model costs and response times
   - **Check**: Are costs reasonable? (Should be < $1 for Phase 0)

   **e) Key Findings**
   - Summary bullet points explaining benchmark results
   - **Check**: Are findings clear and supported by the data above?

   **f) Audit Tables**
   - Expandable sections showing every evaluation pair
   - **Check**: Can you click to expand and see detailed results?

### Key Numbers to Verify

| Metric | Expected |
|--------|----------|
| MedGemma 27B Accuracy | ~70% |
| GPT-4 Baseline Accuracy | 75% |
| MedGemma 4B Accuracy | ~35% |
| Two-Stage (27B + Gemini) Accuracy | ~85% |

---

## Test Scenario 4: Navigation & Layout

**Goal**: Verify overall app usability and responsiveness.

### Steps

1. **Sidebar navigation**:
   - Click between Home, Pipeline Demo, and Benchmark pages
   - **Check**: Does navigation work without errors?
   - **Check**: Does the sidebar remain visible?

2. **Page layout**:
   - Resize browser window (desktop → tablet width)
   - **Check**: Do components reflow gracefully?
   - **Check**: Are there any horizontal scroll bars on normal content?

3. **Home page**:
   - Open **http://localhost:8501**
   - **Check**: Is the pipeline described clearly (INGEST -> PRESCREEN -> VALIDATE)?
   - **Check**: Are the models listed (MedGemma 4B, MedGemma 27B, Gemini 3 Pro)?

4. **Error states**:
   - If any section shows an error, note the exact error message
   - **Check**: Are errors user-friendly (not raw Python tracebacks)?

---

## Test Scenario 5: Live Pipeline (Optional — requires API keys)

> **Warning**: This test costs ~$0.05-$0.10 per run and takes 1-2 minutes.
> Only run if API endpoints are confirmed active.

### Steps

1. Enable Dev Mode (`?dev=1`)
2. Set Pipeline Mode to **"Live"** in the sidebar
3. Select a patient (mpx1016 recommended)
4. Click **"Run Pipeline"**
5. Observe:
   - PRESCREEN agent trace (tool calls to ClinicalTrials.gov API)
   - Real-time status updates
   - Trial candidates appearing as they're found
6. After PRESCREEN completes, observe VALIDATE:
   - Per-criterion evaluation progress
   - Results appearing trial by trial
7. **Check**: Does the live pipeline complete without errors?
8. **Check**: Are results consistent with cached results?

---

## Bug Reporting Template

If you find an issue, please note:

```
Page:        [Home / Pipeline Demo / Benchmark]
Mode:        [Patient / Dev]
Patient:     [e.g., mpx1016]
Steps:       [What you clicked/did]
Expected:    [What should happen]
Actual:      [What actually happened]
Screenshot:  [If possible]
Severity:    [Blocker / Major / Minor / Cosmetic]
```

### Severity Definitions

| Level | Definition |
|-------|-----------|
| **Blocker** | App crashes, page won't load, data is completely wrong |
| **Major** | Feature doesn't work, misleading information displayed |
| **Minor** | UI glitch, awkward wording, minor display issue |
| **Cosmetic** | Spacing, alignment, color consistency |

---

## Key Questions for Stakeholder Feedback

After testing, please share your thoughts on:

1. **Clarity**: Is it clear what each pipeline stage does?
2. **Trust**: Does the two-stage reasoning (MedGemma + Gemini) build confidence in the results?
3. **Patient Mode**: Would a real patient find this useful and understandable?
4. **Benchmark**: Does the benchmark dashboard effectively demonstrate model quality?
5. **Narrative**: Does the multi-model orchestration story come through? (This is not just one AI model — it's specialized medical AI + general AI working together)
6. **Missing Features**: What would you add for the Kaggle submission demo?

---

## Quick Reference

| What | Where |
|------|-------|
| Home page | http://localhost:8501 |
| Pipeline Demo | http://localhost:8501/Pipeline_Demo |
| Pipeline Demo (dev) | http://localhost:8501/Pipeline_Demo?dev=1 |
| Benchmark Dashboard | http://localhost:8501/Benchmark |
| Cached patient | mpx1016 (43F, Lung Adenocarcinoma) |
