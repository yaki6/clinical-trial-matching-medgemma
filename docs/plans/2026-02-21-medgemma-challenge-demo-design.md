# Design: MedGemma Impact Challenge Demo

**Date**: 2026-02-21
**Deadline**: 2026-02-24 (Kaggle submission close)
**Competition**: https://www.kaggle.com/competitions/med-gemma-impact-challenge

## Problem

We have a working CLI benchmark tool (trialmatch) comparing MedGemma 4B, 27B, and Gemini 3 Pro on clinical trial criterion-level matching. We need to turn this into a demo-ready submission for the MedGemma Impact Challenge in 4 days.

## Competition Requirements

1. **3-minute video demonstration**
2. **3-page technical documentation** (approach, architecture, results)
3. **Complete, reproducible source code**
4. **Must use at least one HAI-DEF model** (MedGemma) — we use both 4B and 27B
5. **Judging criteria**: model utilization, problem significance, real-world impact, technical viability, execution quality
6. **Special awards**: agent-based workflows (our PRESCREEN module qualifies)

## Design

### Product Concept

**TrialMatch Demo** — A clinical trial matching tool showcasing a full INGEST -> PRESCREEN -> VALIDATE pipeline with 3 pre-loaded sample patients. Real-time streaming logs show each pipeline step as it runs. MedGemma 4B handles multimodal cases (EHR + images), 27B handles text-only cases.

### Architecture

```
Next.js Frontend (localhost:3000)     FastAPI Backend (localhost:8000)
┌──────────────────────────┐   SSE    ┌───────────────────────────┐
│ Patient Selector (3 cases)│◄───────►│ POST /api/pipeline/run     │
│ Pipeline Viewer (realtime)│ stream  │   ├─ INGEST (4B or 27B)   │
│  ├─ INGEST step + logs   │         │   ├─ PRESCREEN (agent+ctgov)│
│  ├─ PRESCREEN step + logs│         │   └─ VALIDATE (per-criterion│
│  └─ VALIDATE step + logs │         │       matching)             │
│ Results Panel             │         │ GET /api/benchmark/results  │
│ Benchmark Dashboard       │         │ GET /api/health             │
└──────────────────────────┘         └───────────────────────────┘
                                        │          │          │
                                    MedGemma    MedGemma   Gemini
                                     4B (HF)    27B (HF)   3 Pro
```

### Sample Patients

3 pre-loaded cases:
1. **Text-only EHR** — processed by MedGemma 27B (demonstrates text-only model strength)
2. **EHR + Medical Image** — processed by MedGemma 4B (demonstrates multimodal capability)
3. **Complex multi-condition** — processed by both, showing complementary model usage

Exact patient data to be curated from TrialGPT HF dataset or synthesized.

### Pipeline Steps (displayed in real-time)

1. **INGEST**: Extract key facts from patient note (demographics, conditions, labs, medications)
2. **PRESCREEN**: Agentic CT.gov search — MedGemma generates search terms, Gemini orchestrates tool calls, returns candidate trials
3. **VALIDATE**: For each candidate trial's criteria, evaluate MET/NOT_MET/UNKNOWN with reasoning chain

### Real-time Streaming

Backend uses Server-Sent Events (SSE) to stream structured log events:
```json
{"step": "ingest", "status": "running", "message": "Extracting key facts..."}
{"step": "ingest", "status": "complete", "data": {"key_facts": [...]}}
{"step": "prescreen", "status": "running", "message": "Searching CT.gov for NSCLC trials..."}
...
```

Frontend renders each step as a collapsible card with status indicator (spinner/check/error) and expandable log detail.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15 + React 19 + Tailwind CSS + shadcn/ui |
| Backend | FastAPI + SSE (sse-starlette) |
| Models | huggingface_hub InferenceClient + google-genai SDK |
| Testing | pytest (backend), Playwright (e2e demo recording) |

### Deployment

Local-only (localhost) for demo and video recording. No cloud deployment needed.

## Priority

### P0 — Must ship (submission-blocking)

1. FastAPI backend with SSE pipeline streaming
2. Next.js frontend: patient selector + pipeline viewer + results
3. 3 sample patient datasets (curated EHR text)
4. Phase 0 benchmark results (run 3 models, collect metrics)
5. 3-page technical document draft
6. Playwright-recorded demo video (3 min)

### P1 — Strong differentiator

7. MedGemma 4B multimodal image processing for image cases
8. 3-model side-by-side comparison view
9. Benchmark dashboard (charts: accuracy, F1, confusion matrix vs GPT-4 baseline)
10. PRESCREEN agent real-time CT.gov search visualization

### P2 — If time permits

11. ClinicalTrials.gov links for matched trials
12. Export report functionality

## Milestone Plan

| Day | Date | Deliverable |
|-----|------|------------|
| 1 | Feb 21 | FastAPI backend (`/pipeline/run` SSE, `/benchmark/results`). Kick off Phase 0 benchmark run in background. |
| 2 | Feb 22 | Next.js frontend core: Patient Selector -> Pipeline Viewer (real-time logs) -> Results Panel |
| 3 | Feb 23 | 4B multimodal support, sample data polish, benchmark dashboard, UI polish |
| 4 | Feb 24 | 3-page tech doc, Playwright demo recording, final submission |

## Competition Story

**Narrative**: Clinical trial matching is a critical healthcare problem where MedGemma's medical domain knowledge provides measurable advantage over general-purpose models. We demonstrate:

1. **MedGemma 4B** — multimodal understanding of patient records (text + medical images)
2. **MedGemma 27B** — superior text-based clinical reasoning for criterion matching
3. **Agent-based workflow** — PRESCREEN module uses MedGemma + CT.gov API for trial search
4. **Quantitative evidence** — Phase 0 benchmark comparing MedGemma vs Gemini vs GPT-4 baseline

This hits multiple judging dimensions: model utilization (both sizes, multimodal), problem significance (clinical trials), real-world impact (matching patients to trials), technical viability (working benchmark), and qualifies for the agent-based workflows special award.
