# Technical Architecture

## System Overview

CLI-first Python pipeline for benchmarking MedGemma vs Gemini 3 Pro on clinical trial criterion matching.

```
TREC Topic (patient text)
    │
    ▼
[INGEST] understand()                    ← PatientProfileText + KeyFacts
    │   Models: MedGemma 1.5 4B / Gemini 3 Pro
    ▼
[PRESCREEN] generate_search_terms()      ← SearchAnchors
    │   Models: MedGemma 1.5 4B / Gemini 3 Pro
    ▼
[VALIDATE] evaluate_criterion()          ← EligibilityLedger (MET/NOT_MET/UNKNOWN)
        Models: MedGemma 1.5 4B / Gemini 3 Pro
```

## Package Structure

```
src/trialmatch/
├── cli/           # Click/Typer CLI entry points (trialmatch command)
├── ingest/        # INGEST component — patient text → PatientProfile + KeyFacts
├── prescreen/     # PRESCREEN component — PatientProfile → SearchAnchors
├── validate/      # VALIDATE component — (Patient, Criterion) → MET/NOT_MET/UNKNOWN
├── data/          # Data loading: TREC topics, qrels, trial fetching (CT.gov API / TrialGPT)
├── models/        # Model abstraction: MedGemma (HF), Gemini (Vertex/AI Studio) adapters
├── evaluation/    # Metrics computation: accuracy, F1, Cohen's κ, confusion matrix
└── tracing/       # Run tracing, artifact persistence, cost tracking
```

## Key Design Principles

1. **Component isolation** — PRESCREEN and VALIDATE eval use gold INGEST SoT as input, not model output, to isolate errors.
2. **Cache isolation** — Cache keys include `ingest_source=gold|model` to prevent contamination.
3. **Determinism** — Every run writes to `runs/<run_id>/` with config, inputs, artifacts, traces, metrics.
4. **Cost tracking** — Every LLM call logs: model, input_tokens, output_tokens, estimated_cost, latency_ms.

## Model Adapter Pattern

Both models (MedGemma 1.5 4B, Gemini 3 Pro) implement a common interface:

- `generate(prompt, **kwargs) → ModelResponse`
- `ModelResponse` includes: text, input_tokens, output_tokens, latency_ms, estimated_cost

MedGemma uses HuggingFace Inference API; Gemini uses Google AI Studio / Vertex AI.

## Data Flow

1. **Data preparation**: `trialmatch data prepare` fetches trial data from TrialGPT dataset or CT.gov API v2
2. **Evaluation runs**: Each component can be run independently with gold or model inputs
3. **Comparison**: `trialmatch compare` generates cross-model comparison reports

## Rate Limits

| Service | Limit |
|---------|-------|
| HuggingFace Inference (MedGemma) | 5 concurrent |
| Gemini API | 10 concurrent |
| CT.gov API v2 | 40 req/min |
