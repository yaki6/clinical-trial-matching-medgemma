# Technical Architecture

## System Overview

CLI-first Python pipeline for benchmarking MedGemma (4B + 27B) vs Gemini 3 Pro on clinical trial criterion matching. Three-component pipeline: INGEST → PRESCREEN → VALIDATE.

```
HuggingFace Dataset (TrialGPT criterion annotations)
    │
    ▼
[INGEST] hf_loader.load_annotations()        → CriterionAnnotation[]
    │   Data: ncbi/TrialGPT-Criterion-Annotations (1,024 pairs)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[PRESCREEN] run_prescreen_agent()     [VALIDATE] evaluate_criterion()
    │   1. MedGemma 4B clinical           │   Two-stage evaluation:
    │      reasoning (pre-search)         │     Stage 1: MedGemma reasoning
    │   2. Gemini Flash agentic loop      │     Stage 2: Gemini labeling
    │   Tools: search_trials,             │   Output: CriterionResult
    │     get_trial_details               │     (MET / NOT_MET / UNKNOWN)
    │   CT.gov API v2 (AREA[StudyType]    │
    │     Interventional filtering)       │
    │                                      │
    ▼                                      ▼
PresearchResult                       CriterionResult[]
  (TrialCandidate[] + agent trace)        │
                                           ▼
                                    [EVALUATION] compute_metrics()
                                           │   accuracy, F1, Cohen's κ
                                           ▼
                                    [TRACING] RunManager → runs/<run_id>/
```

## Detailed Architecture Documents

| Document | Contents |
|----------|----------|
| [Pipeline Overview](pipeline-overview.md) | Mermaid flowcharts, data contract tables, all Pydantic model field specs, component isolation vs E2E diagrams, tool schema + CT.gov API mapping, run artifact structure |
| [Sequence Diagrams](prescreen-sequence-diagrams.md) | Mermaid sequence diagrams: PRESCREEN agent loop, CT.gov parameter mapping, Phase 0 benchmark flow, future E2E pipeline |

**Reading order for new developers:**
1. This file (overview + package structure)
2. `pipeline-overview.md` (data contracts — what flows between components)
3. `prescreen-sequence-diagrams.md` (interactions — how components talk to each other)

## Package Structure

```
src/trialmatch/
├── cli/           # Click CLI entry points (trialmatch command)
├── ingest/        # INGEST component — profile_adapter converts nsclc JSON → PRESCREEN dict
│   └── profile_adapter.py  # list-of-objects → flat dict, demographics promotion (age/sex → top-level)
├── prescreen/     # PRESCREEN component — MedGemma pre-search + Gemini agentic search
│   ├── agent.py       # MedGemma clinical reasoning + Gemini multi-turn agentic loop
│   ├── ctgov_client.py # Async CT.gov API v2 client (AREA[StudyType], rate-limited, retry)
│   ├── schema.py      # TrialCandidate, ToolCallRecord, PresearchResult
│   └── tools.py       # Gemini FunctionDeclarations (search_trials, get_trial_details) + ToolExecutor
├── validate/      # VALIDATE component — (Patient, Criterion) → MET/NOT_MET/UNKNOWN
│   └── evaluator.py   # Reusable criterion evaluator (model-agnostic)
├── data/          # Data loading: HF dataset (TrialGPT criterion annotations), sampling
├── models/        # Model adapters: MedGemma (HF Inference + Vertex AI) + Gemini (AI Studio)
│   ├── schema.py      # CriterionAnnotation, CriterionVerdict, ModelResponse, CriterionResult
│   ├── medgemma.py    # HF Inference adapter (TGI text_gen + vLLM chat_completion)
│   └── vertex_medgemma.py  # Vertex AI Model Garden adapter (Google Auth, GPU-hour costing)
├── evaluation/    # Metrics: accuracy, F1, Cohen's κ, confusion matrix, trial-level aggregation
└── tracing/       # Run artifact persistence to runs/<run_id>/, cost tracking
```

## Key Design Principles

1. **Component isolation** — PRESCREEN and VALIDATE eval use gold INGEST SoT as input, not model output, to isolate errors.
2. **Cache isolation** — Cache keys include `ingest_source=gold|model` to prevent contamination.
3. **Determinism** — Every run writes to `runs/<run_id>/` with config, inputs, artifacts, traces, metrics.
4. **Cost tracking** — Every LLM call logs: model, input_tokens, output_tokens, estimated_cost, latency_ms.

## Model Adapter Pattern

All models (MedGemma 4B, MedGemma 27B, Gemini 3 Pro) implement a common interface:

- `generate(prompt, **kwargs) → ModelResponse`
- `ModelResponse` includes: text, input_tokens, output_tokens, latency_ms, estimated_cost

MedGemma supports two backends:
- **TGI (4B)**: `text_generation` with manual Gemma chat template — token counts estimated (chars // 4)
- **vLLM (27B)**: `chat_completion` (OpenAI-compatible API) — token counts from API usage stats

Vertex AI adapter (`VertexMedGemmaAdapter`) available for Model Garden deployments with Google Auth + GPU-hour cost estimation.

Gemini uses Google AI Studio (`google-genai` SDK).

## Data Flow

1. **Data preparation**: `trialmatch data prepare` loads TrialGPT criterion annotations from HuggingFace
2. **Evaluation runs**: Each component can be run independently with gold or model inputs
3. **Comparison**: `trialmatch compare` generates cross-model comparison reports

## PRESCREEN Agent Architecture

The PRESCREEN agent uses a two-model architecture (ADR-009):

1. **MedGemma 4B** — Clinical reasoning pre-search: generates structured guidance (condition terms, likely molecular drivers, priority eligibility keywords, treatment line assessment) before the agentic loop begins. Latency: 12-43s, cost: ~$0.05/patient.

2. **Gemini Flash** — Agentic search orchestrator: multi-turn function-calling loop with 2 tools (`search_trials`, `get_trial_details`), max 8 tool calls. Budget guard sends structured FunctionResponse errors when tool cap is reached.

`normalize_medical_terms` was the original third tool (MedGemma-powered term normalization) but was commented out (ADR-011) due to ~25s latency per call with near-zero value — MedGemma echoed inputs unchanged. The clinical reasoning pre-search step provides better guidance.

### Candidate Scoring

Candidates are ranked by a heuristic score: `query_count * 3 + phase_bonus + recruiting_bonus + details_bonus`, capped at MAX_CANDIDATES=20. Phase II/III trials get +2, RECRUITING gets +1, trials with fetched eligibility criteria get +2.

### CT.gov API v2 Filtering

Study type filtering uses Essie `AREA[StudyType]Interventional` syntax in `query.term` (ADR-010), NOT `filter.studyType` which is invalid. Age bounds use `AREA[MinimumAge]RANGE[MIN, X]` and `AREA[MaximumAge]RANGE[X, MAX]`. Phase and sex use `aggFilters` (comma-joined, e.g., `phase:2,sex:f`).

### Demographics Promotion

`profile_adapter.py` promotes nested demographics fields (from `key_facts[].value.age`, `key_facts[].value.sex`) to top-level keys in the key_facts dict. This enables the agent to inject age/sex into CT.gov API filters automatically.

## Rate Limits

| Service | Limit | Enforcement |
|---------|-------|-------------|
| HuggingFace Inference (MedGemma) | 1 concurrent | Adapter-level (reduced from 5 due to CUDA bug) |
| Gemini API | 10 concurrent | Adapter-level |
| CT.gov API v2 | 40 req/min | CTGovClient rate limiter + 429 retry |
