# TrialMatch

**Clinical trial matching powered by MedGemma + Gemini multi-model orchestration.**

TrialMatch is a three-stage pipeline that matches patients to clinical trials using specialized medical AI models. The key innovation is a **two-stage architecture** that pairs MedGemma's clinical reasoning with Gemini's structured output capabilities, achieving 87.5% accuracy on criterion-level matching — competitive with GPT-4 (86.2%).

## Architecture

```
Patient Record ──► INGEST ──► PRESCREEN ──► VALIDATE ──► Matched Trials
                  (MedGemma    (Gemini Pro    (MedGemma 27B
                   4B imaging)  agentic search) + Gemini Pro
                                               two-stage)
```

| Stage | Model | Purpose |
|-------|-------|---------|
| **INGEST** | MedGemma 1.5 4B (multimodal) | Extract key clinical facts from EHR text + medical images |
| **PRESCREEN** | Gemini 3 Pro | Agentic search over ClinicalTrials.gov to find candidate trials |
| **VALIDATE** | MedGemma 27B + Gemini Pro (two-stage) | Criterion-level eligibility matching with structured verdicts |

## Benchmark Results

Evaluated on [TrialGPT criterion-level annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) (n=80 pairs, 4 seeds):

| Model | Accuracy | Macro-F1 | Cohen's k |
|-------|----------|----------|-----------|
| **TrialMatch (MedGemma 27B + Gemini Pro, two-stage)** | **87.5%** | **0.875** | **0.805** |
| GPT-4 (TrialGPT baseline) | 86.2% | — | — |
| Gemini 3 Pro (standalone) | 75.0% | 0.558 | 0.583 |
| MedGemma 27B (standalone) | 70.0% | 0.722 | 0.538 |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Cloud project with Vertex AI enabled (for MedGemma endpoints)
- Google AI Studio API key (for Gemini)

### Setup

```bash
# Install dependencies
uv sync

# Configure environment (copy and fill in your credentials)
cp .env.example .env
```

Required environment variables:

```bash
GCP_PROJECT_ID="your-gcp-project"
GCP_REGION="us-central1"
VERTEX_ENDPOINT_ID="..."         # MedGemma 4B endpoint
VERTEX_ENDPOINT_ID_27B="..."     # MedGemma 27B endpoint
GOOGLE_API_KEY="..."             # Gemini API key
```

### Run the Demo

```bash
uv run streamlit run demo/app.py --server.port 8501
```

The demo supports **cached mode** (deterministic replay) and **live mode** (real-time API calls with preflight checks).

### Run Benchmarks

```bash
# Phase 0 quick check (20 pairs, ~$1)
uv run trialmatch phase0 --config configs/phase0.yaml

# Full Tier A evaluation (1,024 pairs)
uv run trialmatch eval validate --pairs data/hf_cache/trialgpt_criterion_annotations.json --model gemini --tier A
```

### Run Tests

```bash
uv run pytest tests/unit/          # Unit tests
uv run pytest tests/bdd/ -m bdd    # BDD scenarios
uv run ruff check src/ tests/      # Lint
```

## Project Structure

```
src/trialmatch/
  cli/           # CLI commands
  ingest/        # Patient record understanding (MedGemma 4B multimodal)
  prescreen/     # Agentic trial search (Gemini Pro)
  validate/      # Criterion-level eligibility (two-stage)
  models/        # Model adapters (Vertex AI, Gemini, HF fallback)
  evaluation/    # Metrics (accuracy, F1, Cohen's k, confusion matrices)
  tracing/       # Run artifacts and cost tracking

demo/            # Streamlit demo application
runs/            # Reproducible benchmark run artifacts
configs/         # Evaluation configuration files
data/            # Datasets and cached annotations
```

## Key Design Decisions

- **Two-stage VALIDATE**: MedGemma provides clinical reasoning; Gemini extracts structured labels. Neither model alone matches their combined accuracy.
- **Component isolation**: Each pipeline stage is evaluated independently against gold-standard inputs to isolate errors.
- **Run reproducibility**: Every benchmark run is persisted with full config, inputs, outputs, and traces in `runs/`.

## Models

TrialMatch uses models deployed on **Vertex AI Model Garden** (recommended):

| Model | Deployment | Use Case |
|-------|-----------|----------|
| MedGemma 1.5 4B | Vertex AI (2x L4 GPU) | Multimodal ingest (EHR + images) |
| MedGemma 27B | Vertex AI (2x L4, int8) | Clinical reasoning for VALIDATE |
| Gemini 3 Pro | Google AI Studio | Structured output + agentic search |

## License

This project was developed for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) (Kaggle, Feb 2026).
