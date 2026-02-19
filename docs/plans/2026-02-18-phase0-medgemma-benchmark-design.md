# Phase 0 Benchmark Design: MedGemma vs Gemini 3 Pro Clinical Trial Matching

**Date**: 2026-02-18
**Updated**: 2026-02-19
**Status**: Approved (v2 — switched to TrialGPT HF criterion-level data)
**Approach**: Vertical Slice (Option A)

---

## Core Question

> Given a patient note and a single eligibility criterion, can MedGemma 1.5 4B judge criterion-level eligibility more accurately than Gemini 3 Pro?

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Phase 0 full (20 pairs, 2 models, ~$1) | Directional capability check before investing in Tier A |
| Ground truth | TrialGPT HF `expert_eligibility` (criterion-level) | Physician-annotated, criterion-level granularity; replaces trial-level TREC qrels |
| Data source | `ncbi/TrialGPT-Criterion-Annotations` (HuggingFace) | Self-contained Parquet: patient notes + criterion text + expert labels. No snapshot download needed |
| Built-in baseline | GPT-4 predictions (`gpt4_eligibility`) in same dataset | Free comparison — no extra API calls for baseline |
| Test strategy | BDD + Unit Tests | Per CLAUDE.md requirements |
| Component scope | VALIDATE only (skip INGEST/PRESCREEN) | Isolate medical reasoning evaluation |
| Label mapping | 6-class HF → 3-class task (see below) | Aligns with clinical semantics |

## Data Strategy

### Source: TrialGPT Criterion Annotations (HuggingFace)

Single source — no FTP downloads, no corpus files, no qrels TSV:

```python
from datasets import load_dataset
ds = load_dataset("ncbi/TrialGPT-Criterion-Annotations", split="train")
```

| Field | Type | Description |
|-------|------|-------------|
| `annotation_id` | int | Unique row ID |
| `patient_id` | str | Patient identifier |
| `note` | str | Full patient clinical note |
| `trial_id` | str | NCT ID |
| `trial_title` | str | Trial title |
| `criterion_type` | str | `"inclusion"` or `"exclusion"` |
| `criterion_text` | str | Single eligibility criterion text |
| `gpt4_eligibility` | str | GPT-4 prediction (6-class) |
| `expert_eligibility` | str | Expert ground truth (6-class) |
| `gpt4_explanation` | str | GPT-4 reasoning chain |
| `explanation_correctness` | str | Expert rating of GPT-4 explanation (Correct / Incorrect / Partially Correct) |
| `gpt4_sentences` | str | Sentence indices GPT-4 cited as evidence |
| `expert_sentences` | str | Sentence indices expert identified as evidence |

### Dataset Statistics

| Stat | Value |
|------|-------|
| Total rows | 1,024 |
| Unique patients | 53 |
| Unique trials | 103 |
| Criterion types | 2 (inclusion / exclusion) |
| Expert eligibility classes | 6 |
| Format | Parquet (auto-converted by HF) |
| License | Public Domain |
| Download size | < 5 MB |

### Label Mapping (6-class → 3-class)

The HF dataset uses 6 fine-grained labels. We map to 3 classes for our benchmark task:

| HF `expert_eligibility` | Our Label | Meaning |
|--------------------------|-----------|---------|
| `included` | **MET** | Patient meets this inclusion criterion |
| `not excluded` | **MET** | Exclusion criterion does not apply to patient |
| `excluded` | **NOT_MET** | Patient is excluded by this exclusion criterion |
| `not included` | **NOT_MET** | Patient does not meet this inclusion criterion |
| `not enough information` | **UNKNOWN** | Insufficient data to determine |
| `not applicable` | **UNKNOWN** | Criterion not applicable (e.g., pregnancy for male) |

This 3-class mapping (MET / NOT_MET / UNKNOWN) aligns with the VALIDATE component's `evaluate_criterion()` interface.

### Label Distribution (full dataset, to be confirmed after loading)

| Our Label | Expected Count | Role in Benchmark |
|-----------|---------------|-------------------|
| MET | ~400-500 | Core: criterion correctly satisfied |
| NOT_MET | ~300-400 | Core: criterion not satisfied |
| UNKNOWN | ~100-200 | Edge cases: insufficient info |

### Phase 0 Sampling (20 pairs, seed=42)

Stratified sample from 1,024 rows:
- 8 MET (criterion satisfied)
- 8 NOT_MET (criterion not satisfied)
- 4 UNKNOWN (insufficient info / not applicable)

Balanced across `criterion_type` (inclusion vs exclusion) where possible.

## Classification Task

### Three-Class Criterion-Level Prediction

For each (patient_note, criterion_text) pair, models predict:

| Prediction | Meaning |
|------------|---------|
| **MET** | Patient satisfies this criterion |
| **NOT_MET** | Patient does not satisfy this criterion |
| **UNKNOWN** | Insufficient information to determine |

### Why This Tests Medical Reasoning

- **MET vs NOT_MET** requires the model to parse medical terminology in the criterion, match against patient conditions, and reason about clinical semantics
- **UNKNOWN** tests the model's ability to recognize information gaps rather than hallucinate
- Criterion-level evaluation isolates individual reasoning steps — no aggregation noise from trial-level verdicts

### Comparison Axes

| Comparison | Method |
|------------|--------|
| MedGemma vs Gemini 3 Pro | Both models predict on same 20 pairs |
| MedGemma vs GPT-4 | GPT-4 predictions already in dataset (`gpt4_eligibility`) |
| All models vs Expert | Expert labels are ground truth (`expert_eligibility`) |

## Prompt Design

```
You are a clinical trial eligibility assessment expert.

Given a patient's clinical note and a single eligibility criterion from a clinical trial,
determine whether the patient meets this criterion.

Criterion Type: {criterion_type}  (inclusion or exclusion)

Criterion:
{criterion_text}

Patient Note:
{patient_note}

Respond in JSON format:
{
  "verdict": "MET" | "NOT_MET" | "UNKNOWN",
  "reasoning": "Step-by-step explanation citing specific evidence from the patient note",
  "evidence_sentences": "Comma-separated indices of sentences from the patient note that support your verdict"
}

Definitions:
- MET: The patient clearly satisfies this criterion based on the available information
- NOT_MET: The patient clearly does not satisfy this criterion
- UNKNOWN: There is not enough information in the patient note to determine this
```

Evidence sentence indices enable comparison with `expert_sentences` for explanation quality evaluation.

## Model Configuration

| Model | Provider | Model ID | Concurrency | Cost/pair |
|-------|----------|----------|-------------|-----------|
| MedGemma 1.5 4B | HF Inference | `google/medgemma-1-5-4b-it-hae` | 5 | ~$0.00 |
| Gemini 3 Pro | Google AI Studio | `gemini-3-pro-preview` | 10 | ~$0.025 |
| GPT-4 (baseline) | — (from dataset) | — | — | $0.00 (pre-computed) |

**Total Phase 0 cost**: < $1 (20 pairs x 2 models = 40 LLM calls)

## Evaluation Metrics

### Primary Metrics (Criterion-Level)

| Metric | Purpose |
|--------|---------|
| Overall Accuracy | Baseline correctness across 3 classes |
| Macro-F1 | Balanced performance across MET/NOT_MET/UNKNOWN |
| MET/NOT_MET F1 | Core metric: medical reasoning quality |
| Cohen's kappa | Agreement with expert labels beyond chance |
| Confusion Matrix | Error pattern analysis (3x3) |

### Bonus Metrics (from HF dataset)

| Metric | Purpose |
|--------|---------|
| Evidence Overlap | Jaccard similarity between model's cited sentences and `expert_sentences` |
| Explanation Quality | Compare model reasoning against `gpt4_explanation` + `explanation_correctness` patterns |
| Inclusion vs Exclusion accuracy | Does the model handle exclusion criteria differently? |

## Go/No-Go Criteria

| Signal | Decision |
|--------|----------|
| MedGemma MET/NOT_MET F1 > Gemini + 5% | **Go** — clear advantage, invest in Tier A |
| MedGemma ~ Gemini (within 5%) | **Investigate** — analyze reasoning quality + evidence overlap |
| MedGemma < Gemini - 5% | **No-Go** — MedGemma has no advantage on this task |
| MedGemma > GPT-4 baseline (87.3%) | **Strong Go** — exceeds published SOTA |

## Architecture (Vertical Slice)

```
HuggingFace Dataset               Models               VALIDATE            Evaluation
ncbi/TrialGPT-         ──┐
  Criterion-Annotations   ├── sampler ──► MedGemma  ──┐
  (1,024 pairs)         ──┘              Gemini     ──┼── evaluator ──► metrics ──► runs/<id>/
                                         GPT-4*     ──┘                              config
                                         (* from dataset)                            results
                                                                                     traces
```

### Module Design

```
src/trialmatch/
├── data/
│   ├── hf_loader.py            # Load ncbi/TrialGPT-Criterion-Annotations, map labels
│   └── sampler.py              # Phase 0 stratified sampling (20 pairs, seed=42)
├── models/
│   ├── base.py                 # ModelAdapter protocol + ModelResponse dataclass
│   ├── medgemma.py             # HF Inference API adapter
│   └── gemini.py               # Google AI Studio adapter
├── validate/
│   └── evaluator.py            # Prompt template + parse JSON verdict
├── evaluation/
│   └── metrics.py              # accuracy, F1, kappa, confusion matrix, evidence overlap
├── tracing/
│   └── run_manager.py          # run_id generation, save config + all results
└── cli/
    └── phase0.py               # `trialmatch phase0 --config configs/phase0.yaml`
```

### Pydantic Models

```python
class CriterionAnnotation:
    annotation_id: int
    patient_id: str
    note: str                       # full patient clinical note
    trial_id: str                   # NCT ID
    trial_title: str
    criterion_type: Literal["inclusion", "exclusion"]
    criterion_text: str             # single criterion
    expert_label: Literal["MET", "NOT_MET", "UNKNOWN"]  # mapped from 6-class
    expert_label_raw: str           # original 6-class label
    expert_sentences: list[int]     # evidence sentence indices
    gpt4_label: Literal["MET", "NOT_MET", "UNKNOWN"]    # mapped from 6-class
    gpt4_label_raw: str             # original 6-class label
    gpt4_explanation: str           # GPT-4 reasoning
    explanation_correctness: str    # Correct / Incorrect / Partially Correct

class Phase0Sample:
    pairs: list[CriterionAnnotation]  # 20 stratified pairs

class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost: float

class CriterionResult:
    verdict: Literal["MET", "NOT_MET", "UNKNOWN"]
    reasoning: str
    evidence_sentences: list[int]
    model_response: ModelResponse

class RunResult:
    run_id: str
    model_name: str
    results: list[CriterionResult]
    metrics: dict  # accuracy, f1, kappa, confusion_matrix, evidence_overlap
```

### Data Flow

1. `hf_loader.py` loads HuggingFace dataset, maps 6-class → 3-class labels
2. `sampler.py` stratified-samples 20 pairs from 1,024 rows
3. For each pair, `evaluator.py` sends prompt (patient_note + criterion_text) to model adapter, parses JSON verdict
4. `metrics.py` computes all metrics comparing predictions vs expert labels
5. Additionally: compare MedGemma/Gemini predictions against GPT-4 predictions (pre-computed)
6. `run_manager.py` saves everything to `runs/<run_id>/`

### Cost Tracking

Every LLM call logged with: model, input_tokens, output_tokens, estimated_cost, latency_ms.
Aggregated per run. Budget guard: abort if cumulative cost > $5.

## Tier Progression (Updated)

| Tier | Pairs | Data Source | Purpose | Budget |
|------|-------|-------------|---------|--------|
| Phase 0 | 20 (stratified) | HF dataset | Directional capability check | ~$1 |
| A | All 1,024 | HF dataset | Full criterion-level evaluation | ~$25 |
| B | TREC 2021+2022 trial-level | TREC qrels | Trial-level accuracy with statistical power | ~$300 |

## Known Limitations

1. **Small sample (n=20)**: Phase 0 is directional only. Tier A uses all 1,024 for statistical power.
2. **Whole-criterion evaluation**: Each criterion is evaluated as a single text block. Sub-criterion reasoning not tested.
3. **Dataset scope**: 53 patients, 103 trials — not exhaustive coverage of all disease areas.
4. **GPT-4 baseline vintage**: GPT-4 predictions in dataset are from 2023-2024 era; newer models may differ.
5. **Label mapping assumptions**: 6→3 class mapping may lose some nuance (e.g., "not applicable" → UNKNOWN).
6. **Single split**: HF dataset has only a train split — no held-out test set. All 1,024 rows are used.

## Data Dependencies (Simplified)

| What | Source | Size | How |
|------|--------|------|-----|
| Criterion annotations | HuggingFace `ncbi/TrialGPT-Criterion-Annotations` | < 5 MB | `datasets.load_dataset(...)` |

No FTP downloads. No corpus files. No qrels TSV. No 131 MB snapshot.
