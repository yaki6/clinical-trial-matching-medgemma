# Phase 0 Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a 20-pair criterion-level benchmark comparing MedGemma 1.5 4B vs Gemini 3 Pro, using the TrialGPT HuggingFace criterion annotations dataset.

**Acceptance Criteria:**
1. The validate module runs TrialGPT HF dataset to benchmark MedGemma vs Gemini 3 Pro on criterion-level eligibility matching
2. The validate module's core `evaluate_criterion()` function is reusable in the end-to-end clinical trial searching pipeline (no benchmark-specific coupling)

**Architecture:** Vertical slice — HF data loading, model adapters, **reusable** validate evaluator, benchmark runner, metrics, tracing, CLI. The validate module exposes a generic interface (`patient_note + criterion → verdict`) that the benchmark harness calls with HF data, and that the future e2e pipeline will call with live ClinicalTrials.gov data.

**Tech Stack:** Python 3.11+, Click CLI, Pydantic models, datasets (HuggingFace), huggingface_hub (MedGemma), google-genai (Gemini), scikit-learn (metrics), structlog (logging), pytest + pytest-bdd (testing).

**Design Doc:** `docs/plans/2026-02-18-phase0-medgemma-benchmark-design.md`

---

## Key Design Decision: Reusable Validate Module

The validate module has a **dual purpose**:

```
┌────────────────────────────────────────────────┐
│  validate/evaluator.py  (REUSABLE CORE)        │
│                                                 │
│  evaluate_criterion(                            │
│    patient_note: str,                           │
│    criterion_text: str,                         │
│    criterion_type: "inclusion" | "exclusion",   │
│    adapter: ModelAdapter,                       │
│  ) -> CriterionResult                           │
│                                                 │
│  - No knowledge of HF dataset or benchmark      │
│  - No knowledge of TrialGPT data format         │
│  - Pure: text in, verdict out                   │
└──────────────┬─────────────────┬───────────────┘
               │                 │
    ┌──────────▼──────┐  ┌──────▼──────────────┐
    │  BENCHMARK USE  │  │  E2E PIPELINE USE    │
    │  (Phase 0)      │  │  (Future)            │
    │                 │  │                      │
    │  HF dataset →   │  │  CT.gov API →        │
    │  sampler →      │  │  trial parser →      │
    │  evaluate_      │  │  evaluate_           │
    │  criterion()    │  │  criterion()         │
    │  → metrics      │  │  → patient report    │
    └─────────────────┘  └─────────────────────┘
```

This means:
- `evaluator.py` takes raw strings, not domain objects
- The benchmark harness (`cli/phase0.py`) extracts strings from HF dataset rows and passes them to the evaluator
- The future e2e pipeline will extract strings from CT.gov API responses and pass them to the same evaluator

---

## Task 1: Infrastructure Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/bdd/__init__.py`
- Create: `tests/bdd/conftest.py`
- Create: `features/.gitkeep`

**Step 1: Add dependencies**

In `pyproject.toml`, add to `[project] dependencies`:
```toml
"datasets>=2.0",     # HuggingFace datasets for TrialGPT data
```

Add to `[dependency-groups] dev`:
```toml
"pytest-bdd>=7.0",
```

Also add BDD markers to `[tool.pytest.ini_options]`:
```toml
markers = [
    "integration: marks integration tests",
    "e2e: marks end-to-end tests",
    "bdd: marks BDD tests",
    "component_validate: validate module BDD tests",
    "component_data: data module BDD tests",
]
```

**Step 2: Install dependencies**

Run: `uv sync`

**Step 3: Create BDD test infrastructure**

Create `tests/bdd/__init__.py` (empty).
Create `tests/bdd/conftest.py` (empty docstring).
Create `features/` directory with `.gitkeep`.

**Step 4: Verify**

Run: `uv run pytest --markers | grep bdd`
Expected: Shows `@pytest.mark.bdd` marker.

**Step 5: Commit**

```bash
git add pyproject.toml tests/bdd/ features/
git commit -m "chore: add pytest-bdd, HF datasets dependency, BDD test infrastructure"
```

---

## Task 2: Domain Models

**Files:**
- Create: `src/trialmatch/models/schema.py`
- Create: `tests/unit/test_schema.py`

These are the Pydantic models for the TrialGPT HF dataset + evaluator output.

**Step 1: Write the failing test**

Create `tests/unit/test_schema.py`:

```python
"""Tests for domain models."""

from trialmatch.models.schema import (
    CriterionAnnotation,
    CriterionResult,
    CriterionVerdict,
    ModelResponse,
    Phase0Sample,
    RunResult,
)


def test_criterion_verdict_values():
    assert CriterionVerdict.MET == "MET"
    assert CriterionVerdict.NOT_MET == "NOT_MET"
    assert CriterionVerdict.UNKNOWN == "UNKNOWN"


def test_criterion_annotation_creation():
    a = CriterionAnnotation(
        annotation_id=1,
        patient_id="P1",
        note="45-year-old male with lung cancer",
        trial_id="NCT001",
        trial_title="Test Trial",
        criterion_type="inclusion",
        criterion_text="Age >= 18",
        expert_label=CriterionVerdict.MET,
        expert_label_raw="included",
        expert_sentences=[0, 1],
        gpt4_label=CriterionVerdict.MET,
        gpt4_label_raw="included",
        gpt4_explanation="Patient is 45, meeting age criterion.",
        explanation_correctness="Correct",
    )
    assert a.criterion_type == "inclusion"
    assert a.expert_label == CriterionVerdict.MET


def test_model_response():
    r = ModelResponse(
        text='{"verdict": "MET"}',
        input_tokens=100,
        output_tokens=50,
        latency_ms=1200.0,
        estimated_cost=0.01,
    )
    assert r.input_tokens == 100


def test_criterion_result():
    mr = ModelResponse(
        text="test", input_tokens=0, output_tokens=0, latency_ms=0, estimated_cost=0
    )
    cr = CriterionResult(
        verdict=CriterionVerdict.MET,
        reasoning="Patient meets criterion",
        evidence_sentences=[0, 2],
        model_response=mr,
    )
    assert cr.verdict == CriterionVerdict.MET
    assert cr.evidence_sentences == [0, 2]


def test_criterion_result_without_evidence():
    """Evidence sentences are optional."""
    mr = ModelResponse(
        text="test", input_tokens=0, output_tokens=0, latency_ms=0, estimated_cost=0
    )
    cr = CriterionResult(
        verdict=CriterionVerdict.NOT_MET,
        reasoning="Patient does not meet criterion",
        model_response=mr,
    )
    assert cr.evidence_sentences == []


def test_phase0_sample():
    a = CriterionAnnotation(
        annotation_id=1, patient_id="P1", note="patient",
        trial_id="NCT1", trial_title="t", criterion_type="inclusion",
        criterion_text="c", expert_label=CriterionVerdict.MET,
        expert_label_raw="included", expert_sentences=[],
        gpt4_label=CriterionVerdict.MET, gpt4_label_raw="included",
        gpt4_explanation="ok", explanation_correctness="Correct",
    )
    sample = Phase0Sample(pairs=[a])
    assert len(sample.pairs) == 1


def test_run_result():
    rr = RunResult(run_id="test-001", model_name="medgemma", results=[], metrics={})
    assert rr.run_id == "test-001"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/trialmatch/models/schema.py`:

```python
"""Domain models for trialmatch benchmark.

These models serve both the Phase 0 benchmark (TrialGPT HF data)
and the future e2e clinical trial matching pipeline.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class CriterionVerdict(str, enum.Enum):
    """Criterion-level eligibility verdict.

    Used by both the benchmark evaluator and the e2e pipeline.
    """

    MET = "MET"              # Patient satisfies this criterion
    NOT_MET = "NOT_MET"      # Patient does not satisfy this criterion
    UNKNOWN = "UNKNOWN"      # Insufficient information to determine


class CriterionAnnotation(BaseModel):
    """A single criterion annotation from the TrialGPT HF dataset.

    Each row represents one (patient, criterion) pair with expert
    and GPT-4 labels.
    """

    annotation_id: int
    patient_id: str
    note: str                       # full patient clinical note
    trial_id: str                   # NCT ID
    trial_title: str
    criterion_type: str             # "inclusion" or "exclusion"
    criterion_text: str             # single eligibility criterion
    expert_label: CriterionVerdict  # mapped from 6-class
    expert_label_raw: str           # original HF label
    expert_sentences: list[int]     # evidence sentence indices
    gpt4_label: CriterionVerdict    # mapped from 6-class
    gpt4_label_raw: str             # original HF label
    gpt4_explanation: str           # GPT-4 reasoning
    explanation_correctness: str    # Correct / Incorrect / Partially Correct


class ModelResponse(BaseModel):
    """Raw model API response metadata."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost: float


class CriterionResult(BaseModel):
    """Result of evaluating a single criterion against a patient.

    This is the output of evaluate_criterion() — used by both
    the benchmark and the e2e pipeline.
    """

    verdict: CriterionVerdict
    reasoning: str
    evidence_sentences: list[int] = Field(default_factory=list)
    model_response: ModelResponse


class Phase0Sample(BaseModel):
    """Stratified sample of criterion annotations for Phase 0."""

    pairs: list[CriterionAnnotation]


class RunResult(BaseModel):
    """Complete results for one model's benchmark run."""

    run_id: str
    model_name: str
    results: list[CriterionResult]
    metrics: dict[str, Any]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_schema.py -v`
Expected: All 8 tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/models/schema.py tests/unit/test_schema.py
git commit -m "feat: add Pydantic domain models for criterion-level benchmark"
```

---

## Task 3: HuggingFace Data Loader

**Files:**
- Create: `src/trialmatch/data/hf_loader.py`
- Create: `tests/unit/test_hf_loader.py`
- Create: `tests/fixtures/hf_sample.json`

This replaces the old trialgpt_loader.py. Loads from HuggingFace, maps 6-class labels to 3-class.

**Step 1: Create test fixture**

Create `tests/fixtures/hf_sample.json` — 4 sample rows mimicking the HF dataset schema:

```json
[
  {
    "annotation_id": 1,
    "patient_id": "P1",
    "note": "A 45-year-old male with non-small cell lung cancer, ECOG 1, no prior chemotherapy.",
    "trial_id": "NCT001",
    "trial_title": "Phase III Pembrolizumab Trial",
    "criterion_type": "inclusion",
    "criterion_text": "Histologically confirmed NSCLC",
    "gpt4_explanation": "The patient has confirmed NSCLC.",
    "explanation_correctness": "Correct",
    "gpt4_sentences": "0",
    "expert_sentences": "0",
    "gpt4_eligibility": "included",
    "expert_eligibility": "included",
    "training": false,
    "split": "test"
  },
  {
    "annotation_id": 2,
    "patient_id": "P1",
    "note": "A 45-year-old male with non-small cell lung cancer, ECOG 1, no prior chemotherapy.",
    "trial_id": "NCT001",
    "trial_title": "Phase III Pembrolizumab Trial",
    "criterion_type": "exclusion",
    "criterion_text": "Prior treatment with anti-PD-1 therapy",
    "gpt4_explanation": "No prior immunotherapy mentioned.",
    "explanation_correctness": "Correct",
    "gpt4_sentences": "",
    "expert_sentences": "",
    "gpt4_eligibility": "not excluded",
    "expert_eligibility": "not excluded",
    "training": false,
    "split": "test"
  },
  {
    "annotation_id": 3,
    "patient_id": "P2",
    "note": "A 32-year-old female, pregnant, with breast cancer.",
    "trial_id": "NCT002",
    "trial_title": "Adjuvant Chemotherapy Trial",
    "criterion_type": "exclusion",
    "criterion_text": "Pregnant or nursing women",
    "gpt4_explanation": "Patient is pregnant.",
    "explanation_correctness": "Correct",
    "gpt4_sentences": "0",
    "expert_sentences": "0",
    "gpt4_eligibility": "excluded",
    "expert_eligibility": "excluded",
    "training": false,
    "split": "test"
  },
  {
    "annotation_id": 4,
    "patient_id": "P3",
    "note": "A 60-year-old male with hypertension.",
    "trial_id": "NCT003",
    "trial_title": "Diabetes Prevention Trial",
    "criterion_type": "inclusion",
    "criterion_text": "Diagnosis of Type 2 Diabetes",
    "gpt4_explanation": "No mention of diabetes.",
    "explanation_correctness": "Correct",
    "gpt4_sentences": "",
    "expert_sentences": "",
    "gpt4_eligibility": "not enough information",
    "expert_eligibility": "not enough information",
    "training": false,
    "split": "test"
  }
]
```

**Step 2: Write the failing test**

Create `tests/unit/test_hf_loader.py`:

```python
"""Tests for HuggingFace TrialGPT data loader."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from trialmatch.data.hf_loader import (
    map_label,
    parse_sentence_indices,
    load_annotations,
    load_annotations_from_file,
)
from trialmatch.models.schema import CriterionAnnotation, CriterionVerdict

FIXTURES = Path(__file__).parent.parent / "fixtures"


# --- Label mapping tests ---

def test_map_label_included():
    assert map_label("included") == CriterionVerdict.MET


def test_map_label_not_excluded():
    assert map_label("not excluded") == CriterionVerdict.MET


def test_map_label_excluded():
    assert map_label("excluded") == CriterionVerdict.NOT_MET


def test_map_label_not_included():
    assert map_label("not included") == CriterionVerdict.NOT_MET


def test_map_label_not_enough_info():
    assert map_label("not enough information") == CriterionVerdict.UNKNOWN


def test_map_label_not_applicable():
    assert map_label("not applicable") == CriterionVerdict.UNKNOWN


def test_map_label_unknown_fallback():
    assert map_label("something_weird") == CriterionVerdict.UNKNOWN


# --- Sentence parsing tests ---

def test_parse_sentence_indices_valid():
    assert parse_sentence_indices("0, 1, 3") == [0, 1, 3]


def test_parse_sentence_indices_single():
    assert parse_sentence_indices("0") == [0]


def test_parse_sentence_indices_empty():
    assert parse_sentence_indices("") == []


def test_parse_sentence_indices_none():
    assert parse_sentence_indices(None) == []


# --- Loading from fixture file ---

def test_load_annotations_from_file():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    assert len(annotations) == 4
    assert all(isinstance(a, CriterionAnnotation) for a in annotations)


def test_load_annotations_label_mapping():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    # Row 0: "included" -> MET
    assert annotations[0].expert_label == CriterionVerdict.MET
    # Row 1: "not excluded" -> MET
    assert annotations[1].expert_label == CriterionVerdict.MET
    # Row 2: "excluded" -> NOT_MET
    assert annotations[2].expert_label == CriterionVerdict.NOT_MET
    # Row 3: "not enough information" -> UNKNOWN
    assert annotations[3].expert_label == CriterionVerdict.UNKNOWN


def test_load_annotations_preserves_raw_labels():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    assert annotations[0].expert_label_raw == "included"
    assert annotations[2].expert_label_raw == "excluded"


def test_load_annotations_parses_sentences():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    assert annotations[0].expert_sentences == [0]
    assert annotations[1].expert_sentences == []  # empty string
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_hf_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 4: Write minimal implementation**

Create `src/trialmatch/data/hf_loader.py`:

```python
"""Load TrialGPT criterion annotations from HuggingFace.

Primary data source for Phase 0 and Tier A benchmarks (ADR-006).
Maps 6-class HF labels to 3-class CriterionVerdict (MET/NOT_MET/UNKNOWN).

Usage:
    # From HuggingFace (requires internet)
    annotations = load_annotations()

    # From local fixture file (for testing)
    annotations = load_annotations_from_file(path)
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from trialmatch.models.schema import CriterionAnnotation, CriterionVerdict

logger = structlog.get_logger()

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
DATASET_SPLIT = "train"

# 6-class -> 3-class label mapping (ADR-006)
LABEL_MAP: dict[str, CriterionVerdict] = {
    "included": CriterionVerdict.MET,
    "not excluded": CriterionVerdict.MET,
    "excluded": CriterionVerdict.NOT_MET,
    "not included": CriterionVerdict.NOT_MET,
    "not enough information": CriterionVerdict.UNKNOWN,
    "not applicable": CriterionVerdict.UNKNOWN,
}


def map_label(raw_label: str) -> CriterionVerdict:
    """Map a 6-class HF label to 3-class CriterionVerdict."""
    return LABEL_MAP.get(raw_label.strip().lower(), CriterionVerdict.UNKNOWN)


def parse_sentence_indices(raw: str | None) -> list[int]:
    """Parse comma-separated sentence indices from HF dataset field."""
    if not raw or not str(raw).strip():
        return []
    try:
        return [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        return []


def _row_to_annotation(row: dict) -> CriterionAnnotation:
    """Convert a single HF dataset row to CriterionAnnotation."""
    return CriterionAnnotation(
        annotation_id=row["annotation_id"],
        patient_id=str(row["patient_id"]),
        note=row["note"],
        trial_id=str(row["trial_id"]),
        trial_title=row["trial_title"],
        criterion_type=row["criterion_type"],
        criterion_text=row["criterion_text"],
        expert_label=map_label(row["expert_eligibility"]),
        expert_label_raw=row["expert_eligibility"],
        expert_sentences=parse_sentence_indices(row.get("expert_sentences")),
        gpt4_label=map_label(row["gpt4_eligibility"]),
        gpt4_label_raw=row["gpt4_eligibility"],
        gpt4_explanation=row.get("gpt4_explanation", ""),
        explanation_correctness=row.get("explanation_correctness", ""),
    )


def load_annotations() -> list[CriterionAnnotation]:
    """Load all annotations from HuggingFace dataset.

    Requires: `pip install datasets`
    Downloads ~5 MB on first call, cached thereafter.
    """
    from datasets import load_dataset

    logger.info("loading_hf_dataset", dataset_id=DATASET_ID, split=DATASET_SPLIT)
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    annotations = [_row_to_annotation(row) for row in ds]
    logger.info("loaded_annotations", count=len(annotations))
    return annotations


def load_annotations_from_file(path: Path) -> list[CriterionAnnotation]:
    """Load annotations from a local JSON file (for testing / offline use)."""
    with open(path) as f:
        rows = json.load(f)
    return [_row_to_annotation(row) for row in rows]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_hf_loader.py -v`
Expected: All 13 tests PASS.

**Step 6: Commit**

```bash
git add src/trialmatch/data/hf_loader.py tests/unit/test_hf_loader.py tests/fixtures/hf_sample.json
git commit -m "feat: add HF data loader with 6-class to 3-class label mapping (ADR-006)"
```

---

## Task 4: Stratified Sampler

**Files:**
- Create: `src/trialmatch/data/sampler.py`
- Create: `tests/unit/test_sampler.py`

**Step 1: Write the failing test**

Create `tests/unit/test_sampler.py`:

```python
"""Tests for stratified sampler."""

from pathlib import Path

from trialmatch.data.hf_loader import load_annotations_from_file
from trialmatch.data.sampler import stratified_sample
from trialmatch.models.schema import CriterionVerdict

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_stratified_sample_counts():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    sample = stratified_sample(annotations, n_pairs=3, seed=42)
    assert len(sample.pairs) == 3


def test_stratified_sample_deterministic():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    s1 = stratified_sample(annotations, n_pairs=3, seed=42)
    s2 = stratified_sample(annotations, n_pairs=3, seed=42)
    ids1 = [a.annotation_id for a in s1.pairs]
    ids2 = [a.annotation_id for a in s2.pairs]
    assert ids1 == ids2


def test_stratified_sample_covers_labels():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    sample = stratified_sample(annotations, n_pairs=3, seed=42)
    labels = {a.expert_label for a in sample.pairs}
    # Fixture has MET, NOT_MET, UNKNOWN — sample of 3 should cover at least 2
    assert len(labels) >= 2


def test_stratified_sample_respects_n():
    annotations = load_annotations_from_file(FIXTURES / "hf_sample.json")
    sample = stratified_sample(annotations, n_pairs=2, seed=42)
    assert len(sample.pairs) == 2
```

**Step 2: Write minimal implementation**

Create `src/trialmatch/data/sampler.py`:

```python
"""Stratified sampling of criterion annotations for Phase 0.

Samples from CriterionAnnotation list, stratified by expert_label
(MET / NOT_MET / UNKNOWN), with deterministic seed.
"""

from __future__ import annotations

import random

from trialmatch.models.schema import CriterionAnnotation, CriterionVerdict, Phase0Sample


def stratified_sample(
    annotations: list[CriterionAnnotation],
    n_pairs: int,
    seed: int = 42,
    distribution: dict[CriterionVerdict, float] | None = None,
) -> Phase0Sample:
    """Sample n_pairs with stratified label distribution.

    Default distribution: 40% MET, 40% NOT_MET, 20% UNKNOWN.
    """
    if distribution is None:
        distribution = {
            CriterionVerdict.MET: 0.4,
            CriterionVerdict.NOT_MET: 0.4,
            CriterionVerdict.UNKNOWN: 0.2,
        }

    # Group by expert label
    buckets: dict[CriterionVerdict, list[CriterionAnnotation]] = {}
    for a in annotations:
        buckets.setdefault(a.expert_label, []).append(a)

    rng = random.Random(seed)
    selected: list[CriterionAnnotation] = []

    for label, ratio in sorted(distribution.items(), key=lambda x: x[0].value):
        pool = buckets.get(label, [])
        n_target = round(n_pairs * ratio)
        n_take = min(n_target, len(pool))
        if n_take > 0:
            selected.extend(rng.sample(pool, n_take))

    # If rounding left us short, fill from largest pool
    while len(selected) < n_pairs:
        remaining = [a for a in annotations if a not in selected]
        if not remaining:
            break
        selected.append(rng.choice(remaining))

    return Phase0Sample(pairs=selected[:n_pairs])
```

**Step 3: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sampler.py -v`
Expected: All 4 tests PASS.

**Step 4: Commit**

```bash
git add src/trialmatch/data/sampler.py tests/unit/test_sampler.py
git commit -m "feat: add stratified sampler for criterion-level Phase 0 pairs"
```

---

## Task 5: Model Base Protocol

**Files:**
- Create: `src/trialmatch/models/base.py`
- Create: `tests/unit/test_model_base.py`

Same as original plan — `ModelAdapter` ABC with `generate()` and `health_check()`. No changes needed from HF data switch.

**Step 1: Write the failing test**

Create `tests/unit/test_model_base.py`:

```python
"""Tests for model adapter base protocol."""

import asyncio

from trialmatch.models.base import ModelAdapter
from trialmatch.models.schema import ModelResponse


class FakeAdapter(ModelAdapter):
    @property
    def name(self) -> str:
        return "fake"

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        return ModelResponse(
            text='{"verdict": "MET", "reasoning": "test"}',
            input_tokens=len(prompt.split()),
            output_tokens=10,
            latency_ms=100.0,
            estimated_cost=0.0,
        )

    async def health_check(self) -> bool:
        return True


def test_fake_adapter_implements_protocol():
    adapter = FakeAdapter()
    assert adapter.name == "fake"


def test_fake_adapter_generate():
    adapter = FakeAdapter()
    result = asyncio.run(adapter.generate("test prompt"))
    assert isinstance(result, ModelResponse)
    assert result.input_tokens > 0


def test_fake_adapter_health_check():
    adapter = FakeAdapter()
    assert asyncio.run(adapter.health_check()) is True
```

**Step 2: Write minimal implementation**

Create `src/trialmatch/models/base.py`:

```python
"""Base protocol for model adapters.

This interface is shared by both benchmark and e2e pipeline model usage.
"""

from __future__ import annotations

import abc

from trialmatch.models.schema import ModelResponse


class ModelAdapter(abc.ABC):
    """Abstract base for LLM model adapters."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Model name for logging and run tracking."""

    @abc.abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Send prompt to model and return structured response."""

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if model endpoint is reachable."""
```

**Step 3: Verify, commit**

Run: `uv run pytest tests/unit/test_model_base.py -v`

```bash
git add src/trialmatch/models/base.py tests/unit/test_model_base.py
git commit -m "feat: add ModelAdapter abstract base class"
```

---

## Task 6: MedGemma Adapter

Same as original plan — no changes from HF data switch. See original Task 6 for full implementation.

**Commit message:** `feat: add MedGemma adapter with Gemma template and retry logic`

---

## Task 7: Gemini Adapter

Same as original plan — no changes from HF data switch. See original Task 7 for full implementation.

**Commit message:** `feat: add Gemini 3 Pro adapter with JSON output and cost tracking`

---

## Task 8: Validate Evaluator (REUSABLE CORE)

**Files:**
- Create: `src/trialmatch/validate/evaluator.py`
- Create: `tests/unit/test_evaluator.py`

**This is the critical reusable component.** The evaluator takes raw text inputs (patient note + criterion text + criterion type) and returns a `CriterionResult`. It has NO dependency on the HF dataset, TrialGPT format, or benchmark infrastructure.

**Step 1: Write the failing test**

Create `tests/unit/test_evaluator.py`:

```python
"""Tests for validate evaluator.

The evaluator is the REUSABLE CORE — it works with raw text inputs,
not benchmark-specific data structures. Tests verify:
1. Prompt construction from raw text
2. Verdict parsing from model output
3. End-to-end evaluation with mocked model
"""

import asyncio
from unittest.mock import AsyncMock

from trialmatch.models.schema import CriterionVerdict, ModelResponse
from trialmatch.validate.evaluator import (
    build_criterion_prompt,
    parse_criterion_verdict,
    evaluate_criterion,
)


# --- Prompt building tests (reusable interface) ---

def test_build_prompt_contains_patient_note():
    prompt = build_criterion_prompt(
        patient_note="45-year-old male with NSCLC",
        criterion_text="Age >= 18",
        criterion_type="inclusion",
    )
    assert "45-year-old male with NSCLC" in prompt


def test_build_prompt_contains_criterion():
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Histologically confirmed NSCLC",
        criterion_type="inclusion",
    )
    assert "Histologically confirmed NSCLC" in prompt


def test_build_prompt_contains_criterion_type():
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="exclusion",
    )
    assert "exclusion" in prompt.lower()


def test_build_prompt_asks_for_met_not_met_unknown():
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "MET" in prompt
    assert "NOT_MET" in prompt
    assert "UNKNOWN" in prompt


def test_build_prompt_asks_for_evidence_sentences():
    prompt = build_criterion_prompt(
        patient_note="Patient note",
        criterion_text="Criterion",
        criterion_type="inclusion",
    )
    assert "evidence_sentences" in prompt


# --- Verdict parsing tests ---

def test_parse_verdict_met():
    v, r, e = parse_criterion_verdict(
        '{"verdict": "MET", "reasoning": "meets criterion", "evidence_sentences": "0, 2"}'
    )
    assert v == CriterionVerdict.MET
    assert r == "meets criterion"
    assert e == [0, 2]


def test_parse_verdict_not_met():
    v, r, e = parse_criterion_verdict(
        '{"verdict": "NOT_MET", "reasoning": "does not meet", "evidence_sentences": ""}'
    )
    assert v == CriterionVerdict.NOT_MET
    assert e == []


def test_parse_verdict_unknown():
    v, r, e = parse_criterion_verdict(
        '{"verdict": "UNKNOWN", "reasoning": "insufficient info"}'
    )
    assert v == CriterionVerdict.UNKNOWN


def test_parse_verdict_markdown_wrapped():
    raw = '```json\n{"verdict": "MET", "reasoning": "ok"}\n```'
    v, r, e = parse_criterion_verdict(raw)
    assert v == CriterionVerdict.MET


def test_parse_verdict_fallback_met():
    v, r, e = parse_criterion_verdict("The patient MEETS this criterion clearly.")
    assert v == CriterionVerdict.MET


def test_parse_verdict_fallback_not_met():
    v, r, e = parse_criterion_verdict("The patient does NOT_MET this criterion.")
    assert v == CriterionVerdict.NOT_MET


def test_parse_verdict_fallback_unknown():
    v, r, e = parse_criterion_verdict("I cannot determine eligibility from the note.")
    assert v == CriterionVerdict.UNKNOWN


# --- End-to-end evaluation (reusable interface) ---

def test_evaluate_criterion_returns_result():
    """evaluate_criterion takes raw text — no benchmark data structures."""
    mock_adapter = AsyncMock()
    mock_adapter.generate.return_value = ModelResponse(
        text='{"verdict": "NOT_MET", "reasoning": "age exclusion", "evidence_sentences": "0"}',
        input_tokens=200,
        output_tokens=30,
        latency_ms=500.0,
        estimated_cost=0.01,
    )

    result = asyncio.run(evaluate_criterion(
        patient_note="30-year-old female with breast cancer",
        criterion_text="Age >= 40 years",
        criterion_type="inclusion",
        adapter=mock_adapter,
    ))
    assert result.verdict == CriterionVerdict.NOT_MET
    assert result.evidence_sentences == [0]


def test_evaluate_criterion_no_benchmark_coupling():
    """Verify evaluate_criterion doesn't import any benchmark/data modules."""
    import inspect
    from trialmatch.validate import evaluator
    source = inspect.getsource(evaluator)
    # Should NOT import from data/ or cli/ modules
    assert "from trialmatch.data" not in source
    assert "from trialmatch.cli" not in source
    assert "CriterionAnnotation" not in source
    assert "Phase0Sample" not in source
```

**Step 2: Write minimal implementation**

Create `src/trialmatch/validate/evaluator.py`:

```python
"""Criterion-level eligibility evaluator.

REUSABLE CORE: This module evaluates whether a patient meets a single
eligibility criterion. It takes raw text inputs and returns a structured
verdict. No dependency on benchmark data, HF dataset, or TrialGPT format.

Used by:
- Phase 0 benchmark (via cli/phase0.py)
- Future e2e clinical trial searching pipeline
"""

from __future__ import annotations

import json
import re

from trialmatch.models.base import ModelAdapter
from trialmatch.models.schema import CriterionResult, CriterionVerdict, ModelResponse

PROMPT_TEMPLATE = """You are a clinical trial eligibility assessment expert.

Given a patient's clinical note and a single eligibility criterion from a clinical trial,
determine whether the patient meets this criterion.

Criterion Type: {criterion_type}  (inclusion or exclusion)

Criterion:
{criterion_text}

Patient Note:
{patient_note}

Respond in JSON format:
{{"verdict": "MET" | "NOT_MET" | "UNKNOWN", "reasoning": "Step-by-step explanation citing specific evidence from the patient note", "evidence_sentences": "Comma-separated indices of sentences from the patient note that support your verdict"}}

Definitions:
- MET: The patient clearly satisfies this criterion based on the available information
- NOT_MET: The patient clearly does not satisfy this criterion
- UNKNOWN: There is not enough information in the patient note to determine this"""


def build_criterion_prompt(
    patient_note: str,
    criterion_text: str,
    criterion_type: str,
) -> str:
    """Build the evaluation prompt from raw text inputs.

    This is the reusable prompt builder — takes strings, not domain objects.
    """
    return PROMPT_TEMPLATE.format(
        patient_note=patient_note,
        criterion_text=criterion_text,
        criterion_type=criterion_type,
    )


def parse_criterion_verdict(raw_text: str) -> tuple[CriterionVerdict, str, list[int]]:
    """Parse model output into (verdict, reasoning, evidence_sentences).

    Tries JSON first, then markdown-wrapped JSON, then keyword extraction.
    """
    # Try direct JSON parse
    try:
        data = json.loads(raw_text)
        return (
            CriterionVerdict(data["verdict"]),
            data.get("reasoning", ""),
            _parse_evidence(data.get("evidence_sentences", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Try markdown-wrapped JSON
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return (
                CriterionVerdict(data["verdict"]),
                data.get("reasoning", ""),
                _parse_evidence(data.get("evidence_sentences", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fallback: keyword extraction
    upper = raw_text.upper()
    if "NOT_MET" in upper:
        return CriterionVerdict.NOT_MET, raw_text, []
    if "MET" in upper and "MEET" not in upper:
        return CriterionVerdict.MET, raw_text, []
    if "MEETS" in upper:
        return CriterionVerdict.MET, raw_text, []

    return CriterionVerdict.UNKNOWN, raw_text, []


def _parse_evidence(raw: str | int | list | None) -> list[int]:
    """Parse evidence sentence indices from various formats."""
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if not raw or not str(raw).strip():
        return []
    try:
        return [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        return []


async def evaluate_criterion(
    patient_note: str,
    criterion_text: str,
    criterion_type: str,
    adapter: ModelAdapter,
    max_tokens: int = 2048,
) -> CriterionResult:
    """Evaluate a single criterion against a patient note.

    This is the REUSABLE ENTRY POINT. Takes raw text, returns structured result.
    No dependency on benchmark data structures.

    Args:
        patient_note: Full patient clinical note text.
        criterion_text: Single eligibility criterion text.
        criterion_type: "inclusion" or "exclusion".
        adapter: Any ModelAdapter implementation.
        max_tokens: Max tokens for model response.

    Returns:
        CriterionResult with verdict, reasoning, evidence, and model metadata.
    """
    prompt = build_criterion_prompt(
        patient_note=patient_note,
        criterion_text=criterion_text,
        criterion_type=criterion_type,
    )
    response: ModelResponse = await adapter.generate(prompt, max_tokens=max_tokens)
    verdict, reasoning, evidence = parse_criterion_verdict(response.text)

    return CriterionResult(
        verdict=verdict,
        reasoning=reasoning,
        evidence_sentences=evidence,
        model_response=response,
    )
```

**Step 3: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_evaluator.py -v`
Expected: All 14 tests PASS (including the no-coupling assertion).

**Step 4: Commit**

```bash
git add src/trialmatch/validate/evaluator.py tests/unit/test_evaluator.py
git commit -m "feat: add reusable criterion evaluator (benchmark + e2e pipeline compatible)"
```

---

## Task 9: Evaluation Metrics

**Files:**
- Create: `src/trialmatch/evaluation/metrics.py`
- Create: `tests/unit/test_metrics.py`

Updated for `CriterionVerdict` (MET/NOT_MET/UNKNOWN) and evidence overlap.

**Step 1: Write the failing test**

Create `tests/unit/test_metrics.py`:

```python
"""Tests for evaluation metrics."""

from trialmatch.evaluation.metrics import compute_metrics, compute_evidence_overlap
from trialmatch.models.schema import CriterionVerdict


def test_compute_metrics_perfect():
    predicted = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    actual = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    m = compute_metrics(predicted, actual)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["cohens_kappa"] == 1.0


def test_compute_metrics_all_wrong():
    predicted = [CriterionVerdict.UNKNOWN, CriterionVerdict.UNKNOWN, CriterionVerdict.UNKNOWN]
    actual = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    m = compute_metrics(predicted, actual)
    assert m["accuracy"] < 0.5


def test_compute_metrics_confusion_matrix():
    predicted = [CriterionVerdict.MET, CriterionVerdict.NOT_MET]
    actual = [CriterionVerdict.MET, CriterionVerdict.MET]
    m = compute_metrics(predicted, actual)
    assert "confusion_matrix" in m
    assert isinstance(m["confusion_matrix"], list)


def test_compute_metrics_per_class_f1():
    predicted = [CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.NOT_MET]
    actual = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.NOT_MET]
    m = compute_metrics(predicted, actual)
    assert "f1_per_class" in m
    assert "MET" in m["f1_per_class"]
    assert "NOT_MET" in m["f1_per_class"]


def test_compute_metrics_met_not_met_f1():
    """Core metric: F1 on just MET and NOT_MET classes."""
    predicted = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    actual = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
    m = compute_metrics(predicted, actual)
    assert "f1_met_not_met" in m
    assert m["f1_met_not_met"] == 1.0


def test_evidence_overlap_identical():
    assert compute_evidence_overlap([0, 1, 2], [0, 1, 2]) == 1.0


def test_evidence_overlap_disjoint():
    assert compute_evidence_overlap([0, 1], [2, 3]) == 0.0


def test_evidence_overlap_partial():
    overlap = compute_evidence_overlap([0, 1, 2], [1, 2, 3])
    assert 0.0 < overlap < 1.0


def test_evidence_overlap_empty():
    assert compute_evidence_overlap([], []) == 1.0  # both empty = perfect agreement


def test_evidence_overlap_one_empty():
    assert compute_evidence_overlap([0, 1], []) == 0.0
```

**Step 2: Write minimal implementation**

Create `src/trialmatch/evaluation/metrics.py`:

```python
"""Evaluation metrics for criterion-level benchmark."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from trialmatch.models.schema import CriterionVerdict

LABELS = [CriterionVerdict.MET, CriterionVerdict.NOT_MET, CriterionVerdict.UNKNOWN]
LABEL_NAMES = [v.value for v in LABELS]


def compute_metrics(
    predicted: list[CriterionVerdict],
    actual: list[CriterionVerdict],
) -> dict[str, Any]:
    """Compute all benchmark evaluation metrics."""
    pred_str = [v.value for v in predicted]
    actual_str = [v.value for v in actual]

    acc = accuracy_score(actual_str, pred_str)
    f1_mac = f1_score(actual_str, pred_str, labels=LABEL_NAMES, average="macro", zero_division=0)
    kappa = cohen_kappa_score(actual_str, pred_str, labels=LABEL_NAMES)
    cm = confusion_matrix(actual_str, pred_str, labels=LABEL_NAMES)
    f1_per = f1_score(actual_str, pred_str, labels=LABEL_NAMES, average=None, zero_division=0)

    # Core metric: F1 on MET + NOT_MET only (excluding UNKNOWN)
    met_nm_labels = [CriterionVerdict.MET.value, CriterionVerdict.NOT_MET.value]
    f1_met_nm = f1_score(
        actual_str, pred_str, labels=met_nm_labels, average="macro", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_met_not_met": float(f1_met_nm),
        "cohens_kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
        "f1_per_class": {name: float(f1) for name, f1 in zip(LABEL_NAMES, f1_per)},
    }


def compute_evidence_overlap(
    predicted_sentences: list[int],
    expert_sentences: list[int],
) -> float:
    """Compute Jaccard similarity between predicted and expert evidence sentences."""
    pred_set = set(predicted_sentences)
    expert_set = set(expert_sentences)

    if not pred_set and not expert_set:
        return 1.0  # both empty = perfect agreement

    if not pred_set or not expert_set:
        return 0.0

    intersection = pred_set & expert_set
    union = pred_set | expert_set
    return len(intersection) / len(union)
```

**Step 3: Verify, commit**

Run: `uv run pytest tests/unit/test_metrics.py -v`

```bash
git add src/trialmatch/evaluation/metrics.py tests/unit/test_metrics.py
git commit -m "feat: add criterion-level metrics with evidence overlap (Jaccard)"
```

---

## Task 10: Run Manager (Tracing)

Same as original plan but updated for `CriterionVerdict` and `evidence_sentences`. See original Task 10 for full implementation — adjust field names.

**Commit message:** `feat: add run manager for saving benchmark results and cost tracking`

---

## Task 11: CLI Phase0 Command

**Files:**
- Modify: `src/trialmatch/cli/__init__.py`
- Create: `src/trialmatch/cli/phase0.py`
- Create: `tests/unit/test_cli_phase0.py`

This is the benchmark harness that connects HF data → evaluator → metrics.

**Step 1: Write the failing test**

Create `tests/unit/test_cli_phase0.py`:

```python
"""Tests for CLI phase0 command."""

from click.testing import CliRunner

from trialmatch.cli import main


def test_phase0_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["phase0", "--help"])
    assert result.exit_code == 0
    assert "phase0" in result.output.lower() or "Phase 0" in result.output


def test_phase0_dry_run():
    runner = CliRunner()
    result = runner.invoke(main, ["phase0", "--dry-run"])
    # Should not crash — dry-run prints config and exits
    assert result.exit_code == 0
```

**Step 2: Write implementation**

Create `src/trialmatch/cli/phase0.py`:

```python
"""CLI command for Phase 0 criterion-level benchmark.

This is the BENCHMARK HARNESS — it connects:
  HF dataset → sampler → evaluate_criterion() → metrics → run artifacts

The evaluate_criterion() call is the reusable core (validate/evaluator.py).
Everything else here is benchmark-specific orchestration.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import click
import structlog
import yaml

from trialmatch.data.hf_loader import load_annotations, load_annotations_from_file
from trialmatch.data.sampler import stratified_sample
from trialmatch.evaluation.metrics import compute_evidence_overlap, compute_metrics
from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.medgemma import MedGemmaAdapter
from trialmatch.models.schema import CriterionVerdict, RunResult
from trialmatch.tracing.run_manager import RunManager
from trialmatch.validate.evaluator import evaluate_criterion

logger = structlog.get_logger()


async def run_model_benchmark(adapter, sample, budget_max: float = 5.0):
    """Run all sampled criterion pairs through one model.

    Calls the REUSABLE evaluate_criterion() for each pair,
    passing raw text extracted from the HF annotations.
    """
    results = []
    total_cost = 0.0

    for i, annotation in enumerate(sample.pairs):
        logger.info(
            "evaluating",
            pair=f"{i + 1}/{len(sample.pairs)}",
            patient=annotation.patient_id,
            trial=annotation.trial_id,
            criterion_type=annotation.criterion_type,
            model=adapter.name,
        )

        # Call the REUSABLE evaluator with raw text
        result = await evaluate_criterion(
            patient_note=annotation.note,
            criterion_text=annotation.criterion_text,
            criterion_type=annotation.criterion_type,
            adapter=adapter,
        )

        total_cost += result.model_response.estimated_cost
        if total_cost > budget_max:
            logger.warning("budget_exceeded", total_cost=total_cost, max=budget_max)
            break

        results.append(result)

    return results


async def run_phase0(config: dict, dry_run: bool = False):
    """Execute Phase 0 criterion-level benchmark."""
    data_cfg = config.get("data", {})
    fixture_path = data_cfg.get("fixture_path")

    # Load annotations
    if fixture_path:
        annotations = load_annotations_from_file(Path(fixture_path))
    else:
        annotations = load_annotations()

    logger.info("data_loaded", annotations=len(annotations))

    # Sample
    n_pairs = data_cfg.get("n_pairs", 20)
    seed = data_cfg.get("seed", 42)
    sample = stratified_sample(annotations, n_pairs=n_pairs, seed=seed)
    logger.info("sampled", n_pairs=len(sample.pairs))

    if dry_run:
        click.echo(f"Dry run: would evaluate {len(sample.pairs)} pairs with models")
        for i, a in enumerate(sample.pairs):
            click.echo(
                f"  {i + 1}. [{a.criterion_type}] Patient {a.patient_id} x "
                f"Trial {a.trial_id}: \"{a.criterion_text[:60]}...\" "
                f"(expert={a.expert_label.value})"
            )
        return

    budget_max = config.get("budget", {}).get("max_cost_usd", 5.0)
    run_mgr = RunManager()

    # Run each model
    for model_cfg in config.get("models", []):
        if model_cfg["provider"] == "huggingface":
            adapter = MedGemmaAdapter(hf_token=os.environ.get("HF_TOKEN", ""))
        elif model_cfg["provider"] == "google":
            adapter = GeminiAdapter(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        else:
            logger.error("unknown_provider", provider=model_cfg["provider"])
            continue

        logger.info("running_model", model=adapter.name)
        results = await run_model_benchmark(adapter, sample, budget_max=budget_max)

        # Compute metrics vs expert labels
        predicted = [r.verdict for r in results]
        actual = [a.expert_label for a in sample.pairs[: len(results)]]
        metrics = compute_metrics(predicted, actual)

        # Evidence overlap (bonus metric)
        overlaps = [
            compute_evidence_overlap(r.evidence_sentences, a.expert_sentences)
            for r, a in zip(results, sample.pairs)
        ]
        metrics["mean_evidence_overlap"] = sum(overlaps) / len(overlaps) if overlaps else 0.0

        # Compare against GPT-4 baseline (free — from dataset)
        gpt4_labels = [a.gpt4_label for a in sample.pairs[: len(results)]]
        gpt4_metrics = compute_metrics(gpt4_labels, actual)
        metrics["gpt4_baseline_accuracy"] = gpt4_metrics["accuracy"]
        metrics["gpt4_baseline_f1_macro"] = gpt4_metrics["f1_macro"]

        # Save run
        run_id = run_mgr.generate_run_id(adapter.name)
        run_result = RunResult(
            run_id=run_id,
            model_name=adapter.name,
            results=results,
            metrics=metrics,
        )
        run_dir = run_mgr.save_run(run_result, config=config)

        # Print summary
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Model: {adapter.name}")
        click.echo(f"Run ID: {run_id}")
        click.echo(f"Results saved to: {run_dir}")
        click.echo(f"Accuracy: {metrics['accuracy']:.2%}")
        click.echo(f"Macro-F1: {metrics['f1_macro']:.2%}")
        click.echo(f"MET/NOT_MET F1: {metrics['f1_met_not_met']:.2%}")
        click.echo(f"Cohen's κ: {metrics['cohens_kappa']:.3f}")
        click.echo(f"Evidence Overlap: {metrics['mean_evidence_overlap']:.2%}")
        click.echo(f"--- GPT-4 Baseline ---")
        click.echo(f"GPT-4 Accuracy: {metrics['gpt4_baseline_accuracy']:.2%}")
        click.echo(f"GPT-4 Macro-F1: {metrics['gpt4_baseline_f1_macro']:.2%}")
        click.echo(f"{'=' * 60}")


@click.command("phase0")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.option("--dry-run", is_flag=True, help="Show sampled pairs without calling models")
def phase0_cmd(config_path: str | None, dry_run: bool):
    """Run Phase 0 benchmark: 20-pair criterion-level MedGemma vs Gemini comparison."""
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {"n_pairs": 20, "seed": 42},
            "models": [
                {"name": "medgemma-1.5-4b", "provider": "huggingface"},
                {"name": "gemini-3-pro", "provider": "google"},
            ],
            "budget": {"max_cost_usd": 5.0},
        }

    if dry_run:
        click.echo(f"Config: {json.dumps(config, indent=2, default=str)}")

    asyncio.run(run_phase0(config, dry_run=dry_run))
```

Modify `src/trialmatch/cli/__init__.py`:

```python
"""CLI entry point for trialmatch."""

import click

from trialmatch.cli.phase0 import phase0_cmd


@click.group()
def main():
    """MedGemma vs Gemini 3 Pro: Clinical trial criterion matching benchmark."""
    pass


main.add_command(phase0_cmd)
```

**Step 3: Verify, commit**

Run: `uv run pytest tests/unit/test_cli_phase0.py -v`

```bash
git add src/trialmatch/cli/ tests/unit/test_cli_phase0.py
git commit -m "feat: add Phase 0 CLI with HF data loading and GPT-4 baseline comparison"
```

---

## Task 12: BDD Feature File for Validate

**Files:**
- Create: `features/validate/criterion_evaluation.feature`
- Create: `tests/bdd/steps/validate_steps.py`

Updated for criterion-level evaluation with MET/NOT_MET/UNKNOWN.

**Step 1: Write the BDD feature file**

Create `features/validate/criterion_evaluation.feature`:

```gherkin
@component_validate @phase0
Feature: Criterion-level eligibility evaluation
  As a benchmark runner or e2e pipeline user
  I want to evaluate whether a patient meets a single eligibility criterion
  So that I can assess model medical reasoning quality

  @implemented
  Scenario: Prompt contains patient note and criterion
    Given a patient note "45-year-old male with NSCLC, ECOG 1"
    And an inclusion criterion "Histologically confirmed NSCLC"
    When I build the criterion evaluation prompt
    Then the prompt contains the patient note
    And the prompt contains the criterion text
    And the prompt asks for MET, NOT_MET, or UNKNOWN

  @implemented
  Scenario: Parse MET verdict from JSON
    Given the model returns '{"verdict": "MET", "reasoning": "confirmed NSCLC", "evidence_sentences": "0"}'
    When I parse the criterion verdict
    Then the verdict is MET
    And the reasoning contains "NSCLC"
    And evidence sentences include 0

  @implemented
  Scenario: Parse NOT_MET verdict from JSON
    Given the model returns '{"verdict": "NOT_MET", "reasoning": "too young", "evidence_sentences": ""}'
    When I parse the criterion verdict
    Then the verdict is NOT_MET
    And evidence sentences are empty

  @implemented
  Scenario: Parse UNKNOWN verdict from JSON
    Given the model returns '{"verdict": "UNKNOWN", "reasoning": "no lab data available"}'
    When I parse the criterion verdict
    Then the verdict is UNKNOWN

  @implemented
  Scenario: Evaluator is reusable (no benchmark coupling)
    Given the evaluate_criterion function
    Then it accepts raw text inputs only
    And it does not import from trialmatch.data
    And it does not import from trialmatch.cli
```

**Step 2: Write step definitions, run, commit**

```bash
git add features/validate/ tests/bdd/
git commit -m "feat: add BDD scenarios for reusable criterion evaluator"
```

---

## Task 13: Full Test Suite + Lint

Run all tests, lint, format. Fix any issues.

```bash
uv run pytest tests/unit/ -v --tb=short
uv run pytest tests/bdd/ -v
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

---

## Task 14: Integration Test — Dry Run with HF Data

Create `tests/integration/test_phase0_dryrun.py`:

```python
"""Integration test: Phase 0 dry run with fixture data."""

import pytest
from click.testing import CliRunner

from trialmatch.cli import main


@pytest.mark.integration
def test_phase0_dry_run_with_fixture():
    """Dry run with fixture data should list sampled pairs."""
    runner = CliRunner()
    result = runner.invoke(main, [
        "phase0",
        "--dry-run",
        "--config", "configs/phase0_test.yaml",
    ])
    assert result.exit_code == 0
    assert "dry run" in result.output.lower() or "Patient" in result.output
```

Create `configs/phase0_test.yaml` pointing to fixture data for offline testing.

---

## Task 15: Live Benchmark Run (Requires API Keys)

**Prerequisites:** `HF_TOKEN` + `GOOGLE_API_KEY` env vars.

Run: `uv run trialmatch phase0 --config configs/phase0.yaml`

Expected output includes:
- 20 pairs evaluated by MedGemma and Gemini
- Accuracy, Macro-F1, MET/NOT_MET F1, Cohen's κ
- Evidence overlap score
- GPT-4 baseline comparison (free)
- Results saved to `runs/phase0-*/`

---

## Dependency Graph

```
Task 1  (infra)     ─── no dependencies
Task 2  (models)    ─── no dependencies
Task 3  (hf_loader) ─── depends on Task 2
Task 4  (sampler)   ─── depends on Task 2, 3
Task 5  (base)      ─── depends on Task 2
Task 6  (medgemma)  ─── depends on Task 5
Task 7  (gemini)    ─── depends on Task 5
Task 8  (evaluator) ─── depends on Task 2, 5  ← REUSABLE CORE
Task 9  (metrics)   ─── depends on Task 2
Task 10 (tracing)   ─── depends on Task 2
Task 11 (CLI)       ─── depends on Task 3, 4, 6, 7, 8, 9, 10
Task 12 (BDD)       ─── depends on Task 8
Task 13 (lint/test) ─── depends on all above
Task 14 (integ)     ─── depends on Task 11
Task 15 (live run)  ─── depends on all + API keys
```

**Parallelizable groups:**
- Group A: Tasks 1, 2 (can start immediately)
- Group B: Tasks 3, 5, 9, 10 (after Task 2)
- Group C: Tasks 4, 6, 7, 8 (after their deps)
- Group D: Tasks 11, 12 (after Group C)
- Group E: Tasks 13-15 (sequential, end-to-end)
