# Phase 0 Benchmark Implementation Plan

> **SUPERSEDED (2026-02-19):** Data strategy updated by ADR-006. The data loading sections below reference TREC 2021 corpus.jsonl/queries.jsonl/qrels which are no longer used. See updated benchmark design: `docs/plans/2026-02-18-phase0-medgemma-benchmark-design.md` (v2). Data module should now use `ncbi/TrialGPT-Criterion-Annotations` from HuggingFace.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a 20-pair benchmark comparing MedGemma 1.5 4B vs Gemini 3 Pro on criterion-level clinical trial matching using TrialGPT HF data.

**Architecture:** Vertical slice — HF data loading, model adapters, validate evaluator, metrics, tracing, CLI. Each module does one thing. HF dataset provides all data (patient notes, criterion text, expert labels). Models return JSON verdicts (MET/NOT_MET/UNKNOWN). Metrics compare against expert labels.

**Tech Stack:** Python 3.11+, Click CLI, Pydantic models, huggingface_hub (MedGemma), google-genai (Gemini), scikit-learn (metrics), structlog (logging), httpx (downloads), pytest + pytest-bdd (testing).

**Design Doc:** `docs/plans/2026-02-18-phase0-medgemma-benchmark-design.md`

---

## Task 1: Infrastructure Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/bdd/__init__.py`
- Create: `tests/bdd/conftest.py`
- Create: `features/.gitkeep`

**Step 1: Add pytest-bdd to dev dependencies**

In `pyproject.toml`, add `"pytest-bdd>=7.0"` to `[dependency-groups] dev`:

```toml
[dependency-groups]
dev = [
    "pytest>=7.4",
    "pytest-mock>=3.12",
    "pytest-cov>=4.1",
    "pytest-timeout>=2.2",
    "vcrpy>=6.0",
    "ruff>=0.4",
    "pytest-bdd>=7.0",
]
```

Also add the `bdd` marker to `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: marks integration tests (deselect with '-m \"not integration\"')",
    "e2e: marks end-to-end tests (deselect with '-m \"not e2e\"')",
    "bdd: marks BDD tests",
    "component_data: data module BDD tests",
    "component_models: model adapter BDD tests",
    "component_validate: validate module BDD tests",
    "component_evaluation: evaluation module BDD tests",
]
addopts = "-v --tb=short"
```

**Step 2: Install dependencies**

Run: `uv sync`
Expected: All dependencies resolve including pytest-bdd.

**Step 3: Create BDD test infrastructure**

Create `tests/bdd/__init__.py` (empty).

Create `tests/bdd/conftest.py`:

```python
"""BDD test configuration."""
```

Create `features/` directory with `.gitkeep`.

**Step 4: Verify pytest discovers BDD markers**

Run: `uv run pytest --markers | grep bdd`
Expected: Shows `@pytest.mark.bdd` marker.

**Step 5: Commit**

```bash
git add pyproject.toml tests/bdd/ features/
git commit -m "chore: add pytest-bdd dependency and BDD test infrastructure"
```

---

## Task 2: Domain Models

**Files:**
- Create: `src/trialmatch/models/schema.py`
- Create: `tests/unit/test_schema.py`

**Step 1: Write the failing test**

Create `tests/unit/test_schema.py`:

```python
"""Tests for domain models."""

from trialmatch.models.schema import (
    CriterionResult,
    ModelResponse,
    Phase0Sample,
    Qrel,
    RunResult,
    Topic,
    Trial,
    Verdict,
)


def test_topic_creation():
    t = Topic(topic_id="1", text="A 45-year-old male...")
    assert t.topic_id == "1"
    assert "45-year-old" in t.text


def test_trial_creation():
    t = Trial(
        nct_id="NCT00001234",
        brief_title="Test Trial",
        inclusion_criteria="Age >= 18",
        exclusion_criteria="Pregnant women",
    )
    assert t.nct_id == "NCT00001234"


def test_qrel_creation():
    q = Qrel(topic_id="1", nct_id="NCT00001234", relevance=2)
    assert q.relevance == 2


def test_qrel_to_verdict():
    assert Qrel(topic_id="1", nct_id="X", relevance=2).expected_verdict == Verdict.ELIGIBLE
    assert Qrel(topic_id="1", nct_id="X", relevance=1).expected_verdict == Verdict.EXCLUDED
    assert Qrel(topic_id="1", nct_id="X", relevance=0).expected_verdict == Verdict.NOT_RELEVANT


def test_model_response():
    r = ModelResponse(
        text='{"verdict": "ELIGIBLE"}',
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
    cr = CriterionResult(verdict=Verdict.ELIGIBLE, reasoning="Patient meets all criteria", model_response=mr)
    assert cr.verdict == Verdict.ELIGIBLE


def test_phase0_sample():
    topic = Topic(topic_id="1", text="patient")
    trial = Trial(nct_id="NCT1", brief_title="t", inclusion_criteria="i", exclusion_criteria="e")
    qrel = Qrel(topic_id="1", nct_id="NCT1", relevance=2)
    sample = Phase0Sample(pairs=[(topic, trial, qrel)])
    assert len(sample.pairs) == 1


def test_run_result():
    rr = RunResult(run_id="test-001", model_name="medgemma", results=[], metrics={})
    assert rr.run_id == "test-001"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trialmatch.models.schema'`

**Step 3: Write minimal implementation**

Create `src/trialmatch/models/schema.py`:

```python
"""Domain models for trialmatch Phase 0 benchmark."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel


class Verdict(str, enum.Enum):
    """Trial matching verdict."""

    ELIGIBLE = "ELIGIBLE"
    EXCLUDED = "EXCLUDED"
    NOT_RELEVANT = "NOT_RELEVANT"


# Mapping from TREC qrel relevance to Verdict
RELEVANCE_TO_VERDICT = {
    2: Verdict.ELIGIBLE,
    1: Verdict.EXCLUDED,
    0: Verdict.NOT_RELEVANT,
}


class Topic(BaseModel):
    """TREC patient vignette."""

    topic_id: str
    text: str


class Trial(BaseModel):
    """Clinical trial with eligibility criteria."""

    nct_id: str
    brief_title: str
    inclusion_criteria: str
    exclusion_criteria: str


class Qrel(BaseModel):
    """TREC relevance judgment."""

    topic_id: str
    nct_id: str
    relevance: int  # 0, 1, 2

    @property
    def expected_verdict(self) -> Verdict:
        return RELEVANCE_TO_VERDICT[self.relevance]


class ModelResponse(BaseModel):
    """Raw model API response metadata."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost: float


class CriterionResult(BaseModel):
    """Single evaluation result for a patient-trial pair."""

    verdict: Verdict
    reasoning: str
    model_response: ModelResponse


class Phase0Sample(BaseModel):
    """Stratified sample of patient-trial pairs for Phase 0."""

    pairs: list[tuple[Topic, Trial, Qrel]]


class RunResult(BaseModel):
    """Complete results for one model's Phase 0 run."""

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
git commit -m "feat: add Pydantic domain models for Phase 0 benchmark"
```

---

## Task 3: TrialGPT Data Loader

**Files:**
- Create: `src/trialmatch/data/trialgpt_loader.py`
- Create: `tests/unit/test_trialgpt_loader.py`
- Create: `tests/fixtures/trialgpt/queries_sample.jsonl`
- Create: `tests/fixtures/trialgpt/corpus_sample.jsonl`
- Create: `tests/fixtures/trialgpt/qrels_sample.tsv`

**Step 1: Create test fixtures**

Create `tests/fixtures/trialgpt/queries_sample.jsonl` (2 sample queries):

```jsonl
{"_id": "trec-20211", "text": "Patient is a 45-year-old man with a history of anaplastic astrocytoma of the spine complicated by severe lower extremity weakness and urinary retention s/p Foley catheter, high-dose steroids, hypertension, and chronic pain.", "metadata": {}}
{"_id": "trec-20212", "text": "A 58-year-old woman with a history of stage IIIA non-small cell lung cancer, status post left upper lobectomy and adjuvant chemotherapy, now with recurrent disease.", "metadata": {}}
```

Create `tests/fixtures/trialgpt/corpus_sample.jsonl` (2 sample trials):

```jsonl
{"_id": "NCT00995306", "title": "Evaluating Civamide in OA of the Knee", "text": "Summary: To evaluate...", "metadata": {"brief_title": "Evaluating Civamide in OA", "inclusion_criteria": "Subject voluntarily agrees to participate\n\nSubject has OA pain in at least one knee for at least 6 months\n\nSubject is 40-75 years of age", "exclusion_criteria": "Presence of tendonitis\n\nPresence of active skin disease\n\nHistory of cardiac events in past year", "brief_summary": "To evaluate safety", "diseases_list": ["Osteoarthritis"], "drugs_list": ["Civamide"]}}
{"_id": "NCT00002569", "title": "Phase II Radiation for Brain Tumors", "text": "Summary: Radiation therapy...", "metadata": {"brief_title": "Radiation for Brain Tumors", "inclusion_criteria": "Histologically confirmed brain tumor\n\nAge 18 or older\n\nKPS >= 60", "exclusion_criteria": "Prior radiation therapy to brain\n\nPregnant or nursing", "brief_summary": "Radiation therapy study", "diseases_list": ["Brain Tumor"], "drugs_list": ["Radiation"]}}
```

Create `tests/fixtures/trialgpt/qrels_sample.tsv`:

```
query-id	corpus-id	score
trec-20211	NCT00995306	0
trec-20211	NCT00002569	2
trec-20212	NCT00995306	1
trec-20212	NCT00002569	0
```

**Step 2: Write the failing test**

Create `tests/unit/test_trialgpt_loader.py`:

```python
"""Tests for TrialGPT data loader."""

from pathlib import Path

from trialmatch.data.trialgpt_loader import (
    load_corpus,
    load_qrels,
    load_queries,
)
from trialmatch.models.schema import Qrel, Topic, Trial

FIXTURES = Path(__file__).parent.parent / "fixtures" / "trialgpt"


def test_load_queries():
    topics = load_queries(FIXTURES / "queries_sample.jsonl")
    assert len(topics) == 2
    assert isinstance(topics[0], Topic)
    assert topics[0].topic_id == "1"  # "trec-20211" -> "1"
    assert "45-year-old" in topics[0].text


def test_load_corpus():
    trials = load_corpus(FIXTURES / "corpus_sample.jsonl")
    assert len(trials) == 2
    assert isinstance(trials["NCT00995306"], Trial)
    assert "voluntarily agrees" in trials["NCT00995306"].inclusion_criteria
    assert "tendonitis" in trials["NCT00995306"].exclusion_criteria


def test_load_qrels():
    qrels = load_qrels(FIXTURES / "qrels_sample.tsv")
    assert len(qrels) == 4
    assert isinstance(qrels[0], Qrel)
    # Check topic_id parsing: "trec-20211" -> "1"
    assert qrels[0].topic_id == "1"
    assert qrels[0].nct_id == "NCT00995306"
    assert qrels[0].relevance == 0


def test_load_qrels_relevance_distribution():
    qrels = load_qrels(FIXTURES / "qrels_sample.tsv")
    counts = {}
    for q in qrels:
        counts[q.relevance] = counts.get(q.relevance, 0) + 1
    assert counts[0] == 2
    assert counts[1] == 1
    assert counts[2] == 1
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_trialgpt_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 4: Write minimal implementation**

Create `src/trialmatch/data/trialgpt_loader.py`:

```python
"""Load TrialGPT TREC 2021 data files.

Parses:
- queries.jsonl -> list[Topic]
- corpus.jsonl -> dict[nct_id, Trial]
- qrels/test.tsv -> list[Qrel]
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trialmatch.models.schema import Qrel, Topic, Trial


def _parse_topic_id(raw_id: str) -> str:
    """Extract numeric topic ID from TrialGPT format.

    'trec-20211' -> '1', 'trec-202175' -> '75'
    """
    match = re.search(r"trec-2021(\d+)$", raw_id)
    if match:
        return match.group(1)
    return raw_id


def load_queries(path: Path) -> list[Topic]:
    """Load patient vignettes from queries.jsonl."""
    topics = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            topics.append(
                Topic(
                    topic_id=_parse_topic_id(data["_id"]),
                    text=data["text"],
                )
            )
    return topics


def load_corpus(path: Path) -> dict[str, Trial]:
    """Load trial data from corpus.jsonl. Returns dict keyed by NCT ID."""
    trials = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            meta = data.get("metadata", {})
            trials[data["_id"]] = Trial(
                nct_id=data["_id"],
                brief_title=meta.get("brief_title", data.get("title", "")),
                inclusion_criteria=meta.get("inclusion_criteria", ""),
                exclusion_criteria=meta.get("exclusion_criteria", ""),
            )
    return trials


def load_qrels(path: Path) -> list[Qrel]:
    """Load relevance judgments from qrels TSV."""
    qrels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("query-id"):
                continue
            parts = line.split("\t")
            qrels.append(
                Qrel(
                    topic_id=_parse_topic_id(parts[0]),
                    nct_id=parts[1],
                    relevance=int(parts[2]),
                )
            )
    return qrels
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_trialgpt_loader.py -v`
Expected: All 4 tests PASS.

**Step 6: Commit**

```bash
git add src/trialmatch/data/trialgpt_loader.py tests/unit/test_trialgpt_loader.py tests/fixtures/trialgpt/
git commit -m "feat: add TrialGPT data loader for queries, corpus, qrels"
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

from trialmatch.data.sampler import stratified_sample
from trialmatch.data.trialgpt_loader import load_corpus, load_qrels, load_queries

FIXTURES = Path(__file__).parent.parent / "fixtures" / "trialgpt"


def test_stratified_sample_counts():
    topics = load_queries(FIXTURES / "queries_sample.jsonl")
    trials = load_corpus(FIXTURES / "corpus_sample.jsonl")
    qrels = load_qrels(FIXTURES / "qrels_sample.tsv")

    # With 4 qrels (2x0, 1x1, 1x2), requesting 3 pairs
    sample = stratified_sample(
        topics=topics,
        trials=trials,
        qrels=qrels,
        n_pairs=3,
        seed=42,
    )
    assert len(sample.pairs) == 3


def test_stratified_sample_deterministic():
    topics = load_queries(FIXTURES / "queries_sample.jsonl")
    trials = load_corpus(FIXTURES / "corpus_sample.jsonl")
    qrels = load_qrels(FIXTURES / "qrels_sample.tsv")

    s1 = stratified_sample(topics, trials, qrels, n_pairs=3, seed=42)
    s2 = stratified_sample(topics, trials, qrels, n_pairs=3, seed=42)
    ids1 = [(t.topic_id, tr.nct_id) for t, tr, _ in s1.pairs]
    ids2 = [(t.topic_id, tr.nct_id) for t, tr, _ in s2.pairs]
    assert ids1 == ids2


def test_stratified_sample_has_all_labels():
    topics = load_queries(FIXTURES / "queries_sample.jsonl")
    trials = load_corpus(FIXTURES / "corpus_sample.jsonl")
    qrels = load_qrels(FIXTURES / "qrels_sample.tsv")

    sample = stratified_sample(topics, trials, qrels, n_pairs=3, seed=42)
    labels = {q.relevance for _, _, q in sample.pairs}
    # With n=3 from {0,1,2}, should have at least 2 labels
    assert len(labels) >= 2


def test_stratified_sample_skips_missing_trials():
    """Pairs referencing NCT IDs not in corpus are skipped."""
    topics = load_queries(FIXTURES / "queries_sample.jsonl")
    trials = {}  # empty corpus
    qrels = load_qrels(FIXTURES / "qrels_sample.tsv")

    sample = stratified_sample(topics, trials, qrels, n_pairs=3, seed=42)
    assert len(sample.pairs) == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/trialmatch/data/sampler.py`:

```python
"""Stratified sampling of patient-trial pairs for Phase 0."""

from __future__ import annotations

import random

from trialmatch.models.schema import Phase0Sample, Qrel, Topic, Trial


def stratified_sample(
    topics: list[Topic],
    trials: dict[str, Trial],
    qrels: list[Qrel],
    n_pairs: int,
    seed: int = 42,
    distribution: dict[int, float] | None = None,
) -> Phase0Sample:
    """Sample n_pairs from qrels with stratified relevance distribution.

    Args:
        topics: All loaded topics.
        trials: All loaded trials keyed by NCT ID.
        qrels: All loaded qrels.
        n_pairs: Total pairs to sample.
        seed: Random seed for reproducibility.
        distribution: Target ratio per relevance level.
            Default: {2: 0.4, 1: 0.4, 0: 0.2} (8/8/4 for n=20).
    """
    if distribution is None:
        distribution = {2: 0.4, 1: 0.4, 0: 0.2}

    topic_map = {t.topic_id: t for t in topics}

    # Filter to valid pairs (topic and trial both exist)
    valid_qrels: dict[int, list[Qrel]] = {0: [], 1: [], 2: []}
    for q in qrels:
        if q.topic_id in topic_map and q.nct_id in trials:
            valid_qrels.setdefault(q.relevance, []).append(q)

    rng = random.Random(seed)
    selected: list[tuple[Topic, Trial, Qrel]] = []

    for rel, ratio in sorted(distribution.items(), key=lambda x: -x[0]):
        pool = valid_qrels.get(rel, [])
        n_target = round(n_pairs * ratio)
        n_take = min(n_target, len(pool))
        chosen = rng.sample(pool, n_take) if n_take > 0 else []
        for q in chosen:
            selected.append((topic_map[q.topic_id], trials[q.nct_id], q))

    return Phase0Sample(pairs=selected)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sampler.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/data/sampler.py tests/unit/test_sampler.py
git commit -m "feat: add stratified sampler for Phase 0 pair selection"
```

---

## Task 5: Model Base Protocol

**Files:**
- Create: `src/trialmatch/models/base.py`
- Create: `tests/unit/test_model_base.py`

**Step 1: Write the failing test**

Create `tests/unit/test_model_base.py`:

```python
"""Tests for model adapter base protocol."""

import asyncio

from trialmatch.models.base import ModelAdapter, ModelResponse


class FakeAdapter(ModelAdapter):
    """Test adapter that returns canned responses."""

    @property
    def name(self) -> str:
        return "fake"

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        return ModelResponse(
            text='{"verdict": "ELIGIBLE", "reasoning": "test"}',
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

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_model_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/trialmatch/models/base.py`:

```python
"""Base protocol for model adapters."""

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

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_model_base.py -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/models/base.py tests/unit/test_model_base.py
git commit -m "feat: add ModelAdapter abstract base class"
```

---

## Task 6: MedGemma Adapter

**Files:**
- Create: `src/trialmatch/models/medgemma.py`
- Create: `tests/unit/test_medgemma_adapter.py`

**Note:** Unit tests mock the HF API. Integration tests (later) will hit the real endpoint.

**Step 1: Write the failing test**

Create `tests/unit/test_medgemma_adapter.py`:

```python
"""Tests for MedGemma model adapter (mocked)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from trialmatch.models.medgemma import MedGemmaAdapter
from trialmatch.models.schema import ModelResponse


def test_medgemma_name():
    adapter = MedGemmaAdapter(
        endpoint_url="https://fake.endpoint.cloud",
        hf_token="hf_fake",
    )
    assert adapter.name == "medgemma-1.5-4b"


@patch("trialmatch.models.medgemma.InferenceClient")
def test_medgemma_generate(mock_client_cls):
    mock_instance = MagicMock()
    mock_instance.text_generation.return_value = '{"verdict": "ELIGIBLE", "reasoning": "test"}'
    mock_client_cls.return_value = mock_instance

    adapter = MedGemmaAdapter(
        endpoint_url="https://fake.endpoint.cloud",
        hf_token="hf_fake",
    )

    result = asyncio.run(adapter.generate("Test prompt", max_tokens=512))

    assert isinstance(result, ModelResponse)
    assert "ELIGIBLE" in result.text
    assert result.input_tokens > 0
    assert result.latency_ms >= 0


@patch("trialmatch.models.medgemma.InferenceClient")
def test_medgemma_formats_gemma_template(mock_client_cls):
    mock_instance = MagicMock()
    mock_instance.text_generation.return_value = "OK"
    mock_client_cls.return_value = mock_instance

    adapter = MedGemmaAdapter(
        endpoint_url="https://fake.endpoint.cloud",
        hf_token="hf_fake",
    )
    asyncio.run(adapter.generate("Hello world"))

    call_args = mock_instance.text_generation.call_args
    prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt") or call_args[0][0]
    assert "<start_of_turn>user" in prompt
    assert "<start_of_turn>model" in prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_medgemma_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/trialmatch/models/medgemma.py`:

```python
"""MedGemma 1.5 4B model adapter via HuggingFace Inference Endpoint.

Endpoint uses default HF inference image (NOT TGI).
Must use text_generation() with manual Gemma chat template.
"""

from __future__ import annotations

import asyncio
import time

from huggingface_hub import InferenceClient

from trialmatch.models.base import ModelAdapter
from trialmatch.models.schema import ModelResponse

# Default endpoint URL (shared HF Inference Endpoint)
DEFAULT_ENDPOINT = "https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud"


class MedGemmaAdapter(ModelAdapter):
    """Adapter for MedGemma 1.5 4B on HF Inference Endpoint."""

    def __init__(
        self,
        endpoint_url: str = DEFAULT_ENDPOINT,
        hf_token: str = "",
        max_retries: int = 6,
        retry_backoff: float = 2.0,
        cold_start_timeout: float = 60.0,
    ):
        self.endpoint_url = endpoint_url
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.cold_start_timeout = cold_start_timeout
        self._client = InferenceClient(model=endpoint_url, token=hf_token)

    @property
    def name(self) -> str:
        return "medgemma-1.5-4b"

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Send prompt through Gemma template and return response."""
        gemma_prompt = self._format_gemma_prompt(prompt)
        input_tokens = len(gemma_prompt.split())  # approximate

        start = time.monotonic()
        text = await self._call_with_retry(gemma_prompt, max_tokens)
        latency_ms = (time.monotonic() - start) * 1000

        output_tokens = len(text.split())  # approximate
        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost=0.0,  # HF hosted inference
        )

    async def health_check(self) -> bool:
        """Ping endpoint with minimal prompt."""
        try:
            result = await self._call_with_retry(
                "<start_of_turn>user\nRespond OK<end_of_turn>\n<start_of_turn>model\n",
                max_tokens=10,
            )
            return bool(result)
        except Exception:
            return False

    @staticmethod
    def _format_gemma_prompt(user_prompt: str) -> str:
        """Wrap prompt in Gemma chat template.

        System prompt is folded into the first user turn.
        """
        return (
            f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    async def _call_with_retry(self, prompt: str, max_tokens: int) -> str:
        """Call HF endpoint with exponential backoff for 503 cold starts."""
        start = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            if time.monotonic() - start > self.cold_start_timeout:
                break
            try:
                text = await asyncio.to_thread(
                    self._client.text_generation,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                )
                if not text:
                    raise ValueError("MedGemma returned empty response")
                return text
            except Exception as e:
                last_error = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status and 400 <= status < 500:
                    raise
                if attempt < self.max_retries - 1:
                    wait = min(self.retry_backoff**attempt, 60.0)
                    await asyncio.sleep(wait)
                    continue
                raise

        raise last_error or RuntimeError("MedGemma retry budget exhausted")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_medgemma_adapter.py -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/models/medgemma.py tests/unit/test_medgemma_adapter.py
git commit -m "feat: add MedGemma adapter with Gemma template and retry logic"
```

---

## Task 7: Gemini Adapter

**Files:**
- Create: `src/trialmatch/models/gemini.py`
- Create: `tests/unit/test_gemini_adapter.py`

**Step 1: Write the failing test**

Create `tests/unit/test_gemini_adapter.py`:

```python
"""Tests for Gemini model adapter (mocked)."""

import asyncio
from unittest.mock import MagicMock, patch

from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.schema import ModelResponse


def test_gemini_name():
    with patch("trialmatch.models.gemini.genai"):
        adapter = GeminiAdapter(api_key="fake-key")
    assert adapter.name == "gemini-3-pro"


@patch("trialmatch.models.gemini.genai")
def test_gemini_generate(mock_genai):
    # Mock the response object
    mock_response = MagicMock()
    mock_response.text = '{"verdict": "EXCLUDED", "reasoning": "Patient is too young"}'
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 150
    mock_usage.candidates_token_count = 30
    mock_response.usage_metadata = mock_usage
    mock_genai.Client.return_value.models.generate_content.return_value = mock_response

    adapter = GeminiAdapter(api_key="fake-key")
    result = asyncio.run(adapter.generate("Test prompt"))

    assert isinstance(result, ModelResponse)
    assert "EXCLUDED" in result.text
    assert result.input_tokens == 150
    assert result.output_tokens == 30


@patch("trialmatch.models.gemini.genai")
def test_gemini_health_check_success(mock_genai):
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_response.usage_metadata = None
    mock_genai.Client.return_value.models.generate_content.return_value = mock_response

    adapter = GeminiAdapter(api_key="fake-key")
    assert asyncio.run(adapter.health_check()) is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_gemini_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/trialmatch/models/gemini.py`:

```python
"""Gemini 3 Pro model adapter via Google AI Studio.

Uses google-genai SDK with structured JSON output.
"""

from __future__ import annotations

import asyncio
import time

from google import genai

from trialmatch.models.base import ModelAdapter
from trialmatch.models.schema import ModelResponse

DEFAULT_MODEL = "gemini-3-pro-preview"

# Approximate pricing per 1K tokens (Gemini 3 Pro preview)
INPUT_COST_PER_1K = 0.00125
OUTPUT_COST_PER_1K = 0.005


class GeminiAdapter(ModelAdapter):
    """Adapter for Gemini 3 Pro via Google AI Studio."""

    def __init__(
        self,
        api_key: str = "",
        model_id: str = DEFAULT_MODEL,
        max_concurrent: int = 10,
    ):
        self.model_id = model_id
        self._client = genai.Client(api_key=api_key)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def name(self) -> str:
        return "gemini-3-pro"

    async def generate(self, prompt: str, max_tokens: int = 2048) -> ModelResponse:
        """Send prompt to Gemini and return structured response."""
        start = time.monotonic()

        async with self._semaphore:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.model_id,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "max_output_tokens": max_tokens,
                },
            )

        latency_ms = (time.monotonic() - start) * 1000
        text = response.text or ""

        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        estimated_cost = (
            (input_tokens / 1000) * INPUT_COST_PER_1K
            + (output_tokens / 1000) * OUTPUT_COST_PER_1K
        )

        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
        )

    async def health_check(self) -> bool:
        """Quick check that Gemini endpoint responds."""
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.model_id,
                contents='Return JSON: {"status": "ok"}',
                config={
                    "response_mime_type": "application/json",
                    "max_output_tokens": 50,
                },
            )
            return bool(response.text)
        except Exception:
            return False
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_gemini_adapter.py -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/models/gemini.py tests/unit/test_gemini_adapter.py
git commit -m "feat: add Gemini 3 Pro adapter with JSON output and cost tracking"
```

---

## Task 8: Validate Evaluator

**Files:**
- Create: `src/trialmatch/validate/evaluator.py`
- Create: `tests/unit/test_evaluator.py`

**Step 1: Write the failing test**

Create `tests/unit/test_evaluator.py`:

```python
"""Tests for validate evaluator."""

import asyncio
from unittest.mock import AsyncMock

from trialmatch.models.schema import ModelResponse, Topic, Trial, Verdict
from trialmatch.validate.evaluator import build_prompt, parse_verdict, evaluate_pair


def test_build_prompt_contains_patient_and_criteria():
    prompt = build_prompt(
        patient_text="45-year-old male with lung cancer",
        inclusion_criteria="Age >= 18\nDiagnosed with NSCLC",
        exclusion_criteria="Prior chemotherapy",
    )
    assert "45-year-old male" in prompt
    assert "Age >= 18" in prompt
    assert "Prior chemotherapy" in prompt
    assert "ELIGIBLE" in prompt
    assert "EXCLUDED" in prompt
    assert "NOT_RELEVANT" in prompt


def test_parse_verdict_eligible():
    assert parse_verdict('{"verdict": "ELIGIBLE", "reasoning": "meets all"}') == (
        Verdict.ELIGIBLE,
        "meets all",
    )


def test_parse_verdict_excluded():
    assert parse_verdict('{"verdict": "EXCLUDED", "reasoning": "too young"}') == (
        Verdict.EXCLUDED,
        "too young",
    )


def test_parse_verdict_not_relevant():
    result = parse_verdict('{"verdict": "NOT_RELEVANT", "reasoning": "wrong disease"}')
    assert result[0] == Verdict.NOT_RELEVANT


def test_parse_verdict_markdown_wrapped():
    raw = '```json\n{"verdict": "ELIGIBLE", "reasoning": "ok"}\n```'
    assert parse_verdict(raw)[0] == Verdict.ELIGIBLE


def test_parse_verdict_fallback_on_bad_json():
    """If JSON fails, try to extract verdict from raw text."""
    result = parse_verdict("The patient is ELIGIBLE because they meet all criteria.")
    assert result[0] == Verdict.ELIGIBLE


def test_parse_verdict_unknown_returns_not_relevant():
    result = parse_verdict("I cannot determine eligibility.")
    assert result[0] == Verdict.NOT_RELEVANT


def test_evaluate_pair():
    mock_adapter = AsyncMock()
    mock_adapter.generate.return_value = ModelResponse(
        text='{"verdict": "EXCLUDED", "reasoning": "age exclusion"}',
        input_tokens=200,
        output_tokens=30,
        latency_ms=500.0,
        estimated_cost=0.01,
    )
    topic = Topic(topic_id="1", text="30-year-old female")
    trial = Trial(
        nct_id="NCT001",
        brief_title="Test",
        inclusion_criteria="Age >= 40",
        exclusion_criteria="None",
    )

    result = asyncio.run(evaluate_pair(topic, trial, mock_adapter))
    assert result.verdict == Verdict.EXCLUDED
    assert "age" in result.reasoning.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/trialmatch/validate/evaluator.py`:

```python
"""Validate evaluator: assess patient-trial eligibility.

Sends (patient_text, eligibility_criteria) to a model adapter
and parses the verdict as ELIGIBLE/EXCLUDED/NOT_RELEVANT.
"""

from __future__ import annotations

import json
import re

from trialmatch.models.base import ModelAdapter
from trialmatch.models.schema import CriterionResult, ModelResponse, Topic, Trial, Verdict

PROMPT_TEMPLATE = """You are a clinical trial matching expert.

Given a patient description and a clinical trial's eligibility criteria,
determine if the patient is ELIGIBLE, EXCLUDED, or NOT_RELEVANT.

Definitions:
- ELIGIBLE: Patient meets all inclusion criteria and no exclusion criteria apply
- EXCLUDED: Patient meets inclusion criteria but is excluded by one or more exclusion criteria
- NOT_RELEVANT: Patient does not meet the basic inclusion criteria

Patient:
{patient_text}

Inclusion Criteria:
{inclusion_criteria}

Exclusion Criteria:
{exclusion_criteria}

Respond in JSON format:
{{"verdict": "ELIGIBLE" | "EXCLUDED" | "NOT_RELEVANT", "reasoning": "Step-by-step explanation of your assessment"}}"""


def build_prompt(patient_text: str, inclusion_criteria: str, exclusion_criteria: str) -> str:
    """Build the evaluation prompt from patient and trial data."""
    return PROMPT_TEMPLATE.format(
        patient_text=patient_text,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
    )


def parse_verdict(raw_text: str) -> tuple[Verdict, str]:
    """Parse model output into (Verdict, reasoning).

    Tries JSON first, then markdown-wrapped JSON, then keyword extraction.
    """
    # Try direct JSON parse
    try:
        data = json.loads(raw_text)
        return Verdict(data["verdict"]), data.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Try markdown-wrapped JSON
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return Verdict(data["verdict"]), data.get("reasoning", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fallback: keyword extraction from raw text
    upper = raw_text.upper()
    if "ELIGIBLE" in upper and "NOT" not in upper.split("ELIGIBLE")[0][-10:]:
        return Verdict.ELIGIBLE, raw_text
    if "EXCLUDED" in upper:
        return Verdict.EXCLUDED, raw_text

    # Default to NOT_RELEVANT if we can't parse
    return Verdict.NOT_RELEVANT, raw_text


async def evaluate_pair(
    topic: Topic,
    trial: Trial,
    adapter: ModelAdapter,
    max_tokens: int = 2048,
) -> CriterionResult:
    """Evaluate a single patient-trial pair using the given model."""
    prompt = build_prompt(
        patient_text=topic.text,
        inclusion_criteria=trial.inclusion_criteria,
        exclusion_criteria=trial.exclusion_criteria,
    )
    response: ModelResponse = await adapter.generate(prompt, max_tokens=max_tokens)
    verdict, reasoning = parse_verdict(response.text)

    return CriterionResult(
        verdict=verdict,
        reasoning=reasoning,
        model_response=response,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_evaluator.py -v`
Expected: All 8 tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/validate/evaluator.py tests/unit/test_evaluator.py
git commit -m "feat: add validate evaluator with prompt builder and verdict parser"
```

---

## Task 9: Evaluation Metrics

**Files:**
- Create: `src/trialmatch/evaluation/metrics.py`
- Create: `tests/unit/test_metrics.py`

**Step 1: Write the failing test**

Create `tests/unit/test_metrics.py`:

```python
"""Tests for evaluation metrics."""

from trialmatch.evaluation.metrics import compute_metrics
from trialmatch.models.schema import Verdict


def test_compute_metrics_perfect():
    predicted = [Verdict.ELIGIBLE, Verdict.EXCLUDED, Verdict.NOT_RELEVANT]
    actual = [Verdict.ELIGIBLE, Verdict.EXCLUDED, Verdict.NOT_RELEVANT]
    m = compute_metrics(predicted, actual)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["cohens_kappa"] == 1.0


def test_compute_metrics_all_wrong():
    predicted = [Verdict.NOT_RELEVANT, Verdict.NOT_RELEVANT, Verdict.NOT_RELEVANT]
    actual = [Verdict.ELIGIBLE, Verdict.EXCLUDED, Verdict.NOT_RELEVANT]
    m = compute_metrics(predicted, actual)
    assert m["accuracy"] < 0.5


def test_compute_metrics_confusion_matrix():
    predicted = [Verdict.ELIGIBLE, Verdict.EXCLUDED]
    actual = [Verdict.ELIGIBLE, Verdict.ELIGIBLE]
    m = compute_metrics(predicted, actual)
    assert "confusion_matrix" in m
    assert isinstance(m["confusion_matrix"], list)


def test_compute_metrics_per_class_f1():
    predicted = [Verdict.ELIGIBLE, Verdict.ELIGIBLE, Verdict.EXCLUDED]
    actual = [Verdict.ELIGIBLE, Verdict.EXCLUDED, Verdict.EXCLUDED]
    m = compute_metrics(predicted, actual)
    assert "f1_per_class" in m
    assert "ELIGIBLE" in m["f1_per_class"]
    assert "EXCLUDED" in m["f1_per_class"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/trialmatch/evaluation/metrics.py`:

```python
"""Evaluation metrics for Phase 0 benchmark."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from trialmatch.models.schema import Verdict

LABELS = [Verdict.ELIGIBLE, Verdict.EXCLUDED, Verdict.NOT_RELEVANT]
LABEL_NAMES = [v.value for v in LABELS]


def compute_metrics(
    predicted: list[Verdict],
    actual: list[Verdict],
) -> dict[str, Any]:
    """Compute all Phase 0 evaluation metrics.

    Returns dict with: accuracy, f1_macro, cohens_kappa,
    confusion_matrix, f1_per_class.
    """
    pred_str = [v.value for v in predicted]
    actual_str = [v.value for v in actual]

    acc = accuracy_score(actual_str, pred_str)
    f1_mac = f1_score(actual_str, pred_str, labels=LABEL_NAMES, average="macro", zero_division=0)
    kappa = cohen_kappa_score(actual_str, pred_str, labels=LABEL_NAMES)
    cm = confusion_matrix(actual_str, pred_str, labels=LABEL_NAMES)
    f1_per = f1_score(actual_str, pred_str, labels=LABEL_NAMES, average=None, zero_division=0)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "cohens_kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
        "f1_per_class": {name: float(f1) for name, f1 in zip(LABEL_NAMES, f1_per)},
    }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/evaluation/metrics.py tests/unit/test_metrics.py
git commit -m "feat: add evaluation metrics (accuracy, F1, kappa, confusion matrix)"
```

---

## Task 10: Run Manager (Tracing)

**Files:**
- Create: `src/trialmatch/tracing/run_manager.py`
- Create: `tests/unit/test_run_manager.py`

**Step 1: Write the failing test**

Create `tests/unit/test_run_manager.py`:

```python
"""Tests for run manager."""

import json
from pathlib import Path

from trialmatch.models.schema import (
    CriterionResult,
    ModelResponse,
    RunResult,
    Verdict,
)
from trialmatch.tracing.run_manager import RunManager


def test_generate_run_id():
    rm = RunManager(base_dir=Path("/tmp/trialmatch_test_runs"))
    run_id = rm.generate_run_id("medgemma")
    assert "medgemma" in run_id
    assert len(run_id) > 10


def test_save_and_load_run(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    run_id = rm.generate_run_id("test-model")

    mr = ModelResponse(text="test", input_tokens=10, output_tokens=5, latency_ms=100, estimated_cost=0)
    cr = CriterionResult(verdict=Verdict.ELIGIBLE, reasoning="ok", model_response=mr)
    result = RunResult(
        run_id=run_id,
        model_name="test-model",
        results=[cr],
        metrics={"accuracy": 0.8},
    )

    rm.save_run(result, config={"phase": 0})

    run_dir = tmp_path / run_id
    assert run_dir.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "metrics.json").exists()

    with open(run_dir / "metrics.json") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["accuracy"] == 0.8


def test_save_run_includes_cost_summary(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    run_id = rm.generate_run_id("test")

    mr1 = ModelResponse(text="a", input_tokens=100, output_tokens=50, latency_ms=200, estimated_cost=0.01)
    mr2 = ModelResponse(text="b", input_tokens=200, output_tokens=60, latency_ms=300, estimated_cost=0.02)
    results = [
        CriterionResult(verdict=Verdict.ELIGIBLE, reasoning="r1", model_response=mr1),
        CriterionResult(verdict=Verdict.EXCLUDED, reasoning="r2", model_response=mr2),
    ]
    rr = RunResult(run_id=run_id, model_name="test", results=results, metrics={})
    rm.save_run(rr)

    with open(tmp_path / run_id / "cost_summary.json") as f:
        cost = json.load(f)
    assert cost["total_input_tokens"] == 300
    assert cost["total_output_tokens"] == 110
    assert abs(cost["total_estimated_cost"] - 0.03) < 0.001
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_run_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/trialmatch/tracing/run_manager.py`:

```python
"""Run manager for saving Phase 0 benchmark results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trialmatch.models.schema import RunResult


class RunManager:
    """Manages run artifacts in runs/<run_id>/ directory."""

    def __init__(self, base_dir: Path | str = "runs"):
        self.base_dir = Path(base_dir)

    def generate_run_id(self, model_name: str) -> str:
        """Generate a unique run ID: phase0-{model}-{timestamp}."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"phase0-{model_name}-{ts}"

    def save_run(
        self,
        result: RunResult,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Save all run artifacts to runs/<run_id>/."""
        run_dir = self.base_dir / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        if config:
            (run_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=str)
            )

        # Save results (verdict + reasoning per pair)
        results_data = [
            {
                "verdict": cr.verdict.value,
                "reasoning": cr.reasoning,
                "input_tokens": cr.model_response.input_tokens,
                "output_tokens": cr.model_response.output_tokens,
                "latency_ms": cr.model_response.latency_ms,
                "estimated_cost": cr.model_response.estimated_cost,
                "raw_text": cr.model_response.text,
            }
            for cr in result.results
        ]
        (run_dir / "results.json").write_text(json.dumps(results_data, indent=2))

        # Save metrics
        (run_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))

        # Save cost summary
        cost_summary = {
            "total_input_tokens": sum(cr.model_response.input_tokens for cr in result.results),
            "total_output_tokens": sum(cr.model_response.output_tokens for cr in result.results),
            "total_estimated_cost": sum(cr.model_response.estimated_cost for cr in result.results),
            "total_latency_ms": sum(cr.model_response.latency_ms for cr in result.results),
            "n_pairs": len(result.results),
            "model": result.model_name,
        }
        (run_dir / "cost_summary.json").write_text(json.dumps(cost_summary, indent=2))

        return run_dir
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_run_manager.py -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/tracing/run_manager.py tests/unit/test_run_manager.py
git commit -m "feat: add run manager for saving benchmark results and cost tracking"
```

---

## Task 11: TrialGPT Data Downloader

**Files:**
- Create: `src/trialmatch/data/downloader.py`
- Create: `tests/unit/test_downloader.py`

**Step 1: Write the failing test**

Create `tests/unit/test_downloader.py`:

```python
"""Tests for TrialGPT data downloader."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from trialmatch.data.downloader import (
    TRIALGPT_FILES,
    get_data_dir,
    files_exist,
)


def test_trialgpt_files_defined():
    assert "queries" in TRIALGPT_FILES
    assert "qrels" in TRIALGPT_FILES
    assert "corpus" in TRIALGPT_FILES


def test_get_data_dir():
    d = get_data_dir()
    assert "trec2021" in str(d)


def test_files_exist_false(tmp_path):
    assert files_exist(tmp_path) is False


def test_files_exist_true(tmp_path):
    (tmp_path / "queries.jsonl").write_text("{}")
    (tmp_path / "qrels.tsv").write_text("a\tb\t1")
    (tmp_path / "corpus.jsonl").write_text("{}")
    assert files_exist(tmp_path) is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_downloader.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/trialmatch/data/downloader.py`:

```python
"""Download TrialGPT TREC 2021 data files.

Sources:
- queries.jsonl: GitHub raw (ncbi-nlp/TrialGPT)
- qrels/test.tsv: GitHub raw
- corpus.jsonl: FTP (ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger()

_BASE_GITHUB = "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/main/dataset/trec_2021"
_FTP_CORPUS = "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl"

TRIALGPT_FILES = {
    "queries": {
        "url": f"{_BASE_GITHUB}/queries.jsonl",
        "filename": "queries.jsonl",
    },
    "qrels": {
        "url": f"{_BASE_GITHUB}/qrels/test.tsv",
        "filename": "qrels.tsv",
    },
    "corpus": {
        "url": _FTP_CORPUS,
        "filename": "corpus.jsonl",
    },
}


def get_data_dir() -> Path:
    """Return default data directory: data/trec2021/."""
    return Path("data/trec2021")


def files_exist(data_dir: Path) -> bool:
    """Check if all required TrialGPT files are downloaded."""
    return all(
        (data_dir / info["filename"]).exists()
        for info in TRIALGPT_FILES.values()
    )


def download_file(url: str, dest: Path, timeout: float = 300.0) -> None:
    """Download a single file with progress logging."""
    logger.info("downloading", url=url, dest=str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (1024 * 1024) < 8192:
                    pct = (downloaded / total) * 100
                    logger.info("progress", file=dest.name, pct=f"{pct:.0f}%")

    logger.info("downloaded", file=dest.name, size_mb=f"{dest.stat().st_size / 1024 / 1024:.1f}")


def download_all(data_dir: Path | None = None, skip_existing: bool = True) -> Path:
    """Download all TrialGPT TREC 2021 files.

    Args:
        data_dir: Target directory. Defaults to data/trec2021/.
        skip_existing: Skip files that already exist.

    Returns:
        Path to data directory.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, info in TRIALGPT_FILES.items():
        dest = data_dir / info["filename"]
        if skip_existing and dest.exists():
            logger.info("skipping_existing", file=dest.name)
            continue
        download_file(info["url"], dest)

    return data_dir
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_downloader.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/data/downloader.py tests/unit/test_downloader.py
git commit -m "feat: add TrialGPT data downloader (queries, qrels, corpus)"
```

---

## Task 12: CLI Phase0 Command

**Files:**
- Modify: `src/trialmatch/cli/__init__.py`
- Create: `src/trialmatch/cli/phase0.py`
- Create: `tests/unit/test_cli_phase0.py`

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


def test_phase0_requires_config_or_defaults():
    runner = CliRunner()
    result = runner.invoke(main, ["phase0", "--dry-run"])
    # Should not crash — dry-run prints config and exits
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_cli_phase0.py -v`
Expected: FAIL with `No such command 'phase0'`

**Step 3: Write minimal implementation**

Create `src/trialmatch/cli/phase0.py`:

```python
"""CLI command for Phase 0 benchmark."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import click
import structlog
import yaml

from trialmatch.data.downloader import download_all, files_exist, get_data_dir
from trialmatch.data.sampler import stratified_sample
from trialmatch.data.trialgpt_loader import load_corpus, load_qrels, load_queries
from trialmatch.evaluation.metrics import compute_metrics
from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.medgemma import MedGemmaAdapter
from trialmatch.models.schema import RunResult, Verdict
from trialmatch.tracing.run_manager import RunManager
from trialmatch.validate.evaluator import evaluate_pair

logger = structlog.get_logger()


async def run_model(adapter, sample, budget_max: float = 5.0):
    """Run all pairs through one model adapter."""
    results = []
    total_cost = 0.0

    for i, (topic, trial, qrel) in enumerate(sample.pairs):
        logger.info(
            "evaluating",
            pair=f"{i + 1}/{len(sample.pairs)}",
            topic=topic.topic_id,
            trial=trial.nct_id,
            model=adapter.name,
        )
        result = await evaluate_pair(topic, trial, adapter)
        total_cost += result.model_response.estimated_cost

        if total_cost > budget_max:
            logger.warning("budget_exceeded", total_cost=total_cost, max=budget_max)
            break

        results.append(result)

    return results


async def run_phase0(config: dict, dry_run: bool = False):
    """Execute Phase 0 benchmark."""
    data_dir = get_data_dir()

    # Download data if needed
    if not files_exist(data_dir):
        logger.info("downloading_trialgpt_data")
        download_all(data_dir)

    # Load data
    topics = load_queries(data_dir / "queries.jsonl")
    trials = load_corpus(data_dir / "corpus.jsonl")
    qrels = load_qrels(data_dir / "qrels.tsv")

    logger.info("data_loaded", topics=len(topics), trials=len(trials), qrels=len(qrels))

    # Sample
    n_pairs = config.get("data", {}).get("n_pairs", 20)
    seed = config.get("data", {}).get("seed", 42)
    sample = stratified_sample(topics, trials, qrels, n_pairs=n_pairs, seed=seed)
    logger.info("sampled", n_pairs=len(sample.pairs))

    if dry_run:
        click.echo(f"Dry run: would evaluate {len(sample.pairs)} pairs with 2 models")
        for i, (t, tr, q) in enumerate(sample.pairs):
            click.echo(f"  {i + 1}. Topic {t.topic_id} x {tr.nct_id} (qrel={q.relevance})")
        return

    budget_max = config.get("budget", {}).get("max_cost_usd", 5.0)
    run_mgr = RunManager()

    # Run each model
    for model_cfg in config.get("models", []):
        if model_cfg["provider"] == "huggingface":
            adapter = MedGemmaAdapter(hf_token=os.environ.get("HF_TOKEN", ""))
        elif model_cfg["provider"] == "google":
            adapter = GeminiAdapter(api_key=os.environ.get("GEMINI_API_KEY", ""))
        else:
            logger.error("unknown_provider", provider=model_cfg["provider"])
            continue

        logger.info("running_model", model=adapter.name)
        results = await run_model(adapter, sample, budget_max=budget_max)

        # Compute metrics
        predicted = [r.verdict for r in results]
        actual = [q.expected_verdict for _, _, q in sample.pairs[: len(results)]]
        metrics = compute_metrics(predicted, actual)

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
        click.echo(f"Cohen's κ: {metrics['cohens_kappa']:.3f}")
        click.echo(f"Per-class F1: {json.dumps(metrics['f1_per_class'], indent=2)}")
        click.echo(f"{'=' * 60}")


@click.command("phase0")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.option("--dry-run", is_flag=True, help="Show what would be run without calling models")
def phase0_cmd(config_path: str | None, dry_run: bool):
    """Run Phase 0 benchmark: 20-pair MedGemma vs Gemini comparison."""
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
        click.echo(f"Config: {json.dumps(config, indent=2)}")

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

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_cli_phase0.py -v`
Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add src/trialmatch/cli/ tests/unit/test_cli_phase0.py
git commit -m "feat: add CLI phase0 command with dry-run support"
```

---

## Task 13: BDD Feature File for Validate

**Files:**
- Create: `features/validate/trial_matching.feature`
- Create: `tests/bdd/steps/validate_steps.py`

**Step 1: Write the BDD feature file**

Create `features/validate/trial_matching.feature`:

```gherkin
@component_validate @phase0
Feature: Clinical trial matching evaluation
  As a benchmark runner
  I want to evaluate patient-trial matching with LLM models
  So that I can compare MedGemma vs Gemini medical reasoning

  Background:
    Given a patient vignette from TREC 2021 topic "1"
    And a clinical trial "NCT00002569" with eligibility criteria

  @implemented
  Scenario: Prompt contains patient and criteria
    When I build the evaluation prompt
    Then the prompt contains the patient text
    And the prompt contains the inclusion criteria
    And the prompt contains the exclusion criteria
    And the prompt asks for ELIGIBLE, EXCLUDED, or NOT_RELEVANT

  @implemented
  Scenario: Parse ELIGIBLE verdict from JSON
    Given the model returns '{"verdict": "ELIGIBLE", "reasoning": "meets all"}'
    When I parse the verdict
    Then the verdict is ELIGIBLE
    And the reasoning is "meets all"

  @implemented
  Scenario: Parse EXCLUDED verdict from JSON
    Given the model returns '{"verdict": "EXCLUDED", "reasoning": "too young"}'
    When I parse the verdict
    Then the verdict is EXCLUDED

  @implemented
  Scenario: Parse verdict from markdown-wrapped JSON
    Given the model returns '```json\n{"verdict": "ELIGIBLE", "reasoning": "ok"}\n```'
    When I parse the verdict
    Then the verdict is ELIGIBLE

  @implemented
  Scenario: Fallback parsing from raw text
    Given the model returns "The patient is EXCLUDED based on age criteria"
    When I parse the verdict
    Then the verdict is EXCLUDED
```

**Step 2: Write step definitions**

Create `tests/bdd/steps/validate_steps.py`:

```python
"""Step definitions for validate BDD scenarios."""

import pytest
from pytest_bdd import given, when, then, scenarios, parsers

from trialmatch.models.schema import Topic, Trial, Verdict
from trialmatch.validate.evaluator import build_prompt, parse_verdict

scenarios("../../features/validate/trial_matching.feature")


@pytest.fixture
def context():
    return {}


@given('a patient vignette from TREC 2021 topic "1"')
def given_patient(context):
    context["topic"] = Topic(
        topic_id="1",
        text="Patient is a 45-year-old man with anaplastic astrocytoma of the spine.",
    )


@given('a clinical trial "NCT00002569" with eligibility criteria')
def given_trial(context):
    context["trial"] = Trial(
        nct_id="NCT00002569",
        brief_title="Phase II Radiation for Brain Tumors",
        inclusion_criteria="Histologically confirmed brain tumor\nAge 18 or older",
        exclusion_criteria="Prior radiation therapy\nPregnant or nursing",
    )


@when("I build the evaluation prompt")
def build_the_prompt(context):
    context["prompt"] = build_prompt(
        patient_text=context["topic"].text,
        inclusion_criteria=context["trial"].inclusion_criteria,
        exclusion_criteria=context["trial"].exclusion_criteria,
    )


@then("the prompt contains the patient text")
def prompt_has_patient(context):
    assert context["topic"].text in context["prompt"]


@then("the prompt contains the inclusion criteria")
def prompt_has_inclusion(context):
    assert context["trial"].inclusion_criteria in context["prompt"]


@then("the prompt contains the exclusion criteria")
def prompt_has_exclusion(context):
    assert context["trial"].exclusion_criteria in context["prompt"]


@then("the prompt asks for ELIGIBLE, EXCLUDED, or NOT_RELEVANT")
def prompt_has_verdicts(context):
    assert "ELIGIBLE" in context["prompt"]
    assert "EXCLUDED" in context["prompt"]
    assert "NOT_RELEVANT" in context["prompt"]


@given(parsers.parse("the model returns '{raw_text}'"))
def given_model_output(context, raw_text):
    # Unescape \n in feature file strings
    context["raw_text"] = raw_text.replace("\\n", "\n")


@when("I parse the verdict")
def parse_the_verdict(context):
    context["verdict"], context["reasoning"] = parse_verdict(context["raw_text"])


@then(parsers.parse("the verdict is {expected_verdict}"))
def check_verdict(context, expected_verdict):
    assert context["verdict"] == Verdict(expected_verdict)


@then(parsers.parse('the reasoning is "{expected_reasoning}"'))
def check_reasoning(context, expected_reasoning):
    assert context["reasoning"] == expected_reasoning
```

**Step 3: Run BDD tests**

Run: `uv run pytest tests/bdd/ -v`
Expected: All 5 scenarios PASS.

**Step 4: Commit**

```bash
git add features/validate/ tests/bdd/
git commit -m "feat: add BDD feature and step definitions for validate component"
```

---

## Task 14: Full Unit Test Suite Run + Lint

**Step 1: Run all unit tests**

Run: `uv run pytest tests/unit/ -v --tb=short`
Expected: All tests PASS (should be ~30+ tests total).

**Step 2: Run BDD tests**

Run: `uv run pytest tests/bdd/ -v`
Expected: All BDD scenarios PASS.

**Step 3: Run linter**

Run: `uv run ruff check src/ tests/`
Expected: No errors. Fix any issues found.

**Step 4: Run formatter**

Run: `uv run ruff format src/ tests/`
Expected: Files formatted.

**Step 5: Commit any formatting fixes**

```bash
git add -A
git commit -m "chore: lint and format all source and test files"
```

---

## Task 15: Integration Smoke Test (Dry Run)

**Step 1: Test the CLI dry-run end to end**

This requires TrialGPT data to be downloaded. First test with fixtures:

Create `tests/integration/test_phase0_dryrun.py`:

```python
"""Integration test: Phase 0 dry run with fixture data."""

import pytest
from pathlib import Path
from click.testing import CliRunner

from trialmatch.cli import main


@pytest.mark.integration
def test_phase0_dry_run():
    """Dry run should print config and sampled pairs without calling models."""
    runner = CliRunner()
    result = runner.invoke(main, ["phase0", "--dry-run"])
    assert result.exit_code == 0
    # Dry run should output something about pairs or config
    assert "dry run" in result.output.lower() or "config" in result.output.lower()
```

**Step 2: Run integration test**

Run: `uv run pytest tests/integration/test_phase0_dryrun.py -v -m integration`
Expected: PASS (or skip if data not downloaded).

**Step 3: Commit**

```bash
git add tests/integration/test_phase0_dryrun.py
git commit -m "test: add Phase 0 dry-run integration test"
```

---

## Task 16: Download TrialGPT Data & Live Dry Run

**Step 1: Download the data**

Run: `uv run python -c "from trialmatch.data.downloader import download_all; download_all()"`

This downloads:
- `data/trec2021/queries.jsonl` (~100 KB)
- `data/trec2021/qrels.tsv` (~1 MB)
- `data/trec2021/corpus.jsonl` (~131 MB) — may need disk space

**Step 2: Verify data loaded correctly**

Run:
```bash
uv run python -c "
from trialmatch.data.trialgpt_loader import load_queries, load_qrels, load_corpus
from pathlib import Path
d = Path('data/trec2021')
topics = load_queries(d / 'queries.jsonl')
qrels = load_qrels(d / 'qrels.tsv')
print(f'Topics: {len(topics)}, Qrels: {len(qrels)}')
# Skip corpus if too large
"
```

Expected: `Topics: 75, Qrels: 35832`

**Step 3: Run CLI dry-run with real data**

Run: `uv run trialmatch phase0 --dry-run`

Expected: Lists 20 sampled pairs with topic IDs, NCT IDs, and qrel labels.

**Step 4: No commit needed (data files in .gitignore)**

Verify `data/trec2021/corpus.jsonl` is in `.gitignore`. If not, add it.

---

## Task 17: Live Benchmark Run (Requires API Keys)

**Prerequisites:**
- `HF_TOKEN` env var set (for MedGemma)
- `GEMINI_API_KEY` env var set (for Gemini)

**Step 1: Health check both models**

Run:
```bash
uv run python -c "
import asyncio, os
from trialmatch.models.medgemma import MedGemmaAdapter
from trialmatch.models.gemini import GeminiAdapter

async def check():
    mg = MedGemmaAdapter(hf_token=os.environ.get('HF_TOKEN', ''))
    print(f'MedGemma: {await mg.health_check()}')
    gm = GeminiAdapter(api_key=os.environ.get('GEMINI_API_KEY', ''))
    print(f'Gemini: {await gm.health_check()}')

asyncio.run(check())
"
```

Expected: Both return `True`.

**Step 2: Run Phase 0 benchmark**

Run: `uv run trialmatch phase0 --config configs/phase0.yaml`

Expected:
- Evaluates 20 pairs with both models
- Prints accuracy, F1, Cohen's κ for each model
- Saves results to `runs/phase0-medgemma-*` and `runs/phase0-gemini-*`
- Total cost < $1

**Step 3: Review results**

Check `runs/` directory for:
- `config.json` — run configuration
- `results.json` — per-pair verdicts and reasoning
- `metrics.json` — accuracy, F1, κ, confusion matrix
- `cost_summary.json` — token counts and costs

**Step 4: Commit results metadata**

```bash
git add runs/
git commit -m "results: Phase 0 benchmark run — MedGemma vs Gemini 3 Pro"
```

---

## Dependency Graph

```
Task 1 (infrastructure) ─── no dependencies
Task 2 (domain models) ──── no dependencies
Task 3 (data loader) ────── depends on Task 2
Task 4 (sampler) ─────────── depends on Task 2, 3
Task 5 (model base) ─────── depends on Task 2
Task 6 (medgemma) ────────── depends on Task 5
Task 7 (gemini) ──────────── depends on Task 5
Task 8 (evaluator) ────────── depends on Task 2, 5
Task 9 (metrics) ──────────── depends on Task 2
Task 10 (run manager) ────── depends on Task 2
Task 11 (downloader) ─────── no dependencies
Task 12 (CLI) ─────────────── depends on Task 3, 4, 6, 7, 8, 9, 10, 11
Task 13 (BDD) ─────────────── depends on Task 8
Task 14 (lint/test) ──────── depends on all above
Task 15 (integration) ────── depends on Task 12
Task 16 (data download) ──── depends on Task 11, 12
Task 17 (live run) ────────── depends on all above + API keys
```

**Parallelizable groups:**
- Group A: Tasks 1, 2 (can start immediately)
- Group B: Tasks 3, 5, 9, 10, 11 (after Task 2)
- Group C: Tasks 4, 6, 7, 8 (after their deps)
- Group D: Tasks 12, 13 (after Group C)
- Group E: Tasks 14-17 (sequential, end-to-end)
