# Test Strategy

## Overview

Three-tier testing: unit → integration → e2e. Each tier targets different failure modes.

## Unit Tests (`tests/unit/`)

Fast, isolated, no network/API calls. Mock all external dependencies.

| Module | What to test | Key assertions |
|--------|-------------|----------------|
| `ingest/` | KeyFact extraction parsing, profile text generation | Output schema validation, evidence span extraction |
| `prescreen/` | SearchAnchor JSON parsing, negative anchor generation | MUST/SHOULD priority assignment, synonym normalization |
| `validate/` | Criterion decision parsing, confidence scoring, trial-level aggregation | MET/NOT_MET/UNKNOWN classification, HARD/SOFT hardness |
| `data/` | TREC topic/qrel loading, trial JSON parsing, TrialGPT dataset parsing | Schema compliance, NCT ID extraction, drift flagging |
| `models/` | Model adapter response parsing, token counting, cost calculation | Token count accuracy, cost formula correctness |
| `evaluation/` | Metric computation: accuracy, F1, Cohen's κ, confusion matrix | Known-answer tests with hand-computed metrics |
| `tracing/` | Run artifact serialization, config logging | Artifact file structure, idempotent writes |

### Conventions

- Test files: `tests/unit/test_<module>.py`
- Use `pytest` with `pytest-mock` for mocking
- Fixtures in `tests/fixtures/` for sample TREC topics, trial JSONs, model responses
- Target: ≥ 90% line coverage on core logic (parsing, metrics, aggregation)

## Integration Tests (`tests/integration/`)

Test cross-module interactions and real API contracts (with recorded responses).

| Test | What it covers | Dependencies |
|------|---------------|-------------|
| `test_ingest_to_validate.py` | Full INGEST → VALIDATE pipeline with mock model | Model adapter mock returning fixture responses |
| `test_data_prepare.py` | Data loading from TrialGPT dataset files | Local fixture files mimicking TrialGPT format |
| `test_ctgov_api.py` | CT.gov API v2 response parsing (VCR-recorded) | `vcrpy` cassettes with real API responses |
| `test_evaluation_pipeline.py` | Metrics computation from real-format model outputs | Fixtures with known expected metrics |
| `test_run_tracing.py` | Full run artifact write/read cycle | Temp directory |

### Conventions

- Use `vcrpy` or `responses` for HTTP mocking
- Mark with `@pytest.mark.integration`
- Can run with: `pytest tests/integration/ -m integration`

## E2E Tests (`tests/e2e/`)

Full CLI invocation, real model calls (or recorded). Expensive — run manually or in CI nightly.

| Test | What it covers |
|------|---------------|
| `test_phase0_smoke.py` | `trialmatch phase0 --config <yaml>` with 2 topics, both models, whole criteria blocks |
| `test_data_prepare_api.py` | `trialmatch data prepare --year 2021 --source api` with 5 NCT IDs |
| `test_eval_validate.py` | `trialmatch eval validate --pairs <fixture> --model gemini --tier A` with 3 pairs, whole criteria blocks |
| `test_compare.py` | `trialmatch compare --runs <ids>` generates valid markdown report |

### Conventions

- Mark with `@pytest.mark.e2e`
- Require `GOOGLE_API_KEY` and/or `HF_TOKEN` env vars
- Run with: `pytest tests/e2e/ -m e2e --timeout=300`
- Budget guard: each test has max cost assertion (e.g., ≤ $0.10)

## Test Data / Fixtures (`tests/fixtures/`)

```
tests/fixtures/
├── trec_topics/           # Sample TREC topic XML (2-3 topics)
├── trec_qrels/            # Sample qrel files (subset)
├── trials/                # Sample trial JSONs (3-5 trials)
├── model_responses/       # Recorded LLM responses for deterministic tests
├── sot/                   # Sample gold-standard annotations
│   ├── ingest_sot.json
│   ├── prescreen_sot.json
│   └── validate_criterion_sot.json
└── expected_metrics/      # Hand-computed expected evaluation outputs
```

## CI Pipeline

```
1. lint (ruff check + ty) — every push
2. unit tests — every push
3. integration tests — every push (uses recorded responses)
4. e2e tests — nightly / manual trigger (uses real APIs)
```

## Coverage Targets

| Tier | Target | Enforcement |
|------|--------|------------|
| Unit | ≥ 90% line coverage on src/trialmatch/ | CI gate |
| Integration | Key pathways covered | Review gate |
| E2E | Smoke test per CLI command | Nightly |
