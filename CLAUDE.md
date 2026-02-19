# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**trialmatch** — CLI-first Python benchmark comparing MedGemma 1.5 4B vs Gemini 3 Pro on clinical trial criterion-level matching. Uses TrialGPT HF criterion-level annotations (ADR-006) as primary data source. Three-component pipeline: INGEST → PRESCREEN → VALIDATE. Spike phase evaluates individual eligibility criteria (not atomized sub-criteria).

Frontend integration planned for next phase

## Commands

```bash
# Install dependencies
uv sync

# Run all unit tests
uv run pytest tests/unit/

# Run a single test file
uv run pytest tests/unit/test_ingest.py

# Run a single test
uv run pytest tests/unit/test_ingest.py::test_keyfact_extraction -v

# Run integration tests
uv run pytest tests/integration/ -m integration

# Run e2e tests (requires GOOGLE_API_KEY / HF_TOKEN)
uv run pytest tests/e2e/ -m e2e --timeout=300

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run ty check src/

# CLI usage
uv run trialmatch data prepare --source huggingface
uv run trialmatch phase0 --config configs/phase0.yaml
uv run trialmatch eval validate --pairs <path> --model gemini --tier A
uv run trialmatch compare --runs <run_ids>
```

## Architecture

```
src/trialmatch/
├── cli/           # Click CLI: trialmatch command group
├── ingest/        # understand() — patient text → PatientProfileText + KeyFacts
├── prescreen/     # generate_search_terms() — PatientProfile → SearchAnchors
├── validate/      # evaluate_criterion() — (Patient, Criterion) → MET/NOT_MET/UNKNOWN
├── data/          # HF dataset loading (TrialGPT criterion annotations), label mapping, sampling
├── models/        # Model adapters: MedGemma (HF Inference) + Gemini (AI Studio/Vertex)
├── evaluation/    # scikit-learn metrics: accuracy, F1, Cohen's κ, confusion matrices
└── tracing/       # Run artifact persistence to runs/<run_id>/, cost tracking per LLM call
```

### Git Commit Rules

- **Atomic commits**: Each commit addresses exactly one concern (one bug fix, one feature, one refactor). Never bundle unrelated changes.
- **Intent-first messages**: Commit message must explain WHY the change was made, not just WHAT changed. Lead with the problem or goal, then the solution.
- **Format**: `<type>: <intent summary>` on the first line, then a blank line, then bullet points describing what was done. Types: `fix`, `feat`, `refactor`, `docs`, `test`, `chore`.
- **Future-readable**: Write messages as if a teammate 6 months from now needs to understand the change from `git log` alone.

### Critical Design Rules

1. **Component isolation**: When evaluating PRESCREEN or VALIDATE, feed gold INGEST SoT as input (not model output). This isolates component-specific errors. E2E runs test error propagation separately.
2. **Cache key contamination**: Cache keys MUST include `ingest_source=gold|model` to prevent cross-contamination between isolated and end-to-end evaluation runs.
3. **Run determinism**: Every run writes to `runs/<run_id>/` containing: config YAML, all inputs, all model responses, traces, computed metrics. Runs must be fully reproducible.
4. **Cost tracking**: Every LLM call must log `model, input_tokens, output_tokens, estimated_cost, latency_ms`. Aggregated per run. Budget guards in e2e tests.


### Data Sources (Preference Order)

1. **TrialGPT HF criterion annotations** (primary, ADR-006) — `ncbi/TrialGPT-Criterion-Annotations` on HuggingFace. 1,024 criterion-level pairs with expert labels, GPT-4 baseline, evidence sentences. Self-contained, < 5 MB.
2. **TREC 2021+2022 qrels** (Tier B, deferred) — trial-level labels for ranking evaluation. Only needed after Phase 0 / Tier A criterion-level eval is complete.
3. **CT.gov API v2** (fallback) — live data, 40 req/min rate limit. Only needed if supplementing with trials not in HF dataset.

## Memory System (docs/)

| Directory | Purpose |
|-----------|---------|
| `docs/prd/` | Product requirements documents. Current: v3.1 criterion-matching benchmark |
| `docs/architecture/` | Technical architecture diagrams and component design |
| `docs/test-strategy/` | Unit/integration/e2e test strategy, coverage targets, CI pipeline |
| `docs/adr/` | Architecture Decision Records (numbered, using template.md) |
| `docs/decisions/` | Quick-reference decision log table |

When making architectural decisions, create a new ADR in `docs/adr/NNN-<slug>.md` using the template and update `docs/decisions/README.md`.

## Evaluation Tiers

| Tier | Pairs | Data Source | Purpose | Budget |
|------|-------|-------------|---------|--------|
| A | All 1,024 | TrialGPT HF | Full criterion-level evaluation | ~$25 |
| B | TREC 2021+2022 trial-level | TREC qrels | Trial-level ranking with statistical power | ~$300 |
| C | All ~35K (optional) | TREC qrels | Comprehensive metrics if results warrant | ~$4,000 |

Phase 0 uses 20 criterion-level pairs from TrialGPT HF dataset as a fast directional capability check (~$1).

## Key Evaluation Targets

- **VALIDATE criterion accuracy**: ≥ 85% (TrialGPT GPT-4 baseline: 87.3%)
- **INGEST key fact F1**: ≥ 87% (recall ≥ 90%, precision ≥ 85%)
- **PRESCREEN MUST-anchor recall**: 100%
- **MedGemma go/no-go**: Directional advantage + qualitatively better error patterns over Gemini 3 Pro

## Data Layout

```
data/
├── hf_cache/              # HuggingFace datasets cache (auto-managed)
│   └── ncbi___trial_gpt-criterion-annotations/  # 1,024 criterion-level pairs
└── sot/                   # Human expert source-of-truth annotations
    ├── ingest/            # Gold PatientProfile + KeyFacts per topic
    ├── prescreen/         # Gold SearchAnchors per topic
    └── validate/          # Gold criterion-level MET/NOT_MET/UNKNOWN labels
```

Note: TrialGPT HF dataset (ADR-006) is the primary data source. Expert labels (`expert_eligibility`) serve as ground truth. GPT-4 predictions (`gpt4_eligibility`) serve as built-in baseline.

## Rate Limits

| Service | Limit | Used By |
|---------|-------|---------|
| HuggingFace Inference | 5 concurrent | MedGemma 1.5 4B |
| Google AI Studio | 10 concurrent | Gemini 3 Pro |
| CT.gov API v2 | 40 req/min | Tier B trial-level eval (deferred) |

## Agent Session Protocol

### Session Start
1. Read `docs/status/DASHBOARD.md` — situational awareness (phase, blockers, sprint goals)
2. Check freshness: if `<!-- Last updated: ... -->` timestamp is > 48h old, validate against `git log --oneline -10` and update
3. Read deeper docs as needed for your task (see Document Map below)
4. If "Open Questions for Human" has unanswered items > 3 sessions old, flag to user at session start

### Session End
1. Update `docs/status/DASHBOARD.md`: check off completed goals, add session row to Recent Sessions, update Component Readiness if changed
2. If DASHBOARD.md was modified since your session started (check `git diff`), re-read before updating to avoid overwriting concurrent changes
3. If you made an architectural decision, create ADR in `docs/adr/` and update `docs/decisions/README.md`

### Decision Authority

| Level | When | Examples |
|-------|------|---------|
| **DECIDE** | Reversible, local, no user impact | Variable naming, test structure, import ordering, internal refactors |
| **DECIDE+RECORD** | Affects architecture or future sessions | New dependency, data format choice, API design, caching strategy → create ADR |
| **ASK** | Irreversible, budget-impacting, scope-changing | Dropping a requirement, changing evaluation tiers, spending > $5 on API calls, modifying PRD targets |

## Document Map

| I need to... | Read this |
|-------------|-----------|
| Understand the project | `CLAUDE.md` (this file) |
| Know what's done / what's next | `docs/status/DASHBOARD.md` |
| Understand requirements | `docs/prd/v3.1-criterion-matching-benchmark.md` |
| Check architectural decisions | `docs/decisions/README.md` → `docs/adr/NNN-*.md` |
| Understand system design | `docs/architecture/README.md` |
| Know test strategy | `docs/test-strategy/README.md` |
| Write BDD feature files | `docs/bdd/README.md` |
| Understand annotation rules | `docs/sot-annotation-requirements.md` |
| Run Phase 0 | `configs/phase0.yaml` |

## BDD Commands

```bash
# Run all BDD tests
uv run pytest tests/bdd/ -m bdd -v

# Run BDD for a specific component
uv run pytest tests/bdd/ -m component_validate -v

# Run only implemented (passing) scenarios
uv run pytest tests/bdd/ -m "implemented and not wip" -v

# Run WIP scenarios to see what needs implementation
uv run pytest tests/bdd/ -m wip -v --no-header

# Count scenario status
uv run pytest tests/bdd/ --collect-only -q 2>/dev/null | grep -c "scenario"
```
