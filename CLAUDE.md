# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**trialmatch** — Clinical trial matching tool using MedGemma (4B multimodal + 27B text) and Gemini 3 Pro. Three-component pipeline: INGEST → PRESCREEN → VALIDATE. Benchmarked on TrialGPT HF criterion-level annotations (ADR-006). Preparing for MedGemma Impact Challenge submission (Kaggle, deadline Feb 24 2026).

## TODO — MedGemma Challenge Demo (Due Feb 24)

Design doc: `docs/plans/2026-02-21-medgemma-challenge-demo-final.md`
UI: **Streamlit** (no separate backend). Narrative: multi-model orchestration (not MedGemma superiority).

### Day 1 — Adapter + Streamlit Scaffold (Feb 21)
- [ ] Thin adapter: `src/trialmatch/ingest/profile_adapter.py` + unit tests
  - Converts `nsclc_trial_profiles.json` key_facts (list-of-objects) → dict for PRESCREEN
- [x] Phase 0 benchmark (27B + re-run 4B with prompt fix) — **running in separate session**
- [ ] Streamlit scaffold: patient selector, INGEST key_facts display, pipeline skeleton
- [ ] Wire PRESCREEN agent into Streamlit with `st.status()` + agent trace viz

### Day 2 — Pipeline Integration (Feb 22)
- [ ] VALIDATE integration: wire `evaluate_criterion()` into Streamlit, results panel
- [ ] Benchmark dashboard page: load run artifacts, metrics table, confusion matrices
- [ ] Cached/replay mode: record live PRESCREEN run, replay from JSON for demo reliability
- [ ] UI polish: loading states, error handling

### Day 3 — QA + Recording (Feb 23)
- [ ] Playwright QA first pass: select patient → run pipeline → verify results
- [ ] Record demo video (live first, cached backup)
- [ ] Pre-warm all endpoints before recording

### Day 4 — Ship (Feb 24)
- [ ] Kaggle Writeup (3-page technical document)
- [ ] Final Playwright QA, code cleanup, README, `.env.example`
- [ ] Submit to Kaggle

### Models Available

| Model | Endpoint | Use Case | Status |
|-------|----------|----------|--------|
| MedGemma 4B | HF Inference (`pcmy7bkqtqesrrzd`) | Multimodal (EHR + images) | Working (max_tokens=512 limit) |
| MedGemma 27B | HF Inference (`wu5nclwms3ctrwd1`) | Text-only (higher accuracy) | Failed — TGI 3.0 chat template incompatible |
| MedGemma 4B | Vertex AI Model Garden | Multimodal (EHR + images) | Configured, untested |
| MedGemma 27B | Vertex AI Model Garden (int8) | Text-only (higher accuracy) | **Benchmarked** — 70% accuracy, endpoint torn down |
| Gemini 3 Pro | Google AI Studio API | General-purpose baseline | Working |

### Phase 0 Benchmark Results (Feb 21-22, 2026)

| Model | Accuracy | Macro-F1 | Cohen's κ | Evidence Overlap | Notes |
|-------|----------|----------|-----------|------------------|-------|
| GPT-4 (baseline) | 75.0% | 74.6% | — | — | Built into HF dataset |
| Gemini 3 Pro | 75.0% | — | — | — | Post prompt fix (was 60%) |
| **MedGemma 27B (Vertex int8)** | **70.0%** | **72.2%** | **0.538** | **15.0%** | Vertex AI, max_tokens=2048, ~8s/pair |
| MedGemma 4B (HF) | 35.0% | 31.5% | 0.030 | 5.0% | max_tokens=512 workaround; was 55% with max_tokens=2048 |

**Key findings**:
1. **MedGemma 27B is competitive with GPT-4** on criterion-level matching (70% vs 75% accuracy). The 5% gap may narrow with prompt tuning.
2. **27B dramatically outperforms 4B** (70% vs 35%) — larger model has much better instruction-following for structured JSON output.
3. **Vertex AI vLLM path has no TGI CUDA bug** — max_tokens=2048 works reliably, enabling full reasoning chains.
4. MedGemma 4B has a **systematic MET bias**: model reasoning is often correct ("patient does NOT meet criterion") but the JSON `label` field contradicts the reasoning and outputs MET anyway. This is an instruction-following failure, not a reasoning failure.
5. The 4B model confuses "criterion is met" with "patient is eligible" — it says MET when the exclusion condition IS present (should be "excluded" → NOT_MET).
6. The max_tokens=512 workaround (for TGI CUDA bug on 4B) truncates thinking chains, degrading accuracy from 55% → 35%.

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

1. **TrialGPT HF criterion annotations** (primary, ADR-006) — `ncbi/TrialGPT-Criterion-Annotations` on HuggingFace. 1,015 rows, expert labels, GPT-4 baseline, evidence sentences. **Local copy**: `data/hf_cache/trialgpt_criterion_annotations.json` (1.4 MB). All configs use `fixture_path` to load from local — no internet required for benchmarks.
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
├── hf_cache/              # Local copy of HuggingFace dataset (checked in)
│   └── trialgpt_criterion_annotations.json  # 1,015 rows, 1.4 MB — primary data source
└── sot/                   # Human expert source-of-truth annotations
    ├── ingest/            # Gold PatientProfile + KeyFacts per topic
    ├── prescreen/         # Gold SearchAnchors per topic
    └── validate/          # Gold criterion-level MET/NOT_MET/UNKNOWN labels
```

Note: TrialGPT HF dataset (ADR-006) is the primary data source. Expert labels (`expert_eligibility`) serve as ground truth. GPT-4 predictions (`gpt4_eligibility`) serve as built-in baseline.

## HF Inference Endpoint Operations

### Known Issues — TGI + MedGemma CUDA Bug

**Problem**: TGI backend crashes with `CUDA CUBLAS_STATUS_EXECUTION_FAILED` on certain prompt + max_new_tokens combinations. After crash, GPU enters permanent "misaligned address" state — all subsequent requests fail until endpoint restart.

**Evidence** (Feb 21, 2026):
- Crash threshold: max_new_tokens between 500 and 1024 (binary search confirmed 500 OK, 1024 crash)
- NOT hardware-related: identical crash on Nvidia L4 (24GB) and L40S (48GB)
- NOT cumulative memory leak: crashes as first request on fresh GPU
- NOT prompt length: all prompts ~500 tokens, similar sizes
- Crash occurs during generation phase only (max_new_tokens=1 always works)
- Specific prompts trigger it reproducibly; others never crash at same max_new_tokens

**Workaround**: Set `max_tokens=512` in `evaluate_criterion()`. This avoids crashes but truncates MedGemma's thinking chain, degrading accuracy from 55% → 35%.

**Recovery**: After CUDA crash, endpoint must scale-to-zero and restart. Use HF API to pause/resume or wait for scale-to-zero timeout.

### Endpoint Management

```bash
# List endpoints
curl -s https://api.endpoints.huggingface.cloud/v2/endpoint/$HF_USERNAME \
  -H "Authorization: Bearer $HF_TOKEN" | python -m json.tool

# Pause endpoint (triggers scale-to-zero)
curl -X POST https://api.endpoints.huggingface.cloud/v2/endpoint/$HF_USERNAME/$ENDPOINT_NAME/pause \
  -H "Authorization: Bearer $HF_TOKEN"

# Resume endpoint
curl -X POST https://api.endpoints.huggingface.cloud/v2/endpoint/$HF_USERNAME/$ENDPOINT_NAME/resume \
  -H "Authorization: Bearer $HF_TOKEN"

# Delete endpoint
curl -X DELETE https://api.endpoints.huggingface.cloud/v2/endpoint/$HF_USERNAME/$ENDPOINT_NAME \
  -H "Authorization: Bearer $HF_TOKEN"
```

### MedGemma 27B Deployment Failures

| Attempt | Method | Result |
|---------|--------|--------|
| TGI (pytorch framework) | HF Inference Endpoint | OOM — 27B x fp32 = 108GB > A100 80GB |
| TGI (custom framework) | HF Inference Endpoint | Works for inference, but TGI 3.0 chat template breaks Gemma 3 format |
| vLLM (custom Docker) | HF Inference Endpoint | Image build failed, custom container not supported |

**Decision**: Abandon HF Inference for 27B. Use Vertex AI Model Garden instead (configured but untested).

## Rate Limits

| Service | Limit | Used By | Notes |
|---------|-------|---------|-------|
| HuggingFace Inference | 1 concurrent | MedGemma 1.5 4B | Was 5; reduced to 1 due to CUDA bug at higher concurrency |
| HuggingFace Inference | max_tokens=512 | MedGemma 1.5 4B | TGI CUDA bug at ≥1024; see HF Inference Endpoint Operations |
| Google AI Studio | 10 concurrent | Gemini 3 Pro | |
| CT.gov API v2 | 40 req/min | Tier B trial-level eval (deferred) | |

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
| Understand pipeline data flow | `docs/architecture/pipeline-overview.md` |
| Understand PRESCREEN agent interactions | `docs/architecture/prescreen-sequence-diagrams.md` |
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


## HF Inference Endpoint Operations

### Known Issues — TGI + MedGemma CUDA Bug

**Critical**: TGI (Text Generation Inference) + MedGemma 4B has a reproducible CUDA bug:

- **Symptom**: `CUBLAS_STATUS_EXECUTION_FAILED` or `CUDA error: misaligned address` when `max_new_tokens` ≥ ~500-1024 on certain prompts
- **NOT hardware-related**: Same crash on both Nvidia L4 (24GB) and L40S (48GB)
- **NOT cumulative memory leak**: Crashes on first request to fresh GPU
- **NOT prompt length**: All prompts are similar size (~500 tokens)
- **Root cause**: Specific combination of full-length clinical prompt + large `max_new_tokens` triggers TGI kernel bug during generation phase
- **Workaround**: Set `max_tokens=512` in `evaluator.py` (binary search confirmed 500 works, 1024 crashes)
- **Side effect**: 512 token limit truncates MedGemma "thinking tokens" (`<unused94>thought...`), degrading accuracy from 55% to 35%

### GPU State Corruption Recovery

After a CUBLAS error, the GPU enters a permanent "misaligned address" state where ALL subsequent requests fail. Recovery requires:
1. Scale endpoint to zero (or pause)
2. Wait for full shutdown
3. Resume endpoint (fresh GPU allocation)

### Endpoint Management via API

```bash
# List endpoints
curl -s https://api.endpoints.huggingface.cloud/v2/endpoint/$HF_USERNAME \
  -H "Authorization: Bearer $HF_TOKEN" | jq '.[].name'

# Delete failed endpoint
curl -X DELETE https://api.endpoints.huggingface.cloud/v2/endpoint/$HF_USERNAME/$ENDPOINT_NAME \
  -H "Authorization: Bearer $HF_TOKEN"

# Update hardware (e.g., L4 → L40S)
curl -X PUT https://api.endpoints.huggingface.cloud/v2/endpoint/$HF_USERNAME/$ENDPOINT_NAME \
  -H "Authorization: Bearer $HF_TOKEN" \
  -d '{"compute":{"instanceType":"nvidia-l40s-1"}}'
```

### MedGemma 27B Deployment Failures

Two deployment approaches failed for MedGemma 27B:
1. **TGI 3.0**: Does not support Gemma 3 chat template — generates garbled output with prompt echo
2. **vLLM custom image**: Custom Docker image deployment on HF Inference failed to start

**Recommendation**: Use Vertex AI Model Garden for 27B access instead of self-hosted HF endpoints.

## MedGemma Model Behavior Notes

- **Thinking tokens**: MedGemma uses `<unused94>thought...<unused95>` internal monologue tokens that consume output token budget before producing actual JSON response
- **Prompt echo**: Raw TGI `text_generation` API echoes the full prompt + chat template markers; `clean_model_response()` handles stripping
- **MET bias**: 4B model heavily biases toward MET predictions, especially on exclusion criteria where it should predict NOT_MET
- **JSON truncation**: When output budget is exhausted mid-JSON, `parse_criterion_verdict()` falls back to keyword extraction, which loses structured reasoning and evidence sentences
- **CWA (Closed World Assumption)**: Prompt instructs model to assume unmentioned facts are negative (e.g., no mention of allergies = no allergies). This is critical for exclusion criteria accuracy.

## IMPORTANT Rules
you MUST run ALL script, especially bash,python scripts in background so you can do other tasks
API keys are already stored in .env
MUST check the tail log in real time when running benchmark; do not use timeout or sleep
you MUST teardown Vertex medgemma endpoints after using to avoice extra cost