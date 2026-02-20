<!-- Last updated: 2026-02-21T05:10:00Z -->

# Project Dashboard

## Current Phase

**PHASE0_READY** — All implementation code complete, 3 models wired (MedGemma 4B + 27B + Gemini 3 Pro), 137 tests passing + 3 smoke tests. Ready for live benchmark run.

```
[SCAFFOLDING] ✅ DONE
    ↓
[PHASE0_READY] ← YOU ARE HERE (3-way comparison ready)
    ↓
[PHASE0_RUNNING] — Running 20-pair criterion-level benchmark (~$1 per model)
    ↓
[PHASE0_COMPLETE] — Results analyzed, go/no-go decision made
    ↓
[TIER_A] — Full 1,024-pair criterion-level evaluation (~$25)
```

## Current Sprint Goals

- [x] Add pytest-bdd dependency and BDD directory structure
- [x] Write initial BDD feature files for validate/ (core benchmark component)
- [x] Implement data/ module: HF dataset loading via `datasets` library (ADR-006)
- [x] Implement models/ module: MedGemma + Gemini adapters
- [x] Implement validate/ evaluator (reusable core)
- [x] Implement evaluation/ metrics
- [x] Implement tracing/ run manager
- [x] Implement CLI phase0 command
- [x] Deploy MedGemma 27B endpoint (TGI on A100 80GB) + smoke tests passing
- [ ] Run live Phase 0 benchmark (3-way: 4B vs 27B vs Gemini)

## Component Readiness

| Component | Domain Models | Logic | Unit Tests | BDD Scenarios | Status |
|-----------|:---:|:---:|:---:|:---:|--------|
| cli/ | - | phase0_cmd | 2 tests | none | Ready |
| data/ | CriterionAnnotation | hf_loader + sampler | 19 tests | none | Ready |
| models/ | ModelResponse, CriterionVerdict | base + medgemma (4B+27B) + gemini | 10 tests + 3 smoke | none | Ready |
| ingest/ | not started | not started | none | none | Not started (deferred) |
| prescreen/ | ToolCallRecord, TrialCandidate, PresearchResult | CTGovClient, ToolExecutor, agent loop | 33 tests | none | Ready |
| validate/ | CriterionResult | evaluator (REUSABLE) | 14 tests | 4 scenarios | Ready |
| evaluation/ | - | metrics + evidence overlap | 10 tests | none | Ready |
| tracing/ | RunResult | run_manager | 2 tests | none | Ready |

## Test Summary

- **133 unit tests** passing across 11 test files
- **4 BDD scenarios** passing for validate module
- **3 smoke tests** for MedGemma 27B endpoint (health check, criterion eval, template format)
- **140 total tests**, zero lint errors, zero format issues

## Blockers

| Blocker | Impact | Owner | Since |
|---------|--------|-------|-------|
| ~~TREC 2021 data fetch status unknown~~ | ~~Can't verify data loading works~~ | ~~Human~~ | Resolved 2026-02-19 — switched to HF dataset (ADR-006) |
| ~~MedGemma endpoint activation status unknown~~ | ~~Can't test MedGemma adapter~~ | ~~Human~~ | Resolved 2026-02-18 |
| ~~GOOGLE_API_KEY availability for Gemini~~ | ~~Can't test Gemini adapter~~ | ~~Human~~ | Resolved 2026-02-18 |

## Open Questions for Human

| Question | Context | Asked | Answered |
|----------|---------|-------|----------|
| ~~Has TREC 2021 data been fetched via `ir_datasets`?~~ | ~~Needed before data/ module implementation~~ | 2026-02-18 | Superseded by ADR-006 — using HF dataset instead |
| Is MedGemma HF Inference endpoint active and reachable? | Needed for models/ adapter smoke test | 2026-02-18 | Yes — verified 2026-02-18, health_check passes |
| Gemini model exact version and Google AI Studio auth status? | Needed for Gemini adapter config | 2026-02-18 | `gemini-3-pro-preview`, API key validated 2026-02-18 |
| ~~Who will annotate the 100 pairs for Tier A? Timeline?~~ | ~~Blocks Tier A evaluation after Phase 0~~ | 2026-02-18 | Superseded — HF dataset has 1,024 pre-annotated pairs for Tier A |
| ~~Annotation tooling decision (Prodigy, Label Studio, spreadsheet)?~~ | ~~Affects SoT format and loading code~~ | 2026-02-18 | Superseded — expert annotations already in HF dataset |

## Human Directives

_Space for human to communicate intent changes without updating the PRD. Agents: check this section every session._

> 2026-02-19: Switch to TrialGPT HF criterion-level dataset as sole Phase 0 data source. Strip out TREC snapshot complexity. See ADR-006.

## Recent Sessions (last 5)

| Date | Agent | What Was Done | What's Next |
|------|-------|--------------|-------------|
| 2026-02-20 | Claude | Deployed MedGemma 27B as third benchmark model: (1) created deploy script using HF Python API — discovered `pytorch` framework OOM'd on A100 80GB (27B x fp32 = 108GB), fixed by using `framework="custom"` + TGI docker (loads bf16 directly, ~54GB), (2) added `model_name` param to MedGemmaAdapter (backward-compatible), (3) wired `endpoint_url` + `model_name` from YAML config into phase0.py, (4) created 27B-only + 3-way config YAMLs, (5) 3/3 smoke tests passing on live endpoint, (6) 108 unit tests still passing. Endpoint: `https://wu5nclwms3ctrwd1.us-east-1.aws.endpoints.huggingface.cloud` (A100 80GB, scale-to-zero 15min). | Run 3-way Phase 0 benchmark: `uv run trialmatch phase0 --config configs/phase0_three_way.yaml` |
| 2026-02-20 | Claude | PRESCREEN bug fixes (code review): (1) ctgov_client 429-retry now resets `_last_call_time` after backoff sleep — prevents double-wait on next request, (2) agent budget guard now sends `FunctionResponse` per pending tool call instead of plain text — fixes invalid conversation structure that would crash the genai SDK. 2 regression tests added, 125 unit tests passing (was 123). | Run live Phase 0 benchmark with API keys |
| 2026-02-20 | Claude | Phase 0 prompt fix + trial aggregation: (1) PROMPT_TEMPLATE now criterion-type-aware (inclusion vs exclusion instructions), (2) fixed bare `thought` token leak in clean_model_response(), (3) Gemini timeout 120s→300s, (4) TrialVerdict + aggregate_to_trial_verdict() added to metrics.py, (5) trial-level aggregation wired into CLI, (6) filter_by_keywords() added to sampler, (7) NSCLC config + keyword filter CLI support. 123 unit tests passing (was 99). NSCLC dry-run: 0 matches — HF dataset has no NSCLC patients. | Re-run Phase 0 benchmark with API keys; exclusion criterion accuracy should improve |
| 2026-02-20 | Claude | Implemented PRESCREEN module (agentic architecture): schema.py (ToolCallRecord/TrialCandidate/PresearchResult), ctgov_client.py (async CT.gov API v2, rate-limited), tools.py (Gemini FunctionDeclarations + ToolExecutor with MedGemma integration), agent.py (Gemini multi-turn agentic loop). 31 new unit tests, 108 total passing. | Run live Phase 0 benchmark with API keys |
| 2026-02-20 | Claude (3-agent team) | Deep PRESCREEN module assessment: multi-agent gap analysis + direct CT.gov API validation (5 query strategies on NSCLC EGFR L858R patient). Found biomarker-first strategy fails for NCT05456256; phenotype-first (never smoker) is correct. Produced complete revised design with 4D SearchAnchors model, 5-layer query architecture, corrected SoT Harness. See `docs/plans/2026-02-20-prescreen-module-assessment.md` | Implement PRESCREEN module |
| 2026-02-19 | Claude | Implemented full Phase 0 vertical slice: domain models, HF loader, sampler, MedGemma + Gemini adapters, reusable evaluator, metrics, run manager, CLI phase0 command, 4 BDD scenarios. 68 tests passing, zero lint. | Run live Phase 0 benchmark with API keys |
| 2026-02-19 | Claude | Researched TREC 2022 vs TrialGPT HF dataset (agent team); updated Phase 0 plan to use HF criterion-level data (ADR-006); updated CLAUDE.md, phase0.yaml, DASHBOARD, decision log | Implement data/ module: hf_loader.py + sampler.py with TDD |
| 2026-02-18 | Claude | Validated MedGemma + Gemini 3 Pro connectivity; ran diagnostic comparison on 2 patients; created model-connectivity-report.md | Implement models/ adapters with TDD (base.py, medgemma.py, gemini.py, factory.py) |
| 2026-02-18 | Claude | Created memory scaffolding: CLAUDE.md protocol, DASHBOARD.md, BDD conventions, Phase 0 config, agent skills | Add pytest-bdd dependency, write initial BDD feature files |
| 2026-02-17 | Claude | Created all project documentation: PRD v3.1, 5 ADRs, architecture, test strategy, decision log | Set up BDD framework and agent memory system |
