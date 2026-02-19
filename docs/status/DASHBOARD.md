<!-- Last updated: 2026-02-19T00:00:00Z -->

# Project Dashboard

## Current Phase

**SCAFFOLDING** — Setting up BDD framework, agent memory system, and development infrastructure. No implementation code yet.

```
[SCAFFOLDING] ← YOU ARE HERE
    ↓
[PHASE0_READY] — BDD features written, test infra working, configs ready
    ↓
[PHASE0_RUNNING] — Running 20-pair criterion-level benchmark (~$1)
    ↓
[PHASE0_COMPLETE] — Results analyzed, go/no-go decision made
    ↓
[TIER_A] — Full 1,024-pair criterion-level evaluation (~$25)
```

## Current Sprint Goals

- [ ] Add pytest-bdd dependency and BDD directory structure
- [ ] Write initial BDD feature files for data/ and models/ (P0 components)
- [ ] Write initial BDD feature files for validate/ (core benchmark component)
- [ ] Implement data/ module: HF dataset loading via `datasets` library (ADR-006)
- [ ] Implement models/ module: MedGemma + Gemini adapters
- [ ] Create Phase 0 smoke test (20 pairs, both models)

## Component Readiness

| Component | Domain Models | Logic | Unit Tests | BDD Scenarios | Status |
|-----------|:---:|:---:|:---:|:---:|--------|
| cli/ | - | stub (9 lines) | none | none | Stub only |
| data/ | not started | not started | none | none | Not started |
| models/ | not started | not started | none | none | Not started |
| ingest/ | not started | not started | none | none | Not started |
| prescreen/ | not started | not started | none | none | Not started |
| validate/ | not started | not started | none | none | Not started |
| evaluation/ | not started | not started | none | none | Not started |
| tracing/ | not started | not started | none | none | Not started |

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
| 2026-02-19 | Claude | Researched TREC 2022 vs TrialGPT HF dataset (agent team); updated Phase 0 plan to use HF criterion-level data (ADR-006); updated CLAUDE.md, phase0.yaml, DASHBOARD, decision log | Implement data/ module: hf_loader.py + sampler.py with TDD |
| 2026-02-18 | Claude | Validated MedGemma + Gemini 3 Pro connectivity; ran diagnostic comparison on 2 patients; created model-connectivity-report.md | Implement models/ adapters with TDD (base.py, medgemma.py, gemini.py, factory.py) |
| 2026-02-18 | Claude | Created memory scaffolding: CLAUDE.md protocol, DASHBOARD.md, BDD conventions, Phase 0 config, agent skills | Add pytest-bdd dependency, write initial BDD feature files |
| 2026-02-17 | Claude | Created all project documentation: PRD v3.1, 5 ADRs, architecture, test strategy, decision log | Set up BDD framework and agent memory system |
