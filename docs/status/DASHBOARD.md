<!-- Last updated: 2026-02-18T00:00:00Z -->

# Project Dashboard

## Current Phase

**SCAFFOLDING** — Setting up BDD framework, agent memory system, and development infrastructure. No implementation code yet.

```
[SCAFFOLDING] ← YOU ARE HERE
    ↓
[PHASE0_READY] — BDD features written, test infra working, configs ready
    ↓
[PHASE0_RUNNING] — Running 20-pair directional benchmark (~$3)
    ↓
[PHASE0_COMPLETE] — Results analyzed, go/no-go decision made
    ↓
[TIER_A] — 100-pair criterion-level evaluation (~$30)
```

## Current Sprint Goals

- [ ] Add pytest-bdd dependency and BDD directory structure
- [ ] Write initial BDD feature files for data/ and models/ (P0 components)
- [ ] Write initial BDD feature files for validate/ (core benchmark component)
- [ ] Implement data/ module: TREC topic/qrels loading via ir_datasets
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
| TREC 2021 data fetch status unknown | Can't verify data loading works | Human | 2026-02-18 |
| MedGemma endpoint activation status unknown | Can't test MedGemma adapter | Human | 2026-02-18 |
| GOOGLE_API_KEY availability for Gemini | Can't test Gemini adapter | Human | 2026-02-18 |

## Open Questions for Human

| Question | Context | Asked | Answered |
|----------|---------|-------|----------|
| Has TREC 2021 data been fetched via `ir_datasets`? | Needed before data/ module implementation | 2026-02-18 | - |
| Is MedGemma HF Inference endpoint active and reachable? | Needed for models/ adapter smoke test | 2026-02-18 | - |
| Gemini model exact version and Google AI Studio auth status? | Needed for Gemini adapter config | 2026-02-18 | - |
| Who will annotate the 100 pairs for Tier A? Timeline? | Blocks Tier A evaluation after Phase 0 | 2026-02-18 | - |
| Annotation tooling decision (Prodigy, Label Studio, spreadsheet)? | Affects SoT format and loading code | 2026-02-18 | - |

## Human Directives

_Space for human to communicate intent changes without updating the PRD. Agents: check this section every session._

> No directives yet.

## Recent Sessions (last 5)

| Date | Agent | What Was Done | What's Next |
|------|-------|--------------|-------------|
| 2026-02-18 | Claude | Created memory scaffolding: CLAUDE.md protocol, DASHBOARD.md, BDD conventions, Phase 0 config, agent skills | Add pytest-bdd dependency, write initial BDD feature files |
| 2026-02-17 | Claude | Created all project documentation: PRD v3.1, 5 ADRs, architecture, test strategy, decision log | Set up BDD framework and agent memory system |
