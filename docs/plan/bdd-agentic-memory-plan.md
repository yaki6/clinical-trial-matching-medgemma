# Plan: BDD-Driven Development + Agentic Memory System for trialmatch

## Context

**Problem:** trialmatch has excellent traditional documentation (PRD, 4 ADRs, architecture docs, test strategy) but zero implementation code, zero tests, and no mechanism linking product requirements to executable specifications. AI agents starting new sessions must re-read everything and have no structured way to know "what's done" vs "what's next."

**Goal:** Establish a Behavior-Driven Development workflow where (1) PRD requirements drive Gherkin feature files, (2) feature files drive implementation via RED-GREEN-REFACTOR, and (3) a sustainable memory system enables agents to pick up work across sessions with minimal context-loading overhead.

**Current state:** Skeleton project — 9 lines of code (CLI stub), 0 tests, 10 comprehensive docs, all module directories empty.

---

## Gap Analysis

| Gap | Impact | Priority |
|-----|--------|----------|
| No BDD framework (no feature files, no pytest-bdd) | PRD requirements not connected to executable tests | P0 |
| No progress tracking (agents can't tell what's done) | Every session wastes time re-discovering state | P0 |
| No session handoff protocol | Context lost between sessions | P0 |
| No PRD→feature→test traceability | Can't verify PRD coverage | P1 |
| CLAUDE.md lacks agent workflow instructions | Agents don't know HOW to work on this project | P1 |
| No BDD/status skills | Common workflows not codified | P2 |
| No configs/ YAML files | CLI features reference non-existent configs | P2 |

---

## Implementation Plan

### Step 1: Add pytest-bdd dependency

**File:** `pyproject.toml`
- Add `"pytest-bdd>=7.0"` to `[dependency-groups] dev`
- Add BDD-related pytest markers: `bdd`, `phase0`, `phase1`, `component_ingest`, `component_prescreen`, `component_validate`, `component_data`, `component_models`, `wip`, `implemented`
- Add `bdd_features_base_dir = "features/"` to `[tool.pytest.ini_options]`
- Run `uv sync`

**Rationale:** pytest-bdd integrates natively with the existing pytest ecosystem (pytest-mock, pytest-cov, vcrpy). No parallel test runner needed.

### Step 2: Create BDD directory structure

```
features/                        # Gherkin specs (NOT inside tests/)
  ingest/
  prescreen/
  validate/
  data/
  models/
  evaluation/
  cli/
  e2e/

tests/bdd/                       # Step definitions
  conftest.py                    # Shared BDD fixtures
  steps/
    common_steps.py
    ingest_steps.py
    prescreen_steps.py
    validate_steps.py
    data_steps.py
    model_steps.py
    evaluation_steps.py
    cli_steps.py
```

Feature files at root = specifications visible to humans and agents. Step definitions in `tests/bdd/` = test code alongside existing test infrastructure.

### Step 3: Create agentic memory documents

**New directory:** `docs/status/`

**File: `docs/status/PROJECT_STATUS.md`** — The agent entry point. Contains:
- Current phase (SCAFFOLDING → PHASE0_READY → PHASE0_RUNNING → ...)
- Component readiness matrix (models? logic? tests? BDD scenarios?)
- Current sprint goals (checkboxes)
- Blocked items
- Recent completions (last 5)
- Quick reference links

**File: `docs/status/COMPONENT_STATUS.md`** — Per-component detail:
- Owner, status, dependencies, what it blocks
- Domain models needed
- Functions to implement
- Linked BDD scenario files + counts

**File: `docs/status/SESSION_LOG.md`** — Append-only log:
- Each agent session appends: timestamp, what was done, what's next

### Step 4: Create traceability matrix

**New directory:** `docs/traceability/`

**File: `docs/traceability/prd-to-features.md`**
- Maps every PRD section → feature file → scenario count → status (@wip/@implemented)
- Single source of truth for "is this requirement covered?"

### Step 5: Write initial feature files (all @wip)

Translate PRD v3.0 requirements into Gherkin scenarios for each component. Key feature files:

| Feature File | PRD Section | Scenarios | Priority |
|-------------|-------------|-----------|----------|
| `features/validate/criterion_decision.feature` | 6.3, 7.3 | ~8 | P0 (core benchmark) |
| `features/ingest/keyfact_extraction.feature` | 6.1, 7.1 | ~6 | P0 |
| `features/prescreen/search_anchor_generation.feature` | 6.2, 7.2 | ~4 | P0 |
| `features/data/trec_topic_loading.feature` | 5 | ~3 | P0 (foundation) |
| `features/data/trial_fetching.feature` | 5 | ~4 | P0 |
| `features/models/model_adapter.feature` | 10.3 | ~4 | P0 |
| `features/models/cost_tracking.feature` | 10.3 | ~4 | P1 |
| `features/cli/phase0.feature` | 9 | ~3 | P1 |
| `features/evaluation/metric_computation.feature` | 10.1 | ~4 | P1 |
| `features/e2e/phase0_smoke.feature` | 7.0 | ~3 | P1 |

**Tag strategy:**
- `@wip` — scenario written, not yet implemented
- `@implemented` — step defs + source code passing
- `@phase0` / `@phase1` — evaluation phase scope
- `@component_*` — component isolation for selective runs
- `@gold_input` — uses gold SoT (ADR-002 compliance)
- `@needs_api` — requires live API (cost-aware)

### Step 6: Create BDD test infrastructure

**File: `tests/bdd/conftest.py`**
- Shared fixtures: gold SoT loading, sample trials, model adapter selection (mock/vcr/live)
- Wire model_backend fixture to test tier markers

**File: `tests/bdd/steps/common_steps.py`**
- Reusable Given/When/Then steps across components (patient profile loading, trial loading)

**File: `tests/bdd/steps/{component}_steps.py`**
- Component-specific step definitions importing from `src/trialmatch/{component}/`

### Step 7: Enhance CLAUDE.md with agent protocol

Add new sections to CLAUDE.md after "Rate Limits":

**"Agent Session Protocol"** section:
- Starting: read PROJECT_STATUS.md → COMPONENT_STATUS.md → pick sprint goal
- During: BDD RED-GREEN-REFACTOR loop with specific commands
- Ending: update tags, status docs, session log

**"BDD Commands"** section:
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
uv run pytest tests/bdd/ --collect-only -q | grep -c "scenario"
```

**"Feature File Conventions"** section:
- Location, naming, tag rules, metadata comments

### Step 8: Create BDD conventions doc

**File: `docs/bdd/README.md`**
- How to write feature files for this project
- Step definition patterns (with examples)
- Tag reference table
- Integration with 3-tier test strategy (same feature, different mocking depth)

### Step 9: Create new skills

**File: `.claude/skills/bdd-workflow/SKILL.md`**
- Triggers: "implement feature", "BDD", "implement component", "next task"
- Session start/end checklists
- RED-GREEN-REFACTOR loop commands

**File: `.claude/skills/status-check/SKILL.md`**
- Triggers: "status", "what's next", "project status"
- Quick status commands: read PROJECT_STATUS.md, count @wip vs @implemented

### Step 10: Create Phase 0 config

**File: `configs/phase0.yaml`**
- 20 pairs from TREC 2021, 3 models, $5 budget cap
- Referenced by CLI feature files and e2e tests

---

## Files to Create/Modify

### New files (~30 files)

| Category | Count | Key Files |
|----------|-------|-----------|
| Feature files | ~16 | `features/{component}/*.feature` |
| Step definitions | ~9 | `tests/bdd/steps/*_steps.py`, `conftest.py` |
| Status docs | 3 | `docs/status/PROJECT_STATUS.md`, `COMPONENT_STATUS.md`, `SESSION_LOG.md` |
| Traceability | 1 | `docs/traceability/prd-to-features.md` |
| BDD docs | 1 | `docs/bdd/README.md` |
| Skills | 2 | `.claude/skills/{bdd-workflow,status-check}/SKILL.md` |
| Config | 1 | `configs/phase0.yaml` |

### Modified files (2)

| File | Changes |
|------|---------|
| `pyproject.toml` | Add pytest-bdd dep, BDD markers, features base dir |
| `CLAUDE.md` | Add Agent Session Protocol, BDD Commands, Feature Conventions sections |

---

## Agent Workflow: How It All Connects

```
CLAUDE.md
  ├── points to → docs/status/PROJECT_STATUS.md  (agent entry point)
  ├── points to → docs/bdd/README.md             (BDD conventions)
  └── defines   → commands, architecture, rules

docs/status/PROJECT_STATUS.md
  ├── references → features/**/*.feature          (BDD scenario counts)
  ├── references → docs/status/COMPONENT_STATUS.md
  └── tracks    → sprint goals, blocked items

features/**/*.feature
  ├── maps from → docs/traceability/prd-to-features.md → PRD sections
  ├── maps to   → tests/bdd/steps/*_steps.py          → step definitions
  └── tags      → @wip / @implemented                  → progress state

tests/bdd/steps/*_steps.py
  ├── imports   → src/trialmatch/**                    → source code
  └── uses      → tests/fixtures/**                    → test data
```

### Standard Agent Session Workflow

```
1. ORIENT
   Read PROJECT_STATUS.md → COMPONENT_STATUS.md → SESSION_LOG.md (last 3)
   Read relevant feature files

2. PICK TASK
   Choose from "Current Sprint Goals" in PROJECT_STATUS.md
   Prefer: unblocked tasks, lowest dependency depth first

3. RED: Write/review BDD scenarios
   Read features/<component>/*.feature
   Write step definitions in tests/bdd/steps/
   Run tests — confirm they FAIL (red)

4. GREEN: Implement source code
   Write minimal code in src/trialmatch/ to pass
   Run BDD tests again — confirm they PASS

5. REFACTOR: Clean up
   Lint + format + type check
   Ensure all tests still pass

6. HANDOFF: Update status documents
   Change @wip → @implemented on passing scenarios
   Update PROJECT_STATUS.md
   Append to SESSION_LOG.md
```

### Parallel Agent Work Zones (safe to parallelize)

| Agent A | Agent B | Agent C |
|---------|---------|---------|
| data/ module | models/ module | evaluation/ module |
| TREC loading | MedGemma adapter | Metric computation |
| Trial fetching | Gemini adapter | Confusion matrix |

**Sequential dependencies:**
```
data/ → models/ → ingest/ → prescreen/ → validate/ → evaluation/ → cli/
```

---

## Verification

1. `uv sync` — dependency install succeeds
2. `uv run pytest tests/bdd/ --collect-only` — all feature scenarios discovered
3. `uv run pytest tests/bdd/ -m wip --no-header -q` — WIP scenarios listed (expected: all fail)
4. Verify `docs/status/PROJECT_STATUS.md` accurately reflects current state
5. Verify `docs/traceability/prd-to-features.md` covers all PRD requirement sections
6. Verify CLAUDE.md agent protocol instructions are clear and actionable

---

## Research Sources

- [pytest-bdd 8.1.0 documentation](https://pytest-bdd.readthedocs.io/en/latest/)
- [Writing Stable E2E Tests with pytest-bdd](https://qahivelab.github.io/2025/01/29/stable-e2e-tests-pytest-bdd.html)
- [Claude Code Memory Management](https://code.claude.com/docs/en/memory)
- [Writing a good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Claude Code Best Practices for Agentic Coding](https://medium.com/@habib.mrad.83/claude-code-practical-best-practices-for-agentic-coding-2be1b62cfeff)
- [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
