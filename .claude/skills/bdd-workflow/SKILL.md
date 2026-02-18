---
name: bdd-workflow
description: BDD development workflow for trialmatch. Use when implementing features, components, or fixing bugs. Enforces RED-GREEN-REFACTOR cycle.
triggers:
  - implement feature
  - implement component
  - BDD
  - next task
  - red green refactor
---

# BDD Development Workflow

## Before You Start

1. Read `docs/status/DASHBOARD.md` for current state
2. Pick a task from "Current Sprint Goals" (prefer unblocked, lowest dependency)
3. Read relevant feature files in `features/{component}/`
4. Read `docs/bdd/README.md` for conventions

## RED Phase — Write Failing Tests

1. Create or update feature file: `features/{component}/{capability}.feature`
2. Tag new scenarios with `@wip` and `@component_{name}`
3. Write step definitions in `tests/bdd/steps/{component}_steps.py`
4. Run and confirm FAILURE:
   ```bash
   uv run pytest tests/bdd/ -m component_{name} -v --timeout=30
   ```

## GREEN Phase — Make Tests Pass

1. Implement minimal code in `src/trialmatch/{component}/`
2. Run BDD tests and confirm PASS:
   ```bash
   uv run pytest tests/bdd/ -m component_{name} -v --timeout=30
   ```
3. Run unit tests to check for regressions:
   ```bash
   uv run pytest tests/unit/ --timeout=30
   ```

## REFACTOR Phase — Clean Up

1. Lint and format:
   ```bash
   uv run ruff check src/ tests/ && uv run ruff format src/ tests/
   ```
2. Run all tests to confirm nothing broke:
   ```bash
   uv run pytest tests/ --timeout=30
   ```

## After You Finish

1. Change `@wip` → `@implemented` on passing scenarios
2. Update `docs/status/DASHBOARD.md`:
   - Check off completed sprint goals
   - Update Component Readiness table
   - Add row to Recent Sessions
   - Update `<!-- Last updated: ... -->` timestamp to current ISO time
