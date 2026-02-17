# ADR-002: Component Isolation Using Gold SoT Inputs

**Status:** Accepted
**Date:** 2026-02-17
**Decision Makers:** CTO

## Context

The pipeline has three components: INGEST → PRESCREEN → VALIDATE. If a downstream component receives bad input from upstream, errors compound and we can't diagnose which component failed.

## Decision

**Evaluate each component with gold-standard (human-annotated) upstream inputs.**

- PRESCREEN receives gold INGEST SoT, not model INGEST output
- VALIDATE receives gold INGEST SoT, not model INGEST output
- A separate E2E run tests error propagation with model outputs end-to-end

## Rationale

- A bad INGEST extraction would poison search terms, making PRESCREEN look bad when PRESCREEN itself may be fine
- Component isolation is standard in NLP pipeline evaluation
- Gold inputs are available from human expert annotations (§6 of PRD)

## Consequences

- **Pro:** Can precisely attribute failures to the responsible component
- **Pro:** Each component's metrics are independent and comparable across models
- **Con:** Requires human annotation effort to create gold SoT (~60-80 physician-hours)
- **Con:** Gold inputs may not reflect realistic operating conditions

## Implementation Note

Cache keys must include `ingest_source=gold|model` to prevent cache contamination between isolated and E2E runs.
