# ADR-005: Whole Criteria Blocks for Spike (No Atomization)

**Status:** Accepted
**Date:** 2026-02-17
**Decision Makers:** CTO

## Context

TrialGPT and similar systems atomize trial eligibility criteria into individual criterion statements (e.g., "Age ≥ 18", "ECOG 0-1", "No prior immunotherapy") before evaluating each one against the patient profile. This atomization step is non-trivial — it requires either an LLM pass or complex regex/NLP parsing, and criteria formatting varies widely across trials.

For the spike phase, we need to answer: "Does MedGemma add value over Gemini 3 Pro on clinical trial medical reasoning?" Atomization is an engineering concern, not a medical reasoning concern.

## Decision

**In the spike phase (Phase 0 and initial Phase 1), pass the entire inclusion criteria block and exclusion criteria block to the model as-is.** Do not atomize criteria into individual statements.

The model receives:
- Patient profile (from INGEST or gold SoT)
- Full inclusion criteria text block
- Full exclusion criteria text block

And returns:
- Overall MET/NOT_MET/UNKNOWN decision per block
- Reasoning chain explaining which specific criteria drove the decision
- Confidence score

## Rationale

- **Isolates the variable under test:** The spike tests medical reasoning capability, not criteria parsing capability. Both models receive identical input — any difference is attributable to reasoning.
- **Reduces engineering overhead:** No need to build/validate a criteria atomization pipeline for the spike.
- **Reduces annotation burden:** Annotators evaluate per-block decisions, not per-criterion decisions. Cuts SoT annotation time by ~60%.
- **More realistic:** In production, whole-block evaluation is a valid deployment pattern (TrialGPT's atomization is one approach, not the only one).

## Consequences

- **Pro:** Faster time-to-signal. Phase 0 can run as soon as model adapters + data loaders are built.
- **Pro:** Significantly reduced human annotation effort for SoT.
- **Pro:** Models see full context of related criteria together (some criteria interact — e.g., "Age ≥ 18 AND ECOG 0-1").
- **Con:** Cannot compute per-criterion accuracy (only per-block). Less granular error analysis.
- **Con:** Not directly comparable to TrialGPT's published 87.3% criterion-level accuracy (different granularity).
- **Con:** If a model gets the block wrong, harder to pinpoint which specific criterion caused the failure (mitigated by requiring reasoning chains in output).

## Revisit When

- Phase 0 results show that whole-block evaluation is too coarse to differentiate models → add atomization for Phase 1.
- Results are strong enough to warrant direct comparison with TrialGPT → atomization needed for apples-to-apples benchmark.
