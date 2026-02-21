# ADR-008: Eligible Label Experiment Results

<!-- Last updated: 2026-02-21 -->

**Status**: Rejected
**Date**: 2026-02-21
**Decision**: Revert simplified "eligible/not eligible/unknown" labels. Keep original TrialGPT-native labels.

## Context

MedGemma 4B achieved only 35% accuracy on Phase 0 benchmark due to a systematic MET bias. The hypothesis was that TrialGPT-native 6-class labels (`included`/`not included`/`excluded`/`not excluded`/`not enough information`) create semantic confusion, especially the double-negation "not excluded" and the inverted semantics of exclusion criteria.

We replaced the 6-class labels with 3 universal labels: `"eligible"` (MET), `"not eligible"` (NOT_MET), `"unknown"` (UNKNOWN).

## Experiment

- Run ID (MedGemma): `phase0-medgemma-1.5-4b-20260221-013509`
- Run ID (Gemini): `phase0-gemini-3-pro-preview-20260221-013700`
- Baseline comparison: `phase0-medgemma-1.5-4b-20260221-005247` (native labels, 35%)
- Baseline comparison: `phase0-gemini-3-pro-preview-20260220-224941` (native labels, 75%)

## Results

| Model | Native Labels | Eligible Labels | Change |
|-------|--------------|-----------------|--------|
| MedGemma 4B | 35% acc, 0.030 kappa | 35% acc, -0.032 kappa | No improvement |
| Gemini 3 Pro | 75% acc, 0.657 kappa | **45% acc**, 0.257 kappa | **-30% regression** |
| GPT-4 (baseline) | 75% | 75% | N/A (precomputed) |

### Verdict Distribution (MedGemma, eligible labels)

| Verdict | Count | Percentage |
|---------|-------|------------|
| MET | 14 | 70% |
| NOT_MET | 3 | 15% |
| UNKNOWN | 3 | 15% |

MET bias worsened from 60% (native) to 70% (eligible).

### Gemini Regression Root Cause

Gemini output 13/20 UNKNOWN with eligible labels (was well-distributed with native labels). Inspection shows truncated JSON fragments like `{   "label": "` in reasoning — Gemini's JSON formatting changed with the new label options, causing parse failures that fell back to keyword extraction matching "unknown" in the reasoning text.

## Decision

**Reject** the eligible label approach. Revert to native TrialGPT labels.

## Consequences

- MedGemma 4B's MET bias is a model capacity issue, not a label semantics issue
- Gemini 3 Pro is sensitive to label vocabulary changes — keep proven native labels
- Future accuracy improvements must target model capacity (27B) or post-processing, not prompt label engineering
