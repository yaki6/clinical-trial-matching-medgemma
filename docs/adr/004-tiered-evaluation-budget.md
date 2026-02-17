# ADR-004: Tiered Evaluation Budget Strategy

**Status:** Accepted
**Date:** 2026-02-17
**Decision Makers:** CTO

## Context

TREC 2021 qrels contain 35,832 pairs. With whole inclusion/exclusion criteria blocks (not atomized) × 2 models, cost is more manageable but still needs tiering.

**Amendment (2026-02-17):** Updated from 3 models to 2 models (MedGemma 1.5 4B + Gemini 3 Pro). Updated from TREC 2022 to TREC 2021. Spike phase uses whole criteria blocks instead of atomized individual criteria.

## Decision

**Three-tier evaluation strategy:**

| Tier | Pairs | Models | Est. Cost | Purpose |
|------|-------|--------|-----------|---------|
| A | 100 (stratified) | Both 2 | ~$30 | Criterion-level accuracy against SoT |
| B | 1,000 (stratified) | Both 2 | ~$300 | Trial-level accuracy with statistical power |
| C | All ~35K (optional) | Best 1 | ~$4,000 | Comprehensive metrics if results warrant |

**Total budget: Tier A+B = ~$330.**

## Rationale

- 100 pairs gives meaningful criterion-level signal against human SoT
- 1,000 pairs provides adequate statistical power to detect ≥ 3pp accuracy differences (α=0.05, power=0.80)
- Tier C only if results are paper-worthy
- Whole criteria blocks in spike phase reduces per-pair cost (1 call per inclusion + 1 per exclusion, not 10-15 per atomized criterion)

## Stratified Sampling for Tier A

- 35 eligible pairs (qrel = 2)
- 35 excluded pairs (qrel = 1)
- 30 not-relevant pairs (qrel = 0)
- Balanced across cancer vs non-cancer topics
