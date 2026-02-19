# ADR-006: Switch to TrialGPT HF Criterion-Level Annotations

**Status:** Accepted
**Date:** 2026-02-19
**Decision Makers:** Human + Claude (research team evaluation)

## Context

The original Phase 0 plan (ADR-001) used TREC 2021 qrels as ground truth. These qrels are **trial-level** (patient × trial → 0/1/2), meaning individual criterion reasoning errors are hidden in aggregate verdicts. The plan's own Known Limitations section flagged this:

> "Trial-level labels only: TREC qrels are trial-level (not criterion-level). We test overall matching, not individual criterion reasoning."

A research evaluation of `ncbi/TrialGPT-Criterion-Annotations` on HuggingFace revealed it provides **criterion-level** annotations (patient × single criterion → label) with expert ground truth, GPT-4 baseline predictions, and evidence sentence markers — exactly what the VALIDATE component needs.

## Decision

**Replace TREC 2021 qrels + corpus.jsonl + queries.jsonl with the TrialGPT HuggingFace criterion-level dataset as the sole Phase 0 data source.**

Key changes:
- Data source: `datasets.load_dataset("ncbi/TrialGPT-Criterion-Annotations")` — one call, < 5 MB
- Granularity: criterion-level (1,024 patient-criterion pairs) instead of trial-level (35,832 patient-trial pairs)
- Ground truth: `expert_eligibility` field (physician-annotated)
- Labels: 6-class HF → 3-class mapping (MET / NOT_MET / UNKNOWN)
- Built-in baseline: `gpt4_eligibility` column provides GPT-4 predictions at zero cost
- Module rename: `trialgpt_loader.py` → `hf_loader.py`

Eliminated:
- FTP download of `corpus.jsonl` (131 MB) or `trial_info.json` (1.1 GB)
- Download of `queries.jsonl` (100 KB) and `qrels/test.tsv` (1 MB)
- `ir_datasets` dependency for TREC data loading
- Trial corpus parsing and NCT ID joining logic

## Rationale

| Factor | TREC 2021 (old) | TrialGPT HF (new) |
|--------|-----------------|---------------------|
| Annotation granularity | Trial-level only | **Criterion-level** |
| Download complexity | 3 files from FTP, 131+ MB | 1 HF call, < 5 MB |
| Built-in baseline | None | GPT-4 (87.3% accuracy) |
| Evidence annotations | None | Sentence-level indices |
| Explanation quality labels | None | Correct/Incorrect/Partially Correct |
| Self-contained | No — needs join across 3 files | **Yes** — all text inline |
| Expert annotations | TREC assessors (trial-level) | Physicians (criterion-level) |

The VALIDATE component's interface is `evaluate_criterion(patient, criterion) → MET/NOT_MET/UNKNOWN`. Criterion-level ground truth directly matches this interface. Trial-level labels require the model to get every criterion right to match the aggregate — masking which criteria caused errors.

## Consequences

- **Pro:** Directly evaluates the VALIDATE component's core function
- **Pro:** Eliminates 131+ MB data download and complex file joining
- **Pro:** Free GPT-4 baseline comparison (no API cost)
- **Pro:** Evidence sentence overlap enables explanation quality metrics
- **Pro:** Simpler data loading code (~50 lines vs ~200 lines)
- **Con:** Smaller dataset scope (53 patients, 103 trials vs 75 topics, 26K trials)
- **Con:** 6→3 class mapping loses some nuance
- **Con:** Single HF split (no train/test separation) — mitigated by using all 1,024 for Tier A
- **Con:** TREC trial-level metrics (ranking, NDCG) deferred to Tier B

## Label Mapping

| HF `expert_eligibility` | → | Our 3-class |
|--------------------------|---|-------------|
| `included` | → | MET |
| `not excluded` | → | MET |
| `excluded` | → | NOT_MET |
| `not included` | → | NOT_MET |
| `not enough information` | → | UNKNOWN |
| `not applicable` | → | UNKNOWN |

## Revisit When

- Phase 0 / Tier A complete and trial-level evaluation needed → add TREC 2021+2022 qrels for Tier B
- Label mapping proves too lossy → consider 4-class or 6-class evaluation
- Need broader patient/trial coverage → supplement with TREC data
