# ADR-001: Qrels-Only Evaluation (No Full Retrieval Pipeline)

**Status:** Accepted (amended by ADR-006)
**Date:** 2026-02-17
**Decision Makers:** CTO

## Context

The original v2.0 PRD proposed building a full retrieval pipeline (PyTerrier + MedCPT dense index, 375K trial corpus, NDCG@10 evaluation) alongside criterion-level matching. This is ~2 weeks of eng effort and ~$500 in compute.

## Decision

**Defer the full retrieval pipeline. Evaluate criterion matching directly against TREC qrels.**

TREC 2021 qrels provide 35,832 (topic_id, NCT_ID, relevance) triples across 75 topics and 26,162 unique NCT IDs. We fetch only the trials referenced in qrels, not the full corpus.

**Amendment (2026-02-17):** Switched from TREC 2022 to TREC 2021 qrels. TREC 2021 has full support in `ir_datasets` (topics + qrels), while TREC 2022 qrels require manual download. TREC 2021 also has 75 topics (vs 50), providing more selection space for Phase 0. Relevance distribution: 0=24,243, 1=6,019, 2=5,570.

## Rationale

- The retrieval pipeline only pays off if MedGemma's criterion-level reasoning beats Gemini 3 Pro
- If MedGemma doesn't win on reasoning, the pipeline investment is wasted
- This version inverts the order: prove reasoning first (cheap), then invest in pipeline (expensive)
- TREC 2021 is fully available via `ir_datasets` — no manual data wrangling needed

## Consequences

- **Pro:** 10x cheaper and faster to validate the core hypothesis
- **Pro:** Can still compute trial-level accuracy and macro F1 against qrels
- **Pro:** TREC 2021 natively supported in ir_datasets (topics + qrels + docs)
- **Con:** Cannot compute NDCG@10 or ranking metrics
- **Con:** Cannot evaluate retrieval recall

## Amendment (2026-02-19, ADR-006)

Phase 0 and Tier A now use TrialGPT HF criterion-level annotations instead of TREC 2021 trial-level qrels. TREC qrels deferred to Tier B for trial-level ranking evaluation. The qrels-only approach (no retrieval pipeline) still holds — ADR-006 changes the ground truth source, not the evaluation strategy.

## Revisit When

Phase 1 results justify pipeline investment (MedGemma shows directional advantage).
