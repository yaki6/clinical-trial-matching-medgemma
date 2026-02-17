# Decision Log

Quick-reference table of all architectural and strategic decisions. Full details in `docs/adr/`.

| # | Decision | Status | Revisit When |
|---|----------|--------|-------------|
| 001 | Qrels-only eval, TREC 2021 (not 2022) | Accepted (amended) | Phase 1 results justify pipeline |
| 002 | Gold SoT inputs for component isolation | Accepted | E2E-best run tests propagation |
| 003 | Two-model ablation (MedGemma 1.5 4B + Gemini 3 Pro) | Accepted (amended) | Phase 0 exit criteria |
| 004 | Tiered evaluation budget (A+B = ~$330) | Accepted (amended) | If results are paper-worthy → Tier C |
| 005 | Whole criteria blocks for spike (no atomization) | Accepted | Phase 0 too coarse to differentiate models |

## From CTO Review (2026-02-17)

| Decision | Rationale | Revisit When |
|----------|-----------|-------------|
| Drop MedGemma 1.0 27B | A100 GPU unavailable; spike answered by 4B vs Gemini | Never (unless 27B becomes easily available) |
| TREC 2021 only (not 2022) | Full ir_datasets support; 75 topics vs 50; no manual download | Need 2022 for separate eval set |
| Whole criteria blocks (no atomization) | Isolates medical reasoning from parsing; reduces annotation burden | Need per-criterion granularity |
| Phase 0 = directional + qualitative | n=20 too small for statistical significance; log full reasoning chains | Phase 1 for statistical power |
| Gemini via Google AI Studio | GOOGLE_API_KEY auth; simpler than Vertex AI | Need batch pricing or higher rate limits |
| VALIDATE is first implementation priority | Core spike question is medical reasoning quality | After baseline established |

## From PRD v3.0 Decision Log

| Decision | Rationale | Revisit When |
|----------|-----------|-------------|
| 100 pairs for criterion SoT | Balances annotation cost vs statistical power | If inter-annotator κ < 0.7 |
| 1,000 pairs for trial-level eval | Adequate power for 3pp accuracy detection | Tier C if paper-worthy |
| Deferred: PyTerrier, BioMCP, NDCG@10 | Retrieval-pipeline concerns, premature for reasoning spike | Phase 1 shows MedGemma advantage |
