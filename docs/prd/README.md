# PRD Archive

This directory contains the Product Requirements Documents for the MedGemma 1.5 4B vs Gemini 3 Pro criterion matching benchmark.

## Documents

- `v3.1-criterion-matching-benchmark.md` — Current PRD (v3.1). Scoped-down spike focusing on criterion-level matching with whole criteria blocks, TREC 2021 qrels, 2-model ablation.

## Key Decisions from PRD

1. **Qrels-only strategy** — No full TREC corpus download. Use TREC 2021 qrel patient-trial pairs directly (35,832 triples, 75 topics).
2. **Two-phase approach** — Phase 0 (20 pairs, directional + qualitative check) -> Phase 1 (stratified samples, full benchmark).
3. **Component isolation** — Each component (INGEST/PRESCREEN/VALIDATE) tested with gold upstream inputs.
4. **Two-model ablation** — MedGemma 1.5 4B / Gemini 3 Pro. MedGemma 1.0 27B dropped (A100 unavailable).
5. **Whole criteria blocks** — No atomization in spike phase. Tests medical reasoning, not criteria parsing.
6. **Qualitative Phase 0 exit criteria** — Directional signal + error pattern analysis (n=20 too small for statistical significance).
