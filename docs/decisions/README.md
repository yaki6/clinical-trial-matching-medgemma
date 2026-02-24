# Decision Log

Quick-reference table of all architectural and strategic decisions. Full details in `docs/adr/`.

| # | Decision | Status | Revisit When |
|---|----------|--------|-------------|
| 001 | Qrels-only eval, TREC 2021 (not 2022) | Accepted (amended) | Phase 1 results justify pipeline |
| 002 | Gold SoT inputs for component isolation | Accepted | E2E-best run tests propagation |
| 003 | Three-model ablation (MedGemma 1.5 4B + 27B + Gemini 3 Pro) | Accepted (amended) | Phase 0 exit criteria |
| 004 | Tiered evaluation budget (A+B = ~$330) | Accepted (amended) | If results are paper-worthy → Tier C |
| 005 | Whole criteria blocks for spike (no atomization) | Accepted | Phase 0 too coarse to differentiate models |
| 006 | Switch to TrialGPT HF criterion-level annotations | Accepted | Need trial-level eval (Tier B) or broader coverage |
| 007 | TGI CUDA bug — max_tokens=512 workaround for MedGemma 1.5 4B | Accepted | Vertex AI confirmed no TGI bug; 27B uses max_tokens=2048 on Vertex |
| 008 | Eligible label experiment: simplify to eligible/not eligible/unknown | **Rejected** | N/A — experiment disproved hypothesis. See `docs/medgemma-4b-reasoning-analysis.md` |
| 009 | MedGemma clinical reasoning pre-search step in PRESCREEN | Accepted | MedGemma 27B available for pre-search, or Gemini gains sufficient medical domain knowledge |
| 010 | CT.gov study type filtering via AREA[StudyType] Essie syntax (not filter.studyType) | Accepted | CT.gov API v2 adds a proper `filter.studyType` parameter |
| 011 | Comment out normalize_medical_terms tool (~25s/call, near-zero value) | Accepted | Better MedGemma prompts or lightweight vocabulary lookup (UMLS/MeSH) available |

## From PRESCREEN Implementation (2026-02-22)

| Decision | Rationale | Revisit When |
|----------|-----------|-------------|
| MedGemma 1.5 4B for clinical reasoning pre-search (ADR-009) | Provides domain-specific guidance (molecular drivers, condition terms) that Gemini alone lacks; complementary architecture for competition narrative | MedGemma 27B fast inference available |
| AREA[StudyType]Interventional in query.term (ADR-010) | filter.studyType is not a valid CT.gov API v2 parameter; returns 400 errors | CT.gov API adds filter.studyType |
| Comment out normalize_medical_terms (ADR-011) | ~25s latency per call with near-zero value (MedGemma echoes input); replaced by clinical reasoning pre-search | Better prompts or vocabulary lookup |
| Heuristic candidate scoring: query_count*3 + bonuses | Simple, interpretable ranking; phase II/III (+2), RECRUITING (+1), fetched details (+2); MAX_CANDIDATES=20 | ML-based scoring or LLM re-ranking |
| Two-stage evaluation: MedGemma reasoning + Gemini labeling | 80% accuracy vs 35% (MedGemma alone) or 75% (Gemini alone); combines domain reasoning with reliable structured output | MedGemma improves instruction-following for JSON |
| Demographics promotion in profile_adapter.py | Nested dict age/sex values promoted to top-level keys for CT.gov API filter injection | INGEST produces flat key_facts natively |

## From CTO Review (2026-02-17)

| Decision | Rationale | Revisit When |
|----------|-----------|-------------|
| ~~Drop MedGemma 1.0 27B~~ | ~~A100 GPU unavailable~~ | Reversed 2026-02-20: deployed MedGemma 27B via HF Inference Endpoint (TGI on A100 80GB). Now 3-way comparison. |
| ~~TREC 2021 only (not 2022)~~ | ~~Full ir_datasets support; 75 topics vs 50~~ | Superseded by ADR-006: TrialGPT HF |
| TrialGPT HF criterion-level data (ADR-006) | Criterion-level granularity; self-contained; GPT-4 baseline built-in; < 5 MB | Need trial-level eval for Tier B |
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
