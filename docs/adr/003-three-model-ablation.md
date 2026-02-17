# ADR-003: Two-Model Ablation Design

**Status:** Accepted (amended 2026-02-17)
**Date:** 2026-02-17
**Decision Makers:** CTO

## Context

Need to determine whether MedGemma (medical-specialized) adds value over Gemini 3 Pro (general-purpose) for clinical trial matching.

**Amendment (2026-02-17):** Originally three models. MedGemma 1.0 27B dropped — requires A100 GPU (unavailable), is v1.0 (not latest), and adds complexity without clear strategic value for a spike. The spike question is "does medical specialization matter?" which is answered by 4B vs Gemini.

## Decision

**Run two models on every component for ablation:**

| Model | Type | Parameters | Context | Notes |
|-------|------|-----------|---------|-------|
| MedGemma 1.5 4B | Open-weight, medical | 4B | 128K in, 8K out | Multimodal. HF Inference Endpoint (verified). |
| Gemini 3 Pro | Proprietary, general | N/A | 1M in, 64K out | Google AI Studio. GOOGLE_API_KEY auth. |

## Rationale

- MedGemma 1.5 4B: tests if a small medical-specialized model can compete on clinical reasoning
- Gemini 3 Pro: general-purpose baseline, represents "just use the best available model"
- 27B dropped: A100 GPU not available, v1.0 (older architecture), and the core spike question (medical specialization value) is answered by 4B alone

## Known Risks

- MedGemma 1.5 is 4B — may lack reasoning capacity for complex multi-criterion evaluation
- MedGemma has no tool-use capability
- MedGemma MedQA benchmark = 69% (4B), concerning for complex reasoning
- MedGemma is prompt-sensitive

## Phase 0 Exit Criteria

Phase 0 uses directional signal + qualitative error pattern analysis (n=20 is too small for statistical significance). All model responses must log full reasoning chains for qualitative review.

| Outcome | Decision |
|---------|----------|
| MedGemma shows directional advantage + better error patterns | Proceed to Phase 1 |
| MedGemma ties Gemini (similar accuracy and error types) | Proceed to Phase 1 to collect more data |
| MedGemma clearly worse (lower accuracy + worse error patterns) | Report negative result. Phase 1 optional — only if INGEST shows MedGemma advantage. |
