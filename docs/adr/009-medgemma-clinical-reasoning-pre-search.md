# ADR-009: MedGemma Clinical Reasoning Pre-Search Step

**Status:** Accepted
**Date:** 2026-02-22
**Decision Makers:** Claude + Yaqi

## Context

The PRESCREEN agent uses Gemini Flash as the agentic search orchestrator for CT.gov queries. While Gemini excels at structured tool orchestration, it lacks domain-specific clinical reasoning — for example, inferring likely oncogenic driver mutations from demographics and histology (e.g., a young female never-smoker with lung adenocarcinoma has ~50-60% probability of EGFR activating mutation).

Without clinical guidance, Gemini's search strategy relies only on explicit key_facts, missing implicit clinical correlations that an oncologist would know to search for.

## Decision

Add an optional MedGemma 4B clinical reasoning call **before** the Gemini agentic loop. MedGemma generates a structured clinical guidance summary that is injected into Gemini's user prompt as a `## MedGemma Clinical Reasoning` section.

### Implementation

- **Location**: `src/trialmatch/prescreen/agent.py` — `_get_clinical_guidance()` helper
- **Model**: MedGemma 4B (via `medgemma_adapter` parameter)
- **Prompt**: `CLINICAL_REASONING_PROMPT` — asks for:
  1. Standard CT.gov condition terms (2-3 best terms)
  2. Most likely molecular drivers (ranked by probability)
  3. Priority eligibility keywords (3-5 terms)
  4. Treatment line assessment (naive, first-line, later-line)
  5. Clinical phenotype hints
- **Output**: Plain text bullet points injected into Gemini's user message
- **Fallback**: If MedGemma fails, search proceeds without guidance (graceful degradation)
- **Cost**: ~$0.05 per patient, latency 12-43s

### Architecture Flow

```
Patient Note + Key Facts
    |
    v
[MedGemma 4B] Clinical reasoning → guidance text (12-43s, ~$0.05)
    |
    v
[Gemini Flash] Agentic search loop (guided by MedGemma output)
    |
    v
CT.gov API v2 → TrialCandidate[]
```

## Rationale

- **Complementary strengths**: MedGemma provides domain-specific medical knowledge; Gemini provides structured reasoning and tool orchestration. Neither alone achieves what both together can.
- **Competition narrative**: Demonstrates genuine multi-model orchestration for the MedGemma Impact Challenge (targets "agent-based workflows" special award).
- **Low risk**: Pre-search guidance is additive — if MedGemma fails or returns unhelpful guidance, the search proceeds normally.
- **Measurable**: Can A/B test with/without clinical guidance by setting `medgemma_adapter=None`.

## Consequences

- **Pro:** Gemini's search strategy is enriched with implicit clinical correlations
- **Pro:** Demonstrates meaningful MedGemma utilization beyond raw benchmarking
- **Pro:** Graceful degradation — search works without MedGemma
- **Con:** Adds 12-43s latency to PRESCREEN (acceptable for a batch process, visible in live demo)
- **Con:** Extra API cost (~$0.05/patient) — negligible for demo scale

## Revisit When

- MedGemma 27B becomes available for pre-search (may produce better clinical reasoning)
- Gemini gains sufficient medical domain knowledge to not need external guidance
- Fine-tuned MedGemma could replace the prompt-based approach
