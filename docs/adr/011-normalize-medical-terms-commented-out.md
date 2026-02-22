# ADR-011: Commenting Out normalize_medical_terms Tool

**Status:** Accepted
**Date:** 2026-02-22
**Decision Makers:** Claude + Yaqi

## Context

The PRESCREEN agent originally had three tools available to Gemini:
1. `search_trials` — CT.gov API v2 search
2. `get_trial_details` — full eligibility criteria fetch
3. `normalize_medical_terms` — MedGemma-powered medical term normalization

The `normalize_medical_terms` tool called MedGemma 4B to produce CT.gov-optimized search variants for medical terms (e.g., normalizing "NSCLC" to "non-small cell lung cancer" with synonyms and disambiguation).

### Problems Observed

1. **Near-zero value**: In live testing, MedGemma 4B echoed the input term back unchanged in most cases, providing no additional search variants.
2. **High latency**: Each call added ~25 seconds due to MedGemma inference time, significantly slowing the agentic loop.
3. **Budget consumption**: Each normalize call counted against the agent's `max_tool_calls` budget (default 8), reducing the number of actual search_trials calls the agent could make.
4. **Redundancy with clinical reasoning**: The new MedGemma clinical reasoning pre-search step (ADR-009) provides better search term guidance upfront, making per-term normalization redundant.

## Decision

Comment out (not delete) the `normalize_medical_terms` tool declaration, system prompt, and executor method. The tool is preserved in source code for potential future reactivation with better MedGemma prompts.

### What Was Commented Out

- `tools.py`: `_NORMALIZE_TERMS_DECL` function declaration, `MEDGEMMA_NORMALIZE_SYSTEM` prompt, `MEDGEMMA_NORMALIZE_USER` template, `_normalize_medical_terms()` executor method
- `tools.py`: `PRESCREEN_TOOLS` now only includes `search_trials` and `get_trial_details`
- `agent.py`: Docstring updated to reflect two tools instead of three

### What Was Preserved

- `ToolExecutor` still accepts `medgemma` parameter (backward compatibility)
- `ToolExecutor.medgemma_calls` and `medgemma_cost` counters still exist (tracked via clinical reasoning in agent.py)
- All commented-out code includes notes explaining why it was disabled and when to revisit

## Rationale

- The tool's cost/benefit ratio was strongly negative (25s latency, ~$0.05/call, near-zero improvement to search quality)
- Clinical reasoning pre-search (ADR-009) provides a superior alternative — domain guidance is injected once upfront rather than per-term
- Commenting out (vs. deleting) preserves the implementation for future experimentation with better prompts or a more capable MedGemma model

## Consequences

- **Pro:** PRESCREEN agent runs faster (eliminates 25s+ per normalize call)
- **Pro:** More tool call budget available for actual CT.gov searches
- **Pro:** Simpler agent loop (2 tools instead of 3)
- **Con:** No per-term medical normalization — Gemini must generate correct CT.gov terms on its own
- **Con:** Commented-out code adds visual noise to `tools.py`

## Revisit When

- MedGemma 27B is available as a fast inference endpoint (may produce better normalizations)
- Better prompts are developed for medical term normalization
- A lightweight medical vocabulary lookup (e.g., UMLS/MeSH) replaces the LLM-based approach
