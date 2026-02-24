# ADR-007: TGI CUDA Bug — max_tokens=512 Workaround for MedGemma 1.5 4B

**Status:** Accepted
**Date:** 2026-02-21
**Decision Makers:** Claude + Yaqi

## Context

During Phase 0 benchmark runs, the HuggingFace Inference Endpoint (TGI backend) for MedGemma 1.5 4B crashes with `CUDA CUBLAS_STATUS_EXECUTION_FAILED` on specific prompt + max_new_tokens combinations. After the initial crash, the GPU enters a permanent "misaligned address" state where all subsequent requests fail until the endpoint restarts (via scale-to-zero + resume).

### Investigation Results

- **max_new_tokens threshold**: Binary search confirmed 500 works, 1024 crashes. Threshold is between 500-1024.
- **NOT hardware-related**: Identical crash on Nvidia L4 (24GB, $1/h) and L40S (48GB, $1.8/h).
- **NOT cumulative memory leak**: Crashes as first request on a fresh GPU (no prior requests needed).
- **NOT prompt length**: All prompts are ~500 tokens, similar sizes. Input processing always succeeds (max_new_tokens=1 works).
- **Prompt-specific**: Some specific prompts trigger the crash reproducibly at high max_new_tokens; others never crash.
- **Generation-phase crash**: Occurs during output token generation, not during input encoding.

## Decision

Set `max_tokens=512` as the default in `evaluate_criterion()` to avoid TGI CUDA crashes. Accept the accuracy tradeoff.

## Rationale

- max_tokens=512 is the highest safe value below the crash threshold (~500-1024 range).
- MedGemma 1.5 4B's "thinking tokens" (`<unused94>thought...`) consume significant output budget, so 512 tokens often truncates the thinking chain before reaching the JSON output.
- The alternative (max_tokens=2048) gives 55% accuracy but crashes on ~30% of prompts, making benchmark runs unreliable.
- This is a workaround for a TGI bug, not a fundamental model limitation.

## Consequences

- **Pro:** Benchmark runs complete reliably (20/20 pairs, zero crashes)
- **Pro:** No endpoint restarts needed during benchmark
- **Con:** Accuracy dropped from 55% (max_tokens=2048) to 35% (max_tokens=512)
- **Con:** Truncated thinking chains force keyword fallback parsing instead of structured JSON
- **Con:** MedGemma 1.5 4B's MET bias may be partially exacerbated by truncation

## Revisit When

- TGI updates fix the CUDA bug (check TGI release notes)
- Vertex AI Model Garden provides a stable MedGemma 1.5 4B endpoint (no TGI dependency)
- vLLM or other serving frameworks become available on HF Inference Endpoints
