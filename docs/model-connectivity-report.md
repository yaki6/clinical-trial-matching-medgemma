# Model Connectivity Report: MedGemma 4B & Gemini 3 Pro

> **Date:** 2026-02-18 (updated 2026-02-23)
> **Status:** Both endpoints verified and operational
> **Default deployment:** Vertex AI Model Garden (HF Inference is legacy fallback — unstable)
> **Next step:** Implement `src/trialmatch/models/` adapters using TDD

---

## 1. Summary

Both model endpoints have been validated with medical diagnostic prompts. This document captures the connection details, test scripts, findings, and integration recommendations for the `models/` adapter layer.

| Model | Endpoint | Auth | Status | Latency |
|-------|----------|------|--------|---------|
| MedGemma 1.5 4B | HF Inference Endpoint | `HF_TOKEN` | Ready | ~3-5s |
| Gemini 3 Pro (preview) | Google AI Studio | `GEMINI_API_KEY` | Ready | ~4-8s (incl. thinking) |

---

## 2. MedGemma 1.5 4B

> **NOTE:** HF Inference Endpoints are unstable for MedGemma (TGI CUDA bugs,
> max_tokens=512 limit). **Use Vertex AI Model Garden as the default.**
> See `scripts/deploy_vertex_4b.py` for Vertex deployment.

### 2.1 HF Endpoint Details (LEGACY FALLBACK)

| Property | Value |
|----------|-------|
| URL | `https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud` |
| Model | `google/medgemma-1-5-4b-it-hae` (4B param, multimodal) |
| Auth | `HF_TOKEN` env var (Bearer token) |
| API style | Raw `text_generation()` — NOT TGI, `/v1/chat/completions` returns 404 |
| Concurrency | 5 concurrent (HF Inference limit) |
| Cold-start | Endpoint may sleep; returns 503 — retry with exponential backoff |
| **Stability** | **Unstable** — CUDA CUBLAS crashes with max_tokens > 512 |

### 2.2 Critical Constraints

1. **No chat completions API** — must use `text_generation()` with manual Gemma chat template
2. **Gemma template** — system prompt folds into first user turn (no native system role)
3. **Cold-start 503** — endpoint pauses after inactivity; budget 60s for retry
4. **Output parsing** — response includes the full prompt echo; must split on `<start_of_turn>model`

### 2.3 Verified Test Script

```python
import asyncio
import os
import sys

sys.path.insert(0, ".claude/skills/medgemma-endpoint/scripts")
from medgemma_client import MedGemmaClient

async def test_medgemma():
    client = MedGemmaClient(
        endpoint_url="https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud",
        hf_token=os.environ["HF_TOKEN"],
    )

    # Health check
    status = await client.health_check()
    assert status["status"] == "ready", f"Endpoint not ready: {status}"

    # Medical diagnosis generation
    result = await client.generate(
        messages=[
            {"role": "system", "content": "You are a medical diagnostic assistant."},
            {"role": "user", "content": "What is hypertension? Reply in one sentence."},
        ],
        max_tokens=128,
    )

    # Parse: strip prompt echo
    if "<start_of_turn>model" in result:
        result = result.split("<start_of_turn>model")[-1].strip()

    assert len(result) > 0, "Empty response"
    print(f"MedGemma OK: {result[:100]}")

asyncio.run(test_medgemma())
```

### 2.4 Reusable Client Location

The drop-in async client with retry logic lives at:
```
.claude/skills/medgemma-endpoint/scripts/medgemma_client.py
```

For pipeline integration, copy into `src/trialmatch/models/medgemma_adapter.py` and wrap with the common `ModelAdapter` interface.

---

## 3. Gemini 3 Pro (Preview)

### 3.1 Endpoint Details

| Property | Value |
|----------|-------|
| Model ID | `gemini-3-pro-preview` |
| SDK | `google-genai>=1.0` (already in `pyproject.toml`) |
| Auth | `GEMINI_API_KEY` env var |
| API style | `client.models.generate_content()` — standard chat API |
| Concurrency | 10 concurrent (AI Studio limit) |
| Thinking mode | **Required** — `thinking_budget=0` returns 400 error |

### 3.2 Critical Constraints

1. **Thinking mode mandatory** — Gemini 3 Pro only works with `ThinkingConfig`; minimum budget ~128 tokens
2. **Token accounting** — response includes `prompt_token_count`, `candidates_token_count`, and `thoughts_token_count` (thinking tokens are billed)
3. **Preview model** — may change with 2 weeks notice; use `gemini-2.5-pro` for stable production

### 3.3 Verified Test Script

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

config = types.GenerateContentConfig(
    max_output_tokens=4096,
    thinking_config=types.ThinkingConfig(thinking_budget=1024),
)

response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="You are a medical diagnostic assistant.\n\nDiagnose this patient: ...",
    config=config,
)

print(response.text)

# Token usage
um = response.usage_metadata
print(f"Input: {um.prompt_token_count}")
print(f"Output: {um.candidates_token_count}")
print(f"Thinking: {um.thoughts_token_count}")
```

---

## 4. Diagnostic Comparison: MedGemma vs Gemini 3 Pro

Two patient cases were tested with identical system prompts.

### 4.1 Patient 1: 2-year-old boy (Classic Kawasaki Disease)

| Aspect | MedGemma 4B | Gemini 3 Pro |
|--------|-------------|--------------|
| **Primary Dx** | Kawasaki Disease | Kawasaki Disease |
| **Key findings cited** | Fever >5d, conjunctivitis, strawberry tongue, desquamation, lymphadenopathy, sterile pyuria, coronary dilation | Same + explicit AHA criteria mapping |
| **Differentials** | Viral exanthems, Scarlet fever, SJS, TSS | Scarlet fever, MIS-C, Adenovirus |
| **Output tokens** | ~200 | 178 (+1182 thinking) |

**Assessment:** Both correct. Gemini 3 Pro notably included MIS-C as a differential (clinically relevant post-COVID). MedGemma included SJS/TSS (broader but less specific).

### 4.2 Patient 2: 75F Septic Shock

| Aspect | MedGemma 4B | Gemini 3 Pro |
|--------|-------------|--------------|
| **Primary Dx** | Sepsis secondary to Klebsiella UTI + AKI + possible adrenal insufficiency | Septic Shock secondary to GAS bacteremia (source: skin/soft tissue via PVD) |
| **Key findings cited** | Hypoglycemia, hypotension, leukocytosis, elevated creatinine, anuria | Same + linked leg rash to PVD as GAS entry point |
| **Differentials** | Adrenal insufficiency, PVD worsening, MI | Adrenal crisis, Necrotizing fasciitis, Cardiogenic shock |
| **Output tokens** | ~250 | 257 (+1254 thinking) |

**Assessment:** Gemini 3 Pro demonstrated stronger clinical reasoning — correctly identified GAS bacteremia as the primary driver (not UTI), linked the leg rash + PVD as the likely entry point, and flagged necrotizing fasciitis (a critical consideration for GAS + skin involvement). MedGemma focused on the Klebsiella UTI as primary source, which is less precise given the blood culture results.

### 4.3 Summary Observations

| Dimension | MedGemma 4B | Gemini 3 Pro |
|-----------|-------------|--------------|
| Diagnostic accuracy | Correct primary Dx both cases | Correct + more nuanced |
| Clinical reasoning depth | Good, somewhat conservative | Stronger causal chain linking |
| Differential quality | Broader, less specific | More clinically actionable |
| Token efficiency | ~200-250 output tokens | ~180-260 + ~1200 thinking |
| Cost per call (est.) | ~$0.001 (HF endpoint) | ~$0.01-0.02 (AI Studio) |
| Latency | 3-5s | 4-8s (thinking overhead) |

---

## 5. Integration Plan for `src/trialmatch/models/`

### 5.1 Target Architecture

```
src/trialmatch/models/
├── __init__.py          # Export ModelAdapter protocol + factory
├── base.py              # ModelAdapter Protocol + ModelResponse dataclass
├── medgemma.py           # MedGemmaAdapter(ModelAdapter)
├── gemini.py            # GeminiAdapter(ModelAdapter)
└── factory.py           # get_model(name: str) -> ModelAdapter
```

### 5.2 Common Interface (from architecture doc)

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost: float
    model_name: str
    thinking_tokens: int = 0  # Gemini-specific

class ModelAdapter(Protocol):
    async def generate(self, prompt: str, **kwargs) -> ModelResponse: ...
    async def health_check(self) -> dict: ...
```

### 5.3 MedGemma Adapter Notes

- Wrap `MedGemmaClient` from skill script
- Parse output: split on `<start_of_turn>model`, strip template artifacts
- Token counting: HF Inference API does NOT return token counts — estimate from character length or use tokenizer
- Cost estimation: based on HF endpoint pricing (per-second compute)

### 5.4 Gemini Adapter Notes

- Use `google.genai.Client` directly
- `ThinkingConfig(thinking_budget=1024)` as default — tune per component
- Token counting: native via `response.usage_metadata`
- Cost estimation: `prompt_token_count * input_price + candidates_token_count * output_price + thoughts_token_count * thinking_price`
- Structured JSON output available via `response_mime_type="application/json"` + Pydantic schema

### 5.5 Environment Variables Required

```bash
# MedGemma
export HF_TOKEN="hf_..."
export MEDGEMMA_ENDPOINT_URL="https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud"

# Gemini
export GEMINI_API_KEY="AIza..."
export GEMINI_MODEL="gemini-3-pro-preview"
```

---

## 6. TDD Test Plan for `models/` Module

### 6.1 Unit Tests (mocked, fast)

| Test | What it verifies |
|------|-----------------|
| `test_medgemma_format_gemma_prompt` | System prompt folds into first user turn correctly |
| `test_medgemma_parse_output` | `<start_of_turn>model` split works; handles edge cases |
| `test_medgemma_retry_on_503` | Exponential backoff fires on cold-start; respects budget |
| `test_medgemma_fail_on_4xx` | 400/401/404 errors raise immediately (no retry) |
| `test_gemini_thinking_config` | ThinkingConfig is always included (budget > 0) |
| `test_gemini_token_tracking` | usage_metadata mapped to ModelResponse correctly |
| `test_model_response_cost_calc` | Estimated cost computed from token counts + price table |
| `test_factory_returns_correct_adapter` | `get_model("medgemma")` / `get_model("gemini")` return right types |

### 6.2 Integration Tests (live API, `@pytest.mark.integration`)

| Test | What it verifies |
|------|-----------------|
| `test_medgemma_health_check_live` | Endpoint responds with `status: ready` |
| `test_medgemma_generate_medical` | Generates coherent medical text from clinical prompt |
| `test_gemini_health_check_live` | AI Studio API reachable and authenticated |
| `test_gemini_generate_medical` | Generates coherent medical text with token counts |
| `test_both_models_same_prompt` | Both models return valid ModelResponse for identical input |

### 6.3 BDD Scenarios (from existing feature template)

```gherkin
Feature: Model adapters for clinical trial matching

  Scenario: MedGemma generates a medical diagnosis
    Given a MedGemma adapter configured with valid credentials
    When I send a clinical patient description
    Then I receive a ModelResponse with non-empty text
    And the response contains a primary diagnosis
    And the ModelResponse includes latency_ms > 0

  Scenario: Gemini generates a medical diagnosis
    Given a Gemini adapter configured with valid credentials
    When I send the same clinical patient description
    Then I receive a ModelResponse with non-empty text
    And the ModelResponse includes input_tokens > 0
    And the ModelResponse includes thinking_tokens > 0

  Scenario: Model factory resolves adapters by name
    Given a configured environment with both API keys
    When I request get_model("medgemma")
    Then I receive a MedGemmaAdapter instance
    When I request get_model("gemini")
    Then I receive a GeminiAdapter instance
```

---

## 7. Known Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MedGemma endpoint pauses after inactivity | Tests fail with 503 on first call | Warm-up call in test fixtures; 60s retry budget |
| Gemini 3 Pro is preview-only | API may change | Pin to `gemini-3-pro-preview`; fallback to `gemini-2.5-pro` |
| MedGemma lacks native token counting | Cost tracking inaccurate | Estimate from char length or load tokenizer locally |
| Gemini thinking tokens add cost | Budget overruns in Tier B/C | Cap `thinking_budget` per component; track in run artifacts |
| HF endpoint is shared/single-tenant | Contention under load | Queue with semaphore (max 5 concurrent) |

---

## 8. Resolved Blockers (from DASHBOARD.md)

| Blocker | Resolution |
|---------|-----------|
| MedGemma endpoint activation status unknown | **Resolved** — endpoint active, health check passes |
| GOOGLE_API_KEY availability for Gemini | **Resolved** — API key validated, Gemini 3 Pro operational |
