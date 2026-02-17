---
name: medgemma-endpoint
description: >
  Connect to and use the shared MedGemma HF Inference Endpoint
  (google/medgemma-1-5-4b-it-hae, multimodal medical AI).
  Use when any project needs to: (1) call the MedGemma endpoint for
  medical text/image extraction or clinical reasoning, (2) integrate
  MedGemma into a new service or pipeline, (3) debug MedGemma
  connection issues (503 cold-start, 404 chat/completions).
  Triggers: "medgemma", "medical extraction", "HF endpoint",
  "patient profile extraction", "clinical criterion evaluation".
---

# MedGemma HF Endpoint

## Endpoint

- **URL**: `https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud`
- **Model**: `google/medgemma-1-5-4b-it-hae` (4B param, multimodal)
- **Auth**: `HF_TOKEN` env var required

## Critical Constraints

1. **NOT TGI** — `/v1/chat/completions` returns 404. Use `text_generation()` with manual Gemma chat template.
2. **Cold-start 503** — Endpoint may sleep. Retry with exponential backoff (2^attempt seconds, max 60s, budget 60s).
3. **Gemma template** — System prompt folds into first user turn. No native system role.

## Quick Start

Install dependency:
```bash
uv add huggingface_hub
```

Copy the client into your project:
```bash
cp ~/.claude/skills/medgemma-endpoint/scripts/medgemma_client.py your_project/services/
```

Usage:
```python
import os
from your_project.services.medgemma_client import MedGemmaClient

client = MedGemmaClient(
    endpoint_url="https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud",
    hf_token=os.environ["HF_TOKEN"],
)

# Health check
status = await client.health_check()

# Generate
result = await client.generate(
    messages=[
        {"role": "system", "content": "You are a medical assistant."},
        {"role": "user", "content": "Summarize this pathology report: ..."},
    ],
    max_tokens=2048,
)

# Parse JSON output (handles markdown-wrapped responses)
data = MedGemmaClient.parse_json(result)
```

## Gemma Chat Template

Handled automatically by `format_gemma_prompt()`. Manual format:
```
<start_of_turn>user
[system prompt here]

[user message here]<end_of_turn>
<start_of_turn>model
```

## Retry Configuration

Override defaults via constructor:
```python
client = MedGemmaClient(
    endpoint_url="...",
    hf_token="...",
    max_retries=6,
    retry_backoff=2.0,
    max_wait=60.0,
    cold_start_timeout=60.0,
)
```

## Resources

- **`scripts/medgemma_client.py`** — Drop-in async client with retry logic and Gemma template formatting. Copy into any project.
- **`references/endpoint-details.md`** — Full endpoint specs, error codes, env var reference, and common errors table.
