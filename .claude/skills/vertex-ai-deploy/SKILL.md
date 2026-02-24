---
name: vertex-ai-deploy
description: Deploy and manage MedGemma 27B on Vertex AI Model Garden. Use when deploying, undeploying, checking status, or smoke-testing the Vertex AI endpoint. Covers the full e2e lifecycle validated Feb 21 2026. Triggers on "deploy vertex", "vertex AI", "model garden", "undeploy vertex", "teardown vertex", "27b endpoint".
---

# Vertex AI MedGemma 27B — E2E Deploy/Undeploy Workflow

## Quick Reference

```bash
# 1. Check current state (ALWAYS do this first)
uv run python scripts/deploy_vertex_27b.py status

# 2. Deploy (reuses Model Registry, deploys to existing endpoint, auto smoke-tests)
uv run python scripts/deploy_vertex_27b.py deploy

# 3. Smoke test a running endpoint (10 attempts, 30s interval)
uv run python scripts/deploy_vertex_27b.py smoke-test

# 4. Undeploy — STOPS BILLING, keeps endpoint at $0/hr
uv run python scripts/deploy_vertex_27b.py undeploy

# 5. Run with auto-undeploy — deploy → run command → undeploy (PREFERRED for benchmarks/tests/pipelines)
uv run python scripts/deploy_vertex_27b.py run -- uv run trialmatch phase0 --config configs/phase0_vertex_27b.yaml
uv run python scripts/deploy_vertex_27b.py run -- uv run pytest tests/e2e/ -m e2e
uv run python scripts/deploy_vertex_27b.py run -- uv run python scripts/my_pipeline.py
```

**CRITICAL: Use `run` for benchmarks, tests, and pipelines.** It auto-undeploys after completion (even on failure/Ctrl-C). 2x L4 = ~$2.30/hr. Forgotten undeploy = wasted money.

## Auto-Undeploy: When to Use What

| Scenario | Command | Auto-undeploy? |
|----------|---------|----------------|
| Benchmark run | `run -- uv run trialmatch phase0 ...` | Yes, always |
| E2E tests | `run -- uv run pytest tests/e2e/ ...` | Yes, always |
| Pipeline execution | `run -- uv run python scripts/...` | Yes, always |
| Interactive demo (keep alive) | `deploy` then later `undeploy` | No, manual |
| Quick check | `status` | N/A |

## Validated E2E Workflow (Feb 21, 2026)

### Deploy Flow (tested, ~18 min total)

```
status → "No models deployed. Endpoint is idle ($0/hr)."
deploy →
  Step 1: Checks Model Registry → "Found existing model, skipping re-upload"
  Step 2: Deploys to endpoint → ~15-20 min (PREPARING → CREATING_CLUSTER → ADDING_NODES → STARTING_SERVER)
  Step 3: Auto smoke-test → "Attempt 1: OK (1371ms)"
```

**Actual timings observed:**
- Model Registry lookup: instant (skips re-upload)
- Deploy to endpoint: ~18 minutes
- First inference latency: 1371ms
- Subsequent requests: ~5-12s (depends on prompt length)

### Undeploy Flow (tested)

```
undeploy → "Done. All models undeployed. Billing stopped ($0/hr)."
status   → "No models deployed. Endpoint is idle ($0/hr)."
```

### Edge Cases (tested)

- `undeploy` on empty endpoint: "No models deployed. Nothing to undeploy." (safe, exit 0)
- `smoke-test` on empty endpoint: "ERROR: No models deployed. Run 'deploy' first." (exit 1)
- `deploy` on endpoint with existing model: "Endpoint already has 1 deployed model(s). Run 'undeploy' first." (exit 1)

## Endpoint Details (Persisted)

| Item | Value |
|------|-------|
| Endpoint ID | `6588061467889631232` (env: `VERTEX_ENDPOINT_ID_27B`) |
| Endpoint name | `medgemma-27b-endpoint` |
| Model display name | `medgemma-27b-it-int8` |
| Model Registry ID | `projects/189529605434/locations/us-central1/models/803697919927517184` |
| Hardware | `g2-standard-24` with 2x NVIDIA L4 (48GB VRAM) |
| Quantization | bitsandbytes int8 (~27GB weights) |
| Dedicated DNS | `6588061467889631232.us-central1-189529605434.prediction.vertexai.goog` |
| Container | `pytorch-vllm-serve:20250430_0916_RC00_maas` |
| Region | `us-central1` |
| Project | `gen-lang-client-0517724223` |

## .env Variables Required

```bash
GCP_PROJECT_ID=gen-lang-client-0517724223
GCP_REGION=us-central1
VERTEX_ENDPOINT_ID_27B=6588061467889631232
VERTEX_DEDICATED_DNS_27B=6588061467889631232.us-central1-189529605434.prediction.vertexai.goog
HF_TOKEN=<required for model download>
```

## Script Internals (`scripts/deploy_vertex_27b.py`)

### `deploy` subcommand
1. `aiplatform.Model.list(filter='display_name="medgemma-27b-it-int8"')` — find existing model
2. If not found, uploads with vLLM int8 config (container args, env vars, ports)
3. Checks endpoint has no deployed models (refuses if occupied)
4. `model.deploy(endpoint=endpoint, ...)` — 15-30 min blocking call
5. Auto-runs smoke test (30 attempts x 30s = 15 min max wait)

### `undeploy` subcommand
- `endpoint.undeploy_all()` — removes all deployed models, GPU billing stops
- Endpoint resource stays alive (no need to recreate)

### `status` subcommand
- Reads `endpoint.gca_resource.deployed_models` for live info
- Shows model ID, display name, machine type, accelerator details, dedicated DNS

### `smoke-test` subcommand
- Creates `VertexMedGemmaAdapter` with endpoint config
- Calls `adapter.health_check()` (sends "hi" with max_tokens=5)
- Retries up to 10 times with 30s interval

### `run` subcommand (auto-undeploy)
- Deploys if endpoint is empty (skips if already deployed)
- Runs the provided command via `subprocess.run()`
- **Always undeploys after command completes** — even on failure or Ctrl-C
- If endpoint was already deployed before `run`, skips both deploy and undeploy (respects external state)
- Exits with the same exit code as the wrapped command
- If undeploy fails, prints warning with manual action required

```bash
# Examples:
uv run python scripts/deploy_vertex_27b.py run -- uv run trialmatch phase0 --config configs/phase0_vertex_27b.yaml
uv run python scripts/deploy_vertex_27b.py run -- uv run pytest tests/e2e/ -m e2e --timeout=300
uv run python scripts/deploy_vertex_27b.py run -- bash -c "echo 'custom script here'"
```

## MedGemma 1.5 4B (HF Inference) — Companion Endpoint

For e2e demo, both 4B and 27B must be online:

| Model | Endpoint | Health Check |
|-------|----------|-------------|
| MedGemma 1.5 4B | HF Inference (`pcmy7bkqtqesrrzd`) | `MedGemmaAdapter.health_check()` via `InferenceClient.text_generation` |
| MedGemma 27B | Vertex AI (`6588061467889631232`) | `VertexMedGemmaAdapter.health_check()` via REST `:predict` |

4B uses TGI backend with `text_generation` API (NOT `/v1/chat/completions` — that returns 404).
27B uses vLLM backend with `chatCompletions` `@requestFormat` via `:predict`.

```python
# Quick 4B health check
from trialmatch.models.medgemma import MedGemmaAdapter
adapter = MedGemmaAdapter(hf_token=HF_TOKEN)
ok = await adapter.health_check()  # True if endpoint is warm
```

## Request Format (27B Vertex)

```json
{
  "instances": [{
    "@requestFormat": "chatCompletions",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 2048,
    "temperature": 0.2
  }]
}
```

Response: `{"predictions": [{"choices": [{"message": {"content": "..."}}], "usage": {...}}]}`

## Cost & Timing Reference

| Action | Time | Cost |
|--------|------|------|
| Deploy (with existing model) | ~18 min | $0 (no inference) |
| Deploy (fresh upload) | ~25-30 min | $0 (no inference) |
| Running (2x L4) | per hour | ~$2.30/hr |
| 20-pair benchmark | ~3 min | ~$0.12 |
| Undeploy | ~2-5 min | $0 after |
| Idle endpoint (no models) | indefinite | $0/hr |

## Failure Recovery

| Error | Cause | Fix |
|-------|-------|-----|
| `CustomModelServingL4GPUsPerProjectPerRegion` | L4 quota exceeded | Use int8 to halve GPUs, or request quota increase |
| `Model server exited unexpectedly` | OOM | Reduce `--max-model-len`, add quantization |
| Deploy hangs > 35 min | Cluster provisioning stuck | Cancel via GCP Console, retry |
| Smoke test fails all attempts | Model server not ready | Check `gcloud ai operations describe` for errors |
| `dedicated_endpoint_dns` missing | Not yet provisioned | Wait 1 min after deploy, re-run `status` |

## Known Gotchas

1. **predict_route must be `/generate`** — vLLM container routes internally based on `@requestFormat`
2. **`VLLM_USE_V1=0`** — v1 engine has compatibility issues
3. **HF_TOKEN in env vars** — must be in `serving_container_environment_variables`, not just local
4. **TGI vs vLLM** — HF Inference (TGI) has CUDA bugs with MedGemma; Vertex (vLLM) does NOT
5. **int8 quality** — 70% accuracy vs 75% GPT-4 baseline, competitive
6. **Model Registry reuse** — deploy script checks by `display_name` before uploading
7. **Undeploy vs teardown** — always prefer undeploy ($0/hr idle) over full delete
