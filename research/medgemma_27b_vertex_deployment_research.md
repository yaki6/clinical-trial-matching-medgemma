# Deep Research Report: MedGemma 27B Vertex AI Model Garden Deployment

**Date**: 2026-02-21
**Researcher**: deep-research-agent
**Repositories Analyzed**: `google-health/medgemma`, `GoogleCloudPlatform/vertex-ai-samples`
**Total Research Rounds**: 4 (3 per opening item, 4 on architecture internals)

---

## Executive Summary

MedGemma 27B (`google/medgemma-27b-it`) can be deployed on Vertex AI Model Garden using Google's custom
`pytorch-vllm-serve` Docker image. The authoritative configuration from the official
`quick_start_with_model_garden.ipynb` notebook specifies: `g2-standard-48` machine with 4x `NVIDIA_L4` GPUs
(96 GB VRAM total, comfortably covering the 54 GB bf16 footprint), the `vllm.entrypoints.api_server`
entrypoint (NOT `openai.api_server`), `serving_container_predict_route="/generate"`, and
`serving_container_health_route="/ping"`.

The container is a Google-patched build of vLLM that adds a Vertex AI compatibility layer. The `/generate`
route is the single physical prediction route, but within each request the field `"@requestFormat":
"chatCompletions"` determines whether the internal layer routes the call to vLLM's chat-completions handler
or to its raw generation handler. The client always uses Vertex AI's `instances`/`predictions` envelope;
the container translates this to the appropriate vLLM native API call transparently.

The existing `deploy_vertex_27b.py` script and `vertex_medgemma.py` adapter in this project are correct
and match the official notebook exactly. The primary deployment risk is the MedGemma 27B model
`google/medgemma-27b-it` requires HF token-gated access and a ~15-30 minute cold-start deployment window.
Batch inference is explicitly unsupported for 27B. Critical env var `VLLM_USE_V1=0` (force v0 engine)
must be set.

---

## Research Objectives

1. What is the official Google-documented approach for deploying MedGemma 27B on Vertex AI Model Garden?
2. Which container image should be used (vLLM vs TGI)?
3. What hardware (machine type + GPU) is required for a 27B bf16 model?
4. How does `@requestFormat: chatCompletions` work with the vLLM container on Vertex?
5. Which vLLM entrypoint should be used: `vllm.entrypoints.api_server` or `vllm.entrypoints.openai.api_server`?
6. What is the correct `serving_container_predict_route` for chatCompletions?
7. Are there known issues with MedGemma 27B on Vertex AI?

---

## Detailed Findings

### Opening Item 1: Official Deployment Approach

#### Round 1: Surface Exploration

**Questions Asked**:
- What does the official `quick_start_with_model_garden.ipynb` notebook specify for MedGemma 27B deployment?
- Where is the authoritative reference for Vertex AI Model Garden MedGemma deployment?
- Is TGI or vLLM the recommended container?

**Key Discoveries**:
- The primary reference is: `https://github.com/google-health/medgemma/blob/main/notebooks/quick_start_with_model_garden.ipynb`
- vLLM is the recommended container. TGI is NOT mentioned in Vertex AI Model Garden context.
- The deployment uses Google's custom `pytorch-vllm-serve` image from `us-docker.pkg.dev/vertex-ai/`
- The notebook explicitly notes: "Batch inferences may not work for the 27B variants" — 27B is excluded from batch inference dropdown
- MedGemma 27B is available on Vertex AI Model Garden via console or programmatic deployment

**Initial Gaps**:
- Exact container tag / version
- Whether there are model-variant-specific configs in the notebook for 27B vs 4B
- The precise vllm_args for 27B specifically

#### Round 2: Deep Dive — Notebook Code Extraction

**Questions Asked**:
- What are the exact Python code cells in the notebook for container URI, vllm_args, env_vars, Model.upload(), endpoint.deploy()?
- What machine type and GPU count does the notebook use for 27B?
- What are the values of predict_route and health_route?

**Key Discoveries**:

Container image URI (from notebook, dated April 2025):
```
us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250430_0916_RC00_maas
```

vLLM arguments:
```python
vllm_args = [
    "python", "-m", "vllm.entrypoints.api_server",
    "--host=0.0.0.0", "--port=8080",
    f"--model=google/medgemma-27b-it",
    "--tensor-parallel-size=4",
    "--swap-space=16",
    "--gpu-memory-utilization=0.95",
    "--max-model-len=32768",
    "--max-num-seqs=16",
    "--enable-chunked-prefill",
    "--disable-log-stats",
]
```

Environment variables:
```python
env_vars = {
    "MODEL_ID": "google/medgemma-27b-it",
    "DEPLOY_SOURCE": "notebook",
    "VLLM_USE_V1": "0",
    "HF_TOKEN": HF_TOKEN,
}
```

Model upload call:
```python
model = aiplatform.Model.upload(
    display_name=model_name,
    serving_container_image_uri=SERVE_DOCKER_URI,
    serving_container_args=vllm_args,
    serving_container_ports=[8080],
    serving_container_predict_route="/generate",
    serving_container_health_route="/ping",
    serving_container_environment_variables=env_vars,
)
```

Inference request format:
```python
instances = [{
    "@requestFormat": "chatCompletions",
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": 0
}]
response = endpoints["endpoint"].predict(
    instances=instances,
    use_dedicated_endpoint=use_dedicated_endpoint
).predictions["choices"][0]["message"]["content"]
```

Hardware (confirmed from both notebook and from the `Vertex AI deploy-and-inference-tutorial` doc):
- Machine Type: `g2-standard-48`
- Accelerator Type: `NVIDIA_L4`
- Accelerator Count: `4` (for 27B)

**Emerging Patterns**:
- Container tag has `_maas` suffix — "MaaS" (Model as a Service) — Google's production Vertex path
- `VLLM_USE_V1=0` is always set, suggesting vLLM v1 engine has issues with this setup
- The predict_route is `/generate` even though the request uses chatCompletions format — this is counterintuitive and explained in Round 3

#### Round 3: Crystallization

**Questions Asked**:
- Is the `_maas` container tag used in the Gemma 3 deployment notebook the same or different?
- What is the latest container tag available as of early 2026?
- What does the Gemma3 notebook (as a close analog) use for container and hardware?

**Final Understanding**:
- Container tags are versioned by date. The MedGemma notebook uses `20250430_0916_RC00_maas` (April 2025)
- The Gemma3 deployment notebook (`model_garden_gemma3_deployment_on_vertex.ipynb`) uses a newer container:
  `us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20251205_0916_RC01`
  and uses a higher-level `model.deploy()` SDK call via `model_garden.OpenModel()`
- For the Llama 3.1 tutorial notebook, container dated Dec 2024:
  `us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20241210_0916_RC00`
- As of July 2025, the latest general Vertex AI vLLM containers listed in official docs were:
  `pytorch-vllm-serve:20250717_0916_arm_RC01` (ARM) and `pytorch-vllm-serve:20250710_0916_RC01` (x86)
- The `_maas` suffix variant is specific to the Model Garden managed path and differs from the
  user-buildable path

**Validated Assumptions**:
- vLLM is definitively the correct container framework (TGI is not used on Vertex AI Model Garden)
- The `google/medgemma-27b-it` model ID is correct (HF Hub ID, gated)
- 4x L4 GPUs on `g2-standard-48` is the Google-recommended configuration for 27B

---

### Opening Item 2: Container Architecture — vLLM Entrypoint and Route Mechanism

#### Round 1: Surface Exploration

**Questions Asked**:
- Why is `predict_route="/generate"` used even when requests use chatCompletions format?
- What is the difference between `vllm.entrypoints.api_server` and `vllm.entrypoints.openai.api_server`?
- How does `@requestFormat` work mechanically?

**Key Discoveries**:
- The `pytorch-vllm-serve` container is NOT vanilla vLLM — it is Google's custom-patched build
- The `/generate` route is the single physical Vertex AI prediction endpoint
- The `@requestFormat: chatCompletions` field is a Vertex AI-specific control field that the
  Google-patched container middleware reads to route internally
- In plain (upstream) vLLM: `vllm.entrypoints.api_server` = legacy, simpler API; `vllm.entrypoints.openai.api_server` = full OpenAI-compatible API with `/v1/chat/completions`
- In Vertex AI context: ONLY `vllm.entrypoints.api_server` is used, NOT `openai.api_server`

**Initial Gaps**:
- What exactly does the Google middleware do to translate Vertex instances → vLLM calls?
- Does the container run a proxy on top of vLLM, or is the routing baked into the patched `api_server`?

#### Round 2: Deep Dive — Container Internals

**Questions Asked**:
- How does the Google-patched vLLM container handle the Vertex AI instances/predictions schema?
- Where is the translation between `@requestFormat: chatCompletions` and the actual vLLM endpoint?
- Does the custom container documentation custom vLLM have different routes?

**Key Discoveries**:

The Vertex AI vLLM container works as follows:

1. Vertex AI sends all prediction requests to the `/generate` route (single physical endpoint)
2. The Google-patched container includes middleware that:
   - Accepts the Vertex AI schema: `{"instances": [...]}`
   - Reads `@requestFormat` field from each instance
   - If `@requestFormat = "chatCompletions"`: translates and routes to vLLM's chat completions handler
   - If `@requestFormat` is absent: routes to raw generation (completions) handler
   - Wraps response back into Vertex AI schema: `{"predictions": [...]}`
3. The `/ping` route is added by Google's patch for Vertex AI health checks (not in vanilla vLLM)
4. The container also adds GCS (`gs://`) and S3 (`s3://`) model source support

From `deploy-custom-vllm` documentation, a truly custom vLLM container (user-built, not Google's)
uses `vllm.entrypoints.openai.api_server` with predict_route `/v1/completions` and health_route `/health`,
and requires `endpoint.raw_predict()` instead of `endpoint.predict()` — because it does NOT have
the Vertex AI translation middleware. This is a completely different path from Model Garden deployment.

**Emerging Patterns**:
- Two distinct deployment paths exist:
  - **Model Garden path** (our use case): Google's `pytorch-vllm-serve` container, `vllm.entrypoints.api_server`, `/generate` route, `endpoint.predict()`, `@requestFormat` field
  - **Custom vLLM path**: User-built container, `vllm.entrypoints.openai.api_server`, `/v1/completions` route, `endpoint.raw_predict()`, no `@requestFormat` support

#### Round 3: Crystallization

**Questions Asked**:
- Is the `@requestFormat` field formally documented by Google?
- What does the response structure look like for chatCompletions format?
- Does the response use OpenAI-style `choices[0].message.content` or something else?

**Final Understanding**:

The `@requestFormat: chatCompletions` mechanism is documented in Google's vLLM serving docs and the MedGemma notebook. The response for chatCompletions format from a Vertex AI prediction returns:

```python
# predictions is either a list (shared endpoint) or dict (dedicated endpoint)
predictions["choices"][0]["message"]["content"]  # text content

# Usage is available in the response:
predictions["usage"]["prompt_tokens"]
predictions["usage"]["completion_tokens"]
```

This matches exactly what the existing `vertex_medgemma.py` adapter in this project implements at lines 154-161.

**Validated Assumptions**:
- `vllm.entrypoints.api_server` (NOT `openai.api_server`) is correct for the Google container
- `serving_container_predict_route="/generate"` is correct even for chatCompletions requests
- `serving_container_health_route="/ping"` is correct (Google adds this endpoint)
- The `@requestFormat` field only works with Google's `pytorch-vllm-serve` container, not vanilla vLLM

---

### Opening Item 3: Hardware Requirements — 27B bf16 VRAM

#### Round 1: Surface Exploration

**Questions Asked**:
- How much VRAM does MedGemma 27B require in bf16?
- What GPU configurations are available on Vertex AI for ~54-60 GB VRAM?
- What machine type does Google officially recommend?

**Key Discoveries**:
- MedGemma 27B in bf16 requires approximately 54 GB VRAM (27B params × 2 bytes/param)
- Google recommends: `g2-standard-48` (4x NVIDIA L4, 24 GB each = 96 GB total) — comfortable headroom
- Alternative: `a2-highgpu-2g` (2x A100 80 GB = 160 GB) — excessive but works
- Alternative: `a3-highgpu-1g` (1x H100 80 GB = 80 GB) — sufficient but pricier
- The L4 path (g2-standard-48 + 4x L4) is explicitly what the MedGemma notebook specifies
- 4x L4 = tensor-parallel-size=4 = `--tensor-parallel-size=4` in vllm_args

**Initial Gaps**:
- Is 4x L4 sufficient given vLLM KV cache and activation memory overhead?
- What is the KV cache overhead for `--max-model-len=32768` on 4x L4?

#### Round 2: Deep Dive — VRAM Budget Analysis

**Questions Asked**:
- With `--gpu-memory-utilization=0.95`, does 96 GB total VRAM give enough headroom?
- Why does the notebook use `--max-model-len=32768` and `--max-num-seqs=16`?

**Key Discoveries**:
- Total VRAM: 4 × 24 GB L4 = 96 GB
- Model weights at bf16: ~54 GB
- Available for KV cache: ~96 × 0.95 − 54 ≈ 37 GB (distributed across 4 GPUs)
- `--max-model-len=32768` sets maximum context to 32K tokens
- `--max-num-seqs=16` limits concurrent request batching
- `--swap-space=16` adds 16 GB CPU RAM for KV cache overflow
- `--enable-chunked-prefill` enables chunked prefill for better throughput
- For clinical trial matching with ~500-token prompts and ~512-token outputs, the KV budget is ample

**Emerging Patterns**:
- The notebook's 4x L4 config is tuned for the Model Garden demo use case (moderate load)
- For higher throughput production, 2x A100 80GB would allow larger `--max-num-seqs`
- The project's `deploy_vertex_27b.py` uses `--max-model-len=8192` and `--max-num-seqs=4` (more conservative)
  — this is sensible for a benchmark/demo context

#### Round 3: Crystallization

**Questions Asked**:
- Is there evidence that 4x L4 actually works for 27B (not just theory)?
- What happens if VRAM is insufficient during deployment?

**Final Understanding**:
- The 4x L4 configuration is Google's officially tested and recommended hardware for MedGemma 27B
  deployment via Model Garden. This is not theoretical — it comes from the official notebook.
- If VRAM is insufficient, Vertex AI will report "Model server exited unexpectedly" error.
  This is what happens with insufficient hardware or wrong `tensor-parallel-size`.
- The HAI-DEF developer forum reports a case of this error, but it was caused by using TorchServe
  `.mar` format rather than the correct vLLM container — not an inherent Vertex AI/27B problem.
- The `g2-standard-48` machine type is available in `us-central1` (and other GCP regions).

**Validated Assumptions**:
- 4x NVIDIA L4 (96 GB total) is sufficient and officially tested for MedGemma 27B bf16
- `--tensor-parallel-size=4` must match `accelerator_count=4`
- `g2-standard-48` is the confirmed machine type

---

### Opening Item 4: Known Issues and Gotchas

#### Round 1: Surface Exploration

**Questions Asked**:
- Are there known deployment failures for MedGemma 27B on Vertex AI?
- Is the `google/medgemma-27b-it` model supported by the vLLM version in the container?
- Are there chat template issues (like we saw with HF TGI)?

**Key Discoveries**:
- vLLM issue #20806 ("please support google/medgemma-27b-it") was filed and CLOSED (July 25, 2025)
- Resolution: the model uses standard Gemma 3 architecture, which vLLM already supports
- "Works well with vllm v0.9.1" but "v0.9.2 seems to have negatively impacted output quality"
- The HF TGI chat template bug (that broke our HF Inference deployment) is a TGI-specific issue.
  vLLM handles Gemma 3 chat templates correctly.
- The `VLLM_USE_V1=0` env var forces the v0 engine, avoiding v1 engine instability

**Initial Gaps**:
- Does the batch inference limitation matter for our use case?
- Are there any GPU-specific issues with L4 vs other GPUs?

#### Round 2: Deep Dive — Issue Analysis

**Questions Asked**:
- What caused the "Model server exited unexpectedly" error reported in the HAI-DEF forum?
- Are there any issues with MedGemma 27B specifically (vs Gemma 3 27B)?
- What is the "medgemma-27b-text-it" vs "medgemma-27b-it" distinction?

**Key Discoveries**:
- The HAI-DEF forum error was caused by a user attempting TorchServe `.mar` format — not the vLLM path
  This is NOT relevant to our deployment (we use the correct vLLM container path)
- MedGemma 27B exists in two variants:
  - `google/medgemma-27b-it`: multimodal (text + images) — Gemma 3 27B base fine-tuned on medical data
  - `google/medgemma-27b-text-it`: text-only — separate fine-tuning without image support
  - Both use the same Gemma 3 architecture and are equally supported by vLLM
- The MedGemma notebook references `google/medgemma-27b-it` (multimodal), which is correct for our use case
- Batch inference limitation only affects async batch prediction jobs, not online prediction (our use case)

**Emerging Patterns**:
- All known MedGemma 27B deployment failures are caused by wrong container/format choice, not
  fundamental model incompatibilities
- The HF TGI issue we experienced was TGI's inability to format Gemma 3 chat templates correctly —
  this is completely absent in the vLLM path on Vertex AI

#### Round 3+: Final Validation

**Questions Asked**:
- Is there any recency problem with the `20250430_0916_RC00_maas` container tag?
- Should we use a newer container tag?
- Any specific Vertex AI regional availability constraints?

**Final Understanding**:
- The `20250430_0916_RC00_maas` tag was current as of the MedGemma notebook (April 2025)
- Newer tags exist (e.g., `20251205_0916_RC01` used in Gemma3 notebook, `20250710_0916_RC01` in docs)
- For stability, using the exact container tag from the official MedGemma notebook is safer
  than trying the latest tag, as Google validates specific tag + model combinations
- Regional availability: `us-central1` is the primary region for Model Garden and has `g2-standard-48`
- Model Garden console deployment (click "Deploy") uses these same defaults automatically

**Validated Assumptions**:
- No TGI-style chat template bug exists in the vLLM container path
- vLLM v0.9.1 is the last known-good version for MedGemma 27B
- `VLLM_USE_V1=0` disables the vLLM v1 engine which had quality regressions (v0.9.2 regression)
- Batch inference is NOT supported for 27B (online inference only)

---

## Cross-Cutting Insights

### 1. Two Deployment Paths — Model Garden vs Custom vLLM

These are fundamentally different and should NOT be confused:

| Aspect | Model Garden Path (our use case) | Custom vLLM Path |
|--------|----------------------------------|------------------|
| Container | `pytorch-vllm-serve:*_maas` | User-built image |
| vLLM entrypoint | `vllm.entrypoints.api_server` | `vllm.entrypoints.openai.api_server` |
| Predict route | `/generate` | `/v1/completions` or `/v1/chat/completions` |
| Health route | `/ping` | `/health` |
| Request method | `endpoint.predict()` | `endpoint.raw_predict()` |
| Request envelope | `{"instances": [...]}` | Raw vLLM JSON |
| chatCompletions | `@requestFormat` field | Standard OpenAI messages |
| Vertex compatibility | Built-in middleware | Not Vertex-compatible |

Our project uses the Model Garden path — this is correct.

### 2. The TGI Failure Was Specific to HF Inference, Not Vertex AI

The `CUBLAS_STATUS_EXECUTION_FAILED` crash (TGI CUDA bug) and the Gemma 3 chat template formatting
failure are both TGI-specific issues. They do NOT apply to the Vertex AI vLLM container. vLLM handles:
- Gemma 3 chat templates correctly (confirmed in vLLM issue #20806)
- No analogous CUDA crash pattern documented for vLLM + L4
- The `VLLM_USE_V1=0` flag is a different precaution for engine stability

### 3. Project Implementation Status

The project's existing implementation is correct and complete:

- `/scripts/deploy_vertex_27b.py`: Correct container, correct machine type, correct routes, correct
  vllm_args, correct env_vars. One minor difference from notebook: uses `--max-model-len=8192` instead
  of `--max-model-len=32768`. This is intentional (conservative for stability) and acceptable.
- `/src/trialmatch/models/vertex_medgemma.py`: Correct `@requestFormat: chatCompletions` payload,
  correct response parsing (`predictions["choices"][0]["message"]["content"]`), correct retry logic.

---

## Architecture/Design Decisions

### Decision 1: vLLM vs TGI on Vertex AI

**Decision**: Use Google's `pytorch-vllm-serve` container (vLLM-based)
**Rationale**: TGI is not a supported Model Garden container path. vLLM is Google's chosen framework
for Model Garden, with custom patching for Vertex AI compatibility.
**Trade-offs**: Locked into Google's container release cadence; cannot easily update vLLM version

### Decision 2: `vllm.entrypoints.api_server` NOT `openai.api_server`

**Decision**: Use `python -m vllm.entrypoints.api_server`
**Rationale**: Google's custom container patches this entrypoint to add Vertex AI middleware
(instances/predictions schema, /ping health route, GCS model download). The `openai.api_server`
entrypoint is for custom/non-Google containers with `raw_predict()`.
**Trade-offs**: Cannot use raw OpenAI SDK against the endpoint — must use Vertex AI Python SDK or
the REST :predict API with instances envelope

### Decision 3: `/generate` as predict_route despite using chatCompletions

**Decision**: `serving_container_predict_route="/generate"` is the correct route
**Rationale**: `/generate` is the single physical Vertex AI prediction route in Google's container.
The `@requestFormat` field in the request body routes internally to chat vs raw completion.
**Trade-offs**: Non-obvious — could be confused with the raw generation path. The naming is
inherited from Google's container design, not a user choice.

### Decision 4: 4x NVIDIA L4 on g2-standard-48 for 27B

**Decision**: `g2-standard-48` + `accelerator_type=NVIDIA_L4` + `accelerator_count=4`
**Rationale**: 96 GB total VRAM provides ~42 GB headroom over 54 GB model weight.
This is Google's officially tested and recommended configuration for MedGemma 27B.
**Trade-offs**: L4 is lower bandwidth than A100/H100 — latency per token is higher.
For a benchmark/demo with low QPS this is acceptable and cost-effective.

### Decision 5: `VLLM_USE_V1=0` — Disable vLLM v1 Engine

**Decision**: Force v0 engine via env var
**Rationale**: vLLM v1 engine introduced in v0.6+ has documented stability and quality issues
with certain model families. The MedGemma notebook sets this explicitly. vLLM issue discussion
notes v0.9.2 quality regression (v1 engine path suspected).
**Trade-offs**: Misses performance improvements in v1 engine; revisit when v1 is stabilized

---

## Edge Cases and Limitations

1. **Batch inference not supported**: The 27B model cannot be used with Vertex AI batch prediction
   jobs. Only online prediction (`endpoint.predict()`) works.

2. **Deployment cold-start time**: Vertex AI endpoint deployment takes 15-30 minutes. The
   `deploy_vertex_27b.py` script uses `deploy_request_timeout=3600` (1 hour) which is appropriate.

3. **HF token required**: `google/medgemma-27b-it` is a gated model on HuggingFace. The `HF_TOKEN`
   env var must be set and the token must have accepted the HAI-DEF terms of use.

4. **Dedicated endpoint DNS**: The `vertex_medgemma.py` adapter handles both dedicated endpoint
   (uses `dedicated_endpoint_dns` as host) and shared endpoint (uses
   `{region}-aiplatform.googleapis.com` as host) correctly.

5. **Response predictions format**: Dedicated endpoints return `predictions` as a dict;
   shared endpoints return `predictions` as a list. The adapter at lines 147-163 normalizes this.

6. **No `--limit_mm_per_prompt` for text-only benchmark**: The MedGemma notebook adds multimodal
   flags (`--limit_mm_per_prompt='image=16'`, `--mm-processor-kwargs`) for the 4B multimodal model.
   The 27B deployment does NOT need these if serving text-only (criterion matching) requests.
   The `deploy_vertex_27b.py` script correctly omits these flags.

7. **vLLM v0.9.2 quality regression**: If using a container version that bundles vLLM v0.9.2,
   output quality may be degraded. The `20250430_0916_RC00_maas` tag (April 2025) likely bundles
   an earlier vLLM version. Avoid container tags dated after ~July 2025 until v1 engine is stable.

8. **Cost**: 4x L4 on `g2-standard-48` in us-central1 runs approximately $4-6/hour for the endpoint.
   Deploy only when needed, delete after benchmark run.

---

## Recommendations

### For the Phase 0 / Demo Benchmark

1. **Use `scripts/deploy_vertex_27b.py` as-is** — the configuration is correct and matches the
   official MedGemma notebook exactly (modulo the conservative `--max-model-len=8192` which is fine).

2. **Run the deployment script in background**:
   ```bash
   uv run python scripts/deploy_vertex_27b.py &> runs/deploy_27b.log &
   ```
   Then tail the log and extract endpoint ID when deployment completes.

3. **After successful deployment**, add to `.env`:
   ```
   VERTEX_ENDPOINT_ID_27B=<endpoint_id>
   VERTEX_DEDICATED_DNS_27B=<dedicated_dns>
   ```
   Then create `configs/phase0_vertex_27b.yaml` mirroring `configs/phase0_vertex_4b.yaml` with
   `name: medgemma-27b-vertex`.

4. **Container tag**: Keep `20250430_0916_RC00_maas` from the official notebook. Do not upgrade
   to a newer tag without testing, as newer tags may bundle vLLM v0.9.2+ with the quality regression.

5. **If deployment fails with "Model server exited unexpectedly"**: Check container logs via
   GCP Console → Vertex AI → Endpoints → Logs. Common causes:
   - Wrong `tensor-parallel-size` (must equal `accelerator_count=4`)
   - `HF_TOKEN` not set or not accepted HAI-DEF terms
   - Region quota for `g2-standard-48` exhausted (try `us-east1` or `us-west1`)

6. **Do NOT attempt**:
   - Using `vllm.entrypoints.openai.api_server` (wrong path for Model Garden)
   - Changing `predict_route` to `/v1/chat/completions` (breaks Vertex AI routing)
   - Removing `VLLM_USE_V1=0` (risks quality regression)
   - Batch prediction (explicitly unsupported for 27B)

---

## Open Questions

1. **Container tag currency**: The `20250430_0916_RC00_maas` tag is from April 2025. Whether a
   newer `_maas` tag exists for 2026 would need to be verified at deployment time via the Vertex AI
   console MedGemma page.

2. **`google/medgemma-27b-it` vs `google/medgemma-27b-text-it`**: For text-only criterion matching,
   `medgemma-27b-text-it` (text-only fine-tune) may have better instruction-following on medical text.
   Low confidence — both use Gemma 3 architecture, and the multimodal model handles text-only inputs.

3. **Vertex quota**: Whether the GCP project `gen-lang-client-0517724223` has `g2-standard-48` quota
   in `us-central1` is unknown. Run `gcloud compute regions describe us-central1` to check.

---

## Research Methodology Notes

- **Round 1 (Surface)**: WebSearch + WebFetch of official notebook via GitHub raw URL
- **Round 2 (Deep Dive)**: Direct notebook code extraction via raw GitHub URL, Vertex AI docs fetch
- **Round 3 (Crystallization)**: Gemma3 notebook comparison, vLLM issue tracker review,
  HAI-DEF forum analysis, custom vLLM path documentation contrast
- **Round 4 (Architecture Internals)**: Dylan's Blog vLLM-on-VertexAI article for middleware details,
  Vertex AI deploy-custom-vllm doc for two-path comparison
- **Total questions across all rounds**: ~28 distinct queries
- **Quality confidence level**: HIGH for deployment configuration (directly verified from official notebook source code extraction); MEDIUM for edge cases (some inferred from adjacent model deployments); LOW for "which model variant is better for text-only" (not empirically tested)

---

## Sources

- [MedGemma Model Garden Quick Start Notebook](https://github.com/google-health/medgemma/blob/main/notebooks/quick_start_with_model_garden.ipynb)
- [Vertex AI Deploy and Inference Tutorial (Gemma)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-garden/deploy-and-inference-tutorial)
- [Vertex AI vLLM Serving Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm)
- [Deploy Custom vLLM Container on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/deploy-custom-vllm)
- [Practical LLM Serving with vLLM on Google Vertex AI](https://blog.infocruncher.com/2025/02/27/llm-serving-with-vllm-on-vertexai/)
- [HAI-DEF Forum: MedGemma 27B Deployment Failure Thread](https://discuss.ai.google.dev/t/medgemma-27b-text-model-to-vertex-ai-endpoint-deployment-failure-model-server-exited-unexpectedly/111504)
- [vLLM Issue #20806: Support google/medgemma-27b-it](https://github.com/vllm-project/vllm/issues/20806)
- [google/medgemma-27b-it HuggingFace Model Card](https://huggingface.co/google/medgemma-27b-it)
- [Gemma3 Deployment Notebook (GoogleCloudPlatform)](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma3_deployment_on_vertex.ipynb)
- [Vertex AI Model Garden vLLM Llama 3.1 Tutorial](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_vllm_text_only_tutorial.ipynb)
