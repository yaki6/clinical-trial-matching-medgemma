# Deploy MedGemma on Google Vertex AI Model Garden

**Date**: 2026-02-21
**Status**: Implementation complete, deployment blocked on API enablement
**Supersedes**: Round 1 and Round 2 assessments (flaws documented below)

---

## Context

HuggingFace Inference Endpoints have proven unreliable: 4B suffers from unstable connections and 27B deployment fails entirely. This plan migrates MedGemma inference to Google Vertex AI Model Garden while keeping HF as a configurable fallback.

## Flaws Fixed from Previous Rounds

**Round 1 errors:**
- Hard-coded model IDs without verifying via `gcloud ai model-garden models list`
- Predicted request payload used `inputs`/`parameters` only; vLLM containers also support `@requestFormat: chatCompletions`
- A100 pricing stated as ~$3.92/hr instead of ~$4.22/hr
- No quota check step
- No IAM role requirements listed

**Round 2 errors:**
- `list-deployment-config` output never shown so users couldn't act on it
- Still no GPU quota pre-check
- No mention that Vertex deploys from GCS (`gs://vertex-model-garden-restricted-us/...`) -- no HF token needed
- "Scale to zero" framing is misleading: Vertex AI online endpoints do NOT scale to zero; undeploy is the only cost-stop
- Deploy script used wrong model IDs and wrong CLI flags

---

## Implementation Status

### Completed (this session)

| File | Status | Description |
|------|--------|-------------|
| `pyproject.toml` | MODIFIED | Added `google-auth>=2.0` dependency |
| `src/trialmatch/models/vertex_medgemma.py` | CREATED | Vertex AI adapter (google-auth ADC + httpx REST) |
| `tests/unit/test_vertex_medgemma.py` | CREATED | 12 unit tests, all passing |
| `src/trialmatch/cli/phase0.py` | MODIFIED | Added `provider: vertex` routing branch |
| `configs/phase0_vertex_4b.yaml` | CREATED | Vertex 4B benchmark config |
| `configs/phase0_vertex_27b.yaml` | CREATED | Vertex 27B benchmark config |
| `scripts/deploy_vertex.sh` | CREATED | gcloud deployment automation |
| `.env.example` | CREATED | Environment variable template |
| `CLAUDE.md` | MODIFIED | Updated Models Available table |

### Verified

- **12/12 new unit tests passing** (adapter, retry, auth, cost estimation)
- **175/176 full test suite passing** (1 pre-existing failure in HF adapter, unrelated)
- **Lint clean** on all new files
- **gcloud CLI ready**: authenticated, Vertex AI API enabled, ADC configured
- **Model discovery verified**: `gcloud ai model-garden models list --model-filter=medgemma` returns 8 variants

### Blocked

- **Cloud Quotas API** not enabled on project `gen-lang-client-0517724223`
- **Service Usage API** not enabled (needed to enable other APIs via CLI)
- Both must be enabled manually via GCP Console before deployment can proceed

---

## Verified Environment State

| Item | Value |
|------|-------|
| gcloud CLI | v523.0.0 |
| Auth account | `liyaqi1996@gmail.com` |
| Project | `gen-lang-client-0517724223` |
| Vertex AI API | Enabled |
| ADC credentials | Configured (quota project set) |
| Billing quota project | `gen-lang-client-0517724223` |

## Verified Model IDs (from `gcloud ai model-garden models list`)

| MODEL_ID | CAN_DEPLOY |
|----------|------------|
| `google/medgemma@medgemma-4b-it` | Yes |
| `google/medgemma@medgemma-4b-it-dicom` | Yes |
| `google/medgemma@medgemma-4b-pt` | Yes |
| `google/medgemma@medgemma-1.5-4b-it` | Yes |
| `google/medgemma@medgemma-1.5-4b-it-dicom` | Yes |
| `google/medgemma@medgemma-27b-it` | Yes |
| `google/medgemma@medgemma-27b-it-dicom` | Yes |
| `google/medgemma@medgemma-27b-text-it` | Yes |

## Verified Deployment Configs (from `list-deployment-config` for `medgemma-4b-it`)

| Deploy Task | Machine | GPU | Count | Context | Notes |
|-------------|---------|-----|-------|---------|-------|
| vLLM 128K context | g4-standard-48 | NVIDIA_RTX_PRO_6000 | 1 | 128K | Full multimodal |
| vLLM 128K context 16 image limit | a2-ultragpu-1g | NVIDIA_A100_80GB | 1 | 128K | High perf |
| vLLM TPU 128K context text-only | ct6e-standard-1t | TPU | - | 128K | Text only |
| vLLM 128K context 16 image limit | a3-highgpu-1g | NVIDIA_H100_80GB | 1 | 128K | Premium |
| vLLM 128K context 16 image limit | a3-highgpu-8g | NVIDIA_H100_80GB | 8 | 128K | Max throughput |
| **vLLM 32K context 16 image limit** | **g2-standard-24** | **NVIDIA_L4** | **2** | **32K** | **Budget choice** |

> Note: The cheapest verified config for 4B is `g2-standard-24` with 2x L4, not 1x L4 as previously assumed. The `list-deployment-config` output is authoritative.

---

## Actual Region: `us-central1`

Originally targeted europe-west4 (Netherlands) for user's EU location. However:
- **L4 quota in europe-west4**: 1 GPU (need 2 for g2-standard-24)
- **L4 quota in us-central1**: 2 GPUs (just enough)
- gcloud CLI v523.0.0 has a KeyError bug when auto-selecting deployment config; workaround: specify `--machine-type` and `--accelerator-type` explicitly

| Region | L4 Quota | Status |
|--------|:--------:|--------|
| **us-central1** | 2 | **Deployed here** (sufficient quota) |
| europe-west4 | 1 | Insufficient (need 2) |
| europe-west1 | 1 | Insufficient |

### Additional APIs Enabled During Deployment

| API | Reason |
|-----|--------|
| `monitoring.googleapis.com` | Required by deploy command |
| `cloudresourcemanager.googleapis.com` | IAM policy queries |

---

## Hardware & Cost Reference

| Model | Machine Type | GPU | Est. $/hr | Notes |
|-------|-------------|-----|-----------|-------|
| **4B-IT (budget)** | g2-standard-24 | 2x L4 | ~$2.30 | Verified config from Model Garden |
| 4B-IT (premium) | a2-ultragpu-1g | 1x A100 80GB | ~$7.00 | |
| 27B-IT | Verify via `list-deployment-config` | TBD | TBD | Must check before deploying |

> Vertex AI online endpoints do NOT scale to zero. GPU billing starts on deploy and stops only on undeploy/delete. Never leave a test endpoint running overnight.

---

## Deployment Procedure

### Phase 0: Preflight (must complete before deploying)

```bash
# 0a. Enable required APIs (must be done via GCP Console if Service Usage API is disabled)
# https://console.developers.google.com/apis/api/serviceusage.googleapis.com/overview?project=gen-lang-client-0517724223
# https://console.developers.google.com/apis/api/cloudquotas.googleapis.com/overview?project=gen-lang-client-0517724223

# 0b. After APIs are enabled, verify from CLI:
gcloud services list --enabled --filter="cloudquotas.googleapis.com" \
  --project=gen-lang-client-0517724223 --format="value(name)"

# 0c. Check GPU quota in target region
gcloud compute regions describe europe-west4 \
  --project=gen-lang-client-0517724223 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" \
  | grep -i "nvidia_l4\|NVIDIA_L4"

# If quota is 0, request increase at:
# https://console.cloud.google.com/iam-admin/quotas?project=gen-lang-client-0517724223

# 0d. Verify IAM roles
gcloud projects get-iam-policy gen-lang-client-0517724223 \
  --flatten="bindings[].members" \
  --filter="bindings.members:liyaqi1996@gmail.com" \
  --format="table(bindings.role)"
# Required: roles/aiplatform.admin OR (roles/aiplatform.user + roles/storage.objectAdmin)
```

### Phase 1: Deploy

```bash
# Using the deploy script:
bash scripts/deploy_vertex.sh --model-size=4b --region=europe-west4

# Or manually:
gcloud ai model-garden models deploy \
  --model="google/medgemma@medgemma-4b-it" \
  --project=gen-lang-client-0517724223 \
  --region=europe-west4 \
  --endpoint-display-name="medgemma-4b-europe" \
  --accept-eula \
  --asynchronous
```

### Phase 2: Configure & Test

```bash
# After deployment completes, get endpoint ID:
gcloud ai endpoints list \
  --list-model-garden-endpoints-only \
  --region=europe-west4 \
  --project=gen-lang-client-0517724223

# Set env vars:
export GCP_PROJECT_ID="gen-lang-client-0517724223"
export GCP_REGION="europe-west4"
export VERTEX_ENDPOINT_ID="<endpoint-id-from-above>"

# Smoke test with dry run:
uv run trialmatch phase0 --config configs/phase0_vertex_4b.yaml --dry-run

# Full benchmark:
uv run trialmatch phase0 --config configs/phase0_vertex_4b.yaml
```

### Phase 3: Test Prediction Formats

The adapter uses `@requestFormat: chatCompletions` by default. If the endpoint returns errors, test both formats:

```bash
# Option A: chatCompletions format (default, recommended for -it models)
cat > /tmp/request_chat.json <<'JSON'
{
  "instances": [{
    "@requestFormat": "chatCompletions",
    "messages": [
      {"role": "user", "content": "What are typical findings in early-stage pneumonia?"}
    ],
    "max_tokens": 256,
    "temperature": 0.2
  }]
}
JSON
gcloud ai endpoints predict <ENDPOINT_ID> \
  --region=europe-west4 --json-request=/tmp/request_chat.json

# Option B: Standard inputs/parameters format (fallback)
cat > /tmp/request_standard.json <<'JSON'
{
  "instances": [{
    "inputs": "What are typical findings in early-stage pneumonia?",
    "parameters": {"max_new_tokens": 256, "temperature": 0.2}
  }]
}
JSON
gcloud ai endpoints predict <ENDPOINT_ID> \
  --region=europe-west4 --json-request=/tmp/request_standard.json
```

### Phase 4: Teardown

```bash
# When done testing -- CRITICAL to avoid ongoing GPU charges:
bash scripts/deploy_vertex.sh --teardown --endpoint-id=<ENDPOINT_ID> --region=europe-west4

# Or manually:
DEPLOYED_MODEL_ID=$(gcloud ai endpoints describe <ENDPOINT_ID> \
  --region=europe-west4 --format="value(deployedModels.id)")
gcloud ai endpoints undeploy-model <ENDPOINT_ID> \
  --region=europe-west4 --deployed-model-id="$DEPLOYED_MODEL_ID"
gcloud ai endpoints delete <ENDPOINT_ID> --region=europe-west4 --quiet
```

---

## Architecture: Adapter Design

```
src/trialmatch/models/vertex_medgemma.py
  VertexMedGemmaAdapter(ModelAdapter)
    - Auth: google.auth.default() ADC -> auto-refreshing Bearer token
    - HTTP: httpx.post() via asyncio.to_thread() (120s timeout)
    - Payload: chatCompletions format (@requestFormat + messages array)
    - Response: tries choices[0].message.content, falls back to str(predictions[0])
    - Retry: 503/429/SERVICE_UNAVAILABLE/RESOURCE_EXHAUSTED -> exp backoff
    - Cost: GPU-hour based (latency_ms / 3,600,000 * hourly_rate)
    - No HF token needed (Vertex serves from GCS)
```

```
CLI routing (phase0.py):
  provider: huggingface -> MedGemmaAdapter (HF Inference, legacy)
  provider: google      -> GeminiAdapter (AI Studio)
  provider: vertex      -> VertexMedGemmaAdapter (Vertex AI Model Garden)
```

---

## Region Fallback Decision Tree

```
Target: europe-west4 (Netherlands)
  -> Check L4 quota > 0?
    YES -> proceed with g2-standard-24 / 2x L4
    NO  -> check europe-west1 (Belgium)
          -> L4 available?
            YES -> use europe-west1
            NO  -> fall back to us-central1 (L4 guaranteed)
```

---

## Checklist (for Claude agent execution)

- [x] Phase 0a: gcloud auth + ADC configured
- [x] Phase 0b: Vertex AI API enabled
- [ ] Phase 0b: Service Usage API enabled (manual -- GCP Console)
- [ ] Phase 0b: Cloud Quotas API enabled (manual -- GCP Console)
- [ ] Phase 0c: GPU quota verified in europe-west4
- [ ] Phase 0d: IAM roles verified
- [x] Implementation: VertexMedGemmaAdapter created + tests passing
- [x] Implementation: CLI routing added (provider: vertex)
- [x] Implementation: YAML configs created (4b + 27b)
- [x] Implementation: Deploy script created + model IDs fixed
- [x] Implementation: .env.example + CLAUDE.md updated
- [ ] Phase 1: Deploy command issued, OPERATION_ID saved
- [ ] Phase 2: Operation complete, ENDPOINT_ID extracted
- [ ] Phase 3: Test prediction returns valid response
- [ ] Phase 4: Teardown executed (or reminder set)
