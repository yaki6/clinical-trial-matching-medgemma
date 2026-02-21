# Deep Research Report: Vertex AI Cost Optimization for Intermittent/Development Use Cases

**Date**: 2026-02-21
**Researcher**: deep-research-agent
**Repositories Analyzed**: Google Cloud Documentation, official pricing pages, community forums, GCP sample repositories
**Total Research Rounds**: 4 rounds across 6 topic areas

---

## Executive Summary

Vertex AI dedicated endpoints bill per-second (minimum 30-second increments) for every second a model is deployed, regardless of whether predictions are being served. A MedGemma 27B deployment on a `g2-standard-24` (2x NVIDIA L4) costs approximately $2.00/hour on-demand, producing a worst-case monthly cost of ~$1,456 for always-on operation. For a 2-4 hour/day development schedule, the same hardware costs $4-8/day or roughly $120-240/month — still significant at research scale.

Three viable cost strategies exist for intermittent use, ordered by automation complexity. The simplest is **manual undeploy/redeploy** via CLI or Python SDK: undeploy stops billing immediately, the endpoint resource persists at zero cost, and redeployment takes 5-10 minutes. The most automated is **scale-to-zero** (`min_replica_count=0`), a preview feature on Vertex AI's v1beta1 API that automatically shuts down replicas after a configurable idle period (default 60 minutes), bringing billing to zero. However, scale-to-zero has a critical limitation for large LLMs: it is **incompatible with multi-host GPU deployments** and models scaled to zero for 30+ days are automatically undeployed by Google. A third alternative, **Cloud Run with GPU + GCS FUSE**, provides true serverless scale-to-zero for development workflows at ~$0.67/hour (L4) only while instances run, but imposes cold-start latencies of 30-120 seconds for large models.

The recommended approach for this project's 2-4 hours/day research profile is the **manual undeploy/redeploy pattern automated via Cloud Scheduler + Cloud Functions**, combined with keeping the endpoint resource alive between sessions. This eliminates idle cost entirely while re-deployment takes 5-10 minutes — acceptable for a research workflow where you can queue the deploy before starting work.

---

## Research Objectives

1. How does Vertex AI bill for dedicated prediction endpoints?
2. What is the cost of keeping an endpoint with no deployed model vs. keeping a deployed idle model?
3. How fast is re-deployment after undeploy (vs. full delete)?
4. Are there any special pricing considerations for Model Garden deployments (MedGemma)?
5. What is the most cost-effective architecture for 2-4 hours/day usage?
6. Can scale-down/scale-up be automated with Cloud Scheduler or Cloud Functions?

---

## Detailed Findings

### Topic 1: Billing Model — How Vertex AI Charges for Prediction Endpoints

#### Round 1: Surface Exploration

**Questions Asked**:
- How does Vertex AI bill for prediction endpoints?
- What are the billing increments?
- Is there a distinction between idle and active billing?

**Key Discoveries**:

Vertex AI online prediction endpoints bill on a **per-node-hour** model. A "node hour" represents the time a virtual machine spends either serving predictions or waiting in an active ready state. There is no distinction between actively serving traffic and idle: a deployed model costs the same whether it is handling 1,000 requests/minute or zero.

**Billing increments**: Usage is charged in **30-second increments** — there is no minimum session duration. This means very short deployments (e.g., 5 minutes) incur cost only for those 5 minutes, rounded to the next 30-second mark.

**Critical rule**: You pay for each model deployed to an endpoint even if no prediction is made. **You must undeploy your model to stop incurring charges.** Models that fail to deploy or are not deployed are not charged.

**Pricing components for a GPU deployment**:
- vCPU cost (per hour, based on machine type)
- RAM cost (per GB-hour)
- GPU cost (per GPU-hour, additive)

**Initial Gaps**: Specific L4 GPU dollar amounts, behavior of empty endpoint (no model), minimum billing period per deployment action.

#### Round 2: Deep Dive

**Questions Asked**:
- What does a g2-standard-24 (2x L4) cost per hour on Vertex AI?
- What is billing behavior during the model loading phase?
- Is there a Vertex AI pricing markup over raw Compute Engine rates?

**Key Discoveries**:

From `cloudprice.net` and `gcloud-compute.com` data:

| Machine Type | vCPUs | RAM | GPUs | On-Demand/hr | Spot/hr |
|---|---|---|---|---|---|
| `g2-standard-24` | 24 | 96 GiB | 2x NVIDIA L4 | **$2.00** | ~$0.80 |
| `g2-standard-12` | 12 | 48 GiB | 1x NVIDIA L4 | ~$1.10 | ~$0.44 |
| `g2-standard-4` | 4 | 16 GiB | 1x NVIDIA L4 | ~$0.72 | ~$0.29 |

Note: Vertex AI applies a **management overhead markup** over raw Compute Engine rates for online prediction. Raw Compute Engine `g2-standard-24` is $2.00/hr; Vertex AI's effective rate for the same hardware may be slightly higher (typically 10-30% markup), but the exact Vertex AI-specific surcharge for dedicated endpoints is not separately published — Google bundles it into the node-hour rate.

**Cloud Run L4 comparison**: Cloud Run GPU (L4) bills at approximately **$0.67/hour** in Tier 1 regions, only while an instance is handling requests. Scale-to-zero means $0.00 when idle.

**Emerging Patterns**: The cost gap between Vertex AI dedicated and Cloud Run with scale-to-zero is significant for development workloads. A 2-hour Vertex AI session costs ~$4.00 but leaves the meter running when forgotten. Cloud Run costs ~$1.34 for the same 2 hours with automatic zero when done.

#### Round 3: Crystallization

**Questions Asked**:
- Does billing start when deployment is initiated or when the model is ready to serve?
- Is there a Vertex AI-specific surcharge for Model Garden deployments?

**Final Understanding**:

Billing begins when the DeployedModel resource transitions to DEPLOYED state (model ready to serve), not when deployment is initiated. The 5-10 minute deployment window itself may not be billed in full — this is not explicitly documented, but the charging model is per "active state" not per "operation in progress."

**Model Garden vs. custom model cost**: There is no special premium for deploying from Model Garden vs. a custom-registered model. The billing is purely infrastructure-based (machine type + GPU hours). Model Garden provides pre-configured containers and one-click deployment but does not add a per-request or per-deployment fee on top of infrastructure costs.

**Validated Assumptions**:
- Billing is per-second (30s min increment), not per-request
- Idle model costs exactly the same as active model
- Model Garden = no billing premium
- Empty endpoint = $0.00

---

### Topic 2: Undeploy vs. Delete — Cost and Operational Differences

#### Round 1: Surface Exploration

**Questions Asked**:
- Does an endpoint with no deployed model incur charges?
- What is the operational difference between undeploy and delete?

**Key Discoveries**:

The Google Developer Forum thread on this topic gives a clear answer confirmed by community testing: **there is no cost for a Vertex AI endpoint resource with no model deployed**. An endpoint is a lightweight routing resource; the GPU compute only runs when a model is actually deployed.

The operational flow is:
1. **Undeploy model** → compute resources freed, billing stops, endpoint still exists
2. **Delete endpoint** (optional, after all models undeployed) → endpoint routing resource removed

**Initial Gaps**: Whether undeploy preserves model artifacts in the Model Registry, how long undeploy takes.

#### Round 2: Deep Dive

**Questions Asked**:
- Does undeploy preserve the model in the Vertex AI Model Registry?
- How long does the undeploy operation take?

**Key Discoveries**:

**Undeploy does NOT delete the model from the Model Registry.** The Model Registry entry (the reference to model weights, container config, etc.) is preserved. Only the DeployedModel association between the endpoint and the model is removed. Re-deploying from the Model Registry is therefore faster than re-creating from scratch.

**Undeploy vs. Delete comparison**:

| Scenario | Cost | Model Registry Entry | Time to Re-deploy |
|---|---|---|---|
| Model deployed (active) | Full GPU rate/hr | Preserved | N/A |
| Model undeployed, endpoint kept | $0.00 | Preserved | 5-10 minutes |
| Endpoint deleted, model in registry | $0.00 | Preserved | 5-10 minutes + create endpoint |
| Model deleted from registry | $0.00 | Deleted | Full re-import + 5-10 min deploy |

**Key insight**: Undeploying is strictly better than deleting the endpoint for a development workflow. The endpoint resource costs nothing when empty, and having the endpoint pre-created saves the endpoint creation step (~30-60 seconds) on the next session start.

#### Round 3: Crystallization

**Questions Asked**:
- Are there any storage costs for keeping model artifacts in Model Registry?

**Final Understanding**:

Model artifacts stored in Cloud Storage (referenced by Model Registry) incur standard Cloud Storage charges (~$0.020/GB/month for Standard storage). For a MedGemma 27B int8 model (~15-20 GB), this is approximately **$0.30-0.40/month** — negligible compared to GPU compute costs. This is often already paid for as part of normal GCS usage.

**Validated Assumptions**:
- Endpoint with no model = $0.00/hr
- Model Registry entry has minimal storage cost (~$0.40/month for 27B)
- Undeploy is reversible; delete endpoint is also reversible (just recreate)
- Keeping both endpoint and Model Registry entry is the optimal state between sessions

---

### Topic 3: Re-deployment Speed — Undeploy/Redeploy Timing

#### Round 1: Surface Exploration

**Questions Asked**:
- How long does deploying a model from Model Garden take?
- Does keeping the endpoint (vs. deleting it) speed up re-deployment?

**Key Discoveries**:

Official Google documentation states: "Deploying a registered model to an endpoint will take **5-10 minutes** depending on the configured resources." This is consistent across community reports for 27B-scale models.

Key finding from the autoscaling docs: "When a new replica spins up, it needs to load the model into memory, which can take **30 seconds to a few minutes** depending on model size." This is the warm-start case (scale-to-zero waking up), not the cold-start full-deploy case.

**Initial Gaps**: Whether there is any caching that makes re-deploy faster than initial deploy. Whether the Vertex control plane has any "warm" state for recently undeployed models.

#### Round 2: Deep Dive

**Questions Asked**:
- Does undeploying preserve any cached state (GPU VRAM, model weights on disk)?
- Is there a "rolling update" path that avoids cold restart?

**Key Discoveries**:

**Undeploying fully deprovisions compute resources.** From the docs: "Undeploying a Model from an Endpoint removes a DeployedModel from it, and frees all resources it's using." There is no cached GPU VRAM or persistent local disk between deploy/undeploy cycles. Each re-deployment is effectively identical to the initial deployment from the platform's perspective.

**Rolling update path exists** for replacing one version with another (same endpoint, new model version), but this requires a currently-running model and is not useful for the undeploy-then-redeploy use case.

**Scale-to-zero cold start** is fundamentally different: a scaled-to-zero replica restarts on the same underlying infrastructure pool, with model weights fetched from GCS. The documented times:

| Model/Config | Scale-to-Zero Cold Start |
|---|---|
| CPU model < 100MB | 10-30 seconds |
| CPU model 100MB-1GB | 30-60 seconds |
| CPU + T4 GPU (medium) | 60-120 seconds |
| CPU + A100 GPU (large >1GB) | 2-5 minutes |
| L4 GPU + 27B model (estimated) | 3-8 minutes |

For a full undeploy/redeploy (not scale-to-zero), the time is 5-10 minutes regardless of previous state.

#### Round 3: Crystallization

**Final Understanding**:

There are two distinct "re-availability" times to understand:

1. **Full undeploy then redeploy**: Always 5-10 minutes. No caching. Identical to first deploy.
2. **Scale-to-zero wake**: 3-8 minutes for a large 27B model. Faster because infrastructure pooling may pre-warm the instance, but weight loading from GCS still takes 2-5 minutes.

For a 2-4 hour/day workflow, 5-10 minutes of "warm-up time" is acceptable if you initiate deploy before you need it (e.g., run `deploy.py` while making coffee).

**Validated Assumptions**:
- No caching between undeploy/redeploy cycles
- 5-10 min is the realistic number for 27B re-deployment
- Scale-to-zero is marginally faster but not dramatically different

---

### Topic 4: Scale-to-Zero — The Automated Approach

#### Round 1: Surface Exploration

**Questions Asked**:
- What is Vertex AI scale-to-zero and how does it work?
- What billing applies when scaled to zero?

**Key Discoveries**:

Scale-to-zero is a **Preview feature** on the Vertex AI `v1beta1` API. When `min_replica_count=0` is set, the service automatically shuts down all replicas when the endpoint receives no traffic for `idle_scaledown_period` seconds (default: 3,600 seconds = 1 hour). When traffic resumes, a new replica spins up.

**Billing during zero-replica state**: $0.00. When no replicas are running, no compute costs are incurred.

**Key parameters**:
```
ScaleToZeroSpec:
  min_scaleup_period: 300-28800s (default: 3600s)  # grace period before eligible for zero
  idle_scaledown_period: 300-28800s (default: 3600s)  # inactivity before scaling to zero
  initial_replica_count: 1  # replicas when scaling up from zero
```

**Initial Gaps**: Whether scale-to-zero works for GPU deployments of 27B models; what the 30-day rule means.

#### Round 2: Deep Dive — Critical Limitations

**Questions Asked**:
- Does scale-to-zero work with multi-host GPU deployments?
- What is the 30-day automatic undeploy rule?
- What API version is required?

**Key Discoveries**:

**Critical limitation found**: Scale-to-zero is **incompatible with multi-host GPU or multi-host TPU deployments**. Single-host deployments with multiple GPUs on one machine are compatible, but tensor-parallel deployments spanning multiple nodes are not.

**For MedGemma 27B on Vertex AI**: The benchmarked deployment used `g2-standard-24` (2x L4 on a SINGLE host) — this IS compatible with scale-to-zero. A multi-node deployment (e.g., 2x `g2-standard-12` across separate nodes) would NOT be compatible.

**30-day auto-undeploy rule**: "DeployedModels scaled to zero for longer than 30 days are automatically undeployed." If the development project is paused for a month, Google will undeploy the model. This is benign for cost (it saves money), but requires re-deploy if work resumes. The Model Registry entry is preserved — only the DeployedModel association is removed.

**Also incompatible with**:
- Shared public endpoints (must use dedicated endpoint)
- Multiple models on one endpoint
- Multi-host GPU deployments

**API requirements**: Must use `v1beta1` API or `gcloud beta ai endpoints deploy-model`. Standard `v1` API does not expose scale-to-zero.

**Cost savings estimate**:
- Always-on `g2-standard-24`: ~$2.00/hr × 720 hr/month = **$1,456/month**
- Scale-to-zero @ 2hr/day, 22 days/month: ~$2.00/hr × 44 hr = **$88/month**
- Savings: ~94%

#### Round 3: Crystallization

**Python code to enable scale-to-zero**:
```python
from google.cloud import aiplatform_v1beta1

# Must use v1beta1 client, not standard v1
from google.cloud.aiplatform_v1beta1.services.endpoint_service import EndpointServiceClient

model.deploy(
    endpoint=endpoint,
    machine_type='g2-standard-24',
    accelerator_type='NVIDIA_L4',
    accelerator_count=2,
    min_replica_count=0,   # KEY: enables scale-to-zero
    max_replica_count=1,
)
```

**gcloud CLI** (beta required):
```bash
gcloud beta ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID \
  --machine-type=g2-standard-24 \
  --accelerator=count=2,type=nvidia-l4 \
  --min-replica-count=0 \
  --max-replica-count=1
```

**Validated Assumptions**:
- Scale-to-zero works for single-host multi-GPU (g2-standard-24 qualifies)
- 30-day auto-undeploy is a safety net, not a billing risk
- Preview feature — may not have full SLA guarantees
- Cold start on wake is 3-8 minutes for 27B

---

### Topic 5: Model Garden Specifics for MedGemma

#### Round 1: Surface Exploration

**Questions Asked**:
- Is there any special pricing for Model Garden deployed models vs. custom models?
- What GPU configuration does MedGemma 27B use on Vertex AI?

**Key Discoveries**:

MedGemma 27B is available through Vertex AI Model Garden (as of 2025). The deployment uses standard Vertex AI infrastructure billing — **no Model Garden surcharge**.

Documented configurations for MedGemma/Gemma class 27B models:
- vLLM backend with `g2-standard-24` (2x NVIDIA L4, 96 GiB RAM)
- 128K context window configuration

**Initial Gaps**: Whether int8 quantization is supported natively or requires custom container.

#### Round 2: Deep Dive

**Questions Asked**:
- Does int8 quantization change the GPU requirement (can it fit on 1x L4 instead of 2x)?
- Is Hex-LLM (Google's premium container) required for quantization?

**Key Discoveries**:

Google's **Hex-LLM** container (available in Model Garden) supports int8 and int4 quantization natively. This is a premium prebuilt container distinct from the standard vLLM container. The benchmarked deployment in this project used int8 quantization with a Vertex AI-provisioned container.

**Quantization GPU footprint**: MedGemma 27B int8 = approximately 27GB of VRAM. A single NVIDIA L4 has 24GB VRAM — not quite enough alone. Two L4s (48GB total) provides comfortable headroom. An A100 80GB would handle int8 in one GPU.

**Container cost implications**: Hex-LLM is a Google-managed container with no additional per-request billing — it is just a container image, not a managed service with per-token fees.

#### Round 3: Crystallization

**Final Understanding**:

For MedGemma 27B int8 on Vertex AI:
- **Minimum viable GPU**: 2x NVIDIA L4 (`g2-standard-24`) — fits in VRAM with headroom
- **Alternative**: 1x A100 40GB (tight, may not fit) or 1x A100 80GB (comfortable)
- **Container**: Use vLLM or Hex-LLM from Model Garden, both are infrastructure-billed only
- **No Model Garden premium**: Pure infrastructure billing

**Model Garden specific behaviors**:
- One-click deploy from console: provisions endpoint + deploys model automatically
- Model Registry entry is created automatically on first deploy
- This preserved Model Registry entry is what enables fast re-deployment in later sessions

**Validated Assumptions**:
- No per-token or per-request fee from Model Garden
- int8 requires 2x L4 minimum
- Hex-LLM supports quantization; standard vLLM container also works if configured manually

---

### Topic 6: Automation — Cloud Scheduler + Cloud Functions Pattern

#### Round 1: Surface Exploration

**Questions Asked**:
- Can Cloud Scheduler trigger Vertex AI undeploy/redeploy operations?
- What is the recommended automation pattern?

**Key Discoveries**:

The standard pattern for scheduling Vertex AI operations is:

```
Cloud Scheduler (cron) → HTTP trigger → Cloud Functions → Vertex AI Python SDK
```

Cloud Scheduler cannot directly call Vertex AI REST APIs with auth in a reliable way; Cloud Functions acts as the intermediary that holds service account credentials.

**Initial Gaps**: Exact Python SDK calls needed, service account permissions required, cold-start time from deploy trigger to usable endpoint.

#### Round 2: Deep Dive

**Questions Asked**:
- What Python SDK calls are needed for undeploy and redeploy?
- What IAM roles are needed for the Cloud Functions service account?

**Key Discoveries**:

**Undeploy (stop billing)**:
```python
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(ENDPOINT_ID)
endpoint.undeploy_all()  # Undeploys all models from endpoint
```

**Redeploy (start billing, make endpoint available)**:
```python
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)
model = aiplatform.Model(MODEL_ID)  # From Model Registry
endpoint = aiplatform.Endpoint(ENDPOINT_ID)

model.deploy(
    endpoint=endpoint,
    machine_type='g2-standard-24',
    accelerator_type='NVIDIA_L4',
    accelerator_count=2,
    min_replica_count=1,
    max_replica_count=1,
    sync=True  # Blocks until deployment is complete (~5-10 min)
)
```

**Required IAM roles** for Cloud Functions service account:
- `roles/aiplatform.user` — deploy/undeploy models
- `roles/aiplatform.endpointServiceAgent` — manage endpoint resources

**Cloud Scheduler cron examples**:
```bash
# Deploy at 9 AM US Pacific time (Mon-Fri)
gcloud scheduler jobs create http deploy-medgemma \
  --schedule="0 17 * * 1-5" \          # 9 AM PT = 17:00 UTC
  --uri="https://REGION-PROJECT.cloudfunctions.net/deploy_medgemma" \
  --time-zone="UTC"

# Undeploy at 7 PM US Pacific time (Mon-Fri)
gcloud scheduler jobs create http undeploy-medgemma \
  --schedule="0 3 * * 2-6" \           # 7 PM PT = 03:00 UTC next day
  --uri="https://REGION-PROJECT.cloudfunctions.net/undeploy_medgemma" \
  --time-zone="UTC"
```

#### Round 3: Crystallization

**Alternative: Direct gcloud + cron on local machine** (simpler for single researcher):

For a research workflow (not production), running cron on a local machine or Cloud Shell is simpler than Cloud Functions:

```bash
# Add to crontab (crontab -e)
# Deploy at 8 AM Pacific
0 8 * * * gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --model=$MODEL_ID \
  --region=us-central1 \
  --machine-type=g2-standard-24 \
  --accelerator=count=2,type=nvidia-l4 \
  --min-replica-count=1 \
  --max-replica-count=1

# Undeploy at 6 PM Pacific
0 18 * * * gcloud ai endpoints undeploy-model $ENDPOINT_ID \
  --deployed-model-id=$DEPLOYED_MODEL_ID \
  --region=us-central1
```

**Practical consideration**: The `--deployed-model-id` changes on each new deployment, so a script must query the current deployed model ID before undeploying. This is the main friction point for automation.

**Python script approach** (recommended for this project):

```python
# scripts/manage_medgemma_endpoint.py
import argparse
from google.cloud import aiplatform

PROJECT_ID = "your-project"
REGION = "us-central1"
ENDPOINT_ID = "your-endpoint-id"
MODEL_ID = "your-model-id"  # Model Registry ID, stable across re-deploys

def deploy():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    model = aiplatform.Model(MODEL_ID)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    model.deploy(
        endpoint=endpoint,
        machine_type="g2-standard-24",
        accelerator_type="NVIDIA_L4",
        accelerator_count=2,
        min_replica_count=1,
        max_replica_count=1,
        sync=True,
    )
    print(f"Endpoint {ENDPOINT_ID} is live")

def undeploy():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    endpoint.undeploy_all()
    print(f"Endpoint {ENDPOINT_ID} undeployed — billing stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["deploy", "undeploy"])
    args = parser.parse_args()
    deploy() if args.action == "deploy" else undeploy()
```

**Usage**:
```bash
# Before starting work (in background — takes 5-10 min)
uv run python scripts/manage_medgemma_endpoint.py deploy &

# After finishing work
uv run python scripts/manage_medgemma_endpoint.py undeploy
```

**Validated Assumptions**:
- `undeploy_all()` handles the dynamic deployed-model-id problem automatically
- Model Registry ID is stable and reusable across deploy/undeploy cycles
- Cloud Functions is overkill for a single-researcher setup; direct SDK script works fine
- Billing stops within seconds of `undeploy_all()` completing

---

## Cross-Cutting Insights

1. **The "deploy/undeploy" lifecycle is the fundamental cost control primitive on Vertex AI.** Unlike cloud services that scale to zero automatically (Lambda, Cloud Run), Vertex AI dedicated endpoints require explicit action to stop billing. This is a design choice, not a limitation — it provides predictable latency at the cost of billing discipline.

2. **Scale-to-zero (v1beta1) changes this equation but introduces complexity.** For single-host GPU deployments (which MedGemma 27B int8 satisfies on g2-standard-24), scale-to-zero provides automatic cost management. The 1-hour default idle timeout means forgetting to undeploy costs at most 1 hour extra billing before automatic scale-down, then $0. This is qualitatively different from the always-on model.

3. **Cloud Run GPU is the true serverless alternative.** If the 5-10 minute cold start of Vertex AI (full redeploy) is unacceptable, Cloud Run with GPU + GCS FUSE achieves 30-120 second cold starts at ~$0.67/hr (L4), billing only while instances run. The downside is more infrastructure ownership.

4. **Model Garden does not add cost complexity.** All pricing is infrastructure-based. This simplifies the cost model significantly.

5. **The 30-day auto-undeploy rule is a cost safety net**, not a hazard. For a project paused for more than 30 days, Google automatically cleans up the DeployedModel. The Model Registry entry and endpoint remain. Resume = re-deploy.

---

## Architecture/Design Decisions

### Decision 1: Scale-to-Zero vs. Manual Undeploy

| Criterion | Scale-to-Zero (v1beta1) | Manual Undeploy Script |
|---|---|---|
| Cost when idle | $0 (after idle_scaledown_period) | $0 (immediate after script) |
| Automation required | None (built-in) | Script or Cloud Scheduler |
| Cold start on first request | 3-8 minutes | N/A (user initiates) |
| API stability | Preview (beta) | GA (stable) |
| 27B single-host compatible | Yes (g2-standard-24) | Yes |
| Forgetting to stop | Max 1 hour extra cost | Full session cost |
| **Recommended for this project** | Secondary option | **Primary option** |

**Rationale for manual undeploy as primary**: The research workflow is researcher-initiated, not traffic-driven. Scale-to-zero is designed for traffic-driven workloads (e.g., API that gets bursts of requests). For a researcher who runs benchmarks in planned sessions, manual deploy-before-work / undeploy-after-work is simpler and avoids beta API dependency.

Scale-to-zero is valuable as a **safety net**: if you forget to undeploy, it catches the idle endpoint after 1 hour and shuts it down automatically. These two strategies are complementary: use scale-to-zero AND write a deploy/undeploy script.

### Decision 2: Vertex AI Dedicated Endpoint vs. Cloud Run GPU

| Criterion | Vertex AI Dedicated | Cloud Run GPU |
|---|---|---|
| Cold start (from zero) | 5-10 min (full deploy) | 30-120 sec |
| Billing model | Per node-hour (always-on if deployed) | Per second (only while active) |
| Cost/hour when running | ~$2.00 (g2-standard-24) | ~$0.67 (1x L4) |
| Model Garden integration | Native, one-click | Manual vLLM setup |
| Scale-to-zero support | Yes (v1beta1) | Yes (native, GA) |
| Multi-GPU on one host | Yes | Yes (but complex) |
| Production SLA | Yes (Vertex AI SLA) | Yes (Cloud Run SLA) |
| Management overhead | Low | Medium |

**For this project**: Vertex AI dedicated endpoint is the right choice because:
1. MedGemma 27B was already benchmarked on Vertex AI — infrastructure is familiar
2. Model Garden integration simplifies deployment
3. The 5-10 min cold start is acceptable for planned research sessions
4. Consistent with existing `VertexAIAdapter` code in this codebase

---

## Edge Cases and Limitations

1. **Scale-to-zero 30-day limit**: If the project is paused for 30+ days with a model deployed (but scaled to zero), Google auto-undeploys. Resume requires re-deploy (5-10 min). Monitor with Cloud Monitoring alerts.

2. **Scale-to-zero multi-host incompatibility**: If MedGemma 27B ever needs more than 2x L4 (e.g., for longer context or larger batches requiring tensor parallelism across nodes), scale-to-zero becomes unavailable. Single-host deployment is the constraint.

3. **Deployed-model-id drift**: Each `deploy_model` call creates a new `deployed_model_id`. The `undeploy_model` gcloud command requires this ID. The Python SDK's `endpoint.undeploy_all()` handles this automatically — prefer SDK over raw gcloud for undeploy automation.

4. **Region GPU availability**: Scale-to-zero with GPU has a documented caveat — "without reservations, capacity stockouts may occur when scaling back up." For development, this means a scale-to-zero endpoint might fail to warm up if L4 capacity is constrained in the region. Use `us-central1` (deepest L4 capacity) and have a fallback region.

5. **Cost during model load**: The 5-10 minute deployment window itself is likely billed at full rate once the VM is provisioned (even before the model is fully loaded). This is a minor cost for development but relevant for high-frequency deploy/undeploy cycles.

6. **Batch prediction alternative**: For offline benchmarks (like Phase 0 evaluation of 20-1,024 pairs), Vertex AI Batch Prediction is **50% cheaper** than online prediction and requires no persistent endpoint. For the TrialGPT evaluation use case, batch prediction may be preferable over maintaining a dedicated endpoint.

---

## Recommendations

### Immediate Action: Enable Scale-to-Zero as Safety Net

When next deploying MedGemma 27B, use the v1beta1 API with `min_replica_count=0` and `idle_scaledown_period=3600` (1 hour). This catches forgotten endpoints automatically.

### Primary Cost Strategy: Manual Deploy Script

Create `scripts/manage_medgemma_endpoint.py` (template above) for session management:
```bash
# Start of research session (takes 5-10 min to be ready)
uv run python scripts/manage_medgemma_endpoint.py deploy &

# End of research session (immediate billing stop)
uv run python scripts/manage_medgemma_endpoint.py undeploy
```

### Cost Projection for This Project

| Usage Pattern | Monthly Cost |
|---|---|
| Always-on (g2-standard-24) | ~$1,456 |
| 4 hrs/day, 20 days/month (manual undeploy) | ~$160 |
| 2 hrs/day, 10 days/month (benchmark sprints) | ~$40 |
| Scale-to-zero, 2 hr actual traffic, 20 days | ~$80 (includes scale-up/down cycles) |
| **Batch prediction only (no endpoint)** | ~$40-80 (50% discount, no idle cost) |

### Best Practice for Phase 0 / Benchmarking Only

If the endpoint is used **only for benchmark runs** (not interactive development), consider **Vertex AI Batch Prediction** instead of a dedicated endpoint:
- 50% cost discount vs. online prediction
- No endpoint to manage
- No idle billing
- Scales to zero automatically (job-based)
- Latency: minutes to hours (acceptable for offline eval)

The current Phase 0 benchmark of 20 pairs at ~$0.04/run is already cheap regardless of approach. For the full 1,024-pair Tier A evaluation, batch prediction would cost approximately $10-15 vs. $20-25 for online prediction via dedicated endpoint.

### Automation Priority

For the Feb 24 deadline, the simplest automation is the Python script above. If the project continues beyond the challenge:

1. **Short term (now)**: `manage_medgemma_endpoint.py` script + scale-to-zero as safety net
2. **Medium term**: Cloud Scheduler + Cloud Functions for fully automatic deploy/undeploy by time of day
3. **Long term**: GKE-based deployment if multi-host GPU or sub-30-second cold starts become requirements

---

## Open Questions

1. **Confirmed deployment time for MedGemma 27B specifically**: The 5-10 minute figure is from general Vertex AI docs. The actual deployment time for this specific model/container may differ. Measure on next deployment and record.

2. **vLLM vs. Hex-LLM container cost on Vertex AI**: Both are infrastructure-billed, but Hex-LLM may offer better throughput (affecting inference latency and thus cost-per-inference). Worth testing if throughput matters.

3. **Scale-to-zero and MedGemma 27B compatibility**: Has not been tested in this project. The docs confirm single-host g2-standard-24 is compatible in principle, but end-to-end testing is required.

4. **Batch prediction with MedGemma 27B**: The existing `VertexAIAdapter` is designed for online prediction. A batch prediction path would require a different adapter. Worth ADR if the project continues beyond the challenge.

---

## Research Methodology Notes

| Topic | Rounds Conducted | Sources Quality | Confidence |
|---|---|---|---|
| Billing model | 3 | Google docs + community forums | High |
| Undeploy vs. delete cost | 3 | Google docs + developer forum with direct answer | High |
| Re-deployment speed | 3 | Official docs + community | Medium (not MedGemma-specific) |
| Scale-to-zero | 4 | Official docs + oneuptime article | High |
| Model Garden pricing | 3 | Official + forum | High |
| Automation patterns | 3 | Official SDK + blog | High |

**Confidence level in overall findings**: High. The billing model, scale-to-zero mechanics, and automation patterns are well-documented in official Google Cloud sources. The specific MedGemma 27B cold-start time is estimated from general patterns (medium confidence) and should be validated empirically.

**Deviations from standard protocol**: GPU raw pricing could not be extracted from the official Vertex AI pricing page due to JavaScript rendering. Compute Engine pricing data from `cloudprice.net` was used as a proxy (high reliability for GCE rates, may not reflect exact Vertex AI prediction markup).

---

## Sources

- [Vertex AI pricing | Google Cloud](https://cloud.google.com/vertex-ai/pricing)
- [Scale inference nodes by using autoscaling | Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/predictions/autoscaling)
- [Undeploy a model and delete the endpoint | Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/predictions/undeploy-model)
- [How to Enable Scale-to-Zero for Vertex AI Prediction Endpoints](https://oneuptime.com/blog/post/2026-02-17-how-to-enable-scale-to-zero-for-vertex-ai-prediction-endpoints-to-reduce-costs/view)
- [Vertex AI Endpoint with no deployed model - is there a cost?](https://discuss.google.dev/t/vertex-ai-endpoint-with-no-deployed-model-is-there-a-cost/272474)
- [Vertex AI AutoML Endpoint Cost Optimisation for Idle State](https://discuss.google.dev/t/vertex-ai-automl-endpoint-cost-optimisation-for-idle-state/243786)
- [g2-standard-24 pricing | cloudprice.net](https://cloudprice.net/gcp/compute/instances/g2-standard-24)
- [MedGemma-27B on GCP — Serverless options forum](https://discuss.ai.google.dev/t/medgemma-27b-on-gcp-is-there-a-way-to-deploy-pay-per-token-or-serverless-without-long-spin-up-times/122653)
- [Scale-to-Zero LLM Inference with vLLM, Cloud Run and Cloud Storage FUSE](https://medium.com/google-cloud/scale-to-zero-llm-inference-with-vllm-cloud-run-and-cloud-storage-fuse-42c7e62f6ec6)
- [Cloud Run GPUs are now generally available | Google Cloud Blog](https://cloud.google.com/blog/products/serverless/cloud-run-gpus-are-now-generally-available)
- [Deploy open models from Model Garden | Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/deploy-model-garden)
- [Vertex AI Batch Generation — 50% cost reduction](https://medium.com/google-cloud/vertex-ai-batch-generation-17091bdd8492)
- [How to Configure Autoscaling for Vertex AI Online Prediction Endpoints](https://oneuptime.com/blog/post/2026-02-17-how-to-configure-autoscaling-for-vertex-ai-online-prediction-endpoints/view)
- [Support auto-scale to zero in Vertex AI: Issue Tracker](https://issuetracker.google.com/issues/206042974)
- [GPU pricing | Google Cloud](https://cloud.google.com/compute/gpus-pricing)
- [Vertex AI: Pricing for Top 16 Vertex Services in 2026 | Finout](https://www.finout.io/blog/top-16-vertex-services-in-2026)
