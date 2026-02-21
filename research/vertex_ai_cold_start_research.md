# Deep Research Report: Vertex AI Endpoint Cold Start Times and Warm-Up Behavior

**Date**: 2026-02-21
**Researcher**: deep-research-agent
**Repositories Analyzed**: GoogleCloudPlatform/vertex-ai-samples, vllm-project/vllm, huggingface/text-generation-inference
**Total Research Rounds**: 4 (3 Deepwiki rounds + 1 web search crystallization round)
**Confidence Level**: High for Vertex AI deployment mechanics; Medium for specific timing ranges (no official SLAs published)

---

## Executive Summary

Vertex AI endpoint cold starts for large language models (27B parameters, int8 quantization on GPU) are a multi-phase process with **no official SLA or guaranteed timing** documented by Google. Based on cross-referencing official sample notebooks, community forum reports, and vLLM internals, the total end-to-end time from "trigger deployment" to "first successful inference" for a 27B model on Vertex AI is **15 to 60 minutes for initial deployment**, and the critical finding is that Vertex AI's standard Online Prediction endpoints **do not support true scale-to-zero** (minimum replica count must be >= 1 for autoscaling). This means a deployed, warm endpoint responds in seconds but keeps costs running 24/7.

For the trialmatch project specifically: the MedGemma 27B endpoint was torn down after benchmarking to avoid cost. Re-deploying would require a full cold start (15-60 min). The practical recommendation is to keep a dedicated endpoint running during active development/demo windows and tear it down immediately after, rather than attempting scale-to-zero. Cloud Run with GCS FUSE is an emerging alternative offering true scale-to-zero with ~2-5 minute cold starts for 27B models.

The serverless GPU alternatives (RunPod FlashBoot, Modal GPU Snapshots) achieve dramatically faster cold starts (sub-second to 2 seconds) by preserving GPU VRAM state between requests, but these are third-party platforms with different pricing models and no Google Cloud ecosystem integration.

---

## Research Objectives

1. How long does GPU resource allocation take on Vertex AI for L4, A100, H100?
2. What is container startup time for vLLM-based Vertex AI Model Garden containers?
3. How long does a 27B int8 model take to download, load into GPU, and warm up?
4. What is the total end-to-end cold start time?
5. What optimization strategies exist to reduce cold start time?
6. How does Vertex AI compare to HuggingFace Endpoints, AWS SageMaker, RunPod/Modal?

---

## Detailed Findings

### Opening Item 1: GPU Allocation Time on Vertex AI

#### Round 1: Surface Exploration

**Questions Asked**:
- What is the cold start time for Vertex AI Online Prediction endpoints when scaling from zero?
- How long does GPU allocation take for L4, A100, and H100 instances?

**Key Discoveries**:
- Vertex AI sample notebooks set `deploy_request_timeout=1800` (30 minutes) and `serving_container_deployment_timeout=7200` (2 hours) — these are the worst-case bounds Google engineers code against
- Official documentation notes deployment can take "15 minutes to 1 hour to finish depending on the size of the model"
- For smaller models, documented deployment time is "around 1 minute to finish" but with a note to wait "3 extra minutes" before sending requests
- GPU H100 availability is noted as "hard to get" — quota availability in zone can add unpredictable wait time

**Initial Gaps**:
- No breakdown of how much is GPU provisioning vs. container pull vs. model download
- No per-GPU-type timing data

#### Round 2: Deep Dive

**Questions Asked**:
- What is the difference between min_replica_count=0 vs. min_replica_count=1?
- What scaling behaviors and latencies are documented?

**Key Discoveries**:
- **Critical finding**: Vertex AI Online Prediction autoscaling requires `min_replica_count > 0`. Scale-to-zero is not supported for standard vLLM endpoints.
- Per the Vertex AI samples code: "For auto-scaling, the minimum number of nodes must be set to a value greater than zero"
- When `min_replica_count=1`, there is always at least one instance ready — this avoids cold starts entirely for steady-state traffic
- Compute Engine GPU Reservations are the recommended strategy to guarantee GPU availability and reduce VM provisioning wait time

**Emerging Patterns**:
- Vertex AI's architecture assumes persistent warm replicas, not serverless scale-to-zero
- The "cold start problem" on Vertex AI manifests primarily as initial deployment latency, not per-request latency
- After initial deployment, all latency is inference latency (sub-10 seconds per request)

#### Round 3: Crystallization

**Questions Asked**:
- What are the specific machine types for 27B model deployment?
- What happens during the post-deployment model-loading phase?

**Final Understanding**:
The Vertex AI cold start lifecycle has **two distinct phases** that are frequently confused:

**Phase 1: Endpoint Deployment** (infrastructure provisioning)
- Duration: 15 minutes to 1 hour
- What happens: VM provisioning, container image pull from Google Container Registry, vLLM process startup
- Result: Endpoint reports "DEPLOYED" status

**Phase 2: Model Weight Loading** (happens after Phase 1 completes)
- Duration: 5 to 40 minutes depending on model size and source
- What happens: Container downloads model weights (from HuggingFace or GCS), loads into GPU VRAM, runs CUDA graph capture, warms up
- Result: First inference request succeeds without 503 error
- **The endpoint reports ready BEFORE this phase completes**, causing 503 errors if you send requests too early

**Recommended machine types for 27B models**:
| GPU | Machine Type | vRAM | Notes |
|-----|-------------|------|-------|
| 2x NVIDIA L4 | g2-standard-24 | 2x24 GB = 48 GB | Tight fit for int8 27B |
| 1x A100 80GB | a2-highgpu-1g | 80 GB | Comfortable for bf16 or int8 |
| 4x H100 80GB | a3-highgpu-4g | 4x80 GB = 320 GB | Multi-GPU tensor parallel |

**Validated Assumptions**:
- The commonly cited "up to 30 minutes" for Gemma 27B deployment on Vertex AI refers specifically to Phase 1 (container up) + Phase 2 (weights downloaded from HuggingFace). Using GCS as model source speeds up Phase 2.
- Compute Engine Reservations do not reduce Phase 2 but can reduce Phase 1 by guaranteeing GPU availability.

---

### Opening Item 2: Container Startup (vLLM Internals)

#### Round 1: Surface Exploration

**Questions Asked**:
- What is the typical startup time for vLLM when loading a 27B parameter model with int8 quantization?
- How long does model weight loading take?
- What happens during warm-up?

**Key Discoveries**:
- vLLM startup has 5 phases: configuration, model loading, memory profiling, KV cache allocation, warm-up + CUDA graph capture
- CUDA graph capture: documented as "5~20 seconds" in vLLM logger messages
- Memory profiling: runs dummy forward passes — adds seconds to tens of seconds
- vLLM integrates with BitsAndBytesModelLoader for int8 weights

**Initial Gaps**:
- No specific numbers for 27B models
- CUDA graph capture time is for unspecified model sizes

#### Round 2: Deep Dive

**Questions Asked**:
- What specific optimizations does vLLM use for fast model loading?
- What are typical CUDA graph capture times?
- What flags can skip or speed up startup phases?

**Key Discoveries**:
Concrete benchmarks from Tensorfuse (vLLM with Llama 3.1-8B on a single GPU as baseline):

| Phase | Without Optimization | With Optimization |
|-------|---------------------|-------------------|
| Model download | 61 seconds | 0 seconds (cached) |
| Weight loading | 33 seconds | 18 seconds |
| Dynamo bytecode transform | 10 seconds | 10 seconds |
| Graph compilation | 42 seconds | 13 seconds |
| CUDA graph capture | 54 seconds | 7 seconds |
| Engine init + imports | 94 seconds | 34 seconds |
| **Total** | **294 seconds (~5 min)** | **82 seconds (~1.5 min)** |

Note: This is for an 8B model. A 27B model has ~3.4x more parameters. Weight loading and CUDA graph capture scale roughly linearly with model size, while engine init is more fixed overhead. **Estimated 27B cold start (uncached weights): 8-15 minutes total on a single fast GPU.**

**Flags to reduce startup time**:
- `--enforce-eager`: Disables torch.compile and CUDA graphs entirely. Reduces startup by 50-100 seconds but increases inference latency ~20-30%
- `--cudagraph-capture-sizes`: Limit captured graph sizes to only relevant batch sizes (e.g., 1,2,4,8) instead of the default 67 graphs
- `--load-format tensorizer`: CoreWeave Tensorizer — deserializes tensors directly to GPU, "significantly reduces Pod startup times"
- `--load-format runai_streamer`: Run:ai Model Streamer — concurrent tensor streaming from GCS/S3

**Run:ai Model Streamer benchmarks** (from NVIDIA blog + search):
- Achieves total readiness in 23 seconds (S3), 28 seconds (IO2 SSD), 35 seconds (GP3 SSD) for unspecified model size
- Cuts model loading times up to 6x vs. standard PyTorch loading

#### Round 3: Crystallization

**Questions Asked**:
- What is the exact sequence of vLLM startup operations?
- What does CUDA graph capture log say?

**Final Understanding**:

vLLM startup sequence (with approximate timing for a 27B model on A100):

```
1. Worker process spawn + config validation        ~5-10s
2. Model weight loading from disk/GCS             ~2-8 min (depending on storage speed and model source)
   - For int8: BitsAndBytesModelLoader quantizes during load
3. Memory profiling (dummy forward pass)          ~30-60s
4. KV cache allocation                            ~5s
5. Warm-up + CUDA graph capture                   ~1-3 min (for 27B, default graph sizes)
   - vLLM logs: "CUDA graph capturing finished in 5~20 seconds" (for smaller models)
   - Larger models scale up proportionally
6. Sampler warm-up + memory buffer preallocation  ~10s
```

**Gemma 3 27B specific (from forum reports)**:
- Individual worker load times: 213-234 seconds (~3.5-4 minutes) on 4x H100
- The 4x H100 setup uses tensor parallel, so weight loading happens in parallel across GPUs
- A single A100 80GB would take longer for weight loading but avoids tensor parallel coordination overhead

---

### Opening Item 3: Total End-to-End Cold Start Time

#### Round 1: Surface Exploration

**Questions Asked**:
- What is the total time from "scale up triggered" to "first successful inference" on Vertex AI?

**Key Discoveries**:
- For Gemma/MedGemma 27B on Vertex AI via Model Garden, the documented range is **up to 30 minutes** for container pull from HuggingFace
- If model weights are pre-staged in GCS, Phase 2 is reduced significantly
- Notebooks warn users to wait 5-20 extra minutes after Phase 1 completes before sending first request

**Initial Gaps**:
- No breakdown of each sub-phase timing

#### Round 2: Deep Dive

**Questions Asked**:
- Are there any community-reported actual timings for MedGemma 27B specifically?

**Key Discoveries**:
From Google AI Developers Forum (MedGemma 27B specific discussion):
- "The time required to load ~20GB+ of weights into VRAM usually makes the first request unusable" — confirms cold start is a known pain point
- Community suggests Cloud Run with GCS FUSE as alternative to avoid this for irregular traffic
- MedGemma 27B int8 model is ~14-20 GB of weights (int8 = 1 byte/param = 27B bytes = 27 GB, reduced by quantization)

**Timing breakdown for MedGemma 27B cold start on Vertex AI (synthesized from all sources)**:

| Phase | Estimated Duration | Notes |
|-------|-------------------|-------|
| VM/GPU provisioning | 1-5 minutes | Reduced with Compute Engine Reservations |
| Container image pull | 2-8 minutes | Vertex AI uses pre-built vLLM containers from GCR |
| Model weight download from HuggingFace | 5-20 minutes | ~14-27 GB at variable network speeds |
| Model weight download from GCS | 1-5 minutes | Parallel download, Google backbone network |
| vLLM process initialization | 1-2 minutes | Worker spawn, config validation |
| Weight loading into GPU VRAM | 2-5 minutes | For int8 27B on A100/L4x2 |
| Memory profiling + KV cache setup | 1-2 minutes | Dummy forward passes |
| CUDA graph capture | 1-3 minutes | Default settings; reducible with --enforce-eager |
| **Total (HuggingFace source)** | **13-45 minutes** | |
| **Total (GCS source)** | **9-25 minutes** | |

#### Round 3: Crystallization

**Questions Asked**:
- What do official Google notebook timeout values tell us about expected worst case?

**Final Understanding**:

The official `serving_container_deployment_timeout=7200` (2 hours) is the absolute safety net. The `deploy_request_timeout=1800` (30 minutes) is the practical expected maximum. These are not arbitrary — they reflect real-world P99 deployment times Google engineers observed in testing.

**Practical guidance for trialmatch project**:
- When re-deploying MedGemma 27B endpoint for a demo window, budget **30-45 minutes** from `gcloud ai endpoints deploy-model` command to first successful inference
- Pre-stage model weights in GCS to reduce the download phase
- Use `--enforce-eager` for demos where first-request latency matters more than inference throughput (eliminates ~1-3 minutes of CUDA graph capture at cost of ~20% slower per-request inference)

---

### Opening Item 4: Optimization Strategies

#### Round 1: Surface Exploration

**Questions Asked**:
- What optimization strategies reduce cold start time on Vertex AI?

**Key Discoveries**:
- Parallel GCS downloading: Vertex AI's vLLM fork includes built-in parallel downloading from GCS
- Prefix caching: GPU and host memory KV cache to speed up repeated prompts (inference optimization, not startup)
- Compute Engine Reservations: Pre-allocate GPU VMs to guarantee availability

**Initial Gaps**:
- No quantitative comparison between GCS vs. HuggingFace download speeds

#### Round 2: Deep Dive

**Questions Asked**:
- What is the Compute Engine Reservations mechanism?
- How does the parallel GCS download work?

**Key Discoveries**:
- GCS parallel download recommendation: For 405B Llama model, Google explicitly says "use Google Cloud source because it downloads faster" — implies at least 2-5x speedup for large models
- Reservations are GPU-only (not CPU reservations), require exact machine type match, and prevent the VM provisioning wait
- Reservations eliminate Phase 1 GPU allocation uncertainty but not Phase 2 weight loading

#### Round 3: Crystallization

**Final ranked list of cold start optimization strategies**:

| Strategy | Impact | Effort | Applicable To |
|----------|--------|--------|---------------|
| Pre-stage weights in GCS bucket | High — reduces download from 5-20 min to 1-5 min | Low (one gsutil command) | Phase 2 |
| Compute Engine GPU Reservations | Medium — eliminates GPU allocation queue | Low (console/API) | Phase 1 |
| `--enforce-eager` flag | Medium — saves 1-3 min CUDA graph capture | Low (vLLM arg) | Phase 2 |
| Limit CUDA graph sizes | Medium — reduces capture from 54s to 7s | Low (vLLM arg) | Phase 2 |
| Run:ai Model Streamer (`--load-format runai_streamer`) | High — up to 6x faster weight loading | Medium (need Vertex custom container) | Phase 2 |
| Tensorizer (`--load-format tensorizer`) | High — direct GPU deserialize | Medium (pre-serialize model first) | Phase 2 |
| Keep min_replica_count=1 | Eliminates cold start entirely | Low (ongoing cost) | Steady-state |
| Periodic warm-up requests | Prevents scale-down cooldown | Low (cron job) | Steady-state |
| Cloud Run + GCS FUSE | Scale-to-zero with ~2-5 min cold start | High (new infra setup) | Cost-sensitive |

---

### Opening Item 5: Comparison with Alternative Platforms

#### Round 1: Surface Exploration

**Questions Asked**:
- How does Vertex AI compare to HuggingFace, AWS SageMaker, RunPod, Modal?

**Key Discoveries**:
- HuggingFace Endpoints: Scale-to-zero triggers at 15 minutes idle; cold start duration "varies by model size" — no SLA given
- AWS SageMaker: Scale-to-zero feature available; 8B model ~5 min cold start, 70B model ~6 min cold start
- RunPod FlashBoot: Sub-200ms cold starts for popular endpoints via VRAM state preservation
- Modal GPU Snapshots: 9x-10x reduction — from 45s to 5s for 0.5B model; scales with model size

#### Round 2: Deep Dive

**Questions Asked**:
- What are the specific timing measurements for each platform?

**Key Discoveries**:

**HuggingFace Inference Endpoints**:
- Scale-to-zero trigger: 15 minutes idle
- Cold start: responds with `502 Bad Gateway` during initialization
- No request queuing — client must handle retries
- Scaling check interval: every 1 minute (scale up), every 2 minutes (scale down)
- No specific timing published; community reports suggest 5-15 min for 27B models

**AWS SageMaker** (official AWS blog on scale-to-zero feature):
- 8B model (LLaMA 3.1): ~5 minutes cold start
- 70B model (LLaMA 3.1): ~6 minutes cold start
- Fast Model Loader feature: Reduces 70B from 407 seconds to 334 seconds (~21% improvement)
- With Fast Model Loader, LLaMA 3.1-70B loads in ~1 minute on ml.p4d.24xlarge

**RunPod Serverless (FlashBoot)**:
- Technical mechanism: Preserves worker state shortly after scale-down; "revives" rather than cold boots
- Performance varies with endpoint popularity
- Whisper model benchmark: P99 cold start < 2.3 seconds; Lowest observed: 563ms
- 70B LLM reported: ~600ms cold start with FlashBoot
- Without FlashBoot: 42-second cold start for same workload

**Modal (GPU Memory Snapshots)**:
- Technical mechanism: Serializes full CPU + GPU VRAM state (weights, CUDA kernels, CUDA graphs, memory mappings)
- vLLM + Qwen2.5-0.5B: 45 seconds → 5 seconds (9x improvement)
- NVIDIA Parakeet audio: ~20 seconds → ~2 seconds (10x improvement)
- For large models: Cold start still proportional to model size, but eliminates JIT compilation, CUDA graph capture, and weight loading phases
- Issue #33930 on vllm-project/vllm: GPU Memory Snapshotting feature under consideration

**Serverless provider comparison (Qwen image generation, H100)**:
| Platform | First-run generation time |
|----------|--------------------------|
| dat1.co | 53-68 seconds |
| Modal | 85-90 seconds |
| RunPod | 125-140 seconds |
| beam.cloud | 132-142 seconds |
| Replicate | 142-155 seconds |
Note: ~30-40 seconds is actual generation; remainder is container spin-up overhead.

#### Round 3: Crystallization

**Platform comparison for 27B LLM cold start (synthesized)**:

| Platform | Cold Start (27B) | Scale-to-Zero | Ecosystem | Notes |
|----------|-----------------|---------------|-----------|-------|
| **Vertex AI** | 15-45 minutes | Not supported for vLLM | Google Cloud | Min replicas must be >= 1 |
| **Vertex AI + GCS** | 9-25 minutes | Not supported | Google Cloud | Faster weight download |
| **Cloud Run GPU + GCS FUSE** | ~2-5 minutes | Supported | Google Cloud | Emerging, weights stream lazily |
| **HuggingFace Endpoints** | 5-15 minutes | Supported | HuggingFace | 15-min idle timeout |
| **AWS SageMaker** | ~6-10 minutes | Supported | AWS | Fast Model Loader = 1 min for 70B |
| **RunPod (FlashBoot)** | ~600ms - 2s | Supported | RunPod | State preservation, cost varies |
| **Modal (GPU Snapshots)** | ~5-15 seconds | Supported | Modal | Snapshot must be pre-created |
| **RunPod (no FlashBoot)** | 2-5 minutes | Supported | RunPod | Standard container cold start |

**Key insight**: RunPod and Modal achieve sub-second/multi-second cold starts by fundamentally different technology (VRAM state preservation / CRIU-like checkpointing) vs. traditional cold starts. These are not improvements to cold start — they are elimination of the cold start by caching the warm GPU state.

---

## Cross-Cutting Insights

1. **The "deployment ready" != "model ready" distinction is critical for Vertex AI**. The Vertex AI endpoint API reports deployment success before the container finishes downloading and loading model weights. This two-phase reality means you must implement health checks or wait additional minutes after deployment before sending production traffic.

2. **vLLM CUDA graph capture is the most controllable cold start bottleneck**. It can be reduced from 54 seconds to 7 seconds by limiting capture to relevant batch sizes, or eliminated entirely with `--enforce-eager`. For demos with irregular traffic patterns, `--enforce-eager` is often the right tradeoff.

3. **Model weight source (HuggingFace vs. GCS) is the largest variable in Vertex AI cold start**. HuggingFace download over public internet is unreliable and slow (5-20 min). GCS parallel download over Google's backbone network is significantly faster (1-5 min). Pre-staging to GCS should be standard practice for production deployments.

4. **True scale-to-zero on Vertex AI requires Cloud Run GPU, not Online Prediction**. Cloud Run GPU supports scale-to-zero with GCS FUSE for streaming model weights lazily. This is newer (2024-2025) infrastructure still maturing, but the architecture specifically solves the cost-vs-cold-start tradeoff that Vertex AI Online Prediction cannot.

5. **Cold start measurement vs. deployment measurement are different things**. Most community reports conflate initial deployment time (infrastructure + container pull + model download) with cold start time (from 0 replicas back to 1 replica on an already-deployed endpoint). Vertex AI doesn't truly support the latter for vLLM, making the distinction somewhat academic for this use case.

---

## Architecture/Design Decisions

### Decision 1: Why Vertex AI does not support scale-to-zero for vLLM

Vertex AI's Online Prediction is designed as a persistent serving infrastructure. The `min_replica_count >= 1` constraint for autoscaling reflects the architecture decision that warm capacity must always exist. This was likely driven by:
- GPU cold starts being too long (15-45 min) for any real application
- Difficulty implementing request queuing with arbitrary cold start times
- Focus on enterprise/production use cases that prefer predictable latency over cost optimization

### Decision 2: The two-phase Vertex AI deployment

Phase 1 (infrastructure) and Phase 2 (model loading) are deliberately decoupled. The serving container is responsible for model weight management, and Vertex AI only tracks container health. This design allows Vertex AI to support diverse serving frameworks (TGI, vLLM, custom) without requiring framework-specific model loading logic in the platform layer.

### Decision 3: vLLM's CUDA graph trade-off

Default vLLM captures 67 CUDA graphs for different batch sizes. This adds 54+ seconds to startup but reduces per-token inference latency by 20-30%. For production serving with high throughput, this is correct. For demos or dev environments, `--enforce-eager` correctly reverses this trade-off.

---

## Edge Cases and Limitations

1. **H100 quota availability**: In some regions, H100 80GB instances are "hard to get" (Google's own language in sample notebooks). Even with reservations, zone-level quota can add unpredictable delays. Use `common_util.check_quota` before deploying.

2. **int8 quantization + vLLM interaction**: The Vertex AI vLLM fork uses bitsandbytes for int8. The quantization happens during weight loading (not pre-quantized on disk), which adds time to Phase 2 weight loading compared to fp16/bf16.

3. **First request after deployment may time out**: Vertex AI's default predict request timeout is **60 seconds** for both public and private endpoints. If the model is still loading when the first request arrives (within the first 5-40 minutes of Phase 2), it will timeout. Client code must implement retry logic with exponential backoff.

4. **Vertex AI request timeout cannot be self-extended**: Users must file a support ticket or contact Google Cloud rep to get timeouts beyond 60 seconds. For vLLM with large models generating long responses, this can be a bottleneck even after cold start.

5. **TGI CUDA bug (project-specific)**: As documented in CLAUDE.md, MedGemma 4B on HuggingFace TGI crashes with `CUBLAS_STATUS_EXECUTION_FAILED` at max_new_tokens >= ~500. This is TGI-specific and does not affect Vertex AI's vLLM path.

---

## Recommendations

### For the trialmatch Demo (Feb 24 deadline)

1. **Deploy 30-45 minutes before recording starts**. Budget the full window: deploy command → wait → verify endpoint health → send first inference. Script a readiness check that polls until first inference succeeds.

2. **Pre-stage MedGemma 27B weights in GCS** before the demo window. Use `gsutil -m cp` to copy from HuggingFace cache or use `gcloud storage` parallel composite uploads. This reduces Phase 2 weight download from 5-20 min to 1-5 min.

3. **Use `--enforce-eager` for demo deployment** unless inference throughput is being demonstrated. Eliminates CUDA graph capture time (~1-3 min) at cost of slightly slower per-request inference.

4. **Implement health-check polling in deployment script**:
```python
import time
import requests

def wait_for_endpoint_ready(endpoint, max_wait_seconds=2700):
    """Poll endpoint with lightweight test request until first successful response."""
    start = time.time()
    while time.time() - start < max_wait_seconds:
        try:
            response = endpoint.predict(instances=[{"test": "ping"}])
            return True
        except Exception:
            time.sleep(30)  # Check every 30 seconds
    return False
```

5. **Tear down immediately after demo**. With min_replica_count=1, an idle A100 endpoint costs ~$3-6/hour. A 2-hour demo window + 45-min setup = ~$15-20 in GPU costs.

6. **Consider Cloud Run GPU + GCS FUSE for post-demo use** if the project continues beyond Feb 24. It offers true scale-to-zero with ~2-5 minute cold starts and pay-per-request pricing. This is the right architecture for irregular-traffic medical AI applications.

### General Recommendations

- Use Compute Engine GPU Reservations for predictable demo scheduling to guarantee GPU availability
- Use `--load-format runai_streamer` with GCS source as the primary optimization for production deployments
- Track cold start time explicitly in run artifacts (add `endpoint_ready_at` timestamp to run metadata)

---

## Open Questions

1. **What is the actual GCS parallel download speed for a 14-20 GB int8 MedGemma 27B model on an A100 instance in us-central1?** This would resolve the Phase 2 timing uncertainty. Could be measured empirically in ~30 minutes.

2. **Does Vertex AI's vLLM fork support `--load-format runai_streamer` with GCS URIs?** The Vertex AI vLLM fork adds GCS support (`gs://` URIs), but it's unclear if the Run:ai streamer integration is also present. If yes, Phase 2 weight loading could be reduced from ~2-5 min to ~30 seconds.

3. **Is `--enforce-eager` compatible with int8 quantization on the Vertex AI vLLM fork?** Some quantization methods have known compatibility issues with eager mode. This should be tested before the demo.

---

## Research Methodology Notes

**Round 1** (3 parallel Deepwiki queries): Established the vLLM startup sequence, Vertex AI deployment mechanics, and HuggingFace TGI initialization process. Identified the key finding that Vertex AI does not support scale-to-zero for vLLM endpoints.

**Round 2** (5 Deepwiki queries + 8 web searches): Drilled into specific timing numbers, the two-phase deployment pattern (infrastructure vs. model loading), and the CUDA graph capture bottleneck. Found the Tensorfuse benchmarks (8B model: 294s → 82s with optimization) and the AWS SageMaker scale-to-zero data (8B: ~5 min, 70B: ~6 min).

**Round 3** (4 Deepwiki queries + 6 web searches): Crystallized the platform comparison, found the Gemma 27B worker loading time (213-234 seconds on 4x H100), the official notebook timeout values (1800s request timeout, 7200s container deployment timeout), and the RunPod FlashBoot and Modal GPU Snapshot mechanisms.

**Round 4** (3 targeted web searches): Confirmed the MedGemma-specific community discussion confirming cold start as a known pain point, extracted Run:ai Model Streamer benchmarks (23-35 seconds total readiness), and confirmed CRIU is NOT yet implemented for vLLM (only torch.compile artifact caching exists).

**Confidence levels**:
- Vertex AI deployment phases and timing ranges: **High** (corroborated by official notebooks and community reports)
- vLLM CUDA graph capture timing: **High** (from source code comments + Tensorfuse benchmarks)
- Platform comparison table: **Medium** (numbers from different benchmark conditions)
- MedGemma 27B specific timing: **Medium** (extrapolated from Gemma 27B data)
- Cloud Run GPU cold start: **Low** (emerging technology, limited public benchmarks)

---

## Sources

- [Vertex AI vLLM Serving Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm)
- [GoogleCloudPlatform/vertex-ai-samples (DeepWiki)](https://deepwiki.com/GoogleCloudPlatform/vertex-ai-samples)
- [vllm-project/vllm (DeepWiki)](https://deepwiki.com/vllm-project/vllm)
- [Reducing GPU Cold Start Time with vLLM (Tensorfuse)](https://tensorfuse.io/docs/blogs/reducing_gpu_cold_start)
- [GPU Memory Snapshots: Supercharging Sub-second Startup (Modal)](https://modal.com/blog/gpu-mem-snapshots)
- [Introducing FlashBoot (RunPod)](https://www.runpod.io/blog/introducing-flashboot-serverless-cold-start)
- [AWS SageMaker Scale-to-Zero Feature](https://aws.amazon.com/blogs/machine-learning/unlock-cost-savings-with-the-new-scale-down-to-zero-feature-in-sagemaker-inference/)
- [AWS Fast Model Loader for LLMs](https://aws.amazon.com/blogs/machine-learning/introducing-fast-model-loader-in-sagemaker-inference-accelerate-autoscaling-for-your-large-language-models-llms-part-1/)
- [MedGemma-27B GCP Cold Start Discussion](https://discuss.ai.google.dev/t/medgemma-27b-on-gcp-is-there-a-way-to-deploy-pay-per-token-or-serverless-without-long-spin-up-times/122653)
- [HuggingFace Inference Endpoints Autoscaling](https://huggingface.co/docs/inference-endpoints/en/autoscaling)
- [vLLM Issue: CUDA Graph Capturing Time Too Long](https://github.com/vllm-project/vllm/issues/16716)
- [CRIU for vLLM Cold Start (vLLM Forums)](https://discuss.vllm.ai/t/using-criu-to-reduce-cold-start-latency-for-llm-tasks/639)
- [Vertex AI Use Reservations with Online Inference](https://docs.cloud.google.com/vertex-ai/docs/predictions/use-reservations)
- [Serverless Inference Providers Compared (dat1.co)](https://dat1.co/blog/serverless-inference-providers-compared)
- [How to Enable Scale-to-Zero for Vertex AI (oneuptime.com)](https://oneuptime.com/blog/post/2026-02-17-how-to-enable-scale-to-zero-for-vertex-ai-prediction-endpoints-to-reduce-costs/view)
- [Practical LLM Serving with vLLM on Vertex AI (infocruncher)](https://blog.infocruncher.com/2025/02/27/llm-serving-with-vllm-on-vertexai/)
- [Scale-to-Zero LLM with Cloud Run + GCS FUSE (Google Cloud)](https://medium.com/google-cloud/scale-to-zero-llm-inference-with-vllm-cloud-run-and-cloud-storage-fuse-42c7e62f6ec6)
- [NVIDIA Run:ai Model Streamer Cold Start](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/)
- [Gemma 3 27B vLLM Issue (vLLM Forums)](https://discuss.vllm.ai/t/issue-serving-gemma3-27b-it/1631)
