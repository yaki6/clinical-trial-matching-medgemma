# MedPix Multimodal Benchmark Plan: MedGemma 4B vs Gemini 3 Pro

**Date**: 2026-02-22  
**Status**: READY TO EXECUTE  
**Goal**: Benchmark MedGemma 4B (multimodal) against Gemini 3 Pro (multimodal) on radiology diagnosis prediction and imaging findings extraction using MedPix ground-truth data.

---

## 1. Benchmark Objective

Evaluate two multimodal models on a **diagnosis + findings extraction** task:

- **Input**: Radiology image (CT/MRI/X-ray PNG) + clinical history text
- **Output**: Predicted diagnosis + extracted imaging findings
- **Ground truth**: MedPix `Case Diagnosis` and `Findings` fields (expert-authored)
- **Scoring**: LLM-as-judge (Gemini) for diagnosis accuracy; ROUGE-L for findings overlap

### Why This Benchmark Matters

1. MedGemma 4B has a medical-domain SigLIP image encoder — this tests whether it provides meaningful advantage over a general-purpose model (Gemini 3 Pro) on real clinical imaging tasks
2. Both models receive identical image + text inputs — true apples-to-apples multimodal comparison
3. MedPix provides expert-authored diagnosis and radiology findings — no annotation work needed

---

## 2. Dataset Inventory

### 2.1 Primary Dataset

| Asset | Path | Description |
|-------|------|-------------|
| **Full MedPix dataset** | `ingest_design/patient-ehr-image-dataset/full_dataset.jsonl` | 671 cases, JSONL, includes all MedPix cases with clinical text + image references |
| **MedPix images** | `ingest_design/MedPix-2-0/images/` | 2,050 PNG files, 100% coverage, 0 missing |
| **Case metadata** | `ingest_design/MedPix-2-0/Case_topic.json` | Rich clinical ground truth per case |
| **Image metadata** | `ingest_design/MedPix-2-0/Descriptions.json` | Per-image `Caption`, `Modality`, `Plane`, `Location`, `location_category` |

### 2.2 Dataset Counts (Verified)

| Metric | Count |
|--------|-------|
| Total MedPix cases | **671** |
| Cases with non-empty diagnosis | **671 (100%)** |
| Cases with non-empty findings | **654 (97.5%)** |
| **Thorax cases with diagnosis** | **132** |
| Head cases | 223 |
| Abdomen cases | 131 |
| Spine & Muscles cases | 126 |
| Reproductive & Urinary cases | 59 |
| Total images on disk | **2,050 PNGs** |

### 2.3 Ground Truth Fields (per case in `full_dataset.jsonl`)

| Field | Type | Benchmark Role | Example |
|-------|------|---------------|---------|
| `diagnosis` | string | **Primary GT — diagnosis accuracy** | "Adenocarcinoma of the Lung" |
| `findings` | string | **Primary GT — findings extraction** | Free-text radiology report describing imaging findings |
| `history` | string | **Model input** — clinical context | "62-year-old female with persistent cough..." |
| `images[].file_path` | string | **Model input** — radiology image path | `MedPix-2-0/images/MPX1016_synpic34317.png` |
| `images[].modality` | string | Metadata | "CT - noncontrast", "MRI", "XR" |
| `images[].location_category` | string | **Filter field** | "Thorax", "Head", "Abdomen" |
| `images[].caption` | string | Secondary GT for image description | Per-image caption from radiologist |
| `differential_diagnosis` | string | Secondary GT | Ranked differential list |
| `title` | string | Short diagnosis summary | Case title |

### 2.4 Critical Data Path Issue (FLAW IDENTIFIED & FIX)

**Flaw**: Image `file_path` in the JSONL is relative to `ingest_design/` (e.g., `MedPix-2-0/images/MPX1016_synpic34317.png`), but the benchmark script will run from the project root. Path resolution requires prepending `ingest_design/`.

**Fix**: The benchmark data loader MUST resolve paths as:
```python
resolved_path = Path("ingest_design") / case["images"][i]["file_path"]
```

This was verified: `ingest_design/MedPix-2-0/images/MPX1024_synpic40275.png` exists=True, while `MedPix-2-0/images/MPX1024_synpic40275.png` exists=False from the project root.

---

## 3. Evaluation Scope & Phasing

### Phase 1: Pilot (10 Thorax cases)
- 10 deterministically sampled Thorax cases (seed=42)
- Filter: `location_category == "Thorax"` AND `has_diagnosis == true` AND `has_findings == true`
- Purpose: validate pipeline, check prompt quality, tune if needed
- Estimated cost: ~$1-2

### Phase 2: Full Thorax (132 cases)
- All 132 Thorax cases with diagnosis
- Purpose: statistically meaningful comparison on the body region most relevant to our NSCLC use case
- Estimated cost: ~$10-15

### Phase 3: All-Disease (650+ cases)
- All 654 cases with both diagnosis and findings
- Purpose: comprehensive cross-specialty evaluation
- Estimated cost: ~$40-60

**Decision**: Start with Phase 1 (10 cases), validate outputs, then scale.

---

## 4. Model Configuration

### 4.1 MedGemma 4B (Vertex AI Model Garden)

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Deployment** | Vertex AI Model Garden | Avoids HF TGI CUDA bug (max_tokens=512 limit causes hallucinated repetitive output) |
| **Deploy command** | `bash scripts/deploy_vertex.sh --model-size=4b` | Already supports 4B path: `google/medgemma@medgemma-4b-it` |
| **GPU** | 1× L4 (24GB) via `g2-standard-8` | 4B at fp16 ≈ 8GB; no quantization needed |
| **Multimodal API** | OpenAI-compatible `image_url` format via vLLM | vLLM on Vertex accepts `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` |
| **max_tokens** | 2048 | Vertex vLLM has no TGI CUDA bug — full token budget available |
| **Hourly cost** | ~$1.15/hr (L4) | Must tear down immediately after benchmark |
| **Config** | `configs/phase0_vertex_4b.yaml` (modify for multimodal) | Existing config for text-only; needs multimodal extension |

### 4.2 Gemini 3 Pro (Google AI Studio)

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Deployment** | Google AI Studio API | Already working, no deployment needed |
| **Multimodal API** | `google.genai` SDK with `Part.from_bytes()` | Native multimodal; no adapter changes for the API itself |
| **max_tokens** | 32768 (auto-scaled for thinking model) | Existing adapter handles this |
| **Cost** | ~$1.25/1M input + $10/1M output tokens | Negligible for 10-132 cases |
| **Concurrency** | 10 (rate limit) | Existing adapter handles retry/backoff |

### 4.3 LLM-as-Judge (Gemini — separate text-only calls)

| Setting | Value |
|---------|-------|
| **Model** | Gemini 3 Flash (cheaper, sufficient for judging) |
| **Task** | Compare predicted diagnosis vs gold diagnosis for semantic equivalence |
| **Output** | `{"score": "correct|partial|incorrect", "explanation": "..."}` |
| **Safeguard** | Judge model is different from evaluated model (Flash judges Pro's predictions too) |

---

## 5. Required Code Changes

### 5.1 Add `generate_with_image()` to GeminiAdapter (NEW)

**File**: `src/trialmatch/models/gemini.py`

**What**: Override `generate_with_image()` using `google.genai` SDK's `Part`-based content structure.

**Implementation**:
```python
async def generate_with_image(
    self, prompt: str, image_path: Path, max_tokens: int = 2048
) -> ModelResponse:
    """Multimodal generation: image + text via Gemini."""
    import base64
    from google.genai import types as genai_types

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime_type = "image/png" if str(image_path).endswith(".png") else "image/jpeg"

    contents = [
        genai_types.Part(inline_data=genai_types.Blob(
            data=image_bytes, mime_type=mime_type
        )),
        genai_types.Part.from_text(prompt),
    ]

    # ... retry logic, token counting, cost calc (same pattern as generate())
```

**Flaw found**: The existing `generate()` method uses `response_mime_type: "application/json"` config. For multimodal image analysis, we may NOT want to force JSON mode — it can cause models to refuse image analysis or produce truncated output. The benchmark harness should handle JSON parsing separately.

**Fix**: `generate_with_image()` should NOT set `response_mime_type: "application/json"`. Instead, the prompt itself asks for structured output, and the benchmark harness parses it.

### 5.2 Add `generate_with_image()` to VertexMedGemmaAdapter (NEW)

**File**: `src/trialmatch/models/vertex_medgemma.py`

**What**: Add multimodal support using OpenAI-compatible `image_url` format that vLLM accepts.

**Implementation**:
```python
async def generate_with_image(
    self, prompt: str, image_path: Path, max_tokens: int = 2048
) -> ModelResponse:
    """Multimodal generation via Vertex AI vLLM endpoint."""
    import base64

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    mime_type = "image/png" if str(image_path).endswith(".png") else "image/jpeg"

    payload = {
        "instances": [{
            "@requestFormat": "chatCompletions",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime_type};base64,{b64}"
                    }},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }]
    }
    # ... same retry/parse logic as generate()
```

**Flaw found**: The existing `_predict_url` uses the `:predict` endpoint. For `chatCompletions` with multimodal content, the vLLM backend on Vertex should still use `:predict` with `@requestFormat: chatCompletions`. However, base64-encoded images significantly increase payload size (~200KB-1MB per image). The `httpx.post` timeout of 120s may be insufficient for image decoding + generation.

**Fix**: Increase `httpx.post` timeout to 300s for multimodal requests. Add payload size logging.

### 5.3 Benchmark Data Loader (NEW)

**File**: `scripts/build_medpix_benchmark.py`

**What**: Filter and prepare MedPix cases for the benchmark.

**Responsibilities**:
1. Load `full_dataset.jsonl`
2. Filter by `location_category`, `has_diagnosis`, `has_findings`
3. Resolve image paths (prepend `ingest_design/`) and verify existence
4. For cases with multiple images: select the **first Thorax image** (avoid multi-image complexity)
5. Output: `data/benchmark/medpix_thorax_10.json` with fields: `uid`, `history`, `gold_diagnosis`, `gold_findings`, `image_path`, `image_modality`, `image_caption`

**Flaw found**: Some cases have multiple images across different body regions. A case categorized as "Thorax" might also have Head images. The benchmark must select only Thorax-region images, not just any image from a Thorax-categorized case.

**Fix**: Filter images within each case to `location_category == "Thorax"`, then select the first matching image.

### 5.4 Benchmark Runner (NEW)

**File**: `scripts/run_medpix_benchmark.py`

**What**: Execute the multimodal benchmark end-to-end.

**Flow**:
```
Load benchmark JSON
  → For each case:
    → Read image from disk
    → Build prompt: history + task instruction
    → Send to MedGemma 4B (Vertex): image + prompt → response
    → Send to Gemini 3 Pro (AI Studio): image + prompt → response
    → Parse both responses
    → Run LLM-as-judge on diagnosis predictions
    → Compute findings overlap (ROUGE-L)
  → Aggregate metrics
  → Write results to runs/<run_id>/
```

**Flaw found**: Running both models sequentially per case is slow. MedGemma on Vertex has ~8-15s latency per request; Gemini is ~3-5s. For 10 cases that's 2-3 minutes sequential. For 132 cases, 30+ minutes.

**Fix**: Run both models in parallel per case using `asyncio.gather()`. Both adapters are already async. But respect Vertex concurrency=1 (single GPU) — parallelize across models, not across cases for Vertex.

### 5.5 Evaluation Metrics Module (NEW)

**File**: `src/trialmatch/evaluation/multimodal_metrics.py`

**What**: Scoring functions for diagnosis and findings.

**Metrics**:

| Metric | Method | Why |
|--------|--------|-----|
| **Diagnosis exact match** | Case-insensitive string match | Baseline — catches exact hits |
| **Diagnosis substring match** | Gold diagnosis substring in prediction or vice versa | Catches "Adenocarcinoma" in "Adenocarcinoma of the Lung, Stage IV" |
| **Diagnosis LLM-judge** | Gemini Flash judges semantic equivalence | Handles synonym variation: "lung adenocarcinoma" = "adenocarcinoma of the lung" |
| **Findings ROUGE-L** | `rouge-score` library | Standard text overlap metric for generated text vs reference |
| **Findings keyword overlap** | Extract medical entities, compute Jaccard | Catches key terms (tumor size, location, laterality) |

**Flaw found**: ROUGE-L alone penalizes models that give more detailed findings than the gold reference. A model that correctly identifies all gold findings PLUS additional valid findings would get a lower ROUGE-L precision.

**Fix**: Report ROUGE-L recall (how much of gold is captured) separately from precision. Also report F1. A model finding additional valid findings is a feature, not a bug — ROUGE-L recall is the primary metric.

---

## 6. Prompt Design

### 6.1 Diagnosis + Findings Extraction Prompt

```
You are a board-certified radiologist with expertise in diagnostic imaging.

CLINICAL HISTORY:
{history}

TASK:
Analyze the provided medical image in the context of the clinical history above.

Provide your analysis in the following format:

DIAGNOSIS: [Your primary diagnosis based on the imaging findings and clinical history]

FINDINGS: [Detailed description of the imaging findings, including:
- Location and laterality of abnormalities
- Size and morphology of lesions
- Associated findings (effusions, lymphadenopathy, etc.)
- Normal structures and their appearance
- Comparison with expected normal anatomy]

DIFFERENTIAL: [Top 2-3 differential diagnoses if the primary diagnosis is uncertain]
```

**Design rationale**:
- Plain text output (not JSON) — avoids instruction-following failures on 4B model
- Image placed before text in the content list (per MedGemma documentation)
- Clinical history provided — matches real clinical workflow where images are read with context
- Structured sections make parsing reliable via regex

### 6.2 LLM-as-Judge Prompt (Diagnosis Scoring)

```
You are an expert medical evaluation judge.

Compare the MODEL PREDICTION against the GOLD STANDARD diagnosis.

GOLD STANDARD: {gold_diagnosis}
MODEL PREDICTION: {predicted_diagnosis}

Score the prediction:
- "correct": The prediction identifies the same disease/condition as the gold standard (synonyms and abbreviations are acceptable)
- "partial": The prediction identifies a related condition or a broader/narrower diagnosis that overlaps significantly
- "incorrect": The prediction identifies a different disease/condition

Respond in JSON:
{"score": "correct|partial|incorrect", "explanation": "brief reason"}
```

---

## 7. Required Skills

| Skill | File | Why Needed | When Used |
|-------|------|-----------|-----------|
| **vertex-ai-deploy** | `.claude/skills/vertex-ai-deploy/SKILL.md` | Deploy MedGemma 4B on Vertex AI. Existing `deploy_vertex.sh` supports `--model-size=4b`. **Must tear down after benchmark to avoid cost (~$1.15/hr).** | Step 8 (deploy) and Step 11 (teardown) |
| **medgemma-endpoint** | `.claude/skills/medgemma-endpoint/SKILL.md` | Reference for MedGemma 4B API patterns, auth, multimodal format. Vertex vLLM uses OpenAI-compatible image_url format. | Step 5.2 (adapter implementation) |
| **bdd-workflow** | `.claude/skills/bdd-workflow/SKILL.md` | RED-GREEN-REFACTOR for new code: multimodal adapters, data loader, metrics, benchmark runner. | Steps 5.1–5.5 (all new code) |
| **status-check** | `.claude/skills/status-check/SKILL.md` | Session protocol — update DASHBOARD.md with benchmark results. | Step 12 (session end) |

---

## 8. Execution Plan (Ordered Steps)

### Pre-flight
1. ☐ Read `docs/status/DASHBOARD.md` (session start protocol)
2. ☐ Verify `.env` has `GOOGLE_API_KEY` and GCP credentials configured

### Implementation (BDD cycle per component)
3. ☐ **Build benchmark data loader** (`scripts/build_medpix_benchmark.py`)
   - Filter `full_dataset.jsonl` → Thorax + diagnosis + findings
   - Resolve image paths with `ingest_design/` prefix
   - Select first Thorax image per case
   - Sample 10 cases deterministically (seed=42)
   - Output `data/benchmark/medpix_thorax_10.json`
   - Unit test: verify count, paths exist, fields present

4. ☐ **Add `generate_with_image()` to GeminiAdapter** (`src/trialmatch/models/gemini.py`)
   - Use `genai_types.Part(inline_data=...)` for image bytes
   - Do NOT use `response_mime_type: "application/json"` (let prompt control format)
   - Same retry/backoff pattern as `generate()`
   - Unit test: mock API, verify content structure

5. ☐ **Add `generate_with_image()` to VertexMedGemmaAdapter** (`src/trialmatch/models/vertex_medgemma.py`)
   - Use OpenAI-compatible `image_url` format in `chatCompletions` payload
   - Increase timeout to 300s for multimodal requests
   - Add payload size logging
   - Unit test: mock httpx, verify payload structure

6. ☐ **Create evaluation metrics** (`src/trialmatch/evaluation/multimodal_metrics.py`)
   - `score_diagnosis_exact()`: case-insensitive exact match
   - `score_diagnosis_substring()`: bidirectional substring check
   - `score_diagnosis_llm_judge()`: Gemini Flash semantic equivalence
   - `score_findings_rouge()`: ROUGE-L (precision, recall, F1)
   - Unit tests for each scoring function

7. ☐ **Create benchmark runner** (`scripts/run_medpix_benchmark.py`)
   - Load benchmark JSON, initialize both model adapters
   - Per case: send image + prompt to both models in parallel
   - Parse responses (regex for DIAGNOSIS/FINDINGS sections)
   - Score with all metrics
   - Write to `runs/<run_id>/` per project conventions
   - Config YAML for benchmark params

### Deployment & Execution
8. ☐ **Deploy MedGemma 4B on Vertex AI**
   - `bash scripts/deploy_vertex.sh --model-size=4b`
   - Wait for deployment (~10-20 min cold start)
   - Set `VERTEX_ENDPOINT_ID_4B` env var from output
   - Smoke test: 1 case with text-only, then 1 case with image

9. ☐ **Smoke test (1 case × 2 models)**
   - Run benchmark with `n=1` to verify end-to-end
   - Check: image sent correctly, response parsed, metrics computed
   - Fix any issues before full run

10. ☐ **Run Phase 1 benchmark (10 cases)**
    - `uv run python scripts/run_medpix_benchmark.py --config configs/medpix_bench.yaml`
    - Monitor: tail logs in real-time (per CLAUDE.md rules)
    - Expected duration: ~5-10 min
    - Expected output: `runs/<run_id>/` with per-case results + aggregate table

11. ☐ **TEAR DOWN Vertex 4B endpoint** (MANDATORY — avoid charges)
    - `bash scripts/deploy_vertex.sh --teardown --endpoint-id=$VERTEX_ENDPOINT_ID_4B`
    - Verify endpoint deleted

### Analysis
12. ☐ **Analyze results and update DASHBOARD.md**
    - Comparison table: Model × Metric × Score
    - Per-case error analysis: which cases did each model get wrong?
    - Qualitative: compare model reasoning quality on shared errors
    - Record ADR if architectural decisions were made

---

## 9. Flaws Identified & Fixes Applied

| # | Flaw | Severity | Fix |
|---|------|----------|-----|
| 1 | **Image path resolution**: `file_path` in JSONL is relative to `ingest_design/`, not project root | **Critical** — all image loads would fail | Benchmark loader must prepend `ingest_design/` to all image paths |
| 2 | **Gemini JSON mode breaks multimodal**: `response_mime_type: "application/json"` in existing `generate()` forces JSON output, which can cause multimodal refusals | **High** — Gemini may refuse to analyze images or produce truncated output | `generate_with_image()` does NOT set JSON response mode; uses plain text with prompt-directed structure |
| 3 | **httpx timeout too short for images**: 120s timeout in Vertex adapter; base64 image payloads are 200KB-1MB and image processing adds latency | **Medium** — large images may time out | Increase timeout to 300s for multimodal; add payload size logging |
| 4 | **Multi-image cases**: Some Thorax cases have 3-5 images; sending all images adds cost and complexity, and MedGemma 4B was not benchmarked on multi-image input | **Medium** — inconsistent evaluation | Select first Thorax-region image per case; document limitation |
| 5 | **ROUGE-L penalizes verbose models**: A model that finds everything in gold + extra valid findings gets lower precision | **Medium** — misleading metric | Report ROUGE-L recall as primary metric; precision and F1 as secondary |
| 6 | **Cases without findings (17/671)**: 17 cases have no `findings` text — cannot compute findings overlap | **Low** — only affects findings metric | Filter to `has_findings == true` for findings evaluation; still evaluate diagnosis on all cases |
| 7 | **Vertex 4B multimodal untested**: The `deploy_vertex.sh` 4B path deploys `medgemma-4b-it` which IS multimodal, but the OpenAI-compatible `image_url` format on vLLM has not been tested for this specific model | **Medium** — may need format adjustment | Smoke test (Step 9) with 1 image before full run; fallback to HF `chat_completion` with `use_chat_api=True` if Vertex multimodal fails |
| 8 | **MedGemma 4B instruction-following weakness**: Known MET bias and instruction-following issues on 4B model (documented in CLAUDE.md); may produce poorly structured diagnosis output | **Medium** — parsing failures | Use flexible regex parsing with fallbacks; plain text prompt (not JSON) reduces instruction-following burden |
| 9 | **Previous plan estimated ~15 NSCLC Thorax cases**: Actual count is **132** all-disease Thorax cases (much larger) | **Low** — estimate was wrong | Corrected in this plan with verified counts |
| 10 | **No automated teardown on failure**: If benchmark crashes mid-run, Vertex endpoint stays up costing ~$1.15/hr | **Medium** — cost risk | Add try/finally in benchmark runner to print teardown reminder; manual teardown as backup |

---

## 10. Cost Estimate

### Phase 1 (10 cases)

| Component | Calculation | Cost |
|-----------|-------------|------|
| MedGemma 4B (Vertex) | ~15 min deploy/warm-up + ~10 min benchmark = 25 min × $1.15/hr | ~$0.50 |
| Gemini 3 Pro (10 calls) | ~5K input tokens × 10 × $1.25/1M + ~2K output × 10 × $10/1M | ~$0.30 |
| LLM-as-judge (10 calls) | Gemini Flash, minimal tokens | ~$0.05 |
| **Total Phase 1** | | **~$0.85** |

### Phase 2 (132 cases)

| Component | Calculation | Cost |
|-----------|-------------|------|
| MedGemma 4B (Vertex) | ~15 min warm-up + ~60 min benchmark = 75 min × $1.15/hr | ~$1.45 |
| Gemini 3 Pro (132 calls) | ~5K input × 132 × $1.25/1M + ~2K output × 132 × $10/1M | ~$3.50 |
| LLM-as-judge (132 calls) | Gemini Flash | ~$0.50 |
| **Total Phase 2** | | **~$5.45** |

---

## 11. Expected Output

### Run Artifacts (`runs/<run_id>/`)

```
runs/<run_id>/
├── config.yaml                    # Benchmark configuration
├── benchmark_input.json           # All input cases with resolved paths
├── results/
│   ├── medgemma_4b_responses.json # Raw model responses per case
│   ├── gemini_pro_responses.json  # Raw model responses per case
│   ├── llm_judge_scores.json     # Per-case diagnosis judgment
│   └── parsed_predictions.json    # Extracted diagnosis + findings per model
├── metrics/
│   ├── summary.json              # Aggregate metrics table
│   ├── per_case_scores.csv       # Per-case × per-model scores
│   └── confusion.json            # Diagnosis correct/partial/incorrect counts
└── traces/
    └── cost_summary.json         # Total cost per model
```

### Summary Table (Expected Format)

| Metric | MedGemma 4B (Vertex) | Gemini 3 Pro | Delta |
|--------|---------------------|--------------|-------|
| Diagnosis — Correct (LLM-judge) | X% | Y% | ±Z% |
| Diagnosis — Partial | X% | Y% | ±Z% |
| Diagnosis — Exact Match | X% | Y% | ±Z% |
| Findings — ROUGE-L Recall | X | Y | ±Z |
| Findings — ROUGE-L F1 | X | Y | ±Z |
| Avg Latency (ms) | X | Y | — |
| Avg Cost per Case ($) | X | Y | — |

---

## 12. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Vertex 4B multimodal doesn't accept images | Medium | Blocks benchmark | Smoke test first (Step 9); fallback: HF endpoint with `use_chat_api=True` + `max_tokens=512` (degraded but functional) |
| Vertex deploy fails or takes >30 min | Low | Delays benchmark | `deploy_vertex.sh` handles polling; manual status check via `gcloud ai endpoints list` |
| MedGemma 4B produces unparseable output | Medium | Partial data loss | Flexible regex parsing with multiple fallbacks; save raw responses for manual review |
| Gemini 3 Pro throttled (429) | Low | Slows benchmark | Existing retry/backoff (8 retries, exponential) handles this |
| Benchmark script crashes mid-run | Low | Cost risk (Vertex stays up) | try/finally with teardown reminder; manual `deploy_vertex.sh --teardown` |
| LLM-as-judge disagrees with human assessment | Medium | Misleading scores | Spot-check judge decisions on all 10 Phase 1 cases before trusting at scale |

---

## 13. Dependencies & Prerequisites

### Python packages (already in `pyproject.toml`)
- `google-genai` — Gemini API SDK
- `google-auth` — Vertex AI ADC
- `httpx` — Vertex REST calls
- `huggingface-hub` — HF Inference Client
- `structlog` — Logging
- `pydantic` — Schema validation

### Python packages to add
- `rouge-score` — ROUGE-L computation for findings evaluation

### Environment variables (in `.env`)
- `GOOGLE_API_KEY` — for Gemini 3 Pro / Flash via AI Studio
- `GCP_PROJECT_ID` — for Vertex AI
- `GCP_REGION` — for Vertex AI (default: `us-central1`)
- `VERTEX_ENDPOINT_ID_4B` — set after deployment (Step 8)
- `VERTEX_DEDICATED_DNS` — set after deployment if applicable

### CLI tools
- `gcloud` — for Vertex AI deployment and teardown
- `uv` — Python package management and script execution
