# MedPix Multimodal Benchmark Plan: MedGemma 4B vs Gemini 3 Pro

**Date**: 2026-02-22 (updated 2026-02-22)  
**Status**: READY TO EXECUTE  
**Goal**: Benchmark MedGemma 4B (multimodal) against Gemini 3 Pro (multimodal) on two structured criteria — image findings extraction and clinical diagnosis — using MedPix ground-truth data. **Filtered by Diagnosis-By field** to select cases where diagnosis was confirmed through imaging + clinical evidence, maximizing body region diversity. **All available images per case are processed.**

---

## 1. Benchmark Objective

Evaluate two multimodal models on **two separate benchmark criteria**, processing **all images** per case:

### Criterion 1: Image Findings Extraction (per-image)

- **Input**: Single radiology image + clinical context (`history`, `exam`, image `modality`/`plane`)
- **Output**: Predicted imaging findings for that specific image
- **Ground truth**: `images[].caption` — radiologist-authored per-image description (available for 100% of images)
- **Scoring**: ROUGE-L (recall, precision, F1) + LLM-as-judge for semantic equivalence
- **Granularity**: One evaluation per image. A case with 3 Thorax images generates 3 evaluations.

### Criterion 2: Diagnosis Prediction (per-case)

- **Input**: **All** images for the case + clinical context (`history`, `exam`, per-image `modality`/`plane`)
- **Output**: Predicted primary diagnosis
- **Ground truth**: Case-level `diagnosis` field (expert-authored)
- **Scoring**: LLM-as-judge (correct/partial/incorrect) + exact/substring match
- **Granularity**: One evaluation per case. All images are sent together for holistic diagnosis.

### Why Two Criteria

1. **Findings extraction** (Criterion 1) tests **visual perception**: can the model describe what it sees in a single image? Ground truth = per-image `caption` (different for each image in the same case).
2. **Diagnosis prediction** (Criterion 2) tests **clinical reasoning**: can the model synthesize all images + clinical history to arrive at the correct diagnosis? Ground truth = case-level `diagnosis`.
3. These criteria are complementary — a model might describe findings well but fail at diagnosis (or vice versa).

### Why All Images

MedPix cases often contain multiple images from different views, sequences, or time points:
- **445 cases have 2+ images** (66% of dataset)
- **311 multi-image cases have DIFFERENT per-image captions** — each image shows something distinct
- For Criterion 2, multiple images provide the radiologist (and model) more information for diagnosis
- For Criterion 1, each image is evaluated independently against its own caption

### Why This Benchmark Matters

1. MedGemma 4B has a medical-domain SigLIP image encoder — this tests whether it provides meaningful advantage over a general-purpose model (Gemini 3 Pro) on real clinical imaging tasks
2. Both models receive identical image + text inputs — true apples-to-apples multimodal comparison
3. MedPix provides expert-authored diagnosis and per-image radiology findings — no annotation work needed
4. Two-criterion design separates perception (findings) from reasoning (diagnosis) — finer-grained model comparison

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
| Cases with `diagnosis_by` field | **500 (74.5%)** |
| Total images on disk | **2,050 PNGs** |

#### Diagnosis-By Semantic Categorization (expanded regex, 500 cases)

The `diagnosis_by` field describes HOW the diagnosis was confirmed. We classify it using regex keyword matching into the following categories:

| Category | Count | Description |
|----------|-------|-------------|
| **imaging + clinical** | **40** | Diagnosis confirmed by BOTH imaging findings and clinical presentation/history/exam |
| **imaging + clinical + pathology** | **4** | All three modalities contributed to diagnosis |
| imaging only | 191 | Diagnosis by imaging alone (CT, MRI, X-ray, etc.) |
| imaging + pathology | 32 | Imaging plus biopsy/surgery confirmation |
| pathology only | 172 | Biopsy/histology without imaging mention |
| clinical only | 13 | Clinical presentation alone |
| other (uncategorized) | 48 | Various methods (ERCP, genetic testing, lab, etc.) |
| no diagnosis_by field | 171 | Field absent or empty |

**Primary target pool**: 44 cases where `diagnosis_by` mentions BOTH imaging AND clinical evidence (40 imaging+clinical + 4 imaging+clinical+pathology). These are ideal benchmark cases because the ground-truth diagnosis was made the same way the model is expected to work: by analyzing medical images in clinical context.

#### Target Pool Body Region Diversity (44 cases, via ACR code)

| Body Region | Cases | % |
|-------------|-------|---|
| Head/Neck | 17 | 38.6% |
| GU/Reproductive | 11 | 25.0% |
| Breast/MSK | 10 | 22.7% |
| GI/Abdomen | 1 | 2.3% |
| Chest/Thorax | 1 | 2.3% |
| MSK/Extremities | 1 | 2.3% |
| Other | 3 | 6.8% |

**Key insight**: Unlike Thorax-only filtering (132 cases, 1 body region), Diagnosis-By filtering yields 44 cases across **7+ body regions** — dramatically better diversity for benchmarking multimodal medical AI.

#### Target Pool Image Counts (44 cases)

| Metric | Value |
|--------|-------|
| Total images | **135** |
| Avg images/case | **3.1** |
| Max images/case | 10 |
| Min images/case | 1 |
| Cases with complete data (images + history + diagnosis + findings + captions) | **44 (100%)** |

#### Broader Pool: All Imaging-Involved (267 cases)

For Phase 3, we expand to all cases where imaging played ANY role in diagnosis:

| Body Region | Cases |
|-------------|-------|
| Head/Neck | 67 |
| GU/Reproductive | 51 |
| Breast/MSK | 43 |
| GI/Abdomen | 24 |
| Cardiovascular | 18 |
| MSK/Extremities | 15 |
| Spine | 14 |
| Chest/Thorax | 13 |
| Other | 22 |

### 2.3 Field Classification — What Goes to the LLM vs What Does NOT

A complete case in `full_dataset.jsonl` contains 25+ fields. **Most fields leak the ground-truth answer** and must NEVER be sent to the model. The table below classifies every field.

#### SAFE — Model Input Fields

| Field | Role | Content | Example |
|-------|------|---------|--------|
| `history` | **Primary text input** | Chief complaint, demographics, presenting symptoms. Available in 664/671 cases. | "60-year-old woman presents with chest pain and shortness of breath." |
| `exam` | **Secondary text input** | Physical examination findings (vitals, palpation, lab values). Available in 527/671 cases. NOT imaging findings. | "Tender in the right adnexa and right lower quadrant. Negative pregnancy test." |
| `images[].file_path` | **Image input** | Path to radiology image on disk (relative to `ingest_design/`) | `MedPix-2-0/images/MPX1016_synpic34317.png` |
| `images[].modality` | **Image metadata (for prompt)** | Imaging modality — tells model what kind of image it's looking at | "CT - noncontrast", "MRI", "XR" |
| `images[].plane` | **Image metadata (for prompt)** | Imaging plane | "Axial", "Coronal", "Sagittal" |

#### GROUND TRUTH — Evaluation Only (NEVER sent to model)

| Field | Role | Why Excluded |
|-------|------|--------------|
| `diagnosis` | **Primary GT for diagnosis accuracy** | Contains the answer the model must predict |
| `findings` | **Primary GT for findings extraction** | Contains the radiological findings the model must extract from the image |
| `differential_diagnosis` | **Secondary GT** | Contains alternative diagnoses — leaks diagnostic reasoning |
| `images[].caption` | **Secondary GT** | Radiologist-authored image description — IS the answer for findings extraction |

#### FORBIDDEN — Answer Leakage (NEVER sent to model)

| Field | Why Forbidden |
|-------|---------------|
| `title` | Almost always contains the diagnosis verbatim (e.g., "Adenocarcinoma of the Lung") |
| `discussion` | Extended discussion of the diagnosis and disease pathology |
| `treatment` | Treatment plan implies the diagnosis (e.g., "chemotherapy" implies cancer) |
| `disease_discussion` | Disease-level encyclopedia entry — contains diagnosis name |
| `topic_title` | Normalized disease name (e.g., "Lung, lobar collapse") |
| `diagnosis_by` | Describes confirmation method — implies diagnosis |
| `llm_prompt` | **CRITICAL**: This pre-built field concatenates `history` + `findings` — **contains GT findings text** |
| `keywords` | Disease-specific keywords |
| `category` | Disease category (e.g., "Neoplasm, carcinoma") |
| `acr_code` | ACR classification code — maps to disease type |

> **CRITICAL WARNING**: The `llm_prompt` field looks convenient but MUST NOT be used. It includes `Imaging Findings:` which is the ground-truth `findings` text. Using it would give the model the answers.

### 2.4 Exact LLM Input Specification

For each benchmark case, the model receives exactly:

```
┌─────────────────────────────────────────────────────────┐
│  MULTIMODAL INPUT TO MODEL                              │
│                                                         │
│  1. IMAGE: Single radiology PNG/JPEG                    │
│     Source: images[0].file_path (first matching         │
│             image for target body region)                │
│                                                         │
│  2. TEXT PROMPT (constructed from safe fields):          │
│     ┌─────────────────────────────────────────────────┐ │
│     │ System instruction (role + task)                 │ │
│     │                                                 │ │
│     │ IMAGE METADATA:                                 │ │
│     │   Modality: {images[0].modality}                │ │
│     │   Plane: {images[0].plane}                      │ │
│     │                                                 │ │
│     │ CLINICAL HISTORY:                               │ │
│     │   {history}                                     │ │
│     │                                                 │ │
│     │ PHYSICAL EXAMINATION: (if non-empty)            │ │
│     │   {exam}                                        │ │
│     │                                                 │ │
│     │ Task instruction + output format                 │ │
│     └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Field availability**:
- `history`: present in 664/671 (98.9%) cases. 7 cases with empty history are excluded from benchmark.
- `exam`: present in 527/671 (78.5%) cases. When empty, the "PHYSICAL EXAMINATION" section is omitted from the prompt.
- `modality` + `plane`: present on all images. Always included for clinical context.

### 2.5 Ground Truth Fields (for evaluation scoring only)

#### Criterion 1 GT: Per-Image Findings

| Field | Type | Scoring Role | Availability | Example |
|-------|------|-------------|-------------|--------|
| `images[].caption` | string | **Primary GT — per-image findings** | **100% of images** (2,050/2,050) | "Contrast enhanced chest CT shows diffuse increased interstitial markings involving the right middle and lower lobes." |

> **Key insight**: In multi-image cases, **311 out of 445 (70%) have DIFFERENT captions per image**. Each image's caption describes what's visible in that specific image, not a duplicate of the case-level findings. This makes per-image evaluation meaningful.

#### Criterion 2 GT: Case-Level Diagnosis

| Field | Type | Scoring Role | Availability | Example |
|-------|------|-------------|-------------|--------|
| `diagnosis` | string | **Primary GT — diagnosis accuracy** | **100% of cases** (671/671) | "Left upper lobe collapse caused by an enlarging, obstructing small cell lung carcinoma." |
| `differential_diagnosis` | string | **Secondary GT** for partial-match scoring | 654/671 cases | Ranked differential list |

#### Supplementary GT (not directly used in scoring)

| Field | Type | Role | Note |
|-------|------|------|------|
| `findings` | string | **Case-level findings summary** | Aggregated findings across ALL images. Used for qualitative review only — NOT as Criterion 1 GT (since it's case-level, not per-image). Available in 654/671 cases. |

### 2.6 Image Preprocessing & Format Requirements

All 2,050 MedPix images are PNG files on disk. Both models accept PNG and JPEG, but have different optimal configurations.

#### Current Image Statistics (Verified)

| Metric | Value |
|--------|-------|
| Format | 100% PNG (8-bit) |
| Color types | **1,762 RGB** (86%) + **288 RGBA** (14%) |
| Dimensions | Highly variable: 512×512 most common (528 images), range from ~200px to ~800px |
| File sizes | Min 14 KB, Median 151 KB, P95 259 KB, Max 697 KB |
| Over 500 KB | Only 5 images |

#### Model-Specific Image Requirements

| Requirement | MedGemma 4B (Vertex vLLM) | Gemini 3 Pro (AI Studio) |
|-------------|---------------------------|-------------------------|
| **Accepted formats** | JPEG, PNG via base64 `data:` URI in `image_url` | JPEG, PNG, WEBP, GIF via `Part.from_bytes()` |
| **Internal resize** | SigLIP resizes to **896×896** internally | Gemini's vision encoder handles variable resolution |
| **Color mode** | RGB (3-channel). RGBA (4-channel) may cause silent issues — alpha channel has no medical meaning | RGB or RGBA accepted natively |
| **Max payload** | base64 increases size ~33%; 700KB PNG → ~933KB base64. vLLM `httpx` timeout must accommodate. | SDK handles streaming; no payload concern |
| **Optimal format** | **JPEG** (lossy OK for radiology screenshots; smaller payload = faster transfer) | **PNG** preferred (lossless; Gemini handles natively) |

#### Preprocessing Pipeline (in benchmark data loader)

```python
from PIL import Image
import io

def preprocess_image(png_path: Path) -> tuple[bytes, str]:
    """Convert MedPix PNG to format optimal for both models.
    
    Steps:
    1. Load PNG
    2. Convert RGBA → RGB (drop alpha channel — medically meaningless)
    3. Save as JPEG quality=95 (visually lossless for radiology)
    4. Return (jpeg_bytes, "image/jpeg") for MedGemma
    
    For Gemini: use original PNG bytes (lossless, native support).
    """
    with Image.open(png_path) as img:
        if img.mode == "RGBA":
            # White background for alpha compositing
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        
        # For MedGemma: JPEG (smaller base64 payload)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), "image/jpeg"
```

**Decision**: Send **JPEG (quality=95)** to MedGemma 4B (smaller payload, faster vLLM processing, avoids RGBA ambiguity) and **original PNG** to Gemini 3 Pro (lossless, native support, no payload concern). This gives each model its optimal format.

> **Flaw found (#11)**: 288 images (14%) are RGBA with an alpha channel. SigLIP in MedGemma expects 3-channel RGB input. Passing RGBA may cause: (a) silent 4th-channel drop with undefined behavior, or (b) an error. The preprocessing pipeline MUST strip the alpha channel before encoding.

### 2.7 Critical Data Path Issue (FLAW IDENTIFIED & FIX)

**Flaw**: Image `file_path` in the JSONL is relative to `ingest_design/` (e.g., `MedPix-2-0/images/MPX1016_synpic34317.png`), but the benchmark script will run from the project root. Path resolution requires prepending `ingest_design/`.

**Fix**: The benchmark data loader MUST resolve paths as:
```python
resolved_path = Path("ingest_design") / case["images"][i]["file_path"]
```

This was verified: `ingest_design/MedPix-2-0/images/MPX1024_synpic40275.png` exists=True, while `MedPix-2-0/images/MPX1024_synpic40275.png` exists=False from the project root.

---

## 3. Evaluation Scope & Phasing

### Filtering Strategy: Diagnosis-By Field

**Rationale**: Instead of filtering by body region (Thorax-only), we filter by the `diagnosis_by` field to select cases where diagnosis was confirmed through BOTH imaging and clinical evidence. This:
1. Selects cases where the diagnostic process matches the model's task (analyze images in clinical context)
2. Provides body region diversity (7+ regions vs 1 with Thorax-only)
3. Every selected case has a diagnosis that was explicitly made using imaging — making the image data genuinely diagnostic, not incidental

**Filter criteria** (applied to `full_dataset.jsonl`):
```
diagnosis_by matches BOTH:
  - imaging_regex: \b(ct|mri|mr\b|imaging|radiograph|x-ray|ultrasound|sonograph|angiograph|fluoroscop|
    mammograph|pet|spect|echocardiograph|scan|radio[lg]|arteriograph|venograph|cholangiograph|
    computed tomograph|cta\b|mra\b|flair|dwi|t1|t2|hrct|scintigraph|bone scan|nuclear|doppler)\b
  - clinical_regex: \b(clinical|history|physical|exam|presentation|symptom|lab|vital|blood|
    serology|eeg|emg|ecg|ekg)\b
AND has_diagnosis == true
AND has_findings == true
AND has at least 1 image on disk
AND all images have non-empty captions
```

### Phase 1: Pilot (10 imaging+clinical cases, ~31 images)
- 10 deterministically sampled cases from the 44-case target pool (seed=42)
- Filter: `diagnosis_by` matches BOTH imaging AND clinical regex patterns
- **All images per case** processed (not just first image)
- Estimated images: ~31 (avg 3.1 images/case)
- **Criterion 1**: ~31 per-image findings evaluations
- **Criterion 2**: 10 per-case diagnosis evaluations
- Purpose: validate pipeline, check prompt quality, tune if needed
- Estimated cost: ~$2-4

### Phase 2: Full Imaging+Clinical (44 cases, 135 images)
- All 44 cases where `diagnosis_by` mentions both imaging and clinical evidence
- **135 total images** across 44 cases (avg 3.1/case, max 10)
- **Criterion 1**: 135 per-image findings evaluations
- **Criterion 2**: 44 per-case diagnosis evaluations
- Body regions: Head/Neck (17), GU/Reproductive (11), Breast/MSK (10), GI/Abdomen (1), Chest/Thorax (1), MSK/Extremities (1), Other (3)
- Purpose: statistically meaningful comparison across diverse body regions on cases where imaging was diagnostically relevant
- Estimated cost: ~$8-12

### Phase 3: All Imaging-Involved (267 cases, ~820 images)
- All 267 cases where imaging played ANY role in diagnosis (imaging_only + imaging_clinical + imaging_pathology)
- Estimated ~820 images (avg ~3.1/case)
- **Criterion 1**: ~820 per-image findings evaluations
- **Criterion 2**: 267 per-case diagnosis evaluations
- Body regions: 9+ regions represented
- Purpose: comprehensive cross-specialty evaluation with maximum sample size
- Estimated cost: ~$50-75

**Decision**: Start with Phase 1 (10 cases, ~31 images), validate outputs, then scale to Phase 2 (all 44 imaging+clinical cases).

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

### 5.1 Add `generate_with_image()` and `generate_with_images()` to GeminiAdapter (NEW)

**File**: `src/trialmatch/models/gemini.py`

**What**: Override `generate_with_image()` (single image, Criterion 1) and add `generate_with_images()` (multiple images, Criterion 2) using `google.genai` SDK's `Part`-based content structure.

**Implementation** (single image):
```python
async def generate_with_image(
    self, prompt: str, image_path: Path, max_tokens: int = 2048
) -> ModelResponse:
    """Multimodal generation: single image + text via Gemini."""
    # ... same as before
```

**Implementation** (multiple images for Criterion 2):
```python
async def generate_with_images(
    self, prompt: str, image_paths: list[Path], max_tokens: int = 4096
) -> ModelResponse:
    """Multimodal generation: multiple images + text via Gemini."""
    from google.genai import types as genai_types

    contents = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        mime_type = "image/png" if str(image_path).endswith(".png") else "image/jpeg"
        contents.append(genai_types.Part(inline_data=genai_types.Blob(
            data=image_bytes, mime_type=mime_type
        )))
    contents.append(genai_types.Part.from_text(prompt))
    # ... retry logic, token counting, cost calc (same pattern as generate())
```

**Flaw found**: The existing `generate()` method uses `response_mime_type: "application/json"` config. For multimodal image analysis, we may NOT want to force JSON mode — it can cause models to refuse image analysis or produce truncated output. The benchmark harness should handle JSON parsing separately.

**Fix**: `generate_with_image()` should NOT set `response_mime_type: "application/json"`. Instead, the prompt itself asks for structured output, and the benchmark harness parses it.

### 5.2 Add `generate_with_image()` and `generate_with_images()` to VertexMedGemmaAdapter (NEW)

**File**: `src/trialmatch/models/vertex_medgemma.py`

**What**: Add multimodal support using OpenAI-compatible `image_url` format that vLLM accepts. Both single-image (Criterion 1) and multi-image (Criterion 2) variants.

**Implementation** (single image — Criterion 1):
```python
async def generate_with_image(
    self, prompt: str, image_path: Path, max_tokens: int = 2048
) -> ModelResponse:
    """Multimodal generation via Vertex AI vLLM endpoint."""
    # Same as before — single image_url in content array
```

**Implementation** (multiple images — Criterion 2):
```python
async def generate_with_images(
    self, prompt: str, image_paths: list[Path], max_tokens: int = 4096
) -> ModelResponse:
    """Multi-image generation via Vertex AI vLLM endpoint."""
    import base64

    content = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        mime_type = "image/jpeg"  # All pre-converted to JPEG
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{mime_type};base64,{b64}"
        }})
    content.append({"type": "text", "text": prompt})

    payload = {
        "instances": [{
            "@requestFormat": "chatCompletions",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }]
    }
    # ... same retry/parse logic
```

**Multi-image payload considerations**:
- 8 images × 150KB JPEG × 1.33 base64 = ~1.6MB payload. Well within HTTP limits.
- Timeout increased to 300s for multi-image (processing N images takes longer).
- Log total payload size for monitoring.

**Flaw found**: The existing `_predict_url` uses the `:predict` endpoint. For `chatCompletions` with multimodal content, the vLLM backend on Vertex should still use `:predict` with `@requestFormat: chatCompletions`. However, base64-encoded images significantly increase payload size (~200KB-1MB per image). The `httpx.post` timeout of 120s may be insufficient for image decoding + generation.

**Fix**: Increase `httpx.post` timeout to 300s for multimodal requests. Add payload size logging.

### 5.3 Benchmark Data Loader (NEW)

**File**: `scripts/build_medpix_benchmark.py`

**What**: Filter and prepare MedPix cases for the benchmark using Diagnosis-By filtering. **Includes ALL images per case** (not just the first).

**Responsibilities**:
1. Load `full_dataset.jsonl`
2. Filter cases by `diagnosis_by` field matching BOTH imaging AND clinical regex patterns (see §3 Filter Criteria)
3. Also require: `has_diagnosis == true`, `has_findings == true`, at least 1 image on disk, all images have captions
4. Include ALL images per case (no body-region image filtering — unlike Thorax-only, these cases' images are all relevant)
5. Resolve all image paths (prepend `ingest_design/`) and verify existence on disk
6. Preprocess all images: RGBA → RGB conversion, generate JPEG variant for MedGemma (see §2.6)
7. Output: `data/benchmark/medpix_imgclin_10.json` (Phase 1) or `medpix_imgclin_44.json` (Phase 2) — a list of case objects:

```json
{
  "uid": "MPX1063",
  "history": "76 year-old woman with long history of hypertension...",
  "exam": "Blood pressure 180/95...",
  "gold_diagnosis": "thoracic aortic dissection",
  "gold_findings": "Multiple axial CT images demonstrate...",
  "images": [
    {
      "image_path": "/abs/path/to/ingest_design/MedPix-2-0/images/MPX1063_synpic22165.png",
      "modality": "CT w/contrast (IV)",
      "plane": "Axial",
      "gold_caption": "Multiple axial CT images...Stanford type A thoracic aortic dissection..."
    },
    {
      "image_path": "/abs/path/to/ingest_design/MedPix-2-0/images/MPX1063_synpic22166.png",
      "modality": "CT w/contrast (IV)",
      "plane": "Axial",
      "gold_caption": "Multiple axial CT images...clear intimal flap..."
    },
    {
      "image_path": "/abs/path/to/ingest_design/MedPix-2-0/images/MPX1063_synpic22167.png",
      "modality": "CT w/contrast (IV)",
      "plane": "Axial",
      "gold_caption": "Multiple axial CT images...hemopericardium..."
    }
  ]
}
```

8. Validate: assert no forbidden fields (`title`, `discussion`, `llm_prompt`, etc.) are included in output
9. Include `diagnosis_by` field in output metadata (for analysis, not model input)
10. Include case body region (from ACR code mapping) in output metadata
11. Sample `n` cases deterministically (seed=42) for Phase 1

**Per-image GT**: Each image's `gold_caption` is the GT for Criterion 1 (findings). The case-level `gold_diagnosis` is the GT for Criterion 2 (diagnosis).

**Design note**: Unlike the previous Thorax-only filtering which required per-image body-region filtering (a Thorax case might have non-Thorax images), Diagnosis-By filtering selects cases holistically — ALL images in a selected case are relevant because the case-level diagnosis was made using imaging. No intra-case image filtering needed.

### 5.4 Benchmark Runner (NEW)

**File**: `scripts/run_medpix_benchmark.py`

**What**: Execute the multimodal benchmark end-to-end.

**Flow** (two-criterion sequential pipeline):
```
Load benchmark JSON
  → For each case:
    ┌─────────────────────────────────────────────────────────────────┐
    │ CRITERION 1: Image Findings (per-image)                       │
    │                                                               │
    │ For each image in case.images:                                │
    │   → Read image from disk                                     │
    │   → Preprocess: RGBA→RGB, JPEG for MedGemma, PNG for Gemini  │
    │   → Build FINDINGS prompt (single image + history + exam)     │
    │   → Send to MedGemma 4B → parse findings                     │
    │   → Send to Gemini 3 Pro → parse findings                    │
    │   → Score vs gold_caption: ROUGE-L + LLM-judge               │
    └─────────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────┐
    │ CRITERION 2: Diagnosis Prediction (per-case)                  │
    │                                                               │
    │   → Collect ALL preprocessed images for this case             │
    │   → Build DIAGNOSIS prompt (all images + history + exam)      │
    │   → Send to MedGemma 4B → parse diagnosis                    │
    │   → Send to Gemini 3 Pro → parse diagnosis                   │
    │   → Score vs gold_diagnosis: LLM-judge + exact/substring      │
    └─────────────────────────────────────────────────────────────────┘
  → Aggregate Criterion 1 metrics (per-image level)
  → Aggregate Criterion 2 metrics (per-case level)
  → Write results to runs/<run_id>/
```

**API call count per case** (for a case with N images):
- Criterion 1: N × 2 models = 2N calls (each image sent individually)
- Criterion 2: 1 × 2 models = 2 calls (all images sent together)
- LLM-as-judge: N (findings) + 1 (diagnosis) × 2 models = 2N + 2 judge calls
- Total: 4N + 4 calls per case
- For 10 Thorax cases with avg 2.5 images: ~(4×2.5 + 4) × 10 = ~140 calls

**Flaw found**: Running both models sequentially per case is slow. MedGemma on Vertex has ~8-15s latency per request; Gemini is ~3-5s. With all images: 10 cases × 2.5 avg images × 2 criteria × 2 models ≈ 100+ calls.

**Fix**: Run both models in parallel per image/case using `asyncio.gather()`. Both adapters are already async. But respect Vertex concurrency=1 (single GPU) — parallelize across models, not across cases for Vertex.

### 5.5 Evaluation Metrics Module (NEW)

**File**: `src/trialmatch/evaluation/multimodal_metrics.py`

**What**: Scoring functions separated by criterion.

#### Criterion 1 Metrics: Image Findings (per-image)

| Metric | Method | GT Field | Why |
|--------|--------|----------|-----|
| **Findings ROUGE-L** | `rouge-score` library | `images[i].caption` | Standard text overlap: how much of the per-image caption is captured |
| **Findings LLM-judge** | Gemini Flash semantic comparison | `images[i].caption` | Handles paraphrasing: "pleural effusion" vs "fluid in pleural space" |
| **Findings keyword overlap** | Extract medical entities, compute Jaccard | `images[i].caption` | Catches key anatomical terms (location, laterality, lesion type) |

> **Important**: Criterion 1 GT is `images[i].caption` (per-image), NOT `findings` (per-case). The case-level `findings` field aggregates across all images — using it would penalize models for not describing images they weren't shown.

#### Criterion 2 Metrics: Diagnosis (per-case)

| Metric | Method | GT Field | Why |
|--------|--------|----------|-----|
| **Diagnosis exact match** | Case-insensitive string match | `diagnosis` | Baseline — catches exact hits |
| **Diagnosis substring match** | Gold diagnosis substring in prediction or vice versa | `diagnosis` | Catches "Adenocarcinoma" in "Adenocarcinoma of the Lung, Stage IV" |
| **Diagnosis LLM-judge** | Gemini Flash judges semantic equivalence | `diagnosis` | Handles synonym variation: "lung adenocarcinoma" = "adenocarcinoma of the lung" |

**Flaw found**: ROUGE-L alone penalizes models that give more detailed findings than the gold caption. A model that correctly identifies all gold findings PLUS additional valid findings would get a lower ROUGE-L precision.

**Fix**: Report ROUGE-L recall (how much of gold is captured) separately from precision. Also report F1. A model finding additional valid findings is a feature, not a bug — ROUGE-L recall is the primary metric.

---

## 6. Prompt Design

### 6.1 Criterion 1 Prompt: Image Findings Extraction (per-image)

Sent once per image. The model receives **one image** and must describe what it sees. Ground truth = `images[i].caption`.

```
You are a board-certified radiologist with expertise in diagnostic imaging.

IMAGE METADATA:
Modality: {image.modality}
Plane: {image.plane}

CLINICAL HISTORY:
{history}

{conditional: if exam is non-empty}
PHYSICAL EXAMINATION:
{exam}
{end conditional}

TASK:
Describe the imaging findings visible in this single image.
Base your description ONLY on what you observe in the image and the clinical context provided.
Do NOT attempt to make a diagnosis — focus solely on describing the findings.

FINDINGS: [Detailed description of the imaging findings visible in this image, including:
- Location and laterality of abnormalities
- Size and morphology of lesions
- Density/signal characteristics
- Associated findings (effusions, lymphadenopathy, etc.)
- Normal structures and their appearance]
```

**Key design choices**:
- **Single image only** — model describes findings for ONE image at a time
- **No diagnosis requested** — isolates visual perception from clinical reasoning
- **Clinical history included** — matches real radiology workflow (radiologist knows why scan was ordered)
- **Single output section (FINDINGS)** — simpler parsing, avoids model confusion

### 6.2 Criterion 2 Prompt: Diagnosis Prediction (per-case, all images)

Sent once per case. The model receives **ALL images** for the case and must synthesize a diagnosis.

```
You are a board-certified radiologist with expertise in diagnostic imaging.

You are reviewing {N} image(s) from a single patient case.

{for each image i in images:}
IMAGE {i+1} METADATA:
Modality: {images[i].modality}
Plane: {images[i].plane}
{end for}

CLINICAL HISTORY:
{history}

{conditional: if exam is non-empty}
PHYSICAL EXAMINATION:
{exam}
{end conditional}

TASK:
Review ALL provided images together with the clinical history.
Synthesize your observations across all images to arrive at a diagnosis.

Provide your analysis in the following format:

DIAGNOSIS: [Your primary diagnosis based on all imaging findings and clinical history]

KEY FINDINGS: [Brief summary of the most important findings across all images that support your diagnosis]

DIFFERENTIAL: [Top 2-3 differential diagnoses if the primary diagnosis is uncertain]
```

**Key design choices**:
- **ALL images sent together** — model can cross-reference views/sequences for diagnosis
- **Per-image metadata listed** — model knows what each image shows (modality, plane)
- **Images placed before text** in the content list (per MedGemma documentation: images must come first)
- **Diagnosis + differential requested** — matches clinical radiology report structure
- **"KEY FINDINGS" (not "FINDINGS")** — distinct from Criterion 1 output; asks for synthesis not exhaustive description

### 6.3 Field-to-Prompt Mapping (both criteria)

| Prompt section | Source field | Used in Criterion | Why included |
|---------------|-------------|-------------------|---------------|
| IMAGE METADATA: Modality | `images[i].modality` | 1 and 2 | Tells model image type (CT vs MRI vs X-ray) |
| IMAGE METADATA: Plane | `images[i].plane` | 1 and 2 | Tells model viewing angle (Axial vs Coronal) |
| CLINICAL HISTORY | `history` | 1 and 2 | Chief complaint + demographics |
| PHYSICAL EXAMINATION | `exam` (if non-empty) | 1 and 2 | Lab values + physical findings |
| (Image itself) | `images[i].file_path` → preprocessed bytes | 1: single image, 2: all images | The radiology image(s) |

**Fields explicitly EXCLUDED from both prompts** (and why):
- `findings` — case-level findings IS the answer (leaks Criterion 1 answers)
- `diagnosis` — IS the answer for Criterion 2
- `llm_prompt` — contains `findings` text (data leakage)
- `title`, `discussion`, `treatment`, `disease_discussion`, `topic_title` — all leak the diagnosis
- `images[i].caption` — per-image description IS the GT for Criterion 1

### 6.4 LLM-as-Judge Prompt: Findings (Criterion 1)

```
You are an expert radiology evaluation judge.

Compare the MODEL'S FINDINGS DESCRIPTION against the GOLD STANDARD image description.

GOLD STANDARD: {gold_caption}
MODEL PREDICTION: {predicted_findings}

Score the prediction on how well it captures the key imaging findings:
- "correct": The prediction describes the same key findings as the gold standard (wording differences are acceptable)
- "partial": The prediction captures some but not all key findings, or includes the main finding but misses important details
- "incorrect": The prediction describes different findings or misses the main abnormality entirely

Respond in JSON:
{"score": "correct|partial|incorrect", "explanation": "brief reason"}
```

### 6.5 LLM-as-Judge Prompt: Diagnosis (Criterion 2)

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
   - Filter `full_dataset.jsonl` by `diagnosis_by` matching BOTH imaging+clinical regex
   - Also require: `has_diagnosis`, `has_findings`, images on disk, captions present
   - Resolve image paths with `ingest_design/` prefix
   - **Include ALL images per case** (no intra-case body-region filtering needed)
   - Add ACR code → body region mapping for each case (metadata only)
   - Sample 10 cases deterministically (seed=42)
   - Output `data/benchmark/medpix_imgclin_10.json`
   - Unit test: verify count, all image paths exist, per-image captions present, body region diversity

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
   - Criterion 1 (findings): `score_findings_rouge()`, `score_findings_llm_judge()` — per-image scoring vs `caption`
   - Criterion 2 (diagnosis): `score_diagnosis_exact()`, `score_diagnosis_substring()`, `score_diagnosis_llm_judge()`
   - Unit tests for each scoring function

7. ☐ **Create benchmark runner** (`scripts/run_medpix_benchmark.py`)
   - Load benchmark JSON, initialize both model adapters
   - **Criterion 1 loop**: For each case, for each image: send single image + findings prompt → score vs caption
   - **Criterion 2 loop**: For each case: send ALL images + diagnosis prompt → score vs diagnosis
   - Parse responses (regex for FINDINGS / DIAGNOSIS sections)
   - Score with all metrics per criterion
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

10. ☐ **Run Phase 1 benchmark (10 cases, ~31 images)**
    - `uv run python scripts/run_medpix_benchmark.py --config configs/medpix_bench.yaml`
    - Monitor: tail logs in real-time (per CLAUDE.md rules)
    - Expected duration: ~10-20 min
    - Expected output: `runs/<run_id>/` with per-criterion results + aggregate tables

11. ☐ **TEAR DOWN Vertex 4B endpoint** (MANDATORY — avoid charges)
    - `bash scripts/deploy_vertex.sh --teardown --endpoint-id=$VERTEX_ENDPOINT_ID_4B`
    - Verify endpoint deleted

### Analysis
12. ☐ **Analyze results and update DASHBOARD.md**
    - Criterion 1 table: Model × Findings Metric × Score (per-image level)
    - Criterion 2 table: Model × Diagnosis Metric × Score (per-case level)
    - Cross-criterion analysis: does better findings extraction correlate with better diagnosis?
    - Per-case error analysis: which cases did each model get wrong on each criterion?
    - Qualitative: compare model reasoning quality on shared errors
    - Record ADR if architectural decisions were made

---

## 9. Flaws Identified & Fixes Applied

| # | Flaw | Severity | Fix |
|---|------|----------|-----|
| 1 | **Image path resolution**: `file_path` in JSONL is relative to `ingest_design/`, not project root | **Critical** — all image loads would fail | Benchmark loader must prepend `ingest_design/` to all image paths |
| 2 | **Gemini JSON mode breaks multimodal**: `response_mime_type: "application/json"` in existing `generate()` forces JSON output, which can cause multimodal refusals | **High** — Gemini may refuse to analyze images or produce truncated output | `generate_with_image()` does NOT set JSON response mode; uses plain text with prompt-directed structure |
| 3 | **httpx timeout too short for images**: 120s timeout in Vertex adapter; base64 image payloads are 200KB-1MB and image processing adds latency | **Medium** — large images may time out | Increase timeout to 300s for multimodal; add payload size logging |
| 4 | **Multi-image cases**: Target pool has cases with up to 10 images. Sending all images for Criterion 2 (diagnosis) increases cost and payload size. MedGemma 4B vLLM has not been benchmarked on multi-image input. | **Medium** — possible vLLM payload limits or degraded quality with many images | Criterion 1 sends images individually (safe). Criterion 2 sends all images; smoke test with a 5+ image case in Step 9. If vLLM rejects multi-image, fall back to sending only first 3 images for diagnosis. |
| 5 | **ROUGE-L penalizes verbose models**: A model that finds everything in gold + extra valid findings gets lower precision | **Medium** — misleading metric | Report ROUGE-L recall as primary metric; precision and F1 as secondary |
| 6 | **Cases without findings (17/671)**: 17 cases have no `findings` text — cannot compute findings overlap | **Low** — only affects findings metric | Filter to `has_findings == true` for findings evaluation; still evaluate diagnosis on all cases |
| 7 | **Vertex 4B multimodal untested**: The `deploy_vertex.sh` 4B path deploys `medgemma-4b-it` which IS multimodal, but the OpenAI-compatible `image_url` format on vLLM has not been tested for this specific model | **Medium** — may need format adjustment | Smoke test (Step 9) with 1 image before full run; fallback to HF `chat_completion` with `use_chat_api=True` if Vertex multimodal fails |
| 8 | **MedGemma 4B instruction-following weakness**: Known MET bias and instruction-following issues on 4B model (documented in CLAUDE.md); may produce poorly structured diagnosis output | **Medium** — parsing failures | Use flexible regex parsing with fallbacks; plain text prompt (not JSON) reduces instruction-following burden |
| 9 | **Previous plan estimated ~15 NSCLC Thorax cases**: Actual Thorax count is 132. With new Diagnosis-By filtering, target pool is 44 cases across 7+ body regions. | **Low** — estimate was wrong | Corrected in this plan with verified counts from `diagnosis_by` field analysis |
| 10 | **No automated teardown on failure**: If benchmark crashes mid-run, Vertex endpoint stays up costing ~$1.15/hr | **Medium** — cost risk | Add try/finally in benchmark runner to print teardown reminder; manual teardown as backup |
| 11 | **Data leakage via `llm_prompt` field**: The pre-built `llm_prompt` field in `full_dataset.jsonl` concatenates `history` + `findings`. Using it sends ground-truth imaging findings directly to the model. Also: `title`, `discussion`, `treatment`, `disease_discussion`, `topic_title` all contain or imply the diagnosis. | **Critical** — invalidates entire benchmark if used | Strict field isolation: model receives ONLY `history` + `exam` + `images[i].modality` + `images[i].plane` + image bytes. All other text fields are forbidden. See §2.3 for complete classification. |
| 12 | **288 RGBA images (14% of dataset)**: MedGemma's SigLIP encoder expects 3-channel RGB. Alpha channel is medically meaningless (artifact from PNG export) and may cause undefined behavior. | **Medium** — silent accuracy degradation or errors on 14% of images | Preprocessing pipeline converts RGBA → RGB (white background alpha compositing) before encoding. See §2.6. |
| 13 | **Image format mismatch between models**: MedGemma 4B on vLLM sends base64 in HTTP payload (large PNGs = slow); Gemini uses SDK streaming (no payload concern). Sending identical PNG to both wastes bandwidth for MedGemma. | **Low** — latency impact only | Send JPEG (quality=95) to MedGemma (smaller base64), original PNG to Gemini (lossless). Each model gets its optimal format. |
| 14 | **Per-image caption duplication**: In 134/445 multi-image cases, all images share the SAME caption. This means Criterion 1 evaluations for these images are redundant — different images scored against identical GT. | **Low** — inflated sample count for Criterion 1 | Track and report: flag cases with identical captions. Report Criterion 1 metrics both with and without duplicated-caption images. |
| 15 | **Multi-image Criterion 2 payload size**: A case with 10 images × ~150KB each = ~1.5MB raw, ~2.0MB base64. vLLM may have payload limits or OOM on long multi-image sequences. | **Medium** — possible failure on high-image cases | Cap at 8 images per case for Criterion 2. If case has >8 images, select first 8 by image_id ordering. Log cases that were capped. |
| 16 | **Regex-based `diagnosis_by` classification is imperfect**: ~48 cases fall into "other" because their `diagnosis_by` text uses non-standard terminology (e.g., "ERCP", "pathognomonic", "genetic testing"). Some may actually involve imaging. | **Low** — we err on the conservative side (false negatives, not false positives) | Accept the 48 "other" cases as excluded. The 44-case target pool is curated from unambiguous matches. For Phase 3, manual review of "other" cases could recover additional imaging-involved cases. |
| 17 | **Body region distribution is Head/Neck heavy (39%)**: 17/44 target cases are Head/Neck. This means findings extraction is dominated by neuro/head CTs/MRIs. GU/Reproductive (25%) and Breast/MSK (23%) provide some balance, but GI, Chest, and Extremities have only 1 case each. | **Low** — inherent dataset bias | Report metrics stratified by body region. Phase 3 expansion to 267 cases provides much better region balance. |

---

## 10. Cost Estimate

### Phase 1 (10 cases, ~31 images)

| Component | Calculation | Cost |
|-----------|-------------|------|
| MedGemma 4B (Vertex) | ~15 min deploy/warm-up + ~15 min benchmark = 30 min × $1.15/hr | ~$0.58 |
| Gemini 3 Pro: Criterion 1 | ~5K input tokens × 31 images × $1.25/1M + ~1K output × 31 × $10/1M | ~$0.50 |
| Gemini 3 Pro: Criterion 2 | ~15K input tokens × 10 cases × $1.25/1M + ~2K output × 10 × $10/1M | ~$0.40 |
| LLM-as-judge | ~62 findings + 20 diagnosis judge calls (Gemini Flash) | ~$0.10 |
| **Total Phase 1** | | **~$1.60** |

### Phase 2 (44 cases, 135 images)

| Component | Calculation | Cost |
|-----------|-------------|------|
| MedGemma 4B (Vertex) | ~15 min warm-up + ~45 min benchmark = 60 min × $1.15/hr | ~$1.15 |
| Gemini 3 Pro: Criterion 1 | ~5K input × 135 images × $1.25/1M + ~1K output × 135 × $10/1M | ~$2.20 |
| Gemini 3 Pro: Criterion 2 | ~15K input × 44 cases × $1.25/1M + ~2K output × 44 × $10/1M | ~$1.70 |
| LLM-as-judge | ~270 findings + 88 diagnosis judge calls | ~$0.40 |
| **Total Phase 2** | | **~$5.45** |

### Phase 3 (267 cases, ~820 images)

| Component | Calculation | Cost |
|-----------|-------------|------|
| MedGemma 4B (Vertex) | ~15 min warm-up + ~3 hr benchmark = 195 min × $1.15/hr | ~$3.75 |
| Gemini 3 Pro: Criterion 1 | ~5K input × 820 images × $1.25/1M + ~1K output × 820 × $10/1M | ~$13.30 |
| Gemini 3 Pro: Criterion 2 | ~15K input × 267 cases × $1.25/1M + ~2K output × 267 × $10/1M | ~$10.35 |
| LLM-as-judge | ~1,640 findings + 534 diagnosis judge calls | ~$2.40 |
| **Total Phase 3** | | **~$29.80** |

---

## 11. Expected Output

### Run Artifacts (`runs/<run_id>/`)

```
runs/<run_id>/
├── config.yaml                    # Benchmark configuration
├── benchmark_input.json           # All input cases with resolved paths + all images
├── results/
│   ├── criterion1/                # Image Findings (per-image)
│   │   ├── medgemma_4b_findings.json   # Raw model responses per image
│   │   ├── gemini_pro_findings.json    # Raw model responses per image
│   │   ├── llm_judge_findings.json     # Per-image findings judgment
│   │   └── rouge_scores.json           # Per-image ROUGE-L scores
│   ├── criterion2/                # Diagnosis (per-case)
│   │   ├── medgemma_4b_diagnosis.json  # Raw model responses per case
│   │   ├── gemini_pro_diagnosis.json   # Raw model responses per case
│   │   └── llm_judge_diagnosis.json    # Per-case diagnosis judgment
│   └── parsed_predictions.json    # All extracted predictions
├── metrics/
│   ├── criterion1_summary.json    # Aggregate findings metrics
│   ├── criterion2_summary.json    # Aggregate diagnosis metrics
│   ├── per_image_scores.csv       # Per-image × per-model Criterion 1 scores
│   ├── per_case_scores.csv        # Per-case × per-model Criterion 2 scores
│   └── combined_summary.json      # Both criteria side-by-side
└── traces/
    └── cost_summary.json         # Total cost per model per criterion
```

### Criterion 1 Summary Table: Image Findings (per-image)

| Metric | MedGemma 4B (Vertex) | Gemini 3 Pro | Delta |
|--------|---------------------|--------------|-------|
| Findings — Correct (LLM-judge) | X% | Y% | ±Z% |
| Findings — Partial | X% | Y% | ±Z% |
| Findings — Incorrect | X% | Y% | ±Z% |
| Findings — ROUGE-L Recall | X | Y | ±Z |
| Findings — ROUGE-L F1 | X | Y | ±Z |
| # Images Evaluated | N | N | — |
| Avg Latency per Image (ms) | X | Y | — |

### Criterion 2 Summary Table: Diagnosis (per-case)

| Metric | MedGemma 4B (Vertex) | Gemini 3 Pro | Delta |
|--------|---------------------|--------------|-------|
| Diagnosis — Correct (LLM-judge) | X% | Y% | ±Z% |
| Diagnosis — Partial | X% | Y% | ±Z% |
| Diagnosis — Exact Match | X% | Y% | ±Z% |
| # Cases Evaluated | N | N | — |
| Avg Images per Case | X | X | — |
| Avg Latency per Case (ms) | X | Y | — |
| Avg Cost per Case ($) | X | Y | — |

---

## 12. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Vertex 4B multimodal doesn't accept images | Medium | Blocks benchmark | Smoke test first (Step 9); fallback: HF endpoint with `use_chat_api=True` + `max_tokens=512` (degraded but functional) |
| Vertex 4B rejects multi-image input (Criterion 2) | Medium | Blocks Criterion 2 | Smoke test with multi-image case. Fallback: send images sequentially as separate turns, or cap at 1 image for Criterion 2 |
| Multi-image payload too large for vLLM | Low | Fails on high-image cases | Cap at 8 images per case; use JPEG (smaller payload); log payload sizes |
| Vertex deploy fails or takes >30 min | Low | Delays benchmark | `deploy_vertex.sh` handles polling; manual status check via `gcloud ai endpoints list` |
| MedGemma 4B produces unparseable output | Medium | Partial data loss | Flexible regex parsing with multiple fallbacks; save raw responses for manual review |
| Gemini 3 Pro throttled (429) | Low | Slows benchmark | Existing retry/backoff (8 retries, exponential) handles this |
| Benchmark script crashes mid-run | Low | Cost risk (Vertex stays up) | try/finally with teardown reminder; manual `deploy_vertex.sh --teardown` |
| LLM-as-judge disagrees with human assessment | Medium | Misleading scores | Spot-check judge decisions on all Phase 1 cases before trusting at scale |
| Identical captions inflate Criterion 1 metrics | Low | Misleading sample size | Track and report duplicated-caption images; report metrics with/without them |

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
- `Pillow` — Image preprocessing (RGBA→RGB conversion, JPEG encoding)

### Environment variables (in `.env`)
- `GOOGLE_API_KEY` — for Gemini 3 Pro / Flash via AI Studio
- `GCP_PROJECT_ID` — for Vertex AI
- `GCP_REGION` — for Vertex AI (default: `us-central1`)
- `VERTEX_ENDPOINT_ID_4B` — set after deployment (Step 8)
- `VERTEX_DEDICATED_DNS` — set after deployment if applicable

### CLI tools
- `gcloud` — for Vertex AI deployment and teardown
- `uv` — Python package management and script execution
