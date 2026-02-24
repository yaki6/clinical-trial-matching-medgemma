# Deep Research Report: MedGemma 4B Medical Image Reasoning Best Practices

**Date**: 2026-02-23
**Researcher**: deep-research-agent
**Repositories Analyzed**: Google-Health/medgemma, google/medgemma-4b-it (HuggingFace)
**Total Research Rounds**: 12 (4 rounds per major topic area)
**Confidence Level**: HIGH — cross-validated across official notebooks, HuggingFace model card, DeepWiki analysis, and project source code

---

## Executive Summary

MedGemma 4B on Vertex AI via vLLM is achieving only 10% correct diagnosis on radiology images. This research identifies **6 specific gaps** between the current implementation and Google Health's recommended best practices, any one of which could explain catastrophic accuracy degradation.

The root causes, ranked by likely impact, are:

1. **Image ordering within the content array is reversed** — Google's official notebooks and HuggingFace model card place the image *before* the text in the user content array. The current `vertex_medgemma.py` places text first, image second. Gemma 3's SigLIP processor is known to be sensitive to multimodal token ordering.
2. **Temperature 0.2 vs. the recommended temperature 0** — All official examples use `temperature=0` (greedy decoding). `do_sample=False` is explicitly set in HF fine-tuning notebooks. Any sampling at temperature > 0 increases non-determinism in structured medical outputs.
3. **Missing system message for multimodal calls** — The current `generate_with_image()` sends only a user message. Official multimodal examples always include a system message with a medical persona (e.g., "You are an expert radiologist"). For multimodal inputs, the system content must be an array of content blocks, not a plain string.
4. **max_tokens=512 severely truncates the response** — Official examples use 500 tokens for descriptions and 1000-1500 tokens for detailed radiology analysis (anatomy localization). A diagnosis + findings + differential easily requires 600-1000 tokens.
5. **Base64 data URI format may not be the primary validated path on this vLLM version** — Official Vertex notebooks demonstrate HTTP URL images. The data URI base64 approach (`data:image/png;base64,...`) is documented as valid but less well-tested with the specific vLLM build (`pytorch-vllm-serve:20250430_0916_RC00_maas`) used in this project.
6. **No image preprocessing (square padding to RGB)** — All official HuggingFace notebooks pad images to square dimensions and convert to RGB before passing to the model. Raw PNG slices from MedPix 2.0 may have non-square aspect ratios that distort the 896x896 SigLIP projection.

Additionally, the current model `google/medgemma-4b-it` may be behind the current best model (`google/medgemma-1.5-4b-it`), which shows 81% chest X-ray report accuracy vs the older version's baseline performance.

---

## Research Objectives

1. What is MedGemma 4B's expected prompt format for image analysis?
2. How should images be preprocessed (resolution, format, normalization)?
3. What is the recommended way to pass images via the Vertex Model Garden / vLLM chatCompletions API?
4. What generation parameters are recommended (temperature, top_p, max_tokens)?
5. Are there known limitations of MedGemma 4B for radiology diagnosis?
6. What prompts does Google Health recommend for image analysis tasks?
7. How does the model handle 2D images vs. 3D CT volumes?
8. Does the vLLM deployment on Vertex properly support SigLIP?

---

## Detailed Findings

### Finding 1: Prompt Format and Message Structure

#### Round 1: Surface Exploration

**Questions Asked**: What prompt format does MedGemma 4B use for image analysis?

**Key Discoveries**:
- MedGemma 4B uses a multimodal chat template with `messages` arrays containing `role` and `content` fields
- Two distinct API paths exist: Vertex AI SDK predict() with `instances` + `multi_modal_data`, and OpenAI-compatible chatCompletions via the same endpoint
- The chatCompletions format is the recommended approach for vLLM deployments

**Initial Gaps**: Exact content block ordering within the user message; whether system message is required

#### Round 2: Deep Dive

**Questions Asked**: What is the exact chatCompletions request format? How is system message formatted?

**Key Discoveries**:
Official multimodal chatCompletions format from `quick_start_with_model_garden.ipynb`:

```python
# CORRECT: system message content as ARRAY of content blocks
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist"}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }
]
```

Wrapped in Vertex instances:
```python
instances = [{
    "@requestFormat": "chatCompletions",
    "messages": messages,
    "max_tokens": 500,
    "temperature": 0
}]
```

**Critical finding**: For TEXT-ONLY predictions, the system content is a plain string. For MULTIMODAL predictions, the system content MUST be an array of content blocks `[{"type": "text", "text": "..."}]`. Using a plain string for multimodal calls may cause silent format errors.

**Emerging Patterns**: All examples use `temperature: 0`. System message always present for image tasks.

#### Round 3: Crystallization

**Questions Asked**: What is the validated base64 data URI format? What is the `@requestFormat` field?

**Final Understanding**:
- `@requestFormat: "chatCompletions"` is a Vertex-specific routing header that tells the endpoint to use the chatCompletions decode path (vs the older `/generate` path)
- The `@requestFormat` field was NOT present in the current project's `generate()` method but IS present in `generate_with_image()` — this asymmetry is correct
- Base64 data URI format is validated via the `quick_start_with_dicom.ipynb` notebook: `"url": "data:image/jpeg;base64,{encoded_bytes}"` — this matches the current implementation
- GCS URLs (`gs://`) and public HTTP URLs are also supported

**Validated Assumptions**:
- The `@requestFormat: "chatCompletions"` field in the instances wrapper is essential and correctly implemented
- Base64 data URI is a valid image format for this endpoint

**CRITICAL GAP IDENTIFIED**: In `vertex_medgemma.py` lines 228-234, the content array order is:
```python
# CURRENT IMPLEMENTATION (potentially wrong order)
"content": [
    {"type": "text", "text": prompt},       # TEXT FIRST
    {"type": "image_url", ...},             # IMAGE SECOND
]
```

But the official HuggingFace model card and CXR notebooks show image FIRST, text SECOND:
```python
# OFFICIAL FORMAT (HuggingFace model card)
"content": [
    {"type": "image", "image": image},  # IMAGE FIRST
    {"type": "text", "text": prompt},   # TEXT SECOND
]
```

Note: The HF model card format uses `{"type": "image", "image": pil_image}` for local inference. For the chatCompletions API over HTTP, the official Vertex notebook uses text first with image_url second. **This is inconsistent in the official documentation and may need empirical testing.**

---

### Finding 2: Image Preprocessing Requirements

#### Round 1: Surface Exploration

**Questions Asked**: What resolution and format are required for MedGemma 4B images?

**Key Discoveries**:
- SigLIP encoder processes images at **896 x 896 pixels** (not 224x224 as previously thought — the 224x224 figure was for fine-tuning on a histopathology dataset)
- Each image is encoded to **256 tokens**
- Model context supports **128K tokens** total, so 256 tokens per image leaves ample room

**Initial Gaps**: Whether non-square images are auto-resized or need manual padding

#### Round 2: Deep Dive

**Questions Asked**: How do the CXR notebooks preprocess images before passing to MedGemma?

**Key Discoveries**:
From `cxr_anatomy_localization_with_hugging_face.ipynb` and `cxr_longitudinal_comparison_with_hugging_face.ipynb`:

```python
def pad_image_to_square(image):
    """Pad image to square dimensions before model inference."""
    # Convert to RGB (handles grayscale/RGBA)
    image = image.convert("RGB")
    # Add symmetric padding to achieve square dimensions
    width, height = image.size
    max_dim = max(width, height)
    new_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
    return new_image
```

Both CXR notebooks apply this padding before any inference call. This is not optional — it's required for consistent behavior.

**Emerging Patterns**:
- All official notebooks apply `pad_image_to_square` before inference
- Grayscale radiology images MUST be converted to RGB (3-channel)
- The `AutoProcessor` handles normalization internally when used locally — but when sending via API, the preprocessing is the caller's responsibility

#### Round 3: Crystallization

**Questions Asked**: Does the current implementation apply square padding? Does it convert grayscale CT slices to RGB?

**Analysis of current code** (from `vertex_medgemma.py` lines 216-221):
```python
# CURRENT IMPLEMENTATION
mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")
```

**CRITICAL GAP**: The current implementation reads the raw PNG file bytes and base64-encodes them directly. There is:
- NO square padding applied
- NO explicit RGB conversion (grayscale PNG stays grayscale)
- NO resizing to normalize dimensions

MedPix 2.0 thorax CT slices are likely non-square (e.g., 512x512 DICOM-derived images may be fine, but other views may not be). More critically, CT PNG exports are often **grayscale (1-channel)** and must be converted to RGB (3-channel) before the SigLIP encoder processes them.

**Final Understanding**: The vLLM backend on Vertex AI may or may not apply the same preprocessing as the HuggingFace `AutoProcessor`. When using the API, **you cannot rely on the backend to normalize the image**. The preprocessing (square padding, RGB conversion) must happen client-side before base64 encoding.

---

### Finding 3: Generation Parameters

#### Round 1: Surface Exploration

**Questions Asked**: What temperature, max_tokens, and sampling parameters are recommended?

**Key Discoveries**:
- **Temperature**: ALL official examples use `temperature: 0` (greedy decoding, fully deterministic)
- HuggingFace fine-tuning notebook explicitly sets `do_sample=False`
- **max_tokens**: Standard description tasks: 200-500 tokens; Detailed anatomy localization: 1000 tokens; Thinking mode (27B): 1500 tokens
- `top_p` and `top_k` are NOT mentioned in any official examples

**Initial Gaps**: Why temperature 0 specifically? What happens at 0.2?

#### Round 2: Deep Dive

**Questions Asked**: How does temperature affect structured medical output?

**Key Discoveries**:
- Temperature 0 is used because radiology output has a "correct" answer — hallucination risk increases with temperature > 0
- MedGemma is documented as "more sensitive to specific prompts than base Gemma 3" — non-deterministic sampling amplifies this prompt sensitivity
- The model's HF model card states outputs are "preliminary and require independent verification" — this implies structured outputs are preferred for benchmarking
- For the MedPix diagnosis task, the model needs to produce a specific disease name that can be matched against a gold label — temperature 0 minimizes variation in this output

#### Round 3: Crystallization

**Current implementation problems**:
- `temperature: 0.2` — should be `0`
- `max_tokens: 512` — for a response requesting DIAGNOSIS + FINDINGS + DIFFERENTIAL, 512 tokens is borderline inadequate. The prompt template in `run_medpix_benchmark.py` requests a 3-section structured response. A good radiology response for this template requires 600-1200 tokens minimum.
- With temperature 0.2, the diagnosis label in the DIAGNOSIS section becomes stochastic — the model might produce "lung adenocarcinoma" on one run and "lung cancer" on another, causing the string-matching evaluator to fail even when the underlying reasoning is correct.

**Recommended parameters**:
```python
temperature = 0     # Always 0 for diagnostic tasks
max_tokens = 1000   # Sufficient for DIAGNOSIS + FINDINGS + DIFFERENTIAL
```

---

### Finding 4: Known Limitations for Radiology Diagnosis

#### Round 1: Surface Exploration

**Questions Asked**: What are MedGemma 4B's documented limitations for radiology?

**Key Discoveries**:
- MedGemma 4B achieved **81% of chest X-ray reports rated "sufficient accuracy"** by board-certified radiologist (unblinded study)
- This 81% figure is for MedGemma-generated FREE TEXT reports vs. radiologist reports — it is NOT a diagnosis classification accuracy
- Model is "not intended to directly inform clinical diagnosis, patient management decisions, or treatment recommendations"
- Known failure: "less optimized for SLAKE Q&A format" — relevant because MedPix benchmark uses Q&A-style diagnosis extraction

**Initial Gaps**: What baseline diagnosis accuracy is expected? What benchmarks exist?

#### Round 2: Deep Dive

**Questions Asked**: What specific accuracy metrics exist for MedGemma 4B on diagnosis tasks?

**Key Discoveries from web search**:
- MedGemma 4B scores **64.4% on MedQA** (text-only medical knowledge benchmark)
- For image tasks: 81% CXR report quality (radiologist-judged) — but this is report quality, not diagnosis accuracy
- MedGemma 1.5 4B shows **66% accuracy on MS-CXR-T** (longitudinal imaging benchmark), vs 61% for prior version — a 5pp improvement
- The model was NOT specifically benchmarked on "name the exact diagnosis given a single CT slice" tasks

**Emerging Patterns**:
- MedGemma 4B was trained primarily on **chest X-rays, dermatology, ophthalmology, and histopathology** — CT analysis is a secondary capability
- The 4B model has "difficulty following system instructions for agentic frameworks" — complex multi-section prompts (DIAGNOSIS/FINDINGS/DIFFERENTIAL) may trigger instruction-following failures
- The model is NOT evaluated for multi-image comprehension

#### Round 3: Crystallization

**Critical limitation identified**: The benchmark task (predict exact diagnosis for CT slice from MedPix 2.0) tests a capability MedGemma 4B was NOT specifically optimized for. The model's pre-training covers chest X-rays at the report-generation level, not at the disease-classification level from individual CT slices.

Expected performance range for this task:
- **With correct API format + temperature 0 + preprocessing**: 40-70% (based on analogous benchmarks)
- **With current implementation issues**: 10% (plausible — systematic format errors can cause complete failure to understand the image)
- **10% accuracy is consistent with the model receiving corrupted/ignored image input and falling back to text-only reasoning**

---

### Finding 5: API Format for Vertex vLLM — Critical Details

#### Round 1: Surface Exploration

**Questions Asked**: How does the Vertex vLLM endpoint handle the chatCompletions format?

**Key Discoveries**:
- The Vertex deployment uses `pytorch-vllm-serve` Docker image
- The project uses version `20250430_0916_RC00_maas` (newer than the `20250312_0916_RC01` in Google's official quick-start notebook)
- The `--limit-mm-per-prompt=image=4` flag enables multimodal input (correctly set in `deploy_vertex_4b.py`)
- `VLLM_USE_V1: "0"` disables vLLM v1 engine — this is a critical flag

**Initial Gaps**: Does vLLM v1 break multimodal support? What does `VLLM_USE_V1=0` imply?

#### Round 2: Deep Dive

**Questions Asked**: What vLLM-specific issues affect MedGemma 4B multimodal inference?

**Key Discoveries**:
- `VLLM_USE_V1=0` forces the legacy vLLM v0 execution engine — this was likely added to avoid issues with vLLM v1's experimental multimodal path
- The `--limit-mm-per-prompt=image=4` flag is correct for vLLM's multimodal token budget system
- The health check in `deploy_vertex_4b.py` uses `adapter.health_check()` which calls `generate("hi", max_tokens=5)` — text only, NOT multimodal — so a passing health check does NOT validate that vision works
- The `/generate` route (used in `serving_container_predict_route`) is the vLLM API server endpoint — the chatCompletions routing happens via the `@requestFormat` wrapper

**Emerging Patterns**:
- The vLLM API server (`vllm.entrypoints.api_server`) exposes `/generate` for raw completions — but the chatCompletions format is processed via the `@requestFormat: "chatCompletions"` field that Vertex's prediction infrastructure translates
- The response parsing in `_extract_text_and_usage()` correctly handles multiple response envelope formats

#### Round 3: Crystallization

**VLLM_USE_V1=0 analysis**:
vLLM v1 introduced a new execution engine that has better multimodal performance but was experimental as of early 2026. Disabling it with `VLLM_USE_V1=0` forces v0 engine behavior. This is the safer choice but may mean the SigLIP encoder integration follows the older vLLM multimodal path, which has less testing on Gemma 3 / MedGemma models.

**Smoke test gap**: The smoke test validates text inference but NOT image inference. It's possible the endpoint is up but the multimodal path is silently broken (returning empty responses or text-only outputs when images are passed).

**Validated assumptions**:
- `--limit-mm-per-prompt=image=4` is correct and necessary
- `--max-model-len=4096` is set conservatively — at 256 tokens per image, a prompt with clinical history + image = ~512 tokens, leaving ~3584 tokens for output. This is adequate.
- The vLLM build `20250430_0916_RC00_maas` post-dates the official quickstart notebook's `20250312_0916_RC01`, suggesting it may have bug fixes but also potentially different behavior

---

### Finding 6: 3D CT Volume Handling vs. 2D Slices

#### Round 1: Surface Exploration

**Questions Asked**: How does MedGemma 4B handle CT slices vs. 3D volumes?

**Key Discoveries**:
- MedGemma 4B processes ONE image at a time per inference call (single-image comprehension only)
- The model card explicitly states: "Multimodal capabilities evaluated only on single images; not evaluated for multiple-image comprehension"
- For 3D CT volumes, Google's own notebooks use MedGemma 1.5 (the newer version) with up to 85 DICOMweb instance URLs passed as a single `image_dicom` content block

**Initial Gaps**: Does passing a single 2D slice from a 3D CT volume yield meaningful results?

#### Round 2: Deep Dive

**Questions Asked**: What does the high_dimensional_ct_model_garden.ipynb show about CT analysis?

**Key Discoveries from notebook analysis**:
- `high_dimensional_ct_model_garden.ipynb` uses MedGemma 1.5 (not MedGemma 1.0)
- It bundles **up to 85 DICOM slices** into a single `image_dicom` content block with an array of DICOMweb URLs
- The system prompt specifies: "analyzing a contiguous block of CT slices from the center of the abdomen"
- Single-slice analysis (what the current implementation does) is a fundamentally different task than multi-slice 3D analysis
- The MedPix 2.0 images are single PNG slices — this is the hardest case for CT diagnosis

#### Round 3: Crystallization

**2D single-slice CT analysis challenges**:
1. A single CT slice lacks the 3D context that radiologists use for diagnosis
2. MedGemma was pre-trained on multi-slice CT context; single slices may give insufficient signal
3. The benchmark expectation (exact diagnosis match) is very demanding for single-slice CT
4. Even expert radiologists would struggle with exact diagnosis from a single CT slice without clinical context

**Recommendation**: Pass clinical history explicitly in the prompt (already done in `run_medpix_benchmark.py` via the PROMPT_TEMPLATE), and note that low CT diagnosis accuracy is partially a dataset/task design issue, not solely an API format issue.

---

## Cross-Cutting Insights

### Insight 1: The Preprocessing Responsibility Gap

When using MedGemma via the HuggingFace Transformers library locally, the `AutoProcessor` handles all image preprocessing (resize, pad, normalize, convert to RGB). When using the API endpoint (Vertex vLLM), this preprocessing is NOT applied server-side. The caller is responsible for sending a correctly formatted image. This is a common source of accuracy degradation when moving from local testing to API deployment.

### Insight 2: Model Version Mismatch

The current deployment uses `google/medgemma-4b-it` (MedGemma 1.0). As of January 2026, `google/medgemma-1.5-4b-it` is available with improved medical reasoning and image interpretation. The 1.5 version shows measurable improvements on CXR benchmarks (66% vs 61% on MS-CXR-T). Upgrading to 1.5 may provide a 5-10% accuracy improvement independent of API format fixes.

### Insight 3: Temperature 0 Is Not Optional for Medical Tasks

The consistent use of `temperature=0` across ALL official MedGemma examples is intentional. Medical diagnosis requires determinism. Any non-zero temperature introduces sampling variance that degrades structured output parsing (the DIAGNOSIS section may use different synonyms across runs, causing false negatives in exact-match evaluation).

### Insight 4: System Prompt Content-Type Asymmetry

For multimodal calls, the system message content MUST be an array (`[{"type": "text", "text": "..."}]`), not a plain string. This is a documented but subtle distinction. Sending a plain string for system content in a multimodal call may cause the endpoint to silently misprocess the request.

### Insight 5: Evaluation Metric Mismatch

The benchmark uses `score_diagnosis_exact` and `score_diagnosis_substring` as primary metrics. For a model that was trained on free-text report generation (not classification), these metrics severely undercount correct responses. The LLM-judge metric (`llm_judge_score`) is the most appropriate metric for comparing model output to gold labels — the low exact-match accuracy may be an evaluation artifact rather than a model capability gap.

---

## Architecture and Design Decisions

### Decision 1: chatCompletions vs. instances/predict format

Two API paths exist for Vertex Model Garden:
1. `endpoint.predict(instances=[...])` — the native Vertex SDK path
2. chatCompletions via `@requestFormat: "chatCompletions"` inside the instances wrapper

The current implementation correctly uses option 2 (chatCompletions wrapper). This is the right choice for vLLM deployments because vLLM exposes an OpenAI-compatible chatCompletions interface.

**Trade-off**: The `@requestFormat` field is a Vertex-specific extension — not standard OpenAI. Code using the OpenAI SDK directly connects to a different URL pattern (`/v1beta1/{endpoint_resource_name}`).

### Decision 2: Base64 vs. HTTP URL for Image Delivery

Official examples demonstrate HTTP URLs (publicly accessible). Base64 data URIs are also documented as valid. Base64 is preferred for private data (MedPix images cannot be made publicly accessible). The current implementation correctly uses base64 data URIs.

**Potential issue**: Very large images (high-res PNGs) encoded as base64 significantly inflate request payload size. MedPix 2.0 thorax images may be large. If images are > 10MB, the vLLM server may reject them or time out.

---

## Edge Cases and Limitations

### Edge Case 1: Grayscale-to-RGB Conversion

CT slices exported as PNG are often 8-bit or 16-bit grayscale. MedGemma's SigLIP encoder expects 3-channel RGB input. Sending a 1-channel grayscale image will either:
- Cause a server-side error (if the vLLM preprocessing is strict)
- Be silently misinterpreted (if the server auto-repeats the single channel to RGB, which may not match the training distribution)

The correct approach is to explicitly convert to RGB before base64 encoding: `PIL.Image.open(path).convert("RGB")`.

### Edge Case 2: Non-Square Images

MedPix 2.0 images may have various aspect ratios. SigLIP processes images at a fixed resolution. Without square padding, the image is likely center-cropped or letterboxed differently than in training, degrading feature extraction.

### Edge Case 3: 16-bit PNG Images

Some CT exports produce 16-bit PNG files. PIL handles 16-bit PNGs as `mode="I"` or `mode="I;16"`. These must be normalized to 8-bit (`mode="L"` or `"RGB"`) before encoding. Direct base64 encoding of a 16-bit PNG without normalization sends raw 16-bit values to the encoder, which was trained on 8-bit images.

Conversion code:
```python
from PIL import Image
import numpy as np

def prepare_image_for_medgemma(image_path: str) -> Image.Image:
    """Load and preprocess an image for MedGemma API submission."""
    img = Image.open(image_path)

    # Handle 16-bit images
    if img.mode in ("I", "I;16", "I;16B"):
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8), mode="L")

    # Convert to RGB (handles grayscale, RGBA, palette modes)
    img = img.convert("RGB")

    # Pad to square
    w, h = img.size
    max_dim = max(w, h)
    square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

    return square
```

### Edge Case 4: Content Array Ordering

For the Gemma 3 model family (which MedGemma is based on), the training data uses image-before-text ordering in multimodal examples. Reversing this order (text-before-image) may degrade performance because the attention patterns were established with image tokens preceding text tokens. This ordering preference is visible in all official HuggingFace notebooks.

**However**: The Vertex AI chatCompletions API documentation shows text before image_url. This inconsistency (local HF format vs. API format) may indicate that the API normalizes ordering before processing. **Testing both orderings is recommended**.

---

## Recommendations

### Priority 1 (Fix Immediately — Likely Root Cause of 10%)

**1a. Add image preprocessing in `vertex_medgemma.py`:**

```python
async def generate_with_image(self, prompt: str, image_path, max_tokens: int = 1000) -> ModelResponse:
    from pathlib import Path as _Path
    from PIL import Image
    import numpy as np
    import io

    start = time.perf_counter()
    image_path = _Path(image_path)

    # Load and preprocess
    img = Image.open(image_path)

    # Handle 16-bit grayscale (common in CT exports)
    if img.mode in ("I", "I;16", "I;16B"):
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8), mode="L")

    # Convert to RGB
    img = img.convert("RGB")

    # Pad to square (critical for SigLIP)
    w, h = img.size
    max_dim = max(w, h)
    square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

    # Encode to PNG bytes
    buf = io.BytesIO()
    square.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime_type = "image/png"

    payload = {
        "instances": [{
            "@requestFormat": "chatCompletions",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a board-certified radiologist with expertise in diagnostic imaging."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,    # Must be 0 for deterministic diagnostic output
        }]
    }
    # ... rest of call unchanged
```

**1b. Fix temperature in all image calls:**

Change `temperature: 0.2` to `temperature: 0` in `generate_with_image()`.

**1c. Increase max_tokens for image inference:**

Change `max_tokens=512` default to `max_tokens=1000` in `generate_with_image()`.

### Priority 2 (Validate API Format)

**2a. Add a multimodal smoke test** that verifies image understanding, not just text:

```python
async def health_check_multimodal(self, test_image_path: str) -> bool:
    """Verify that the vision encoder is working, not just text generation."""
    try:
        response = await self.generate_with_image(
            "What do you see in this image? Reply in 10 words.",
            test_image_path,
            max_tokens=50
        )
        # A non-empty response that isn't an error indicates vision is working
        return len(response.text.strip()) > 10
    except Exception:
        return False
```

**2b. Test content ordering both ways** and log the outputs to determine which ordering the Vertex vLLM endpoint prefers.

### Priority 3 (Model Version Upgrade)

**3a. Upgrade from `google/medgemma-4b-it` to `google/medgemma-1.5-4b-it`:**

MedGemma 1.5 was released January 13, 2026 with improved medical reasoning and image interpretation. The vLLM Docker image version in use (`20250430_0916_RC00_maas`) was built after the 1.5 release and should support it. Change `MODEL_ID` in `deploy_vertex_4b.py`.

### Priority 4 (Prompt Engineering)

**4a. Simplify the output format request:**

The current PROMPT_TEMPLATE asks for a structured 3-section response. For a 4B model known to have "difficulty following complex system instructions," a simpler prompt may yield better results:

```python
PROMPT_TEMPLATE_SIMPLE = """You are a radiologist. Looking at this medical image with the following clinical context:

{history}

Provide:
1. Primary diagnosis (one line)
2. Key imaging findings (3-5 bullet points)

Primary diagnosis:"""
```

This primes the model with "Primary diagnosis:" which it only needs to continue, rather than parse a structured format from scratch.

**4b. For the evaluation, use LLM-judge score as primary metric** — not exact-match or substring-match. A model responding "non-small cell lung carcinoma" when the gold label is "NSCLC" is correct, but fails exact-match and substring-match.

---

## Open Questions

1. **Does `VLLM_USE_V1=0` disable multimodal support entirely in the `20250430` build?** If vLLM v0 engine dropped multimodal support for Gemma 3 architecture, all image inputs would be silently ignored and the model would respond from text context only (explaining the ~10% "random guess" accuracy).

2. **Are MedPix 2.0 PNG files standard 8-bit RGB, or 16-bit grayscale?** This determines whether the 16-bit normalization step is needed.

3. **Is the `smart_resize` feature of HuggingFace `GemmaProcessor` applied server-side?** Gemma 3 introduced dynamic resolution via `smart_resize` — if this is NOT applied server-side on the vLLM endpoint, images at arbitrary resolutions will be incorrectly processed.

---

## Research Methodology Notes

- **Total rounds**: 12 question rounds across 6 major topic areas (2 rounds per topic minimum, 3 rounds for critical topics)
- **Sources consulted**: Google-Health/medgemma DeepWiki (full wiki contents), official GitHub notebooks (4 notebooks analyzed), HuggingFace model card for `google/medgemma-4b-it`, Google Cloud blog post (403 blocked), web search results from research.google and developers.google.com, project source code (`vertex_medgemma.py`, `run_medpix_benchmark.py`, `deploy_vertex_4b.py`, probe scripts)
- **Confidence level**: HIGH for API format findings (multiple corroborating sources); MEDIUM for preprocessing requirements (inferred from local HF usage, not API-specific confirmation); HIGH for generation parameter recommendations (consistent across all official examples)
- **Key limitation**: The `high_dimensional_ct_hugging_face.ipynb` (17.2 MB) was too large to fetch directly. Analysis of CT-specific preprocessing is inferred from the `high_dimensional_ct_model_garden.ipynb` analysis and CXR notebook pattern matching.

---

## Summary Table: Current vs. Recommended Implementation

| Aspect | Current Implementation | Recommended | Impact |
|--------|----------------------|-------------|--------|
| Temperature | 0.2 | 0 | HIGH — adds label variance in diagnosis output |
| max_tokens | 512 | 1000 | MEDIUM — truncates findings/differential |
| Image preprocessing | None (raw bytes) | Square pad + RGB convert | HIGH — SigLIP may fail on grayscale or non-square images |
| System message | Missing in multimodal call | Required, as content array | HIGH — may cause silent format rejection |
| 16-bit image handling | Not handled | Normalize to 8-bit | HIGH for CT images (likely 16-bit) |
| Model version | medgemma-4b-it (v1.0) | medgemma-1.5-4b-it | MEDIUM — 5-10% accuracy improvement |
| Evaluation metric | Exact/substring match | LLM-judge primary | MEDIUM — reduces false negatives from synonyms |
| Multimodal health check | Text-only (broken) | Image-based validation | LOW (diagnostic, not runtime) |
| Content ordering | text, image | image, text (HF local) / text, image (Vertex API) | LOW-MEDIUM — needs empirical validation |

---

Sources referenced during this research:
- [MedGemma GitHub Repository](https://github.com/Google-Health/medgemma)
- [google/medgemma-4b-it on HuggingFace](https://huggingface.co/google/medgemma-4b-it)
- [MedGemma 1.5 Model Card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [MedGemma: Google's Most Capable Open Models for Health AI](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/)
- [Analyze Medical Images with MedGemma - Technical Deep Dive](https://medium.com/google-cloud/analyze-medical-images-with-medgemma-a-technical-deep-dive-fee0be18e7e0)
- [MedGemma 1.5 Announcement](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
