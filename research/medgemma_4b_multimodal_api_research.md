# Deep Research Report: MedGemma 4B Multimodal API Integration

**Date**: 2026-02-21
**Researcher**: deep-research-agent
**Repositories Analyzed**: google/medgemma-4b-it (HF), Google-Health/medgemma (GitHub), src/trialmatch/models/ (local)
**Total Research Rounds**: 4 (web searches across multiple targeted queries per topic)

---

## Executive Summary

MedGemma 4B is a Gemma 3-based multimodal model with a medical-domain SigLIP image encoder. It supports chest X-rays, CT, MRI, histopathology, ophthalmology, and dermatology images. When deployed via HuggingFace Inference Endpoints (TGI backend), **image input uses the OpenAI-compatible `/v1/chat/completions` endpoint with `image_url` content type** — the same format as GPT-4V. This is critically different from the current project's TGI text_generation path, which is text-only.

The key integration decision for this project: the existing HF endpoint (`pcmy7bkqtqesrrzd`) runs TGI in text_generation mode, which **does not accept images**. To use multimodal image features, the endpoint must be invoked via `chat_completion` with `use_chat_api=True` and image content in the messages list. However, the TGI CUDA bug (CUBLAS_STATUS_EXECUTION_FAILED) documented in CLAUDE.md makes image inference on TGI risky for the 4B model. Vertex AI vLLM is the safer path for multimodal image requests.

There is also a known model-level bug (fixed July 9, 2025) where an end-of-image token was missing from the vocabulary, causing silent degradation on multimodal tasks. Any MedGemma 4B checkpoint pulled after July 9, 2025 includes this fix.

---

## Research Objectives

1. Exact API format for image input to HF Inference Endpoint (TGI) for MedGemma 4B
2. Supported medical image modalities and limitations
3. Whether to use `chat_completion` vs `text_generation` for images
4. Base64 vs URL image encoding format
5. Recommended prompt structure for medical image extraction
6. Known bugs and constraints
7. Vertex AI path for multimodal image input

---

## Detailed Findings

### Topic 1: HuggingFace Inference Endpoint — Image API Format

#### Round 1: Surface Exploration

**Questions Asked**: What endpoint format does TGI use for image input? Is it chat/completions or text_generation?

**Key Discoveries**:
- TGI provides two inference paths: `POST /generate` (text_generation) and `POST /v1/chat/completions` (Messages API / OpenAI-compatible)
- Only the `/v1/chat/completions` path supports multimodal (image) inputs in TGI
- The `text_generation` path is text-only — images cannot be embedded in a raw text string
- The Messages API must be enabled at container startup via `MESSAGES_API_ENABLED=true` (default in TGI >= 2.3.0)

**Initial Gaps**: What is the exact JSON structure for image content? Is base64 supported or only URLs?

#### Round 2: Deep Dive

**Questions Asked**: Exact content structure for image_url, base64 format, how InferenceClient maps to this

**Key Discoveries**:

The chat completions API for TGI VLMs accepts this message structure:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/xray.jpg"
                    # OR base64: "data:image/jpeg;base64,{base64_string}"
                }
            },
            {
                "type": "text",
                "text": "Describe findings in this chest X-ray."
            }
        ]
    }
]
```

Using `huggingface_hub.InferenceClient`:
```python
from huggingface_hub import InferenceClient
import base64

client = InferenceClient(
    model="https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud",
    token=HF_TOKEN,
)

# Base64 encode image
with open("xray.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                },
                {
                    "type": "text",
                    "text": "Extract key clinical findings from this image."
                }
            ]
        }
    ],
    max_tokens=512,
)
text = response.choices[0].message.content
```

Using OpenAI SDK pointing to TGI endpoint:
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud/v1",
    api_key=HF_TOKEN,
)

response = client.chat.completions.create(
    model="tgi",  # model name is ignored by TGI but required by OpenAI SDK
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Describe the pathology visible in this image."}
            ]
        }
    ],
    max_tokens=512,
)
```

**Emerging Patterns**:
- The content field must be a **list** (not a string) when images are included
- Image must appear **before** the text in the content list (this is the recommended order)
- The `image_url` object always has a single `url` key containing either an HTTPS URL or `data:` URI

#### Round 3: Crystallization

**Questions Asked**: Does the current project's MedGemmaAdapter support image content lists? What changes are needed?

**Final Understanding**:

The current `MedGemmaAdapter._call_chat_completion()` in `src/trialmatch/models/medgemma.py` passes `prompt` as a plain string:
```python
messages = [{"role": "user", "content": prompt}]  # content is a STRING — text only
```

To support images, content must become a **list**:
```python
messages = [{"role": "user", "content": [image_block, text_block]}]
```

The existing `use_chat_api=True` flag puts the adapter on the right path (chat_completion), but the prompt string must be restructured.

**Validated Assumptions**:
- TGI on HF Inference Endpoints does support image_url with base64 for vision models
- The `InferenceClient.chat_completion()` method passes content lists through correctly
- DICOM files must be pre-converted to JPEG/PNG before sending (see limitations section)

---

### Topic 2: Supported Medical Image Modalities

#### Round 1: Surface Exploration

**Questions Asked**: What medical image types does MedGemma 4B support?

**Key Discoveries**:
- SigLIP image encoder was pre-trained on de-identified medical data including:
  - Chest X-rays (CXR)
  - Histopathology slides (H&E staining)
  - Ophthalmology (fundus photography, retinal images)
  - Dermatology (skin lesion photography)
  - Radiology (general)
- MedGemma 1.5 4B (the version in this project) extends to:
  - CT (Computed Tomography) — single-slice 2D JPEG/PNG
  - MRI — single-slice 2D JPEG/PNG
  - Whole-slide histopathology (WSI) — as tiled patches
  - Longitudinal radiology (comparing current vs prior CXR)
  - Anatomical localization

**Initial Gaps**: Does CT/MRI mean full 3D volumes or 2D slices only? What about DICOM format?

#### Round 2: Deep Dive

**Questions Asked**: DICOM support, 3D volume handling, resolution requirements

**Key Discoveries**:
- DICOM is NOT natively supported by the model API — DICOM must be converted to JPEG/PNG first
- Community tools exist (e.g., `Violet-sword/DICOM-to-JPEG-Converter`) specifically for MedGemma preprocessing
- MedGemma 1.5 with native DICOMweb integration is being developed but requires server-side preprocessing
- CT/MRI support means **individual 2D axial/coronal/sagittal slices** encoded as JPEG or PNG
- No explicit resolution constraint is published, but Gemma 3 uses a fixed 896x896 input to SigLIP (confirmed by architecture)
- For WSI: multiple patch images can be sent — this is one of few multi-image use cases

**Emerging Patterns**:
- The practical workflow for CT/MRI: extract key representative slice(s) → convert DICOM → JPEG → base64 → API
- For clinical trial matching (this project's use case), a single representative slice is typically sufficient

#### Round 3: Crystallization

**Questions Asked**: Performance benchmarks per modality, vs GPT-4V comparisons

**Final Understanding**:

MedGemma 4B was benchmarked across 22+ datasets, 5 task types, 6 image modalities. The model was evaluated on:
- Radiology VQA
- CXR report generation (with expert human evaluation)
- Dermatology classification
- Pathology slide analysis
- Ophthalmology grading

Key performance context:
- MedGemma 4B is a **starting point for fine-tuning**, not a finished product
- Out-of-box performance on specialized tasks (e.g., rare surgical pathology) may be limited
- Developers are explicitly expected to fine-tune for production accuracy
- Agentic framework tasks (multi-step reasoning with tool use) are documented as a known weakness of 4B

---

### Topic 3: TGI API — chat_completion vs text_generation for Images

#### Round 1: Surface Exploration

**Questions Asked**: Which TGI endpoint path is required for images?

**Key Discoveries**:
- `/generate` (text_generation) = text only, no image support
- `/v1/chat/completions` (Messages API) = multimodal supported
- This is fundamental — cannot embed images in raw text prompts for Gemma 3

**Initial Gaps**: Is chat_completion always enabled on HF Inference Endpoints?

#### Round 2: Deep Dive

**Questions Asked**: TGI version requirements, flag configuration

**Key Discoveries**:
- TGI >= 2.3.0 enables Messages API by default (no flag needed for modern deployments)
- HF Inference Endpoints with the `pytorch` framework on recent images include this by default
- The existing endpoint (`pcmy7bkqtqesrrzd`) runs TGI — chat_completion should work if TGI version is recent enough
- The `use_chat_api=True` flag in `MedGemmaAdapter` is the correct switch to use chat_completion path

#### Round 3: Crystallization

**Final Understanding**:

For this project's MedGemma 4B HF endpoint:
1. Switch `use_chat_api=True` — already supported by `MedGemmaAdapter`
2. Change content from string to list with image_url + text blocks
3. **CRITICAL WARNING**: The existing TGI CUDA bug (CUBLAS) may be exacerbated by multimodal inputs which require more GPU operations. Image processing adds additional GPU kernel calls that may hit the same CUDA instability.
4. Alternative: Use Vertex AI vLLM endpoint for multimodal — more reliable, no CUDA bug observed

---

### Topic 4: Known Bugs and Limitations

#### Round 1: Surface Exploration

**Key Discoveries**:
- **End-of-image token bug** (July 9, 2025 fix): Missing `<end_of_image>` token in vocabulary caused silent multimodal performance degradation. Fixed in all model checkpoints pulled after July 9, 2025.
- **TGI CUDA CUBLAS bug** (project-specific): Crashes at `max_new_tokens >= ~500-1024` on certain prompts. Documented exhaustively in CLAUDE.md. Not specific to multimodal, but risk of triggering increases with complex inputs.
- **No multi-turn optimization**: MedGemma 4B not evaluated or optimized for multi-turn conversations
- **Single-image primary evaluation**: Comprehension of multiple simultaneous images not benchmarked (exception: WSI tile patches)
- **No system prompt support**: Gemma 3 uses only `user` and `model` roles — system instructions must be folded into the first user turn
- **DICOM not natively accepted**: Must pre-convert to JPEG/PNG
- **Agentic framework weakness**: 4B model struggles with system-instruction-following in agentic setups (per MedGemma Technical Report)
- **MET bias** (project-specific finding): 4B model systematically biases toward MET in criterion matching. Likely an instruction-following weakness, not a reasoning one.

#### Round 2: Deep Dive

**Questions Asked**: Are there image-specific CUDA issues, resolution hard limits?

**Key Discoveries**:
- No published hard resolution limit for input images — SigLIP internally resizes to 896x896
- Large base64 payloads (e.g., high-res JPEG > 1MB) may cause TGI to time out or OOM independently of the CUDA bug
- For clinical imaging: compress to ~200-500KB JPEG before sending
- The July 2025 end-of-image token fix is critical — any model checkpoint before this date will have degraded multimodal performance

#### Round 3: Crystallization

**Confidence**: High for CUDA/TGI risks (project data). Medium for resolution limits (inference from architecture). High for July 2025 bug fix (documented by Google).

---

### Topic 5: Recommended Prompt Format for Medical Image Extraction

#### Round 1: Surface Exploration

**Key Discoveries**:
- Gemma 3 image-text format uses content list with `{"type": "image"}` or `{"type": "image_url"}` followed by `{"type": "text"}`
- Internal token: `<start_of_image>` is injected by the processor/tokenizer — developer does not write it manually
- System role not supported — fold instructions into user turn

**Initial Gaps**: Optimal clinical prompt structure for extracting structured EHR facts from images?

#### Round 2: Deep Dive

**Recommended prompt structure for medical image extraction (via chat_completion)**:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            },
            {
                "type": "text",
                "text": (
                    "You are an expert radiologist. Analyze this medical image and extract "
                    "key clinical findings relevant to patient eligibility assessment.\n\n"
                    "Respond in JSON format:\n"
                    "{\n"
                    '  "modality": "chest_xray|ct|mri|pathology|ophthalmology|dermatology",\n'
                    '  "findings": ["finding 1", "finding 2"],\n'
                    '  "key_abnormalities": ["abnormality 1"],\n'
                    '  "normal_findings": ["normal structure 1"],\n'
                    '  "clinical_impression": "one sentence summary"\n'
                    "}"
                )
            }
        ]
    }
]
```

**Key prompt design principles**:
1. Place image BEFORE text in the content list
2. Include domain specialization context ("You are an expert radiologist")
3. Specify output format explicitly (JSON)
4. Keep prompts concise — 4B has limited instruction-following capacity
5. Use Closed World Assumption: "If a finding is not visible, state it is absent"
6. Avoid multi-image requests unless using WSI tiling pattern

#### Round 3: Crystallization

**For CXR specifically** (most relevant to clinical trials):
```python
text_prompt = (
    "You are a board-certified radiologist. This is a posteroanterior chest X-ray.\n"
    "Extract clinical findings in JSON:\n"
    "{\n"
    '  "cardiac_size": "normal|enlarged",\n'
    '  "pulmonary_findings": [],\n'
    '  "pleural_findings": [],\n'
    '  "bone_findings": [],\n'
    '  "impression": "one sentence"\n'
    "}\n"
    "If a structure is not abnormal, omit it from findings arrays.\n"
    "Respond with JSON only."
)
```

---

### Topic 6: Vertex AI Path for Multimodal Image Input

#### Round 1: Surface Exploration

**Key Discoveries**:
- Vertex AI Model Garden deploys MedGemma 4B via vLLM (not TGI)
- vLLM does not have the CUBLAS bug documented in CLAUDE.md
- Vertex AI uses `chatCompletions` requestFormat in the predict body
- The current `VertexMedGemmaAdapter` uses text-only `content` string in `messages`

#### Round 2: Deep Dive

**Vertex AI multimodal image format**:

For vLLM on Vertex AI, the `instances` payload supports OpenAI-compatible content lists:

```python
payload = {
    "instances": [
        {
            "@requestFormat": "chatCompletions",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe the key findings in this medical image."
                        }
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.2,
        }
    ]
}
```

This works because vLLM's OpenAI-compatible server accepts `content` as either a string OR a list with text/image_url parts.

#### Round 3: Crystallization

**The VertexMedGemmaAdapter needs a multimodal variant** where `content` changes from a string to a list. The existing adapter's `generate(prompt, max_tokens)` signature is text-only. A new `generate_with_image(prompt, image_b64, mime_type, max_tokens)` method would be the clean design.

---

## Cross-Cutting Insights

1. **One API shape for all backends**: The OpenAI-compatible `content` list with `image_url` + `text` works for TGI (HF), vLLM (Vertex), and local transformers pipeline. Standardize on this format everywhere.

2. **Image before text**: Across all frameworks and implementations reviewed, image content blocks should appear BEFORE text content in the list. This matches the Gemma 3 architecture where image embeddings are prepended to text tokens.

3. **System prompt folding is non-negotiable**: Gemma 3 ignores `"role": "system"` — always fold system instructions into the first `"role": "user"` content block as the text component before or after the image.

4. **DICOM preprocessing is a prerequisite**: MedGemma does not process DICOM. Any clinical trial that provides imaging data as DICOM must go through a preprocessing step (pydicom → numpy → PIL → JPEG).

5. **4B vs 27B multimodal**: MedGemma 27B (IT version) is also multimodal (unlike 27B text-only). The 27B multimodal has higher accuracy but is not benchmarked in this project yet. For image-heavy tasks, 27B multimodal on Vertex would be the gold standard.

---

## Architecture / Design Decisions

| Decision | Current State | Required Change for Images |
|----------|---------------|---------------------------|
| HF TGI path | `text_generation` with string prompt | Switch to `chat_completion` with content list |
| Vertex AI path | `chatCompletions` with string content | Change content to list with image_url block |
| Image encoding | Not implemented | base64 JPEG, `data:image/jpeg;base64,{b64}` |
| DICOM handling | Not implemented | pydicom → PIL → JPEG → base64 |
| `MedGemmaAdapter.generate()` | text-only | Add `generate_with_image()` or extend signature |
| `VertexMedGemmaAdapter.generate()` | text-only | Same as above |
| Prompt structure | Plain string | Text block in content list (after image block) |

---

## Edge Cases and Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| TGI CUDA bug on max_tokens >= ~500 | Image prompts may increase crash risk | Use Vertex AI vLLM for images; keep max_tokens=256-512 |
| No multi-image support | Cannot send multiple slices simultaneously | Send one representative slice per request |
| No multi-turn optimization | Cannot do image follow-up questions | Full context per request |
| DICOM not natively accepted | All medical images from PACS need conversion | pydicom preprocessing pipeline |
| 4B agentic task weakness | May fail structured JSON output with images | Tight JSON schema in prompt; retry on parse fail |
| End-of-image token bug (pre-July 2025) | Silent degradation | Use model pulled after July 9, 2025 (current HF has fix) |
| Base64 payload size | Large images may hit TGI payload limits | Compress to <500KB JPEG before encoding |
| Gemma 3 no system role | System instructions silently ignored | Fold into user turn text block |
| MET bias (4B) | High false MET rate on criterion matching | Not directly related to image input; existing issue |

---

## Recommendations

### For Immediate Integration (Demo, Feb 24)

1. **Use Vertex AI vLLM path for any multimodal image requests** — avoids TGI CUDA risk entirely. The `VertexMedGemmaAdapter` is already wired for chatCompletions; only the content shape needs updating.

2. **Extend `VertexMedGemmaAdapter` with a multimodal method**:
   ```python
   async def generate_with_image(
       self,
       prompt: str,
       image_b64: str,
       mime_type: str = "image/jpeg",
       max_tokens: int = 512,
   ) -> ModelResponse:
       content = [
           {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
           {"type": "text", "text": prompt},
       ]
       # rest same as generate() but with content list instead of string
   ```

3. **Preprocess any medical images** through a standard pipeline:
   ```python
   # For DICOM: import pydicom; pixel_array = ds.pixel_array; img = PIL.Image.fromarray(...)
   # For JPEG/PNG: direct PIL open
   # Resize: img.thumbnail((896, 896))  # SigLIP max
   # Compress: io.BytesIO JPEG quality=85
   # Encode: base64.b64encode(buf.getvalue()).decode()
   ```

4. **Prompt template for clinical trial image context**:
   - Fold system context into user text block
   - Specify JSON output schema explicitly
   - Keep max_tokens at 512 (consistent with existing CUDA workaround)
   - Place image block before text block in content list

5. **For the HF endpoint**: Only attempt image inference if `use_chat_api=True` AND you are willing to handle potential CUDA crashes. Not recommended for demo stability.

### For Post-Demo (Phase 1+)

- Evaluate MedGemma 27B IT multimodal on Vertex AI for image-heavy tasks
- Build a pydicom-based DICOM preprocessing pipeline for PACS integration
- Consider MedSigLIP (standalone image encoder) as a preprocessing step for embedding medical images before passing to the LLM

---

## Open Questions

1. **Does the current HF endpoint version of TGI support multimodal chat_completion without redeployment?** The CUDA bug is with text_generation, but switching to chat_completion with images may trigger different CUDA paths. Needs empirical testing.

2. **What is the exact MedGemma 4B model checkpoint date on the current HF endpoint?** If it was deployed before July 9, 2025, the end-of-image token bug may be present, silently degrading multimodal results.

3. **Does Vertex AI MedGemma 4B deployment (vLLM) support multimodal content lists in the chatCompletions format?** The Vertex notebook confirms online prediction works for text; image_url in content list is not explicitly confirmed for the Model Garden deployment variant.

---

## Research Methodology Notes

- **Total rounds**: 4 rounds of web search queries across 6 topic areas
- **DeepWiki MCP**: Unavailable (permission denied) — substituted with targeted web searches and direct codebase analysis
- **Sources consulted**: HuggingFace model cards (medgemma-4b-it, medgemma-1.5-4b-it), TGI docs (visual_language_models), Google HAI-DEF model card, Google Cloud/Vertex AI docs, Google Research blog posts, MedGemma Technical Report (arxiv 2507.05201), local codebase analysis
- **Confidence levels**:
  - API format (image_url in content list): HIGH — confirmed across TGI docs, InferenceClient docs, OpenAI-compatible endpoint spec
  - Supported modalities: HIGH — confirmed from official Google model card
  - TGI CUDA bug with images: MEDIUM — CUDA bug is documented for text; image path untested in this project
  - Vertex AI multimodal path: MEDIUM — text confirmed working; image content list inferred from vLLM spec
  - Prompt format recommendations: MEDIUM — derived from Gemma 3 template spec + MedGemma VQA examples

---

## Sources

- [google/medgemma-4b-it on HuggingFace](https://huggingface.co/google/medgemma-4b-it)
- [google/medgemma-1.5-4b-it on HuggingFace](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma 1.5 Model Card — Google HAI-DEF](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [Vision Language Model Inference in TGI](https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/visual_language_models)
- [HuggingFace InferenceClient Reference](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client)
- [TGI Messages API Blog Post](https://huggingface.co/blog/tgi-messages-api)
- [Google-Health/medgemma GitHub](https://github.com/Google-Health/medgemma)
- [MedGemma Vertex AI Notebook (quick_start_with_model_garden.ipynb)](https://github.com/google-health/medgemma/blob/main/notebooks/quick_start_with_model_garden.ipynb)
- [MedGemma Technical Report (arxiv 2507.05201)](https://arxiv.org/abs/2507.05201)
- [Analyze Medical Images with MedGemma — Google Cloud Community](https://medium.com/google-cloud/analyze-medical-images-with-medgemma-a-technical-deep-dive-fee0be18e7e0)
- [Next generation medical image interpretation with MedGemma 1.5](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
- [LlamaEdge MedGemma-4b Quick Start](https://llamaedge.com/docs/ai-models/multimodal/medgemma-4b/)
- [DICOM to JPEG Converter for MedGemma](https://github.com/Violet-sword/DICOM-to-JPEG-Converter)
- [Integrating MedGemma into clinical workflows](https://cloud.google.com/blog/topics/developers-practitioners/integrating-medgemma-into-clinical-workflows-just-got-easier)
