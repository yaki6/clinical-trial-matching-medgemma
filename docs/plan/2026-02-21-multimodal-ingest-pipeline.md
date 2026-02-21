# Plan v2: Integrate Real-World NSCLC Patient Data into INGEST Pipeline

> **Revision**: v2 (2026-02-21). Supersedes v1 after live endpoint probing revealed critical API incompatibilities.

## Critical Findings from v1 Review

The original plan had **5 fatal flaws** discovered through live endpoint testing:

| # | Flaw | Evidence | Impact |
|---|------|----------|--------|
| **F1** | Plan assumes `chat_completion` API — endpoint returns **404** | `probe_medgemma_multimodal.py` test: `HfHubHTTPError: 404 Not Found for /v1/chat/completions` | Entire multimodal code path was wrong |
| **F2** | `text_generation()` is text-only — **cannot accept images** | HF InferenceClient signature: `text_generation(prompt: str, ...)` — no image param | Current adapter physically cannot send images |
| **F3** | Gemma chat template **breaks multimodal** — model refuses to see images | Test 3: model responds "I am a text-based AI and cannot interpret medical images" when template is used | Must NOT use `format_gemma_prompt()` for multimodal |
| **F4** | Plan says "demo never needs live image extraction" — **contradicts AC** | AC: "the real connection is required in the final demo" | Pre-cache-only strategy rejected |
| **F5** | `encode_image()` in skill client is **dead code** — `format_gemma_prompt()` strips images at line 103-105 | `text_parts = [p["text"] for p in content if p.get("type") == "text"]` | No existing multimodal code actually works |

### What Actually Works (Discovered via Probing)

The correct multimodal API is a **raw HTTP POST** with structured JSON:

```python
import requests, base64

resp = requests.post(
    ENDPOINT_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
    json={
        "inputs": {
            "text": "Describe findings in this medical image.",  # Plain text, NO Gemma template
            "image": base64.b64encode(open("image.png","rb").read()).decode(),
        },
        "parameters": {"max_new_tokens": 256},
    },
)
# Returns: [{"input_text": "...", "generated_text": "...prompt echo + response..."}]
```

**Confirmed working**: Test 2 returned real clinical findings ("heart appears enlarged... cardiomegaly... lung fields clear") in 19.7s for a 69KB PNG.

**Gotchas discovered**:
- Response echoes `input_text` inside `generated_text` — must strip prompt from output
- VQA pipeline causes **CUDA OOM** — don't use `visual_question_answering()`
- Multipart form upload returns 400 — only `application/json` supported
- Single image per request is safe; multi-image untested (5-image MPX1201 = OOM risk)

---

## Context

The MedGemma Impact Challenge demo (deadline Feb 24) needs realistic patient data to showcase the full E2E pipeline: INGEST → PRESCREEN → VALIDATE.

**AC**: The demo must use a **live MedGemma connection** to process medical images and create patient profiles. Cached results serve as fallback for demo reliability, not as the primary path.

**Goal**: Create a curated SoT data harness of 5 diverse NSCLC patients (3 multimodal + 2 text-only) that flows through the full pipeline, with live MedGemma 4B multimodal image extraction during the INGEST step.

---

## Step 0: Validate Endpoint Stability (GATE — do first)

**Already done** via `scripts/probe_medgemma_multimodal.py` + `probe_medgemma_multimodal_v2.py`.

Results:
| API Path | Status | Latency | Notes |
|----------|--------|---------|-------|
| Raw HTTP POST `{"inputs": {"text","image"}}` | **WORKS** | 17-20s | Correct multimodal path |
| `text_generation()` text-only | Works | 4.4s | Existing path, no images |
| `chat_completion` | 404 | — | NOT supported on this endpoint |
| VQA pipeline | CUDA OOM | — | Do not use |
| Gemma template + image | Model refuses | 31s | Template breaks multimodal |

**Gate passed**: Multimodal endpoint works. Proceed with implementation.

---

## Step 1: Curate 5 Representative NSCLC Patients

**Create**: `data/sot/ingest/nsclc_demo_harness.json`

**Reduced from 7 to 5** — fewer patients = faster demo runs, less endpoint stress, more time for polish.

**Dropped MPX1201** (5 images, OOM risk) and **6001149-1** (least differentiated text case).

| # | topic_id | Mode | Demographics | Diagnosis | Biomarkers | Why Selected |
|---|----------|------|-------------|-----------|------------|--------------|
| 1 | `mpx1016` | IMG (1 CT) | 43F, never-smoker | Adenocarcinoma + signet-ring | None | Young non-smoker; ambiguity = multimodal value-add. **Send 1 image only** (synpic34317, 69KB). |
| 2 | `mpx1875` | IMG (1 CT) | Smoker, cachectic | Large-cell/clear-cell NSCLC | None | Rare histology; single clear CT image (181KB). |
| 3 | `mpx1575` | IMG (1 CT) | 86M, comorbid | Squamous cell + IPF | None | Elderly + comorbidities; eligibility filtering showcase (212KB). |
| 4 | `6031552-1` | TEXT | 63M, 45 pack-yr | Adenocarcinoma, Stage IV | EGFR+ALK+PD-L1 | Richest biomarker profile; most trial-matchable. |
| 5 | `6000873-1` | TEXT | 64M | EGFR+ adenocarcinoma → pituitary mets | EGFR (L858R+T790M) | Dual EGFR mutations; targeted therapy matching. |

**Design decision: 1 image per patient max**. Multi-image requests are untested and risk CUDA OOM (VQA OOM'd at 69KB). Single-image is proven stable at 17-20s latency.

**Script**: `scripts/build_demo_harness.py` — reads `nsclc_dataset.jsonl` + `nsclc_trial_profiles.json`, emits harness.

**Harness schema**:
```json
{
  "version": "2.0",
  "created_at": "2026-02-21T...",
  "patients": [{
    "topic_id": "mpx1016",
    "source_dataset": "MedPix",
    "ingest_mode": "multimodal",
    "ehr_text": "...(raw history/findings from JSONL)...",
    "profile_text": "...(Gemini-structured profile)...",
    "key_facts": [{"field": "...", "value": "...", ...}],
    "image": {
      "file_path": "ingest_design/MedPix-2-0/images/MPX1016_synpic34317.png",
      "modality": "CT",
      "location": "Chest"
    },
    "medgemma_image_findings": null,
    "ambiguities": [...]
  }]
}
```

**Key schema change from v1**: `images` (array) → `image` (single object or null). Eliminates multi-image complexity.

---

## Step 2: Add `generate_with_image()` to MedGemmaAdapter

**Modify**: `src/trialmatch/models/medgemma.py`

Add a new method using **raw HTTP POST** (not `chat_completion`, not `text_generation`):

```python
async def generate_with_image(
    self, prompt: str, image_path: Path, max_tokens: int = 512
) -> ModelResponse:
    """Multimodal generation: image + text → findings.

    CRITICAL CONSTRAINTS (from endpoint probing):
    1. Must use raw HTTP POST to endpoint root, NOT chat_completion (404) or text_generation (text-only)
    2. Must NOT use Gemma chat template — template causes model to refuse image analysis
    3. Payload format: {"inputs": {"text": "...", "image": "<base64>"}, "parameters": {"max_new_tokens": N}}
    4. Response echoes input_text in generated_text — must strip prompt echo
    5. Single image only — multi-image causes CUDA OOM
    """
```

Implementation details:
- `base64.b64encode()` the PNG image
- POST to `self._endpoint_url` (root, not `/v1/chat/completions`)
- Use `requests` library (not `InferenceClient`) since no built-in method supports this format
- Parse response JSON: `response[0]["generated_text"]`
- Strip prompt echo: `generated_text.replace(input_text, "", 1).strip()`
- Same retry logic as `generate()` for 503 cold-start
- Token estimation: `len(prompt) // 4` for input (image tokens unknown), `len(output) // 4` for output

**Modify**: `src/trialmatch/models/base.py` — add optional `generate_with_image()` (singular, not plural) with default `NotImplementedError`.

**Do NOT modify** existing `generate()` or `format_gemma_prompt()` — multimodal is a completely separate code path.

---

## Step 3: Extend profile_adapter.py for Multimodal

**Modify**: `src/trialmatch/ingest/profile_adapter.py`

```python
def load_demo_harness(path: Path | str | None = None) -> list[dict]:
    """Load nsclc_demo_harness.json — 5 curated patients."""

def get_image_path(patient: dict) -> Path | None:
    """Return resolved image path for multimodal patients, None for text-only."""

def merge_image_findings(key_facts: dict, image_findings: dict) -> dict:
    """Merge MedGemma image extraction results into key_facts dict.
    Adds 'medgemma_imaging' key with findings, impression, modality."""

def adapt_harness_patient(patient: dict, image_findings: dict | None = None
) -> tuple[str, dict[str, Any]]:
    """Adapt harness patient for PRESCREEN.
    Returns (patient_note, key_facts).
    Reuses adapt_profile_for_prescreen() internally.
    If image_findings provided, merges into key_facts."""
```

---

## Step 4: Dual-Mode Demo Architecture (Live + Cached Fallback)

The demo runs in **live-first, cache-fallback** mode:

```
Demo Start
  → Warm up MedGemma endpoint (health_check)
  → If warm: LIVE MODE (real MedGemma image extraction)
  → If cold/error: CACHED MODE (pre-computed results, warn user)
```

### 4a: Pre-compute Cache (safety net)

**Create**: `scripts/extract_image_findings.py`

For each of the 3 image patients:
1. Load single image from `ingest_design/MedPix-2-0/images/`
2. Call `MedGemmaAdapter.generate_with_image()` with extraction prompt
3. Save to `data/sot/ingest/medgemma_image_cache/{topic_id}.json`

**Image extraction prompt** (plain text, NO Gemma template):
```
You are a radiologist analyzing a medical image for clinical trial eligibility screening.

Extract structured findings:
1. Primary abnormality location and characteristics
2. Tumor/mass description if present (size, margins, density)
3. Evidence of metastasis or invasion
4. Lymph node status if visible
5. Other clinically significant findings

Respond with JSON only:
{"findings": ["finding 1", "finding 2"], "tumor_characteristics": "description or null", "impression": "one-sentence clinical impression", "modality_observed": "CT/MRI/XR"}
```

Cache is committed to repo. Demo loads from cache if live extraction fails.

### 4b: Live Extraction in Demo

**Modify**: `demo/pages/1_Pipeline_Demo.py`

The INGEST step becomes interactive:
1. User selects patient from dropdown (5 curated patients)
2. For multimodal patients:
   - Display image via `st.image()`
   - Show "Extracting findings with MedGemma 4B..." spinner
   - Call `generate_with_image()` live
   - If succeeds: display live results, save to session
   - If fails (timeout/503/CUDA): load from cache, show "(cached)" badge
3. For text-only patients:
   - Display EHR text and key_facts as before
4. Merged key_facts (text + image findings) flow to PRESCREEN

---

## Step 5: Update Streamlit Demo UI

**Modify**: `demo/pages/1_Pipeline_Demo.py`
- Switch patient selector from `load_profiles()` to `load_demo_harness()` (5 patients)
- Add image display for multimodal patients
- Add live MedGemma extraction with `st.status()` progress
- Show extraction results in expandable section
- Fallback to cache on error with visual indicator

**Modify**: `demo/components/pipeline_viewer.py`
- Add `render_ingest_step_multimodal()`:
  - Shows image thumbnail + MedGemma findings
  - Shows text-derived key facts
  - Single column layout (not side-by-side — simpler for deadline)

**Modify**: `demo/components/patient_card.py`
- Add `st.image()` rendering for multimodal patients
- Show modality label ("CT Chest") under image

---

## Step 6: Unit Tests

**Create**: `tests/unit/test_demo_harness.py`

```python
def test_load_demo_harness():
    """5 patients, all have required fields."""

def test_harness_image_patients_have_image_field():
    """3 multimodal patients have image.file_path that resolves."""

def test_harness_text_patients_have_null_image():
    """2 text-only patients have image=null."""

def test_adapt_harness_patient_text_only():
    """Text patient → (patient_note, key_facts) with no image data."""

def test_adapt_harness_patient_with_image_findings():
    """Multimodal patient + findings → key_facts includes medgemma_imaging."""

def test_merge_image_findings():
    """Merge adds medgemma_imaging key without overwriting existing facts."""

def test_generate_with_image_mock():
    """MedGemmaAdapter.generate_with_image() with mocked HTTP response.
    Verifies: plain text prompt (no Gemma template), base64 image, prompt echo stripping."""

def test_generate_with_image_strips_prompt_echo():
    """Response that echoes input_text is correctly cleaned."""

def test_generate_with_image_retries_on_503():
    """Cold-start 503 triggers retry with backoff."""
```

---

## Files to Modify/Create

| Action | File | Purpose |
|--------|------|---------|
| Create | `scripts/build_demo_harness.py` | Assemble 5-patient harness |
| Create | `data/sot/ingest/nsclc_demo_harness.json` | SoT data harness (output) |
| Modify | `src/trialmatch/models/base.py` | Add optional `generate_with_image()` |
| Modify | `src/trialmatch/models/medgemma.py` | Add `generate_with_image()` via raw HTTP POST |
| Modify | `src/trialmatch/ingest/profile_adapter.py` | Add `load_demo_harness()` + `merge_image_findings()` + `adapt_harness_patient()` |
| Create | `scripts/extract_image_findings.py` | Pre-compute image cache |
| Create | `data/sot/ingest/medgemma_image_cache/` | Cached extraction results |
| Modify | `demo/pages/1_Pipeline_Demo.py` | Live multimodal INGEST UI |
| Modify | `demo/components/pipeline_viewer.py` | Image + findings display |
| Modify | `demo/components/patient_card.py` | Image rendering |
| Create | `tests/unit/test_demo_harness.py` | Unit tests |
| Keep | `scripts/probe_medgemma_multimodal*.py` | Endpoint validation evidence |

---

## Verification

1. `uv run pytest tests/unit/test_demo_harness.py -v` — all harness tests pass
2. `uv run pytest tests/unit/ -v` — no regressions in existing tests
3. `uv run python scripts/build_demo_harness.py` — generates harness with 5 patients
4. `uv run python scripts/extract_image_findings.py` — extracts for 3 multimodal patients (background, check logs)
5. `streamlit run demo/app.py` — verify:
   - Patient selector shows 5 curated patients
   - Multimodal patient: image displays, live MedGemma extraction works (~20s)
   - Text-only patient: shows key facts, no image section
   - PRESCREEN receives enriched key_facts and runs
6. Resilience test: pause HF endpoint → demo falls back to cache with "(cached)" indicator

---

## Execution Order & Dependencies

```
Step 0 [DONE] ──→ Step 1 ──→ Step 2 ──→ Step 3 ──→ Step 4a (background)
                                                  ↘ Step 5 ──→ Step 6
                                                  ↗ Step 4b (needs Step 2+5)
```

- **Steps 1-3**: Sequential, blocking (data → adapter → integration)
- **Step 4a + Step 5**: Can run in parallel after Step 3
- **Step 4b**: Needs both Step 2 (adapter) and Step 5 (UI) complete
- **Step 6**: Final lock-down, runs after everything

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CUDA OOM on image processing | Low (single 69-212KB images tested OK) | Use smallest image per patient; cap at 1 image; if OOM, retry once then fall back to cache |
| Endpoint cold-start during demo | Medium (60s budget) | Pre-warm endpoint 2min before demo; live health_check indicator in UI |
| CUDA CUBLAS crash (known TGI bug) | Medium | Use raw HTTP POST (not text_generation); max_tokens=256 (not 512); pre-cached fallback |
| Prompt echo in response | Certain | `strip_prompt_echo()` utility in adapter — already confirmed this happens |
| Gemma template applied accidentally | Prevented by design | `generate_with_image()` is a separate method that does NOT call `format_gemma_prompt()` |
| Multi-image request causes OOM | Prevented by design | Schema enforces single `image` (not `images` array) |
| MedGemma image findings low quality | Medium | Honest results = writeup material; cache allows hand-review before demo |
| `requests` library not installed | Low | Check `pyproject.toml`; add if missing |

---

## Appendix: Endpoint Probe Evidence

### Working Format (Test 2, 19.7s)
```json
POST https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud
Content-Type: application/json
Authorization: Bearer $HF_TOKEN

{"inputs": {"text": "Describe the key clinical findings...", "image": "<base64>"}, "parameters": {"max_new_tokens": 256}}

→ 200 OK
[{"input_text": "Describe the key clinical findings...",
  "generated_text": "Describe the key clinical findings...\nThe image shows a chest X-ray. The key clinical findings are:\n1. **Heart Size:** The heart appears enlarged...cardiomegaly.\n2. **Lung Fields:** The lung fields are clear bilaterally..."}]
```

### Gemma Template Breaks Multimodal (Test 3, 31.3s)
```
"generated_text": "...I am unable to provide a description of the clinical findings in the medical image you have provided. I am a text-based AI and cannot interpret medical images."
```

### API 404 Confirmation (Test 1)
```
HfHubHTTPError: Client error '404 Not Found' for url '.../v1/chat/completions'
```

### CUDA OOM on VQA (Test 5)
```
CUDA out of memory. Tried to allocate 16.51 GiB. GPU 0 has a total capacity of 44.39 GiB of which 11.11 GiB is free.
```
