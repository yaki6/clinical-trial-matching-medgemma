# Plan: Integrate Real-World NSCLC Patient Data into INGEST Pipeline

## Context

The MedGemma Impact Challenge demo (deadline Feb 24) needs realistic patient data to showcase the full E2E pipeline: INGEST -> PRESCREEN -> VALIDATE. Currently, the system uses pre-generated `nsclc_trial_profiles.json` (37 profiles from Gemini 2.5 Pro) but:
1. INGEST is a thin adapter — no actual MedGemma extraction happens
2. Medical images in `ingest_design/MedPix-2-0/images/` are unused
3. MedGemma 4B's multimodal capability (HAE = Health AI Enterprise) is not demonstrated
4. Judges score on "Effective use of HAI-DEF models" — showing MedGemma extracting from images is a strong differentiator

**Goal**: Create a curated SoT data harness of 7 diverse NSCLC patients (4 multimodal + 3 text-only) that flows through the full pipeline, showcasing MedGemma 4B multimodal image extraction.

---

## Step 1: Curate 7 Representative NSCLC Patients

**Create**: `data/sot/ingest/nsclc_demo_harness.json`

Select from the 37 cases in `ingest_design/nsclc-dataset/nsclc_dataset.jsonl` + `nsclc_trial_profiles.json`:

| # | topic_id | Mode | Demographics | Diagnosis | Biomarkers | Why Selected |
|---|----------|------|-------------|-----------|------------|--------------|
| 1 | `mpx1016` | IMG (2 CT) | 43F, never-smoker | Adenocarcinoma + signet-ring | None | Young non-smoker; image shows mass NOT in EHR text (ambiguity = multimodal value-add) |
| 2 | `mpx1201` | IMG (5 CT) | 47M, heavy smoker | Metastatic adenocarcinoma | None | Multi-site mets (femur, lung, liver); most images; family hx of cancer |
| 3 | `mpx1875` | IMG (1 CT) | Smoker, cachectic | Large-cell/clear-cell NSCLC | None | Rare histology variant; single clear CT image |
| 4 | `mpx1575` | IMG (1 CT) | 86M, comorbid | Squamous cell + IPF | None | Elderly + complex comorbidities (AFib, dementia, hypothyroidism, IPF); eligibility filtering showcase |
| 5 | `6031552-1` | TEXT | 63M, 45 pack-yr | Adenocarcinoma, Stage IV | EGFR+ALK+PD-L1 | **Richest biomarker profile**: triple-positive; most trial-matchable |
| 6 | `6000873-1` | TEXT | 64M | EGFR+ adenocarcinoma → pituitary mets | EGFR (L858R+T790M) | Dual EGFR mutations + resistance mutation; on erlotinib; targeted therapy trial matching |
| 7 | `6001149-1` | TEXT | 56F | Unresectable NSCLC, pT3N0M0 | PD-L1+ | On nivolumab immunotherapy; 2nd/3rd-line trial matching; adverse event (vitiligo) |

**Diversity matrix**: 4 histology types, 3 biomarker combos, age 43-86, F/M mix, never/heavy/ex-smokers, Stage I to IV, 4 multimodal + 3 text-only.

**Script**: `scripts/build_demo_harness.py` — reads both JSONL + profiles JSON, assembles the harness.

**Harness schema** (extends nsclc_trial_profiles.json):
```json
{
  "version": "1.0",
  "patients": [{
    "topic_id": "mpx1016",
    "source_dataset": "MedPix",
    "ingest_mode": "multimodal",
    "ehr_text": "...(raw history/findings from JSONL)...",
    "profile_text": "...(Gemini-structured profile)...",
    "key_facts": [{"field": "...", "value": "...", ...}],
    "images": [{"image_id": "...", "file_path": "ingest_design/MedPix-2-0/images/MPX1016_synpic34317.png", "modality": "CT w/contrast", "plane": "Axial", "location": "Chest", "caption": "..."}],
    "medgemma_image_findings": null,
    "ambiguities": [...]
  }]
}
```

---

## Step 2: Add `generate_with_images()` to MedGemmaAdapter

**Modify**: `src/trialmatch/models/medgemma.py`

Add multimodal method using HF `chat_completion` API (OpenAI-compatible format):

```python
async def generate_with_images(
    self, prompt: str, image_paths: list[Path], max_tokens: int = 512
) -> ModelResponse:
```

- Base64-encode PNG images as `data:image/png;base64,...` data URIs
- Send via `chat_completion` with `[{"type": "image_url", ...}, {"type": "text", ...}]` content
- Same retry logic as existing `generate()`
- 512 token limit is sufficient for structured image findings JSON (~200-350 tokens)

**Modify**: `src/trialmatch/models/base.py` — add optional `generate_with_images()` with default `NotImplementedError`.

---

## Step 3: Extend profile_adapter.py for Multimodal

**Modify**: `src/trialmatch/ingest/profile_adapter.py`

Add two functions:

```python
def load_demo_harness(path=None) -> list[dict]:
    """Load nsclc_demo_harness.json — 7 curated patients."""

def adapt_multimodal_profile(patient: dict, medgemma_image_findings: dict | None = None
) -> tuple[str, dict[str, Any], list[Path]]:
    """Adapt harness patient for PRESCREEN. Returns (patient_note, key_facts, image_paths).
    Merges MedGemma image findings into key_facts if provided."""
```

Reuses existing `adapt_profile_for_prescreen()` flattening logic internally.

---

## Step 4: Create Image Extraction Script (Pre-computation)

**Create**: `scripts/extract_image_findings.py`

For each of the 4 image patients:
1. Load images from `ingest_design/MedPix-2-0/images/`
2. Call `MedGemmaAdapter.generate_with_images()` with medical imaging extraction prompt
3. Parse structured JSON response (findings, tumor characteristics, modality)
4. Save to `data/sot/ingest/medgemma_image_cache/{topic_id}.json`
5. Update harness JSON `medgemma_image_findings` field

**Image extraction prompt** (designed for <512 token output):
```
Analyze the medical image(s). Extract structured findings for clinical trial matching:
1. Tumor location, size, characteristics
2. Metastasis/invasion evidence
3. Lymph node involvement
4. Measurable disease (RECIST)
Respond with JSON: {"findings": [...], "tumor_characteristics": {...}, "modality_observed": "..."}
```

Results are **cached and committed** to repo — demo never needs live image extraction.

---

## Step 5: Update Streamlit Demo for Multimodal

**Modify**: `demo/pages/1_Pipeline_Demo.py`
- Switch patient selector from `load_profiles()` to `load_demo_harness()` (7 curated patients)
- Add image thumbnails via `st.image()` for multimodal cases
- Add "MedGemma Image Analysis" section showing extracted findings (from cache)

**Modify**: `demo/components/pipeline_viewer.py`
- Add `render_ingest_step_multimodal()` with side-by-side layout:
  - Left: Text-derived key facts (from Gemini profile)
  - Right: Image-derived findings (from MedGemma 4B)
  - Highlight discrepancies (e.g., mpx1016: image shows mass not mentioned in EHR text)

**Modify**: `demo/components/patient_card.py`
- Add medical image rendering with modality/plane captions

---

## Step 6: Unit Tests

**Create**: `tests/unit/test_demo_harness.py`

- `test_load_demo_harness()` — 7 patients, all have required fields
- `test_adapt_multimodal_profile_with_images()` — image paths resolve, key_facts includes medgemma_imaging
- `test_adapt_multimodal_profile_text_only()` — text-only patients get empty image_paths
- `test_merge_image_findings()` — MedGemma findings merge correctly
- `test_roundtrip_multimodal_to_format_key_facts()` — enriched key_facts work with `_format_key_facts()`
- `test_generate_with_images_mock()` — MedGemmaAdapter multimodal method with mocked HF client

---

## Files to Modify/Create

| Action | File | Purpose |
|--------|------|---------|
| Create | `scripts/build_demo_harness.py` | Assemble 7-patient harness from JSONL + profiles |
| Create | `data/sot/ingest/nsclc_demo_harness.json` | SoT data harness (output of above) |
| Modify | `src/trialmatch/models/base.py` | Add optional `generate_with_images()` to base |
| Modify | `src/trialmatch/models/medgemma.py` | Add `generate_with_images()` multimodal method |
| Modify | `src/trialmatch/ingest/profile_adapter.py` | Add `load_demo_harness()` + `adapt_multimodal_profile()` |
| Create | `scripts/extract_image_findings.py` | Pre-compute MedGemma image extraction |
| Create | `data/sot/ingest/medgemma_image_cache/` | Cached image extraction results |
| Modify | `demo/pages/1_Pipeline_Demo.py` | Multimodal INGEST UI |
| Modify | `demo/components/pipeline_viewer.py` | Side-by-side text vs image findings |
| Modify | `demo/components/patient_card.py` | Image rendering |
| Create | `tests/unit/test_demo_harness.py` | Unit tests for harness + multimodal adapter |

---

## Verification

1. `uv run pytest tests/unit/test_demo_harness.py -v` — all harness tests pass
2. `uv run pytest tests/unit/test_profile_adapter.py -v` — existing adapter tests still pass (no regression)
3. `uv run python scripts/build_demo_harness.py` — generates harness JSON with 7 patients
4. `uv run python scripts/extract_image_findings.py` — extracts image findings for 4 multimodal patients (run in background, check logs)
5. `streamlit run demo/app.py` — verify:
   - Patient selector shows 7 curated patients
   - Multimodal patients display images + MedGemma findings
   - Text-only patients show "No images" gracefully
   - PRESCREEN receives enriched key_facts and runs successfully
6. Playwright QA: full pipeline run for 1 multimodal + 1 text-only patient

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MedGemma 4B CUDA crash during image extraction | Pre-compute and cache; use `chat_completion` API (not `text_generation`); max_tokens=512 |
| HF endpoint cold start | All image extraction is pre-cached; demo runs in cached mode |
| Image base64 too large for context | MedPix PNGs are 68-158KB; HF handles image tokenization server-side |
| MedGemma image quality poor | Honest result = writeup material; mpx1016 ambiguity is pre-validated |
| Deadline pressure | Steps 1-3 are blocking; Steps 4-5 can run in parallel; Step 6 tests lock it down |
