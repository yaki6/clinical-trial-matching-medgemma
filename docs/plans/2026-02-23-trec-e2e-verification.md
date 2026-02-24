# Plan: TREC-2022 Pipeline Verification (Simplified)

## Goal

Validate the **logic** of PRESCREEN and VALIDATE modules against TREC 2022 ground truth (patient trec-20226, 647 trial judgments).

- **PRESCREEN**: Can the search strategy find relevant trials via CT.gov API? Measure recall against gold eligible trial list.
- **VALIDATE**: Can criterion-level evaluation produce correct trial-level judgments? Sample 10 trials, compare to gold qrels.

## Key Constraints

- TREC data is a 2022 snapshot; CT.gov API returns 2026 data (trials may have changed status/criteria)
- `corpus.jsonl` is NOT in the ground truth file — trial criteria must be fetched via CT.gov API
- TREC qrels use 3-level scoring (0/1/2) vs VALIDATE's criterion-level verdicts
- Budget: 10 trials for VALIDATE (~$1-2 LLM cost), PRESCREEN is API-only (free)

## Label Mapping (方案 A)

```python
VERDICT_TO_QREL = {
    "ELIGIBLE": 2,
    "EXCLUDED": 1,       # exclusion triggered
    "UNCERTAIN": 1,      # cannot determine → partial
    "NOT_RELEVANT": 0,
}
```

---

## Step 1: Extract Ground Truth Data

**What:** Parse `data/trec2022_ground_truth.md` → standalone files.

**Script:** `scripts/extract_trec_ground_truth.py`

**Output:**
- `data/trec2022_ground_truth/patient.jsonl` — line 1773
- `data/trec2022_ground_truth/qrels.tsv` — lines 1780-2428 (header + 647 data rows)
- `data/trec2022_ground_truth/expected_outcomes.json` — lines 182-1769

**Validation:** assert 647 qrels entries, 1 patient record, 118+20+509 distribution.

---

## Step 2: PRESCREEN Verification

**What:** Run PRESCREEN for the TREC patient, measure recall against gold eligible trials (118 NCT IDs).

**Script:** `scripts/run_trec_prescreen.py`

**Logic:**
1. Load patient from `patient.jsonl`, inline-adapt to (patient_note, key_facts):
   - `patient_note` = `record["text"]`
   - `key_facts` = flat dict from `structured_profile` (age, sex, diagnosis, comorbidities, smoking)
2. Load qrels → `{nct_id: score}`, extract eligible set (score=2, 118 trials)
3. Run `run_prescreen_agent()` **twice**:
   - Run A: `status=None` (all statuses) — theoretical recall ceiling
   - Run B: `status=["RECRUITING"]` — real-world behavior
4. Post-filter: intersect returned candidates with qrels NCT IDs
5. Compute metrics:
   - **Recall@all (eligible):** fraction of 118 eligible trials found
   - **Recall@20 (eligible):** eligible trials in top-20
   - **Precision (in-qrels):** fraction of returned candidates in 647 judged set
6. **Search term tracing:** Log all `SearchAnchors` generated and which gold NCT IDs each search term retrieves. If recall is low, this diagnoses WHY.
7. Save results to `runs/trec_prescreen_<timestamp>/`

**Important:** PRESCREEN defaults to RECRUITING filter. Running with `status=None` separates "search strategy is bad" from "trials expired."

---

## Step 3: Fetch Trial Criteria for VALIDATE Sample

**What:** Sample 10 trials (stratified: ~3-4 per qrel level), fetch criteria from CT.gov API.

**Script:** `scripts/fetch_trec_trial_criteria.py`

**Logic:**
1. Load qrels, stratified sample 10 trials (seed=42)
2. For each NCT ID: `CTGovClient.get_details(nct_id)`
3. Extract `eligibilityModule.eligibilityCriteria`, parse into inclusion/exclusion lists
4. Cache to `data/trec2022_ground_truth/trial_criteria_cache.json`
5. Log `lastUpdatePostDate` for each trial — flag any updated after 2022 (criteria drift)
6. Log 404s (trial removed from CT.gov)

**Output format:**
```json
{
  "NCT00005636": {
    "brief_title": "...",
    "inclusion_criteria": ["criterion 1", ...],
    "exclusion_criteria": ["criterion 1", ...],
    "status": "COMPLETED",
    "last_update": "2023-04-15",
    "criteria_drift_warning": false,
    "fetched_at": "2026-02-23T..."
  }
}
```

---

## Step 4: VALIDATE Verification

**What:** Run criterion-level VALIDATE on 10 sampled trials, aggregate to trial-level, compare to gold qrels.

**Script:** `scripts/run_trec_validate.py`

**Logic:**
1. Load patient (inline adapt, same as Step 2)
2. Load cached trial criteria (Step 3)
3. For each of 10 trials:
   a. Parse criteria into individual criterion strings
   b. For each criterion: `evaluate_criterion_two_stage(patient_note, criterion_text, criterion_type)`
   c. Collect `CriterionResult` list
4. Aggregate criterion verdicts → `TrialVerdict` via `aggregate_to_trial_verdict()`
5. Map `TrialVerdict` → qrel score via 方案 A mapping
6. Compare predicted vs gold qrel scores
7. Compute: accuracy, macro F1, confusion matrix (3x3) using sklearn directly
8. **Criterion-level debugging:** For wrong trial verdicts, save which specific criteria were misjudged and the model's reasoning
9. Save to `runs/trec_validate_<timestamp>/`:
   - `results.json` — criterion-level results per trial
   - `trial_verdicts.json` — trial-level verdicts + gold comparison
   - `metrics.json` — accuracy, F1, confusion matrix

**Model config:** Gemini 3 Pro for both stages (no Vertex endpoint needed).

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/extract_trec_ground_truth.py` | Parse .md → standalone data files |
| `scripts/run_trec_prescreen.py` | PRESCREEN recall verification (2 runs) |
| `scripts/fetch_trec_trial_criteria.py` | Fetch + cache 10 trial criteria |
| `scripts/run_trec_validate.py` | VALIDATE trial-level verification |

**4 scripts. No new modules. No new Pydantic models. No unit tests for loaders.**

Data loading (~20 lines) and patient adaptation (~5 lines) are inlined in each script. Label mapping is a 4-line dict.

---

## Execution Order

```
Step 1 (extract data) → Step 2 (PRESCREEN, independent)
                       → Step 3 (fetch criteria) → Step 4 (VALIDATE)
```

Steps 2 and 3 can run in parallel after Step 1.
Step 1 is pure file parsing. Step 2 is CT.gov API only. Step 3 is 10 API calls. Step 4 is LLM calls (~$1-2).

---

## Success Criteria

| Module | Metric | Target |
|--------|--------|--------|
| PRESCREEN (status=None) | Recall@all on 118 eligible trials | Directional (expect low due to API snapshot) |
| PRESCREEN | Search terms cover patient condition | All generated anchors logged |
| VALIDATE | Trial-level accuracy (10 trials) | ≥ 70% |
| VALIDATE | Per-trial criterion debugging | Wrong verdicts traced to specific criteria |

---

## What This Plan Does NOT Do (By Design)

- No reusable `trec_loader.py` module — inline loading is fine for validation
- No `TRECTrialAnnotation` Pydantic model — just dicts
- No unit tests for loaders/adapters — the scripts ARE the tests
- No E2E report generator — metrics from Steps 2+4 are the report
- No YAML config file — hardcoded paths in scripts
- No formal adapter function — 5-line inline dict mapping
