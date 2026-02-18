# SoT Harness Design — Integration Test Fixtures for INGEST / PRESCREEN / VALIDATE

> **Date:** 2026-02-19
> **Status:** Approved
> **Purpose:** Define 3 source-of-truth (SoT) patient scenarios with real CT.gov API data to serve as integration test harness for coding agents building production-ready pipeline code.

---

## 1. Overview

### Goal

Create 3 end-to-end patient scenarios that flow through all 3 pipeline components (INGEST → PRESCREEN → VALIDATE), providing gold-standard input/output at each stage. Coding agents use these to verify their implementation against known-correct data with real-world API response formats.

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Fictional patients + real trials** | Patients hand-written by domain expert; trial data recorded from CT.gov API to ensure schema fidelity |
| **Component isolation** | PRESCREEN and VALIDATE receive gold INGEST output as input (not model output), per PRD §6 |
| **Single real trial anchor** | All 3 scenarios evaluate against the same real trial (NCT05456256) — different patient profiles produce different verdicts |
| **TrialGPT label alignment** | SoT records TrialGPT 6-label taxonomy alongside our MET/NOT_MET/UNKNOWN for future comparison with 87.3% benchmark |
| **Criteria parsing tested** | SoT includes both raw `eligibility_criteria` blob AND pre-split inclusion/exclusion blocks to verify parsing logic |
| **Precise API query mapping** | PRESCREEN SoT includes exact CT.gov API v2 query parameters (`query.cond`, `query.term=AREA[...]`) |

### Key Findings from Research

1. **TrialGPT published ground truth exists** — `ncbi/TrialGPT-Criterion-Annotations` on HuggingFace (1,024 rows) with `expert_eligibility` labels. Usable for Phase 0/1 but NOT for this harness (harness uses fictional patients).
2. **CT.gov API returns a single `eligibility_criteria` text blob** — must be parsed by splitting on `"Inclusion Criteria:"` / `"Exclusion Criteria:"` headers, then `\n\n` paragraph boundaries.
3. **eGFR false positive risk** — searching `EGFR` in eligibility criteria also matches `eGFR` (renal function). PRESCREEN must use exact phrases like `"EGFR mutation"` or `"EGFR L858R"`.
4. **VALIDATE is a single LLM call** — inclusion + exclusion blocks sent together, model returns one `Verdict` (ELIGIBLE/EXCLUDED/NOT_RELEVANT). Per-block MET/NOT_MET is annotator reasoning documentation only.
5. **TrialGPT label taxonomy** — 6 labels (included, not included, excluded, not excluded, not enough information, not applicable). Our 3-label mapping: included→MET, not included→NOT_MET, not enough info→UNKNOWN, not applicable→UNKNOWN.

---

## 2. Scenario Design

### 2.1 Anchor Trial: NCT05456256

**"Phase II Trial of LP-300 in Combination With Carboplatin and Pemetrexed in Never Smoker Patients With Relapsed Advanced Primary Adenocarcinoma of the Lung After Treatment With Tyrosine Kinase Inhibitors (The HARMONIC Study)"**

Selected because:
- Real recruiting NSCLC Phase II trial with detailed, well-structured eligibility criteria
- 14 inclusion criteria + 16 exclusion criteria covering demographics, molecular markers, prior treatment, labs, comorbidities
- Rich enough to differentiate ELIGIBLE vs EXCLUDED vs NOT_RELEVANT patient profiles

### 2.2 Three Scenarios

| Scenario | Patient Profile | Key Differentiators | Expected Verdict |
|----------|----------------|--------------------|-----------------:|
| **1: ELIGIBLE** | NSCLC adenocarcinoma, EGFR L858R, never-smoker, ECOG 1, prior TKI with progression, labs normal | Matches all inclusion; no exclusion triggered | `ELIGIBLE` (qrel=2) |
| **2: EXCLUDED** | NSCLC adenocarcinoma, EGFR L858R, never-smoker, but progressed on prior chemotherapy + immunotherapy | Meets inclusion #1-3 but triggers exclusion #4 (prior chemo progression) | `EXCLUDED` (qrel=1) |
| **3: NOT_RELEVANT** | NSCLC but KRAS G12C (not an actionable EGFR/ALK/MET mutation), current smoker | Fails inclusion #1 (no actionable genomic alteration) and #3 (not a never-smoker) | `NOT_RELEVANT` (qrel=0) |

All 3 patients are NSCLC — this keeps PRESCREEN search logic consistent (all search for NSCLC trials) while producing different VALIDATE outcomes through patient-specific factors.

---

## 3. File Structure

```
data/sot/harness/
├── scenario_1_eligible/
│   ├── 01_ingest_input.json              # Patient EHR text (human expert)
│   ├── 02_ingest_gold_output.json        # Gold PatientProfileText + KeyFacts
│   ├── 03_prescreen_gold_output.json     # Gold SearchAnchors
│   ├── 04_api_query_spec.json            # Exact CT.gov API v2 query parameters
│   ├── 05_api_search_response.json       # Recorded CT.gov search results
│   ├── 06_trial_detail_ctgov.json        # Recorded CT.gov trial detail (raw)
│   ├── 07_trial_detail_parsed.json       # Parsed trial with split inclusion/exclusion blocks
│   ├── 08_validate_gold_output.json      # Gold Verdict + reasoning + annotation
│   ├── 09_mock_model_responses/          # Canned LLM responses for parsing tests
│   │   ├── ingest_medgemma.json
│   │   ├── ingest_gemini.json
│   │   ├── validate_medgemma.json
│   │   └── validate_gemini.json
│   └── scenario_manifest.json            # Metadata + expected metrics
├── scenario_2_excluded/
│   └── ... (same structure)
├── scenario_3_not_relevant/
│   └── ... (same structure)
└── expected_aggregate_metrics.json        # Aggregate metrics across all 3 scenarios
```

---

## 4. Component SoT Schemas

### 4.1 INGEST

#### Input (`01_ingest_input.json`)

```json
{
  "topic_id": "sot_001",
  "text": "<EHR clinical narrative, 100-200 words>",
  "medical_images": [],
  "metadata": {
    "source": "human_expert",
    "author": "<annotator_id>",
    "date": "2026-02-XX"
  }
}
```

- `topic_id`: prefixed `sot_` to distinguish from TREC IDs
- `text`: free-text clinical narrative in TREC topic style, compatible with `Topic(topic_id, text)` Pydantic model
- `medical_images`: list of image file paths (optional, MedGemma multimodal support)

#### Gold Output (`02_ingest_gold_output.json`)

```json
{
  "topic_id": "sot_001",
  "profile_text": "<structured summary following INGEST template headings>",
  "key_facts": [
    {
      "field": "primary_diagnosis",
      "value": "Stage IV NSCLC, adenocarcinoma",
      "evidence_span": "<exact substring from input text>",
      "required": true,
      "notes": null
    },
    {
      "field": "biomarkers",
      "value": ["EGFR exon 21 L858R mutation"],
      "evidence_span": "EGFR exon 21 L858R mutation positive",
      "required": true,
      "notes": null
    },
    {
      "field": "demographics",
      "value": {"age": "58", "sex": "female"},
      "evidence_span": "58-year-old female",
      "required": true,
      "notes": null
    },
    {
      "field": "prior_therapies",
      "value": ["osimertinib (first-line, 14 months, progressed)"],
      "evidence_span": "first-line osimertinib for 14 months, now with disease progression",
      "required": true,
      "notes": null
    },
    {
      "field": "ecog_kps",
      "value": "ECOG 1",
      "evidence_span": "ECOG performance status 1",
      "required": true,
      "notes": null
    },
    {
      "field": "labs_organ_function",
      "value": [
        {"test": "ANC", "value": "2.1", "unit": "x10^9/L"},
        {"test": "Hemoglobin", "value": "11.2", "unit": "g/dL"},
        {"test": "Platelets", "value": "180", "unit": "x10^9/L"},
        {"test": "Creatinine", "value": "0.9", "unit": "mg/dL"},
        {"test": "Total bilirubin", "value": "0.8", "unit": "mg/dL"}
      ],
      "evidence_span": "Labs: ANC 2.1, Hgb 11.2, Plt 180, creatinine 0.9, total bilirubin 0.8",
      "required": true,
      "notes": null
    },
    {
      "field": "comorbidities",
      "value": [],
      "evidence_span": null,
      "required": false,
      "notes": null
    },
    {
      "field": "missing_info",
      "value": ["smoking pack-years detail", "CNS imaging date"],
      "evidence_span": null,
      "required": false,
      "notes": null
    }
  ],
  "ambiguities": []
}
```

**Required fields** (per PRD §6.1): `primary_diagnosis`, `stage_grade`, `biomarkers`, `demographics`, `prior_therapies`, `comorbidities`, `labs_organ_function`, `ecog_kps`, `missing_info`

### 4.2 PRESCREEN

#### Gold Output (`03_prescreen_gold_output.json`)

```json
{
  "topic_id": "sot_001",
  "search_anchors": {
    "conditions": [
      {
        "term": "non-small cell lung cancer",
        "priority": "MUST",
        "synonyms": ["NSCLC", "lung adenocarcinoma"]
      }
    ],
    "biomarkers": [
      {
        "term": "EGFR L858R",
        "priority": "MUST",
        "normalized_form": "EGFR exon 21 L858R",
        "notes": "Use exact phrase 'EGFR L858R' or 'EGFR mutation' to avoid eGFR false positives"
      }
    ],
    "interventions": [
      {
        "term": "osimertinib",
        "priority": "SHOULD",
        "notes": "Prior therapy — search for post-progression options"
      }
    ],
    "constraints": {
      "age": {"value": "58"},
      "sex": {"value": "female"},
      "phase": {"value": ["PHASE2", "PHASE3"]},
      "status": {"value": ["RECRUITING"]}
    },
    "negative_anchors": [
      {"term": "first-line", "reason": "Patient progressed on prior TKI — first-line trials would exclude"},
      {"term": "treatment-naive", "reason": "Patient has prior osimertinib therapy"}
    ]
  }
}
```

#### API Query Spec (`04_api_query_spec.json`)

Defines exact CT.gov API v2 query parameters for each search strategy:

```json
{
  "topic_id": "sot_001",
  "queries": [
    {
      "query_id": "q1_condition_biomarker",
      "description": "Primary: condition + biomarker in eligibility criteria",
      "ct_gov_api_v2_params": {
        "query.cond": "non-small cell lung cancer",
        "query.term": "AREA[EligibilityCriteria]\"EGFR L858R\"",
        "filter.overallStatus": "RECRUITING",
        "pageSize": 50,
        "countTotal": true
      }
    },
    {
      "query_id": "q2_condition_intervention",
      "description": "Secondary: condition + prior intervention",
      "ct_gov_api_v2_params": {
        "query.cond": "NSCLC",
        "query.intr": "osimertinib",
        "filter.overallStatus": "RECRUITING",
        "pageSize": 50
      }
    },
    {
      "query_id": "q3_broad_biomarker",
      "description": "Broad: biomarker synonym expansion",
      "ct_gov_api_v2_params": {
        "query.cond": "lung cancer",
        "query.term": "AREA[EligibilityCriteria](\"EGFR L858R\" OR \"EGFR exon 21\" OR \"EGFR activating mutation\")",
        "filter.overallStatus": "RECRUITING",
        "pageSize": 50
      }
    }
  ],
  "expected_nct_ids_in_results": ["NCT05456256"],
  "notes": {
    "egfr_edge_case": "AREA[EligibilityCriteria] with exact phrase avoids eGFR (renal) false positives",
    "negative_anchors": "CT.gov API v2 supports NOT in Essie syntax but has no dedicated exclude parameter",
    "phase_filter": "Phase filtering via AREA[Phase] in advanced query syntax, not a direct filter parameter"
  }
}
```

#### Recorded API Response (`05_api_search_response.json`)

Recorded from real CT.gov API call. Structure:

```json
{
  "_recording_metadata": {
    "recorded_at": "2026-02-19",
    "api_version": "v2",
    "tool": "mcp__claude_ai_Clinical_Trials__search_trials"
  },
  "query_params": {
    "condition": "non-small cell lung cancer",
    "status": ["RECRUITING"],
    "phase": ["PHASE2", "PHASE3"]
  },
  "response": {
    "count": 3,
    "items": [
      {
        "nct_id": "NCT05456256",
        "title": "Phase II Trial of LP-300...",
        "conditions": ["Adenocarcinoma of Lung", "Carcinoma, Non-Small-Cell Lung"],
        "interventions": ["LP-300", "Pemetrexed", "Carboplatin"],
        "enrollment": 90,
        "phase": ["PHASE2"],
        "sponsor": "Lantern Pharma Inc.",
        "status": "RECRUITING"
      }
    ]
  }
}
```

### 4.3 Data Layer: Criteria Parsing

#### Raw Trial Detail (`06_trial_detail_ctgov.json`)

Complete `get_trial_details` response for NCT05456256 — recorded verbatim from CT.gov API. Contains:
- `eligibility_criteria`: single text blob with both "Inclusion Criteria:" and "Exclusion Criteria:" sections
- All other trial metadata (conditions, interventions, outcomes, locations, etc.)

#### Parsed Trial Detail (`07_trial_detail_parsed.json`)

Verifies the criteria parsing step:

```json
{
  "nct_id": "NCT05456256",
  "brief_title": "A Study of LP-300 With Carboplatin and Pemetrexed in Never Smokers With Advanced Lung Adenocarcinoma",
  "data_source": "api_current",
  "last_update_date": "2025-XX-XX",
  "eligibility_criteria_raw": "<full blob from API>",
  "inclusion_criteria": "<extracted inclusion section>",
  "exclusion_criteria": "<extracted exclusion section>",
  "inclusion_criteria_count": 14,
  "exclusion_criteria_count": 16,
  "minimum_age": "18 Years",
  "maximum_age": null,
  "sex": "ALL",
  "parsing_method": "split on 'Inclusion Criteria:' and 'Exclusion Criteria:' headers, then paragraph-level by double newline",
  "parsing_notes": "Follows TrialGPT parse_criteria() convention: filter headers, filter entries <5 chars"
}
```

### 4.4 VALIDATE

#### Gold Output (`08_validate_gold_output.json`)

Top-level `verdict` matches the pipeline's actual output schema (single `Verdict` enum). The `annotation` block is human reasoning documentation for qualitative analysis.

```json
{
  "topic_id": "sot_001",
  "nct_id": "NCT05456256",
  "verdict": "ELIGIBLE",
  "confidence": 0.95,
  "reasoning": "Patient meets all 14 inclusion criteria and triggers no exclusion criteria. Key matches: (1) Stage IV NSCLC adenocarcinoma with EGFR L858R actionable mutation; (3) never smoker; (4) prior TKI with progression; (6) ECOG 1; (10) labs within range. No exclusion factors: adenocarcinoma histology, primary lung origin, no prior investigational agents beyond TKI, no prior chemo/immuno progression.",
  "annotation": {
    "inclusion_assessment": {
      "overall": "MET",
      "key_criterion_matches": [
        {"criterion": 1, "status": "MET", "note": "EGFR L858R is an actionable genomic alteration"},
        {"criterion": 3, "status": "MET", "note": "Never smoker"},
        {"criterion": 4, "status": "MET", "note": "Prior TKI (osimertinib) with progression"},
        {"criterion": 6, "status": "MET", "note": "ECOG 1"},
        {"criterion": 7, "status": "MET", "note": "Age 58 >= 18"},
        {"criterion": 10, "status": "MET", "note": "ANC, Hgb, Plt, creatinine, bilirubin all within range"}
      ],
      "blocking_factors": []
    },
    "exclusion_assessment": {
      "overall": "NOT_MET",
      "triggered_exclusions": [],
      "close_calls": []
    },
    "difficulty": "EASY",
    "difficulty_reason": "Patient profile clearly matches trial requirements with no ambiguity",
    "trialgpt_label_mapping": {
      "methodology": "TrialGPT uses 6 labels; our 3-label mapping below",
      "inclusion_labels": "included -> MET, not included -> NOT_MET, not enough information -> UNKNOWN, not applicable -> UNKNOWN",
      "exclusion_labels": "excluded -> NOT_MET (for patient), not excluded -> MET (for patient), not enough information -> UNKNOWN, not applicable -> UNKNOWN"
    }
  }
}
```

### 4.5 Mock Model Responses

Each scenario includes canned LLM responses for parsing and cost tracking tests:

```json
{
  "model": "gemini-3-pro",
  "prompt_hash": "<sha256 of prompt>",
  "raw_text": "{\"verdict\": \"ELIGIBLE\", \"reasoning\": \"The patient meets all inclusion criteria...\"}",
  "input_tokens": 2847,
  "output_tokens": 312,
  "latency_ms": 1450,
  "estimated_cost": 0.02
}
```

### 4.6 Scenario Manifest

Each scenario's `scenario_manifest.json`:

```json
{
  "scenario_id": "sot_001_eligible",
  "description": "NSCLC patient with EGFR L858R, never-smoker, post-TKI progression — ELIGIBLE for NCT05456256",
  "expected_verdict": "ELIGIBLE",
  "expected_qrel": 2,
  "anchor_trial_nct_id": "NCT05456256",
  "data_files": {
    "ingest_input": "01_ingest_input.json",
    "ingest_gold": "02_ingest_gold_output.json",
    "prescreen_gold": "03_prescreen_gold_output.json",
    "api_query_spec": "04_api_query_spec.json",
    "api_search_response": "05_api_search_response.json",
    "trial_detail_raw": "06_trial_detail_ctgov.json",
    "trial_detail_parsed": "07_trial_detail_parsed.json",
    "validate_gold": "08_validate_gold_output.json",
    "mock_responses": "09_mock_model_responses/"
  },
  "ingest_expected_metrics": {
    "key_fact_count": 8,
    "required_key_fact_count": 6
  }
}
```

### 4.7 Aggregate Metrics

`expected_aggregate_metrics.json`:

```json
{
  "description": "Hand-computed expected metrics when all 3 scenarios are predicted correctly",
  "validate_metrics": {
    "n_scenarios": 3,
    "accuracy": 1.0,
    "per_class_distribution": {"ELIGIBLE": 1, "EXCLUDED": 1, "NOT_RELEVANT": 1},
    "macro_f1": 1.0,
    "note": "Perfect scores expected when model matches gold on all 3 harness scenarios"
  },
  "ingest_metrics": {
    "total_key_facts_across_scenarios": 24,
    "total_required_key_facts": 18,
    "note": "If model extracts all required key facts: recall=1.0, precision depends on extra extractions"
  },
  "prescreen_metrics": {
    "total_must_anchors": 3,
    "total_should_anchors": 3,
    "must_recall_target": 1.0,
    "note": "Each scenario has ~1 MUST condition + ~1 MUST biomarker"
  }
}
```

---

## 5. Harness Usage by Coding Agents

### 5.1 INGEST Development

```
Load 01_ingest_input.json
  → Run through INGEST component (understand())
  → Compare output against 02_ingest_gold_output.json
  → Verify: key_fact fields match, evidence_spans are valid substrings of input text
```

### 5.2 PRESCREEN Development

```
Load 02_ingest_gold_output.json (component isolation — gold INGEST, not model output)
  → Run through PRESCREEN component (generate_search_terms())
  → Compare output against 03_prescreen_gold_output.json
  → Verify: MUST-anchor recall = 100%, SHOULD-anchor recall >= 80%

Load 04_api_query_spec.json
  → Verify API query construction maps SearchAnchors to correct CT.gov parameters
  → Use 05_api_search_response.json as VCR cassette for API call mocking
  → Verify: expected_nct_ids appear in search results
```

### 5.3 Data Layer (Criteria Parsing)

```
Load 06_trial_detail_ctgov.json
  → Run criteria parser (parse_criteria())
  → Compare against 07_trial_detail_parsed.json
  → Verify: inclusion_criteria_count = 14, exclusion_criteria_count = 16
```

### 5.4 VALIDATE Development

```
Load 02_ingest_gold_output.json + 07_trial_detail_parsed.json
  → Run through VALIDATE component (evaluate_criterion())
  → Compare output.verdict against 08_validate_gold_output.json.verdict
  → Verify: verdict matches, reasoning references correct criteria numbers
```

### 5.5 Evaluation Module

```
Collect all 3 scenario verdicts
  → Run through evaluation module (compute_metrics())
  → Compare against expected_aggregate_metrics.json
  → Verify: accuracy, F1, confusion matrix are correctly computed
```

---

## 6. Human Expert Deliverables

| # | Deliverable | Description | Est. Effort |
|---|-------------|-------------|-------------|
| 1 | 3 patient EHR texts | NSCLC scenarios, ~100-200 words each, covering eligible/excluded/not-relevant profiles | ~30 min |
| 2 | 3 gold INGEST outputs | KeyFacts with evidence_spans, per PRD §6.1 required fields | ~1 hr |
| 3 | 3 gold SearchAnchors | MUST/SHOULD priorities, negative anchors, per PRD §6.2 | ~30 min |
| 4 | Review NCT05456256 | Confirm trial criteria are appropriate for all 3 scenarios | ~15 min |
| 5 | 3 gold VALIDATE verdicts | Verdict + reasoning + per-criterion annotation against real trial criteria | ~1 hr |
| 6 | Review API query mapping | Confirm CT.gov query parameter strategies are medically sound | ~15 min |

**Total estimated effort: ~3.5 hours**

## 7. Engineering Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | Record CT.gov API responses | Execute queries from `04_api_query_spec.json`, save to `05_api_search_response.json` and `06_trial_detail_ctgov.json` |
| 2 | Parse criteria blocks | Generate `07_trial_detail_parsed.json` from raw API response |
| 3 | Create mock model responses | Canned LLM responses matching expected output schemas |
| 4 | Compute expected metrics | Hand-calculate aggregate metrics for `expected_aggregate_metrics.json` |
| 5 | Build fixture loader | Utility to load scenario data into test framework |

---

## 8. Flaws Identified in Original Design (Resolved)

| # | Flaw | Severity | Resolution |
|---|------|----------|------------|
| 1 | VALIDATE output schema showed per-block MET/NOT_MET instead of single Verdict | Critical | Top-level `verdict` now matches pipeline's `Verdict` enum; per-block assessment moved to `annotation` |
| 2 | Missed TrialGPT published criterion-level ground truth | Critical | Documented for Phase 0/1 reuse; harness uses fictional patients so not directly applicable here |
| 3 | Trial data source only CT.gov API, not TrialGPT dataset | Critical | Criteria parsing SoT bridges both formats; `07_trial_detail_parsed.json` aligns with TrialGPT corpus.jsonl structure |
| 4 | No CT.gov API query parameter mapping | Critical | `04_api_query_spec.json` defines exact `query.cond`, `query.term=AREA[...]`, `filter.overallStatus` params |
| 5 | NOT_RELEVANT scenario used breast cancer (flow inconsistency) | Medium | Changed to NSCLC + KRAS G12C (same disease, different molecular profile — flow stays consistent) |
| 6 | Missing criteria block parsing test | Medium | Added `06_trial_detail_ctgov.json` → `07_trial_detail_parsed.json` with count verification |
| 7 | Label taxonomy not aligned with TrialGPT | Medium | Mapping documented in `trialgpt_label_mapping` field of VALIDATE gold output |
| 8 | eGFR false positive not addressed | Medium | Noted in SearchAnchors `notes` and API query spec `egfr_edge_case` |
| 9 | No expected aggregate metrics | Low | Added `expected_aggregate_metrics.json` |
| 10 | File location confusion (data/sot vs tests/fixtures) | Low | Canonical location: `data/sot/harness/`; tests can symlink or load directly |
| 11 | No mock model responses for parsing tests | Low | Added `09_mock_model_responses/` per scenario |
| 12 | INGEST input format not aligned with codebase | Low | `topic_id` + `text` fields match `Topic(topic_id, text)` Pydantic model |

---

## 9. Open Items

1. **Human expert availability** — Need oncologist to write 3 patient narratives and annotate gold outputs (~3.5 hrs). Critical path.
2. **NCT05456256 temporal drift** — Trial criteria may change. Record API response with timestamp; re-record if harness tests break.
3. **MedGemma multimodal SoT** — No medical images in current 3 scenarios. Add a 4th scenario with chest CT image when image pipeline is ready.
4. **TrialGPT dataset integration** — When `data prepare --source trialGPT` is implemented, verify that parsed output matches `07_trial_detail_parsed.json` format.
