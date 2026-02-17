# Human Expert Source-of-Truth (SoT) Annotation Requirements

> Extracted from PRD v3.1 §6 and adapted for whole-criteria block evaluation.
> Last updated: 2026-02-17

---

## Overview

The benchmark evaluates three pipeline components (INGEST, PRESCREEN, VALIDATE) against human expert annotations. Without component-level ground truth, we can only evaluate end-to-end — if the system gets a trial-level label wrong, we cannot diagnose whether INGEST misunderstood the patient, PRESCREEN generated bad search terms, or VALIDATE misreasoned about criteria.

**Evaluation dataset:** TREC 2021 Clinical Trials (75 topics, 35,832 qrels, loaded via `ir_datasets`).

**Models under test:** MedGemma 1.5 4B (HF Inference) and Gemini 3 Pro (Google AI Studio).

---

## 1. INGEST SoT — Patient Profile Gold Standard

### What
For each TREC topic, a clinician produces a gold-standard structured patient profile with annotated key_facts.

### Who Annotates
Board-certified physician (oncologist preferred given TREC's cancer-heavy topic distribution). Minimum 2 annotators for inter-rater reliability on a 20% subset.

### Annotation Scope

| Phase | Topics to annotate | Purpose |
|-------|-------------------|---------|
| Phase 0 | 10 topics from TREC 2021 | Prompt tuning + initial capability check |
| Phase 1 | All 50 topics from TREC 2021 (selected subset) | Full benchmark |

### Annotation Schema (per topic)

```json
{
  "topic_id": "string",
  "annotator_id": "string",
  "annotation_date": "YYYY-MM-DD",
  "gold_key_facts": [
    {
      "field": "primary_diagnosis",
      "value": "Stage IIIA non-small cell lung cancer, adenocarcinoma",
      "evidence_span": "exact substring from topic text",
      "required": true,
      "notes": "optional clarification"
    }
  ],
  "gold_profile_text": "Free-text structured summary following INGEST template headings",
  "ambiguities": [
    {
      "field": "ecog_status",
      "note": "Topic mentions 'ambulatory' but does not state explicit ECOG score."
    }
  ]
}
```

### Required Fields

| Field | Type | Description | Evaluation Criteria |
|-------|------|-------------|-------------------|
| `primary_diagnosis` | string | Canonical disease name + subtype | Exact match or clinically equivalent synonym |
| `stage_grade` | string | TNM stage, grade, or severity | Exact match |
| `biomarkers` | list[string] | Gene mutations, receptor status, molecular markers | Set match (order-insensitive) |
| `demographics` | object | Age, sex, ethnicity (if stated) | Exact match on stated values |
| `prior_therapies` | list[string] | Treatments received, lines of therapy | Set match with partial credit |
| `comorbidities` | list[string] | Co-existing conditions | Set match |
| `labs_organ_function` | list[object] | Lab values with units | Value + unit match |
| `ecog_kps` | string | Performance status if stated | Exact match; null if not stated |
| `missing_info` | list[string] | Clinically relevant info NOT in the topic | Set overlap |

### Evaluation Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Key fact recall | % of gold key_facts correctly extracted by model | >= 90% |
| Key fact precision | % of model-extracted key_facts that match gold | >= 85% |
| Key fact F1 | Harmonic mean | >= 87% |
| Hallucination rate | % of model key_facts with no evidence in source text | <= 2% |
| Missing info recall | % of gold missing_info items identified by model | >= 50% |

### Matching Rules
1. **Automatic:** Normalize whitespace, case, common abbreviations (NSCLC = non-small cell lung cancer). Check string containment and UMLS CUI match.
2. **Clinician adjudication:** For ambiguous matches (~10-20% of comparisons), a clinician reviews and marks as match/no-match/partial.

### Estimated Effort
- Phase 0 (10 topics): ~2-3 hours
- Phase 1 (50 topics): ~10-15 hours
- Inter-rater subset (10 topics x 2 annotators): +3 hours

---

## 2. PRESCREEN SoT — Search Anchor Gold Standard

### What
For each TREC topic, a clinician + information specialist produce gold-standard search anchors.

### Why This Is Tricky
Unlike INGEST (extractive) and VALIDATE (qrels exist), PRESCREEN ground truth is **generative** — many valid sets of search terms exist. Two oncologists might generate different but equally effective search strategies.

### Who Annotates
Clinician + clinical trial search specialist (CRA or trial navigator).

### Annotation Scope

| Phase | Topics | Purpose |
|-------|--------|---------|
| Phase 0 | 10 topics (TREC 2021) | Calibrate expectations |
| Phase 1 | 25-50 topics (TREC 2021) | Full benchmark |

### Annotation Schema (per topic)

```json
{
  "topic_id": "string",
  "annotator_ids": ["clinician_id", "search_specialist_id"],
  "gold_search_anchors": {
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
        "normalized_form": "EGFR exon 21 L858R"
      }
    ],
    "interventions": [
      {
        "term": "osimertinib",
        "priority": "SHOULD",
        "notes": "Patient previously received; search for next-line options"
      }
    ],
    "constraints": {
      "age": {"value": "62", "source": "topic text"},
      "sex": {"value": "female", "source": "topic text"},
      "phase": {"value": ["PHASE2", "PHASE3"]},
      "status": {"value": ["RECRUITING"]}
    },
    "negative_anchors": [
      {
        "term": "first-line",
        "reason": "Patient has had prior therapy; first-line trials would exclude"
      }
    ]
  },
  "search_strategy_notes": "Free-text explanation of search rationale"
}
```

### Priority Levels
- **MUST:** Missing this = search is fundamentally broken (e.g., wrong disease)
- **SHOULD:** Missing this significantly degrades search quality (e.g., key biomarker)
- **NICE_TO_HAVE:** Improves precision but not critical

### Evaluation Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| MUST-anchor recall | % of MUST-priority gold anchors covered | 100% (any miss = critical failure) |
| SHOULD-anchor recall | % of SHOULD-priority gold anchors covered | >= 80% |
| Anchor precision | % of model anchors rated relevant by expert | >= 75% |
| Negative anchor compliance | % of negative anchors correctly avoided | >= 70% |
| Constraint accuracy | % of gold constraints correctly specified | >= 90% |

### Matching Rules
- Condition: UMLS CUI normalization. "NSCLC" matches "non-small cell lung cancer."
- Biomarker: Normalize gene names (HUGO), mutation notation. "EGFR L858R" matches "EGFR exon 21 L858R."
- Intervention: Map to MeSH or RxNorm. "osimertinib" matches "Tagrisso."

### Estimated Effort
- Phase 0 (10 topics): ~3-4 hours (clinician + search specialist)
- Phase 1 (25-50 topics): ~10-20 hours
- This is the highest annotation cost. Annotate 25 topics in Phase 1, extend to 50 only if signal is promising.

---

## 3. VALIDATE SoT — Block-Level Eligibility Labels

### What
TREC 2021 qrels provide trial-level labels (0=not relevant, 1=excluded, 2=eligible). These are the primary VALIDATE ground truth.

**No additional criterion-level annotation needed for the spike phase** because we use whole criteria blocks (not atomized criteria). The qrel labels directly apply.

### Block-Level Reasoning Annotation (Phase 0 only, small-scale)

For Phase 0 pairs (20 pairs), a clinician annotates the **reasoning** behind the qrel label — which inclusion/exclusion criteria drove the eligibility decision. This is NOT per-criterion MET/NOT_MET annotation; it is a rationale annotation for qualitative analysis.

### Annotation Schema (per pair)

```json
{
  "topic_id": "string",
  "nct_id": "string",
  "qrel_label": 0,
  "annotator_id": "string",
  "inclusion_assessment": {
    "overall": "MET | NOT_MET | UNKNOWN",
    "key_factors": [
      "Patient has Stage IIIA NSCLC which matches the required diagnosis",
      "EGFR L858R mutation meets the biomarker requirement"
    ],
    "blocking_factors": [],
    "confidence": 0.95
  },
  "exclusion_assessment": {
    "overall": "NOT_MET",
    "key_factors": [],
    "blocking_factors": [
      "Patient has prior immunotherapy which is listed as exclusion criterion"
    ],
    "confidence": 0.90
  },
  "overall_rationale": "Free-text explanation of why this trial is excluded for this patient",
  "difficulty": "EASY | MEDIUM | HARD",
  "difficulty_reason": "Requires understanding of prior therapy lines and their interaction with exclusion criteria"
}
```

### Trial-Level Label Mapping

| Model output | TREC qrel |
|-------------|-----------|
| ELIGIBLE (inclusion MET + exclusion NOT_MET) | 2 |
| EXCLUDED (inclusion MET + exclusion MET) | 1 |
| NOT RELEVANT (inclusion NOT_MET) | 0 |
| UNCERTAIN | 0 (for scoring) but flagged separately |

### Evaluation Metrics (against qrels)

| Metric | Definition | Target |
|--------|-----------|--------|
| Trial-level accuracy | 3-class accuracy (eligible/excluded/not-relevant) | >= 75% |
| Trial-level F1 (macro) | Macro-averaged F1 across 3 classes | >= 70% |
| Eligible recall | % of qrel-eligible trials correctly classified | >= 80% |
| Excluded precision | When model says EXCLUDED, how often is qrel=excluded? | >= 70% |

### Estimated Effort
- Phase 0 reasoning annotation (20 pairs): ~4-6 hours
- Phase 1: No additional annotation needed (qrels are the SoT)

---

## 4. Annotation Timeline

```
Week 1:   Recruit annotators. Prepare annotation tool (spreadsheet or Prodigy).
          Download TREC 2021 topics via ir_datasets.
          Select 10 topics for Phase 0.

Week 2:   Phase 0 annotation:
          - INGEST SoT: 10 topics (~3 hrs)
          - PRESCREEN SoT: 10 topics (~4 hrs)
          - VALIDATE reasoning: 20 pairs (~5 hrs)
          In parallel: build pipeline scaffolding, model adapters.

Week 3:   Run Phase 0 benchmark. Analyze results. Go/no-go decision.
          Begin Phase 1 annotation if go.

Week 4-5: Phase 1 annotation:
          - INGEST: 50 topics (~12 hrs)
          - PRESCREEN: 25-50 topics (~15 hrs)
          In parallel: run Phase 1 benchmark as annotations complete.

Week 6:   Final analysis and report.
```

**Critical path:** Annotator availability. If physician annotators are part-time (10 hrs/week), Phase 1 annotations take ~2-3 weeks.

---

## 5. Annotation Quality Assurance

### Inter-Annotator Agreement (IAA)
- Compute Cohen's kappa on the 20% dual-annotated subset for each SoT type.
- Minimum acceptable kappa: >= 0.7 for eligibility decisions, >= 0.6 for key_fact extraction.
- If kappa < threshold: review annotation guidelines, adjudicate disagreements, re-annotate.

### Adjudication Process
- Disagreements resolved by a third senior clinician.
- Adjudicated labels become the final gold standard.
- Track disagreement types: ambiguous patient info vs. annotator error vs. genuine medical disagreement.

---

## 6. Output File Locations

```
data/sot/
├── ingest/
│   ├── phase0/          # 10 topics
│   │   └── topic_{id}.json
│   └── phase1/          # 50 topics
│       └── topic_{id}.json
├── prescreen/
│   ├── phase0/          # 10 topics
│   │   └── topic_{id}.json
│   └── phase1/          # 25-50 topics
│       └── topic_{id}.json
└── validate/
    └── phase0/          # 20 pairs (reasoning annotations only)
        └── topic_{id}_nct_{id}.json
```
