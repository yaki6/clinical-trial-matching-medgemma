# MedGemma 4B Reasoning Analysis: Model vs Human Expert vs GPT-4

<!-- Last updated: 2026-02-21 -->
<!-- Based on Phase 0 benchmark run: phase0-medgemma-1.5-4b-20260221-013509 -->

## Overview

This document compares MedGemma 4B's medical reasoning against human expert annotations and GPT-4 explanations from the TrialGPT HF criterion-level dataset (20-pair Phase 0 sample).

### Data Sources

| Source | Labels | Reasoning | Evidence |
|--------|--------|-----------|----------|
| Human Expert | Ground truth (MET/NOT_MET/UNKNOWN) | Not available (labels only) | Sentence indices (sparse) |
| GPT-4 | 75% accurate | Free-text explanation per pair | Not structured |
| MedGemma 4B | 35% accurate | Step-by-step reasoning chain | Sentence indices |

## Error Taxonomy

MedGemma 4B made 13 errors in 20 pairs. These cluster into 4 distinct failure modes:

### Type A: Reasoning Correct, Label Contradicts (3/13 errors)

The model's reasoning reaches the correct clinical conclusion, but the output label contradicts it. This is an **instruction-following failure**, not a reasoning failure.

#### Pair #10 — `COPD, non smoker` (inclusion, expert=NOT_MET)

| Source | Reasoning | Label |
|--------|-----------|-------|
| Expert | Patient has smoking history, does not satisfy "non smoker" | NOT_MET |
| GPT-4 | "significant smoking history contradicts non-smoker" | NOT_MET |
| **MedGemma** | "significant smoking history strongly suggests the patient is **not a non-smoker**. Therefore, the patient is **not eligible** for this criterion." | **MET** (wrong) |

The model explicitly states "not eligible" in its reasoning but outputs an `"eligible"` JSON label.

#### Pair #12 — `fever more than one week` (inclusion, expert=NOT_MET)

| Source | Reasoning | Label |
|--------|-----------|-------|
| Expert | Fever only 2 days, does not meet 1 week | NOT_MET |
| GPT-4 | "fever for 2 days, does not meet one week" | NOT_MET |
| **MedGemma** | "fever has **not lasted more than one week**" | **MET** (wrong) |

Reasoning correctly identifies 2 days < 1 week, but label says MET.

#### Pair #15 — `elective carotid endarterectomy` (inclusion, expert=NOT_MET)

| Source | Reasoning | Label |
|--------|-----------|-------|
| Expert | Note does not mention this procedure | NOT_MET |
| GPT-4 | "does not mention elective carotid endarterectomy" | NOT_MET |
| **MedGemma** | "The note does not mention an elective carotid endarterectomy" | **MET** (wrong) |

Same pattern: correct negative reasoning, wrong positive label.

**Root cause**: MedGemma 4B has insufficient instruction-following capacity to reliably map its own reasoning conclusions to the correct JSON label value. The model treats label output as partially independent from its reasoning chain.

---

### Type B: Exclusion Criteria Semantic Inversion (4/13 errors)

The model cannot handle the logical inversion required for exclusion criteria: "patient does NOT have excluded condition" → eligible (MET).

#### Pair #6 — `traumatic cause of dyspnea` (exclusion, expert=MET)

| Source | Reasoning | Label |
|--------|-----------|-------|
| Expert | No traumatic dyspnea → not excluded → MET | MET |
| GPT-4 | "not applicable, no mention of trauma" | UNKNOWN |
| **MedGemma** | "note does not mention traumatic cause → patient does not meet the criterion → **not eligible**" | NOT_MET (wrong) |

The model's clinical assessment is correct (no trauma mentioned). But it maps "does not meet the exclusion condition" to "not eligible", when the correct mapping for exclusion criteria is:
- Patient does NOT have the excluded condition → patient IS eligible → MET
- Patient HAS the excluded condition → patient is NOT eligible → NOT_MET

MedGemma consistently applies inclusion-criteria logic to exclusion criteria.

#### Pair #8 — `Mild Cognitive Impairment` (exclusion, expert=MET)

| Source | Reasoning | Label |
|--------|-----------|-------|
| Expert | Patient's condition exceeds MCI → not excluded by this criterion → MET | MET |
| GPT-4 | "condition is beyond MCI" → not excluded | MET |
| **MedGemma** | "note does not explicitly state MCI diagnosis... functional impairment not characteristic of MCI" → "not eligible" | NOT_MET (wrong) |

Correct clinical reasoning (patient has dementia-level impairment, not MCI), but wrong label direction for exclusion.

#### Pair #2 — `fever for non-infectious diseases` (exclusion, expert=MET)

MedGemma: "not eligible" → NOT_MET. Expert: not excluded → MET. Same inversion pattern.

#### Pair #16 — `uterine pathology other than leiomyoma` (exclusion, expert=NOT_MET)

MedGemma: "eligible" → MET. Expert: excluded → NOT_MET. Inverted again — the model identified the pathology correctly but mapped it to "eligible" instead of "not eligible".

**Root cause**: 4B model capacity is insufficient to maintain the logical mapping: `exclusion criterion NOT present → eligible`. The model defaults to a simpler heuristic: "condition not found → not eligible" regardless of criterion type.

---

### Type C: Medical Reasoning Errors (3/13 errors)

The model makes genuine clinical reasoning mistakes — confusing similar but distinct medical concepts.

#### Pair #9 — `mild cognitive impairment` (inclusion, expert=NOT_MET)

| Source | Reasoning | Label |
|--------|-----------|-------|
| Expert | Patient has **severe** cognitive deficits, not **mild** → NOT_MET | NOT_MET |
| GPT-4 | Incorrectly equates "severe cognitive deficits" with "symptoms of mild cognitive impairment" | MET (wrong) |
| **MedGemma** | "patient note explicitly mentions progressive memory loss and severe cognitive deficits" → eligible | MET (wrong) |

Both GPT-4 and MedGemma fail to distinguish "severe" from "mild" cognitive impairment. The expert correctly recognizes that severe cognitive deficits with functional impairment exceeds the MCI threshold, making the patient ineligible for a study requiring mild impairment.

**Clinical insight**: This is a nuanced distinction — MCI is specifically defined as cognitive decline that does NOT significantly impair daily functioning. The patient in this case cannot dress, bathe, or walk independently, which places them well beyond MCI into dementia territory.

#### Pair #13 — `Pulmonary nodule on a recent CT` (inclusion, expert=NOT_MET)

| Source | Reasoning | Label |
|--------|-----------|-------|
| Expert | Chest X-ray showed lung **mass**, not CT-detected **nodule** → NOT_MET | NOT_MET |
| GPT-4 | Notes it's X-ray not CT, considers "not enough information" | UNKNOWN |
| **MedGemma** | "lung mass is a type of pulmonary nodule" + "CT was likely performed" | MET (wrong) |

Two reasoning errors:
1. **mass ≠ nodule**: A lung mass (>3cm) is clinically distinct from a nodule (<3cm). Different workup, different differential diagnosis.
2. **Speculative inference**: Assumes CT was performed based on X-ray finding, but the note only mentions chest X-ray.

The expert applies strict literal matching (criterion says "CT", note says "X-ray"). GPT-4 partially catches this. MedGemma over-infers.

#### Pair #14 — `visible skin disease` (exclusion, expert=NOT_MET)

| Source | Reasoning | Label |
|--------|-----------|-------|
| Expert | Patient HAS skin lesions → excluded → NOT_MET | NOT_MET |
| GPT-4 | "patient has lesions, meets exclusion criterion" | NOT_MET |
| **MedGemma** | Correctly identifies skin lesions, but outputs MET | MET (wrong) |

Mixed error: correct clinical finding (skin lesions present) + exclusion criteria inversion (outputs eligible when should be not eligible).

**Root cause**: Genuine gaps in clinical concept granularity (mass vs nodule, severe vs mild) combined with over-inferencing from limited evidence.

---

### Type D: Output Truncation (3/13 errors)

The 512 max_tokens limit (TGI CUDA bug workaround) truncates MedGemma's thinking chain before it produces complete JSON output.

#### Affected pairs: #2, #3, #9 (partial)

Example — Pair #3 (`Acute exacerbation of COPD`):
- Output tokens: 1133 (thinking consumed most of the budget)
- Captured reasoning: `` ```json\n{ `` (truncated immediately)
- Parsed as: UNKNOWN (fallback)

MedGemma uses `<unused94>...<unused95>` thinking tokens before producing actual JSON. With 512 token budget, the thinking phase can exhaust the entire allocation, leaving no room for the structured response.

**Root cause**: TGI CUDA bug forces max_tokens=512. MedGemma's internal monologue tokens compete with the actual response for this limited budget.

---

## Quantitative Comparison

### Accuracy by Error Type

| Error Type | Count | Fixable by prompt? | Fixable by model upgrade? |
|------------|-------|--------------------|---------------------------|
| A: Reasoning-label contradiction | 3 | No (tried, failed) | Yes (better instruction following) |
| B: Exclusion semantic inversion | 4 | No (tried, failed) | Yes (better reasoning capacity) |
| C: Medical reasoning error | 3 | Partially (better examples) | Yes (domain fine-tuning) |
| D: Output truncation | 3 | No (hardware bug) | N/A (infrastructure issue) |

### Reasoning Quality vs Label Accuracy

| Metric | Expert | GPT-4 | MedGemma 4B |
|--------|--------|-------|-------------|
| Label accuracy | 100% | 75% | 35% |
| Reasoning-label consistency | N/A | High (85% "Correct" by TrialGPT annotation) | Low (~50% — reasoning often correct, label wrong) |
| Exclusion criteria handling | Perfect | Mostly correct | Systematic failure |
| Clinical concept precision | Distinguishes mass/nodule, severe/mild | Occasional confusion (Pair #9) | Frequent confusion + over-inference |
| Information sufficiency judgment | Conservative (UNKNOWN when unsure) | Conservative | Over-confident (defaults to MET) |
| CWA application | Correct | Correct | Inconsistent |

### Verdict Distribution Comparison

| Verdict | Ground Truth | GPT-4 | MedGemma 4B |
|---------|-------------|-------|-------------|
| MET | 8 (40%) | 8 (40%) | 14 (70%) |
| NOT_MET | 8 (40%) | 8 (40%) | 3 (15%) |
| UNKNOWN | 4 (20%) | 4 (20%) | 3 (15%) |

MedGemma's distribution is heavily skewed toward MET, while both the ground truth and GPT-4 are balanced.

### Evidence Overlap

| Model | Mean Evidence Overlap (Jaccard) |
|-------|-------------------------------|
| MedGemma 4B (native labels) | 5% |
| MedGemma 4B (eligible labels) | 35% |
| Gemini 3 Pro (native labels) | 75% |

Note: Evidence overlap improved with eligible labels because more pairs produced parseable JSON with evidence_sentences, even though accuracy did not improve.

---

## Key Takeaways

### 1. MedGemma 4B's medical reasoning is better than its accuracy suggests

At least 7 of 13 errors (Types A + B) involve correct or partially correct clinical reasoning with wrong label output. The model's **understanding** of clinical content is substantially better than its **instruction-following** on the label mapping task.

### 2. The exclusion criteria inversion is the hardest problem

The logical chain `exclusion criterion not present → patient eligible → MET` requires maintaining criterion type context through the entire reasoning chain. This is a working memory / instruction following challenge that correlates with model size.

### 3. GPT-4 and MedGemma share some reasoning blind spots

Pair #9 shows both models confusing severe cognitive deficits with mild cognitive impairment. This suggests the distinction requires clinical training data that may be underrepresented in both models' training corpora.

### 4. Label engineering cannot fix capacity limitations

The eligible/not eligible experiment (ADR-008) proved that simplifying labels does not help when the core issue is model capacity for multi-step logical reasoning with context-dependent semantics.

### 5. Path forward

| Approach | Expected Impact | Effort |
|----------|----------------|--------|
| **MedGemma 27B** (Vertex AI) | Fix Types A+B (instruction following + inversion) | Medium — deployment configured |
| **Post-processing reasoning→label alignment** | Fix Type A (extract verdict from reasoning text) | Low — regex/LLM post-processing |
| **Remove max_tokens cap** (fix TGI or use Vertex) | Fix Type D (truncation) | Low-Medium |
| **Few-shot examples in prompt** | Partially fix Type B (show exclusion examples) | Low |
| **Fine-tuning on criterion evaluation** | Fix Types B+C | High |
