# Two-Stage Prompt Changelog

Source of truth for the two-stage clinical trial criterion reasoning prompts.
Each version documents the exact Stage 1 (MedGemma reasoning) and Stage 2 (Gemini labeling) prompts,
along with benchmark results and lessons learned.

## Version Summary

| Version | Commit | Seed 42 Acc | Seed 123 Acc | Key Change | Outcome |
|---------|--------|-------------|--------------|------------|---------|
| v1 | 30fb2fd | 80% | — | Initial two-stage split | Beats GPT-4 75% baseline |
| v2 | 7131225 | **95%** | 75% | CWA exceptions, severity/staging Q4, explicit label mapping rules | Best ever on seed 42 |
| v3 | 2f77531 | 85% | 70% | Expanded CWA categories, negation handling, contradiction check blocks re-derivation | **REGRESSION** — contradiction check + "don't re-derive" hurt |
| v4 | c7a05ad | **95%** | TBD (needs Vertex redeploy) | Revert Stage 2 to v2-style re-derivation + add severity gating + diagnosis vs symptoms | **Recovered** — matches v2 best |

---

## v1 — Initial Two-Stage Split (commit 30fb2fd)

**Date**: 2026-02-21
**Result**: 80% accuracy, 0.796 F1, 0.697 kappa (seed 42)

### Stage 1 (MedGemma Reasoning)
- Simple 4-question format: requirement → evidence → YES/NO/INSUFFICIENT DATA → confidence
- Basic CWA: "if note doesn't mention a fact, assume not true"
- No severity/staging/negation guidance

### Stage 2 (Gemini Labeling)
- Minimal: "assign correct label based on analysis"
- Single rule: "label MUST be consistent with medical analysis"
- No explicit mapping rules for inclusion vs exclusion

### Lessons
- Two-stage already beats single-stage (80% vs 70% standalone 27B)
- Stage 2 implicitly handles exclusion logic correctly most of the time

---

## v2 — CWA Exceptions + Severity Matching (commit 7131225)

**Date**: 2026-02-22
**Result**: 95% accuracy, 0.958 F1, 0.922 kappa (seed 42) — **BEST EVER**

### Stage 1 Changes (vs v1)
1. **CWA with exceptions**: Explicit list of what CWA does NOT apply to (procedures, tests, treatments)
2. **Severity/staging/modality matching** in Q4: "Does it match SPECIFIC requirements?"
3. **Negation handling** in Q1 (implicit via "does criterion require")
4. 5 questions instead of 4 (added Q4 specifics + Q5 confidence)

### Stage 2 Changes (vs v1)
1. **Explicit LABEL MAPPING RULES** for both inclusion and exclusion criteria
2. **CONTRADICTION CHECK**: "If analysis conclusion contradicts its own reasoning, rely on the REASONING CONTENT, not the conclusion. If ambiguous, output 'unknown'"
3. Key design: Stage 2 is ALLOWED to re-derive from reasoning text to fix Stage 1 keyword errors

### Lessons
- v2 Stage 2 correctly fixed Stage 1 errors (e.g., mild vs severe in pair 8)
- Explicit mapping rules give the labeler a decision tree to follow
- Allowing re-derivation from reasoning text is a STRENGTH, not a weakness

### v2 Stage 1 Prompt (reference)
```
You are a medical expert analyzing a patient's clinical note.

Closed World Assumption (CWA):
If the note does not mention a MEDICAL CONDITION, assume the patient does NOT have it
(e.g., no mention of allergies = no known allergies; no mention of diabetes = no diabetes).

EXCEPTION — Do NOT apply CWA to:
- Procedural/safety requirements (contraception use, pregnancy tests, informed consent)
- Active treatments or scheduled interventions (currently on medication X, scheduled for surgery Y)
- Test results not yet obtained (spirometry values, specific imaging modality results)
For these, if not documented, answer INSUFFICIENT DATA.

Criterion Type: {criterion_type}
Criterion: {criterion_text}

Patient Note:
{patient_note}

Analyze this criterion against the patient note. Answer these questions:

1. What does this criterion specifically require?
   Note any severity, staging, timing, or specificity requirements.

2. What does the patient note explicitly state about this?
   Cite specific sentences by index.

3. Does the patient have the GENERAL condition described in this criterion?
   Answer: YES / NO / INSUFFICIENT DATA

4. If YES to #3: Does it match the SPECIFIC requirements?
   - Severity match? (e.g., "mild" vs "severe")
   - Type/staging match? (e.g., "Rutherford stage 2")
   - Modality/test match? (e.g., "on CT scan")
   Answer: MATCHES / DOES NOT MATCH / INSUFFICIENT DATA
   If NO to #3, skip this question.

5. How confident are you? HIGH / MEDIUM / LOW

Respond in plain text (no JSON). Focus on clinical accuracy.
```

### v2 Stage 2 Prompt (reference)
```
You are a clinical trial eligibility label assignment system.

A medical AI analyzed a patient's clinical note against an eligibility criterion.
Your task: assign the correct eligibility label based on the medical analysis.

Criterion Type: {criterion_type}
{criterion_type_instructions}

Criterion: {criterion_text}

Medical Analysis:
{stage1_reasoning}

LABEL MAPPING RULES:

For INCLUSION criteria:
- Analysis says YES + MATCHES specifics → "eligible"
- Analysis says YES + DOES NOT MATCH specifics → "not eligible"
- Analysis says YES + INSUFFICIENT DATA on specifics → "unknown"
- Analysis says NO → "not eligible"
- Analysis says INSUFFICIENT DATA → "unknown"

For EXCLUSION criteria:
- Analysis says YES (patient HAS the excluded condition) → "not eligible"
- Analysis says NO (patient does NOT have it) → "eligible"
- Analysis says INSUFFICIENT DATA → "unknown"

IMPORTANT RULES:
1. When the analysis gives a clear NO with HIGH confidence, output "not eligible" (inclusion)
   or "eligible" (exclusion). Do NOT downgrade to "unknown" unless you find a specific error.
2. CONTRADICTION CHECK: If the analysis conclusion (YES/NO) contradicts its own reasoning
   (e.g., says "NO" but reasoning describes the patient HAS the condition), rely on the
   REASONING CONTENT, not the conclusion. If ambiguous, output "unknown".

Respond ONLY with valid JSON:
{
  "label": "<eligible|not eligible|unknown>",
  "reasoning": "Brief explanation of how you mapped the clinical finding to eligibility",
  "evidence_sentences": [0, 1, 2]
}
```

---

## v3 — Expanded CWA + Contradiction Blocks Re-Derivation (commit 2f77531)

**Date**: 2026-02-22
**Result**: 85% accuracy, 0.855 F1, 0.769 kappa (seed 42) — **REGRESSION from v2**

### Stage 1 Changes (vs v2)
1. **Expanded CWA categories**: Added "Actions, behaviors, compliance", "Prior surgeries", "Lifestyle/social history"
2. **Negation parsing guidance**: "If negated, identify the UNDERLYING CONDITION being checked"
3. **N/A option** added to Q4 for criteria with no specific qualifiers

### Stage 2 Changes (vs v2)
1. **Broke contradiction check**: Changed from "rely on REASONING CONTENT" → "flag and output unknown"
2. **Added "Do NOT re-derive"**: Blocked Stage 2 from using clinical judgment
3. Added explicit DOES NOT MATCH path for exclusion (YES + DOES NOT MATCH → "eligible")

### What Regressed and Why
- **Pair 8** (mild vs severe cognitive impairment): v2 caught severity mismatch by re-deriving from text → v3 couldn't re-derive → MET (wrong)
- **Pair 15** (uterine pathology): v2 re-derived correct NOT_MET from reasoning despite wrong keyword → v3 flagged contradiction as UNKNOWN (wrong)
- **Net**: "don't re-derive" rule cost 2 correct answers, gained 0

### Lessons
- Allowing Stage 2 to re-derive from reasoning text is critical
- The contradiction check should CORRECT errors, not punt to UNKNOWN
- Stage 1's expanded CWA categories are good in principle but MedGemma-27B doesn't always follow them

---

## v4 — Severity Gating + Diagnosis Distinction + Reverted Re-Derivation (current)

**Date**: 2026-02-22
**Result**: 95% accuracy, 0.958 F1, 0.922 kappa (seed 42) — **RECOVERED from v3 regression**

### Design Rationale

v4 combines v2's working Stage 2 approach with v3's better Stage 1 structure, plus new targeted fixes:

1. **Stage 1**: Keep v3's CWA nuance and negation handling, ADD severity gating question and diagnosis vs symptoms distinction
2. **Stage 2**: REVERT to v2's "rely on REASONING CONTENT" contradiction handling, ADD severity attention rule

### Targeted Fixes

| Failure Mode | Fix Location | Change |
|-------------|-------------|--------|
| Over-inference from symptoms to diagnosis | Stage 1 Q3 | Distinguish formal diagnosis vs symptoms |
| Severity mismatch ignored | Stage 1 Q3b (new) + Stage 2 | Explicit severity gate + Stage 2 attention |
| Contradiction check backfires | Stage 2 | Revert to v2: re-derive from reasoning text |
| CWA over-application to behavioral items | Stage 1 CWA | Keep v3's expanded categories (already there) |
| Differential diagnosis not considered | Stage 1 Q3 | Add "consider alternative explanations" |

### v4 Benchmark Results (2026-02-22)

| Model Combo | Seed | Accuracy | F1-macro | Kappa | Evidence Overlap |
|-------------|------|----------|----------|-------|------------------|
| 27B + Flash v4 | 42 | **95.0%** | 0.958 | 0.922 | 30% |
| 27B + Pro v4 | 42 | **95.0%** | 0.958 | 0.922 | 20% |

- Inclusion accuracy: 100% (both)
- Exclusion accuracy: 87.5% (both)
- Only error: Pair 7 (dementia diagnosis over-inference — persistent across all versions)
- Flash and Pro perform identically on Stage 2 labeling
- v4 successfully recovered from v3 regression (85% → 95%)

### v4 Stage 1 Prompt
See `src/trialmatch/validate/evaluator.py` STAGE1_REASONING_PROMPT

### v4 Stage 2 Prompt
See `src/trialmatch/validate/evaluator.py` STAGE2_LABELING_PROMPT
