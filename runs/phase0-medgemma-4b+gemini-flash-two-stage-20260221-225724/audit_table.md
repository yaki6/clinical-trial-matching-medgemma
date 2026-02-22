# Audit Table — medgemma-4b+gemini-flash-two-stage

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | NOT_MET | ✗ | {   "label": "not eligible |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | UNKNOWN | ✗ | {   "label": |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | NOT_MET | ✗ | {   "label": "not eligible", |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The clinical note states the patient is 21 years old, which satisfies the inclus |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The patient's age is explicitly stated as 52 years, which meets the inclusion cr |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | MET | ✓ | {   "label": "eligible",   "reasoning": "The patient's dyspnea does not have a t |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | NOT_MET | ✗ | The patient has progressive memory loss and severe cognitive deficits, which sat |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | UNKNOWN | ✗ | {   "label": " |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | UNKNOWN | ✗ | {   "label": " |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient does not satisfy the 'non smoker' component of the inclusion criteri |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | {   "label": "not eligible", |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | UNKNOWN | ✗ | { |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | MET | ✗ | {   "label": "eligible", |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | UNKNOWN | ✗ | {   "label": " |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | UNKNOWN | ✗ | {   "label": "unknown", |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | UNKNOWN | ✗ | {   "label": " |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | {   "label": "unknown", |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | {   "label |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis indicates that the patient's note does not mention an aller |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | {   "label": "unknown", |

**Correct: 9/20 (45%)**
