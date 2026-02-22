# Audit Table — gemini-3-pro-preview

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | NOT_MET | ✗ | The patient note describes a new clinical presentation following a cardiac cathe |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | NOT_MET | ✗ | The patient presents with fever, dyspnea, and cough (Sentence 0), and a chest x- |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | UNKNOWN | ✗ | Timeout: model did not respond within 120s |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The patient is explicitly described as a '21-year-old female', which satisfies t |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | UNKNOWN | ✗ | Timeout: model did not respond within 120s |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | NOT_MET | ✗ | {   "verdict": "NOT_MET",   "reasoning": "The criterion excludes patients with a |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | MET | ✓ | The patient presents with progressive memory loss and severe cognitive deficits  |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | NOT_MET | ✗ | The eligibility criterion defines Mild Cognitive Impairment (MCI) as having cogn |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | NOT_MET | ✓ | The inclusion criterion specifically requires participants to have symptoms of ' |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires the patient to be a 'non smoker'. The patient n |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | MET | ✗ | The criterion excludes patients with respiratory distress. The patient note expl |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires patients to have had a fever for more than one  |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | UNKNOWN | ✗ | The patient note describes a 'left lung mass' found on a 'chest x-ray' (Sentence |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | MET | ✗ | The criterion excludes subjects with visible skin disease, skin condition, or ab |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires patients to be undergoing an elective carotid e |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | MET | ✗ | {   "verdict": "MET",   "reasoning": "The criterion excludes patients with evide |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The inclusion criterion requires a specific GOLD Grade (2-3) based on spirometry |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | Timeout: model did not respond within 120s |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The inclusion criterion requires the patient to have a negative allergy skin tes |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires adequate cardiac function. The patient note describes a h |

**Correct: 10/20 (50%)**
