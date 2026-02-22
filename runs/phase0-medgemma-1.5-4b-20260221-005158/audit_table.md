# Audit Table — medgemma-1.5-4b

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | MET | ✓ | The criterion requires the patient to have been previously enrolled in the trial |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | UNKNOWN | ✗ | The criterion requires the patient to have a fever for non-infectious diseases s |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | UNKNOWN | ✗ | The user wants me to determine if a patient is eligible for a specific inclusion |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The patient note explicitly states the patient is 21 years old. The criterion re |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The criterion is Age ≥18 years. The patient note states '0. A 52-year-old Africa |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | UNKNOWN | ✗ | The criterion requires a traumatic cause of dyspnea. The patient note does not m |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | MET | ✓ | The criterion is 'Diagnosis of Dementia'. The patient note explicitly states 'pr |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | UNKNOWN | ✗ | The criterion requires a diagnosis of Mild Cognitive Impairment (MCI). The patie |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | MET | ✗ | The criterion requires participants to be experiencing symptoms of mild cognitiv |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | MET | ✗ | The criterion requires the patient to have COPD and be a non-smoker. The patient |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | MET | ✗ | The criterion is 'Respiratory distress'. The patient note explicitly states 'On  |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | MET | ✗ | The criterion requires patients to have fever more than one week. The patient no |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | MET | ✗ | The criterion requires the presence of a pulmonary nodule on a recent CT scan. T |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | MET | ✗ | The criterion requires subjects with visible skin disease, tattoos, skin conditi |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | MET | ✗ | The criterion requires patients undergoing elective carotid endarterectomy. The  |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | MET | ✗ | The criterion requires evidence of uterine pathology other than leiomyoma. The p |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires the patient to have a Grade of COPD 2-3 (GOLD) at the lat |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires the patient to be non-visible on standard-size bronchosco |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires a negative allergy skin test (AST). The patient note does |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The criterion requires assessment of cardiac function. The patient note does not |

**Correct: 7/20 (35%)**
