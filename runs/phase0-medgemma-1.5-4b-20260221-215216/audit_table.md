# Audit Table — medgemma-1.5-4b

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | UNKNOWN | ✗ | The criterion requires determining if the patient was previously enrolled in the |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | MET | ✓ | The criterion is 'fever for non-infectious diseases such as rheumatic autoimmune |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | MET | ✓ | The criterion requires the patient to have an acute exacerbation of COPD type I  |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The patient note explicitly states the patient is a 21-year-old female. The incl |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The inclusion criterion is Age ≥18 years. The patient note explicitly states 'A  |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | MET | ✓ | The criterion is that patients presenting with a traumatic cause of dyspnea are  |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | MET | ✓ | The criterion is 'Diagnosis of Dementia'. The patient note does not explicitly s |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | MET | ✓ | {   "label": "eligible",   "reasoning": "The criterion requires a diagnosis of M |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | MET | ✗ | The criterion requires the patient to be experiencing symptoms of mild cognitive |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | MET | ✗ | The criterion is COPD, non smoker. The patient note explicitly states 'significa |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | MET | ✗ | The criterion is 'Respiratory distress'. The patient note states 'On examination |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | NOT_MET | ✓ | The user wants me to determine the eligibility of a patient for a specific inclu |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | NOT_MET | ✓ | The criterion requires a pulmonary nodule on a recent CT scan. The patient note  |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | MET | ✗ | The criterion requires subjects with visible skin disease, tattoos, skin conditi |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion requires patients undergoing elective carotid endarterectomy. The  |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | MET | ✗ | The criterion is 'Evidence of uterine pathology other than leiomyoma'. The patie |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The user wants me to determine the eligibility of a patient for a specific inclu |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | MET | ✗ | The criterion is 'non-visible on standard-size bronchoscopy'. The patient note s |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The criterion requires a negative allergy skin test (AST). The patient note does |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | MET | ✗ | ```json {   "label": "eligible",   "reasoning": "The criterion requires adequate |

**Correct: 11/20 (55%)**
