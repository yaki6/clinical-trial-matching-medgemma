# Audit Table — medgemma-1.5-4b

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | UNKNOWN | ✗ | The criterion requires the patient to have been previously enrolled in the trial |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | MET | ✓ | The criterion requires the patient to have a fever and a non-infectious disease  |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | MET | ✓ | The criterion requires the patient to have an acute exacerbation of COPD type I  |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The criterion requires the patient to be older than 18 years at the time of sign |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The criterion is Age ≥18 years. The patient note explicitly states '0. A 52-year |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | MET | ✓ | The criterion requires the patient to have a traumatic cause of dyspnea. The pat |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | UNKNOWN | ✗ | The criterion requires a diagnosis of Dementia. The patient note mentions 'progr |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | MET | ✓ | ```json {   "verdict": "MET",   "reasoning": "The criterion requires a diagnosis |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | MET | ✗ | The criterion requires the patient to be experiencing symptoms of mild cognitive |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion requires the patient to have COPD and be a non-smoker. The patient |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | MET | ✗ | The criterion is 'Respiratory distress'. The patient note explicitly states 'On  |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion requires fever for more than one week. The patient note explicitly |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | NOT_MET | ✓ | The criterion requires a pulmonary nodule on a recent CT scan. The patient note  |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | MET | ✗ | The criterion requires the presence of visible skin disease, tattoos, skin condi |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion requires the patient to be undergoing an elective carotid endarter |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | MET | ✗ | The criterion requires evidence of uterine pathology other than leiomyoma. The p |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | MET | ✗ | The criterion requires the patient to have a Grade of COPD 2-3 (GOLD) at the lat |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires the patient's lung mass to be non-visible on standard-siz |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | MET | ✗ | The criterion requires a negative allergy skin test (AST). The patient note expl |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | MET | ✗ | The criterion requires adequate cardiac function. The patient note explicitly st |

**Correct: 11/20 (55%)**
