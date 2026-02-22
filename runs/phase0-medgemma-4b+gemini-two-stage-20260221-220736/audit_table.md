# Audit Table — medgemma-4b+gemini-two-stage

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | NOT_MET | ✗ | The medical analysis concludes that the patient satisfies the exclusion criterio |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | MET | ✓ | The medical analysis concludes that the patient does not satisfy the exclusion c |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | NOT_MET | ✗ | The medical analysis explicitly states that the patient does not satisfy the inc |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The patient is a 21-year-old female, which satisfies the inclusion criterion of  |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The patient is noted to be 52 years old, which meets the inclusion requirement o |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | MET | ✓ | The criterion excludes patients with dyspnea caused by trauma. The medical analy |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | NOT_MET | ✗ | The medical analysis indicates the patient has 'progressive memory loss' and 'se |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | MET | ✓ | The criterion excludes patients with Mild Cognitive Impairment (MCI), specifical |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | NOT_MET | ✓ | The inclusion criterion requires participants to have mild cognitive impairment. |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion requires the patient to be a non-smoker. The medical analysis indi |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes patients with respiratory distress. The medical analysis  |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires a fever lasting more than one week. The analysi |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | MET | ✗ | The medical analysis explicitly states that the patient satisfies the inclusion  |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient note describes 'Multiple lesions' with colors varying to 'slightly h |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion requires patients to be undergoing elective carotid endarterectomy |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient's ultrasound shows cystic areas with a differential diagnosis includ |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires a specific COPD GOLD grade (2-3) from a spirometry test.  |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis indicates that the patient note does not mention bronchosco |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires a negative allergy skin test (AST). The medical analysis  |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The inclusion criterion requires adequate cardiac function and mandates a cardio |

**Correct: 16/20 (80%)**
