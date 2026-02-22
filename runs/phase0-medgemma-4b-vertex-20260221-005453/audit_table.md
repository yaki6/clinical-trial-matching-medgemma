# Audit Table — medgemma-4b-vertex

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | NOT_MET | ✗ | The patient note explicitly states that the patient underwent cardiac catheteriz |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | NOT_MET | ✗ | The patient has fever for non-infectious diseases such as rheumatic autoimmune d |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | MET | ✓ | The patient presents with cough, shortness of breath, productive cough, purulent |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | NOT_MET | ✗ | The patient is 21 years old, which is less than 18 years old. Therefore, the pat |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The patient is 52 years old, which is greater than or equal to 18 years. The not |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | MET | ✓ | The patient note does not mention any traumatic cause of dyspnea. The patient's  |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | MET | ✓ | The patient note explicitly states that the patient has 'severe cognitive defici |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | NOT_MET | ✗ | The patient note states that the patient is unable to dress, bathe, use the toil |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | NOT_MET | ✓ | The patient note mentions cognitive deficits, memory dysfunction, severe cogniti |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient note explicitly states that the patient has a significant smoking hi |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient note explicitly states that the patient is in respiratory distress a |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient note states that the patient has a fever up to 39 C, dyspnea and cou |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | NOT_MET | ✓ | The patient note mentions a left lung mass on chest x-ray, but does not mention  |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | MET | ✗ | The patient note mentions lesions on the neck, varying colors from flesh-colored |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient note describes a patient who underwent cardiac catheterization and i |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | MET | ✗ | The patient note mentions an enlarged uterus (12 cm x 9 cm x 7 cms) with multipl |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The note does not mention the patient's grade of COPD according to the GOLD crit |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The patient note does not mention whether the lung mass is visible on bronchosco |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The patient note mentions a history of allergic rhinitis, but no history of whee |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The note does not mention any cardiac history, such as myocardial infarction, he |

**Correct: 12/20 (60%)**
