# Audit Table — medgemma-1.5-4b

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | MET | ✓ | The patient's clinical note indicates that she underwent a cardiac catheterizati |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | UNKNOWN | ✗ | The patient presents with a fever (sentence 0). The criterion is for fever assoc |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | MET | ✓ | The patient presents with symptoms consistent with an acute exacerbation of COPD |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The patient note states that the patient is a 21-year-old female. The eligibilit |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The patient note explicitly states the patient is a '52-year-old man'. The eligi |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | MET | ✓ | The patient's clinical note explicitly states that she presents with shortness o |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | MET | ✓ | The patient note explicitly states 'severe cognitive deficits and memory dysfunc |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | NOT_MET | ✗ | The patient exhibits subjective complaints from his wife and son regarding cogni |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | MET | ✗ | The patient presents with progressive memory loss and severe cognitive deficits, |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient has a significant smoking history (1 to 2 packs per day for 47 years |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | MET | ✗ | The patient note explicitly states that the patient is 'in respiratory distress' |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient note states the patient has had fever for '2 days'. The eligibility  |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | NOT_MET | ✓ | The criterion requires a pulmonary nodule on a recent CT scan. The patient note  |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | MET | ✗ | The patient has multiple lesions on her neck. The description of the lesions (sm |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient note states the patient underwent cardiac catheterization, not a car |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | MET | ✗ | thought The user wants me to determine if the patient meets the exclusion criter |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The patient has a significant smoking history and symptoms suggestive of COPD (c |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The patient has a left lung mass, as stated in sentence 0. However, the clinical |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The patient note does not contain any information regarding whether an allergy s |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | MET | ✗ | The patient's clinical note does not mention any history of cardiac problems suc |

**Correct: 13/20 (65%)**
