# Audit Table — medgemma-27b+gemini-pro-two-stage-v4

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | MET | ✓ | The criterion is an exclusion for patients previously enrolled. The analysis con |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | MET | ✓ | The criterion excludes fever caused by non-infectious diseases like rheumatic au |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | MET | ✓ | The medical analysis concludes that the patient satisfies the inclusion criterio |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The patient is identified as a 21-year-old female, which satisfies the inclusion |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The patient is 52 years old, which satisfies the inclusion requirement of being  |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | MET | ✓ | The medical analysis confirms that the patient's dyspnea does not have a traumat |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | NOT_MET | ✗ | The medical analysis interprets the patient's 'severe cognitive deficits and mem |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | MET | ✓ | The exclusion criterion specifies Mild Cognitive Impairment (MCI), explicitly de |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | NOT_MET | ✓ | The criterion specifies 'mild cognitive impairment' (MCI), but the patient note  |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires the patient to be a non-smoker. The patient not |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes patients with respiratory distress. The medical analysis  |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion requires a fever duration of more than one week. The analysis indi |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | NOT_MET | ✓ | The analysis concludes NO because the patient has a lung mass identified on a ch |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes subjects with visible skin conditions. The medical analys |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The patient is presenting with complications following a cardiac catheterization |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes patients with uterine pathology other than leiomyoma. The |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis explicitly states that the patient note does not provide sp |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The clinical note does not provide information regarding whether the lung mass i |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The analysis indicates that the patient note does not mention any allergy skin t |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The medical analysis explicitly states there is insufficient data because the pa |

**Correct: 19/20 (95%)**
