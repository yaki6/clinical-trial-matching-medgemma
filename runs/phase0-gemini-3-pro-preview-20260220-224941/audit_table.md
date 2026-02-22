# Audit Table — gemini-3-pro-preview

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | MET | ✓ | The criterion excludes patients who were previously enrolled in the trial and ar |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | MET | ✓ | The exclusion criterion applies to patients with fever caused by non-infectious  |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | MET | ✓ | {   "label": "included",   "reasoning": "The criterion requires an acute exacerb |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The patient is explicitly stated to be a '21-year-old female' in sentence 0, whi |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The criterion requires the patient to be aged 18 years or older. Sentence 0 expl |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | MET | ✓ | The patient presents with shortness of breath (sentence 0), but the clinical his |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | NOT_MET | ✗ | The exclusion criterion is a diagnosis of dementia. The patient note describes a |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | MET | ✓ | The exclusion criterion applies to patients with Mild Cognitive Impairment (MCI) |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | NOT_MET | ✓ | The inclusion criterion requires participants to be experiencing symptoms of 'mi |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires the patient to be a 'non smoker'. The patient n |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes patients with respiratory distress. Sentence 4 explicitly |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion requires patients to have had a fever for more than one  |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | NOT_MET | ✓ | The inclusion criterion requires a 'Pulmonary nodule on a recent CT'. The patien |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes subjects with visible skin disease, skin conditions, or a |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | NOT_MET | ✓ | The inclusion criterion specifically requires patients to be undergoing an 'elec |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | NOT_MET | ✓ | The criterion excludes patients with evidence of uterine pathology other than le |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | NOT_MET | ✗ | {   "label": "not included",   "reasoning": "The inclusion criterion explicitly  |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | NOT_MET | ✗ | {   "label": "not included",   "reasoning": "The inclusion criterion requires th |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | NOT_MET | ✗ | The inclusion criterion requires a documented negative allergy skin test (AST).  |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | MET | ✗ | The inclusion criterion requires adequate cardiac function, noting that patients |

**Correct: 15/20 (75%)**
