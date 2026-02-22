# Audit Table — medgemma-4b+gemini-pro-two-stage

| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |
|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|
| 1 | sigir-201410 | NCT02097186 | excl | Patients previously enrolled in the trial representing for a | MET | MET | UNKNOWN | ✗ | Error: (Request ID: xRnKlV)  Bad request: CUDA error: misaligned address CUDA ke |
| 2 | sigir-20142 | NCT02618655 | excl | fever for non-infectious diseases such as rheumatic autoimmu | MET | UNKNOWN | UNKNOWN | ✗ | Error: (Request ID: LQ-hOq)  Bad request: CUDA error: misaligned address CUDA ke |
| 3 | sigir-201423 | NCT00170222 | incl | Acute exacerbation of COPD type I or II according to GOLD | MET | UNKNOWN | UNKNOWN | ✗ | Error: (Request ID: _ZaPSP)  Bad request: CUDA error: misaligned address CUDA ke |
| 4 | sigir-201421 | NCT00997100 | incl | Age > 18 years at the time of signing the informed consent f | MET | MET | MET | ✓ | The patient is explicitly identified as a 21-year-old female in the note, which  |
| 5 | sigir-201419 | NCT02509286 | incl | Age ≥18 years | MET | MET | MET | ✓ | The criterion requires the patient to be at least 18 years old. The analysis ide |
| 6 | sigir-201413 | NCT00163709 | excl | Patients presenting with a traumatic cause of dyspnea | MET | UNKNOWN | MET | ✓ | The criterion excludes patients with dyspnea caused by trauma. The medical analy |
| 7 | sigir-20148 | NCT01519271 | excl | Diagnosis of Dementia | MET | MET | MET | ✓ | The criterion excludes patients with a specific 'Diagnosis of Dementia'. The med |
| 8 | sigir-201520 | NCT00880347 | excl | Diagnostic of Mild Cognitive Impairment defined by subjectiv | MET | MET | MET | ✓ | The criterion excludes patients with Mild Cognitive Impairment (MCI), explicitly |
| 9 | sigir-20148 | NCT01519271 | incl | Participants must be experiencing symptoms of mild cognitive | NOT_MET | MET | UNKNOWN | ✗ | Error: (Request ID: g2ORMq)  Bad request: Bad Request: The endpoint is paused, a |
| 10 | sigir-201519 | NCT00806091 | incl | COPD, non smoker | NOT_MET | NOT_MET | UNKNOWN | ✗ | Error: (Request ID: Deol1B)  Bad request: Bad Request: The endpoint is paused, a |
| 11 | sigir-20142 | NCT00711399 | excl | Respiratory distress | NOT_MET | NOT_MET | UNKNOWN | ✗ | Error: (Request ID: Cj8VwR)  Bad request: Bad Request: The endpoint is paused, a |
| 12 | sigir-20142 | NCT02618655 | incl | patients have fever more than one week | NOT_MET | NOT_MET | UNKNOWN | ✗ | Error: (Request ID: m2HujJ)  Bad request: Bad Request: The endpoint is paused, a |
| 13 | sigir-20143 | NCT02490059 | incl | Pulmonary nodule on a recent CT | NOT_MET | UNKNOWN | UNKNOWN | ✗ | Error: (Request ID: ahC6u4)  Bad request: Bad Request: The endpoint is paused, a |
| 14 | sigir-20149 | NCT02615912 | excl | Subjects with visible skin disease, tattoos, skin condition, | NOT_MET | NOT_MET | UNKNOWN | ✗ | Error: (Request ID: dyhKmV)  Bad request: Bad Request: The endpoint is paused, a |
| 15 | sigir-201410 | NCT02097186 | incl | Patients undergoing elective carotid endarterectomy | NOT_MET | NOT_MET | UNKNOWN | ✗ | Error: (Request ID: mKQoGD)  Bad request: Bad Request: The endpoint is paused, a |
| 16 | sigir-201415 | NCT00180739 | excl | 13. Evidence of uterine pathology other than leiomyoma | NOT_MET | NOT_MET | UNKNOWN | ✗ | Error: (Request ID: Ndh7lT)  Bad request: Bad Request: The endpoint is paused, a |
| 17 | sigir-201519 | NCT02213809 | incl | Grade of COPD 2-3 (GOLD) at the latest spirometry completed  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The provided medical analysis is truncated and cuts off before describing the pa |
| 18 | sigir-20143 | NCT02490059 | incl | non-visible on standard-size bronchoscopy | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires the condition to be non-visible on standard-size bronchos |
| 19 | sigir-201516 | NCT00839124 | incl | Negative allergy skin test (AST) | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires a negative allergy skin test (AST). The medical analysis  |
| 20 | sigir-201419 | NCT02509286 | incl | Adequate cardiac function ( Patients with a cardiac history  | UNKNOWN | UNKNOWN | UNKNOWN | ✓ | The criterion requires adequate cardiac function, with a specific requirement fo |

**Correct: 9/20 (45%)**
